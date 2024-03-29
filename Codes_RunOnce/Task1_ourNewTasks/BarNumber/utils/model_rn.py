from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
try:
    import tfplot
except:
    pass

from utils.ops import conv2d, fc
from utils.util import log
import numpy as np
from sklearn.metrics import mean_squared_error

class ModelRN(object):

    def __init__(self,learning_rate = 0.0005, batch_size = 64, c_dim = 1, a_dim = 6,
                 debug_information=False,
                 is_train=True):
        self.debug = debug_information

        self.batch_size = batch_size
        self.img_size = 100
        self.c_dim = c_dim
        self.a_dim = a_dim
        self.conv_info = np.array([24, 24, 24, 24])
        self.learning_rate = learning_rate

        # create placeholders for the input
        # (image, question) -> (answer)
        self.img = tf.placeholder(
            name='input_image', dtype=tf.float32,
            shape=[self.batch_size, self.img_size, self.img_size, self.c_dim],
        )
        self.a = tf.placeholder(
            name='a', dtype=tf.float32, shape=[self.batch_size, self.a_dim],
        )

        self.is_training = tf.placeholder_with_default(bool(is_train), [], name='is_training')

        self.build(is_train=is_train)

    def get_feed_dict(self, batch_chunk, step=None, is_training=None):
        fd = {
            self.img: batch_chunk['img'],  # [B, h, w, c]
            self.q: batch_chunk['q'],  # [B, n]
            self.a: batch_chunk['a'],  # [B, m]
        }
        if is_training is not None:
            fd[self.is_training] = is_training

        return fd

    def build(self, is_train=True):

        n = self.a_dim
        conv_info = self.conv_info

        # build loss and accuracy {{{
        def build_loss(logits, labels):
            # Cross-entropy loss
            #loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)  # [lzh]
            loss = tf.reduce_mean(tf.square(logits-labels))

            # Classification accuracy
            #correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.square(logits-labels))
            return tf.reduce_mean(loss), accuracy
        # }}}

        def concat_coor(o, i, d):
            coor = tf.tile(tf.expand_dims(
                [float(int(i / d)) / d, (i % d) / d], axis=0), [self.batch_size, 1])
            o = tf.concat([o, tf.to_float(coor)], axis=1)
            return o

        def g_theta(o_i, o_j, scope='g_theta', reuse=True):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                if not reuse: log.warn(scope.name)
                g_1 = fc(tf.concat([o_i, o_j], axis=1), 256, name='g_1')
                g_2 = fc(g_1, 256, name='g_2')
                g_3 = fc(g_2, 256, name='g_3')
                g_4 = fc(g_3, 256, name='g_4')
                return g_4

        # Classifier: takes images as input and outputs class label [B, m]
        def CONV(img, scope='CONV'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                conv_1 = conv2d(img, conv_info[0], is_train, s_h=3, s_w=3, name='conv_1')
                conv_2 = conv2d(conv_1, conv_info[1], is_train, s_h=3, s_w=3, name='conv_2')
                conv_3 = conv2d(conv_2, conv_info[2], is_train, name='conv_3')
                conv_4 = conv2d(conv_3, conv_info[3], is_train, name='conv_4')

                # eq.1 in the paper
                # g_theta = (o_i, o_j, q)
                # conv_4 [B, d, d, k]
                d = conv_4.get_shape().as_list()[1]
                all_g = []
                for i in range(d*d):
                    o_i = conv_4[:, int(i / d), int(i % d), :]
                    o_i = concat_coor(o_i, i, d)
                    for j in range(d*d):
                        o_j = conv_4[:, int(j / d), int(j % d), :]
                        o_j = concat_coor(o_j, j, d)
                        if i == 0 and j == 0:
                            g_i_j = g_theta(o_i, o_j, reuse=False)
                        else:
                            g_i_j = g_theta(o_i, o_j,reuse=True)
                        all_g.append(g_i_j)

                all_g = tf.stack(all_g, axis=0)
                all_g = tf.reduce_mean(all_g, axis=0, name='all_g')
                return all_g

        def f_phi(g,output_dim, scope='f_phi'):
            with tf.variable_scope(scope) as scope:
                log.warn(scope.name)
                fc_1 = fc(g, 256, name='fc_1')
                fc_2 = fc(fc_1, 256, name='fc_2')
                fc_2 = slim.dropout(fc_2, keep_prob=0.5, is_training=is_train, scope='fc_3/')
                fc_3 = fc(fc_2, output_dim, activation_fn=None, name='fc_3')
                return fc_3

        g = CONV(self.img, scope='CONV')
        logits = f_phi(g, self.a_dim, scope='f_phi')

        self.all_preds = logits

        self.loss, self.accuracy = build_loss(logits, self.a)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def Run_one_batch(self,sess,x_batch,y_batch):
        _,l = sess.run([self.train_step,self.loss], feed_dict={self.img: x_batch, self.a: y_batch})
        return l


    def GetTotalLoss(self,sess, x_data,y_data, max_num):
        m_batchSize = self.batch_size
        batch_amount = max_num // m_batchSize

        # delta = []
        #
        # for bid in range(batch_amount):
        #     x_batch = x_data[bid * m_batchSize: (bid + 1) * m_batchSize]
        #     y_batch = y_data[bid * m_batchSize: (bid + 1) * m_batchSize]
        #
        #     pred_batch = sess.run(self.all_preds, feed_dict={self.img: x_batch, self.a: y_batch})
        #
        #     delta.append(y_batch-pred_batch)
        #
        # preds = np.vstack(delta)
        # # print(preds.shape)
        # total_loss = sess.run(tf.reduce_mean(tf.square(preds)))
        #
        # return total_loss
        preds = []

        for bid in range(batch_amount):
            x_batch = x_data[bid * m_batchSize: (bid + 1) * m_batchSize]
            y_batch = y_data[bid * m_batchSize: (bid + 1) * m_batchSize]

            pred_batch = sess.run(self.all_preds, feed_dict={self.img: x_batch, self.a: y_batch})

            preds.append(pred_batch)

        preds = np.vstack(preds)
        # print(preds.shape)
        total_loss = mean_squared_error(preds, y_data[:preds.shape[0]])

        return total_loss

    def GetPredictions(self,sess, x_data,y_data, max_num):
        m_batchSize = self.batch_size
        batch_amount = max_num // m_batchSize

        results = []

        for bid in range(batch_amount):
            x_batch = x_data[bid * m_batchSize: (bid + 1) * m_batchSize]
            y_batch = y_data[bid * m_batchSize: (bid + 1) * m_batchSize]

            pred_batch = sess.run(self.all_preds, feed_dict={self.img: x_batch, self.a: y_batch})

            results.append(pred_batch)

        preds = np.vstack(results)
        return preds



