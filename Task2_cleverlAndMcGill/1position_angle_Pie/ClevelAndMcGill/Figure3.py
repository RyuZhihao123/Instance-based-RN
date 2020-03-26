import numpy as np
import os
import cv2
import sys
class Figure3:
    SIZE = (100, 100)

    @staticmethod
    def AddNoise(image,noises):
        image = image + noises
        _min = image.min()
        _max = image.max()
        image -= _min
        image /= (_max - _min)
        return image

    @staticmethod
    def generate_datapoint():

        def randomize_data():
            max = 36;
            min = 3;
            diff = 0.1;

            d = []

            while len(d) < 5:
                randomnumber = np.ceil(np.random.random() * 36 + 3);
                found = False;
                for i in range(len(d)):
                    if not ensure_difference(d, randomnumber):
                        found = True
                        break

                if not found:
                    d.append(randomnumber)

            return d;

        def ensure_difference(A, c):
            result = True;
            for i in range(len(A)):
                if c > (A[i] - 3) and c < (A[i] + 3):
                    result = False

            return result

        sum = -1
        while (sum != 100):
            data = randomize_data()
            sum = data[0] + data[1] + data[2] + data[3] + data[4]

        labels = np.zeros((5), dtype=np.float32)
        for i, d in enumerate(data):
            labels[i] = d / float(np.max(data))

        labels = np.roll(labels, 5 - np.where(labels == 1)[0])

        return data, list(labels)

    @staticmethod
    def data_to_barchart(data):
        '''
        '''
        barchart = np.ones((100, 100), dtype=np.float32)
        subcharts = []
        for i, d in enumerate(data):
            sub = np.ones((100, 100), dtype=np.float32)

            if i == 0:
                start = 2
            else:
                start = 0

            left_bar = start + 3 + i * 3 + i * 16
            right_bar = 3 + i * 3 + i * 16 + 16

            cv2.line(barchart,(left_bar, 99), (left_bar,99 - int(d)),0,1)
            cv2.line(barchart, (right_bar, 99), (right_bar, 99 - int(d) ),0,1)
            cv2.line(barchart, (left_bar,99 - int(d)), (right_bar,99 - int(d) ),0,1)

            cv2.line(sub,(left_bar, 99), (left_bar,99 - int(d)),0,1)
            cv2.line(sub, (right_bar, 99), (right_bar, 99 - int(d) ),0,1)
            cv2.line(sub, (left_bar,99 - int(d)), (right_bar,99 - int(d) ),0,1)

            if d == np.max(data):
                # mark the max
                barchart[90:91, left_bar + 8:left_bar + 9] = 0
                sub[90:91, left_bar + 8:left_bar + 9] = 0
            subcharts.append(sub)

        noises = np.random.uniform(0, 0.05, (100, 100))
        barchart = Figure3.AddNoise(barchart,noises)
        for i in range(len(subcharts)):
            subcharts[i] = Figure3.AddNoise(subcharts[i],noises)

        return barchart, subcharts

    @staticmethod
    def data_to_piechart_aa(data):
        '''
        '''
        # print('data', data/np.max(data))
        piechart = np.ones((100, 100), dtype=np.float32)
        RADIUS = 30
        # rr, cc, val = skimage.draw.circle_perimeter_aa(50, 50, RADIUS)
        # piechart[rr, cc] = val
        cv2.ellipse(piechart,(50,50),(RADIUS,RADIUS),0,0,360.0,0)
        random_direction = np.random.randint(2,0.8*(data[0] / 100.) * 360.)
        theta = -(np.pi / 180.0) * random_direction
        END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
        # rr, cc, val = skimage.draw.line_aa(50, 50, int(np.round(END[0])), int(np.round(END[1])))
        # piechart[rr, cc] = val
        cv2.line(piechart, (50, 50), (int(np.round(END[0])), int(np.round(END[1]))),0,1)

        subcharts = [np.ones((100, 100), dtype=np.float32) for i in range(len(data))]

        all_angles = np.zeros(len(data))

        for i, d in enumerate(data):

            current_value = data[i]
            current_angle = (current_value / 100.) * 360.

            theta = -(np.pi / 180.0) * (random_direction - current_angle)
            END = (50 - RADIUS * np.cos(theta), 50 - RADIUS * np.sin(theta))
            cv2.line(piechart, (50, 50), (int(np.round(END[0])), int(np.round(END[1]))), 0, 1)
            cv2.line(subcharts[i], (50, 50), (int(np.round(END[0])), int(np.round(END[1]))), 0, 1)
            cv2.line(subcharts[(i+1)%len(data)], (50, 50), (int(np.round(END[0])), int(np.round(END[1]))), 0, 1)
            all_angles[i] = 180.0*theta/np.pi
            if d == np.max(data):
                # this is the max spot
                theta = -(np.pi / 180.0) * (random_direction - current_angle / 2.)
                END = (50 - RADIUS / 2 * np.cos(theta), 50 - RADIUS / 2 * np.sin(theta))
                cv2.line(piechart, (int(np.round(END[0])), int(np.round(END[1]))),
                         (int(np.round(END[0])),int(np.round(END[1]))), 0, 1)
                cv2.line(subcharts[i], (int(np.round(END[0])), int(np.round(END[1]))),
                         (int(np.round(END[0])), int(np.round(END[1]))), 0, 1)

            random_direction -= current_angle
        # print('angle',all_angles)
        for i, d in enumerate(data):
            if i !=0 :
                cv2.ellipse(subcharts[i],(50,50),(RADIUS,RADIUS),180.,all_angles[(i-1)%len(data)], all_angles[i],0,1)
            if i ==0 :
                # print(all_angles[(i-1)%len(data)], all_angles[i])
                cv2.ellipse(subcharts[i],(50,50),(RADIUS,RADIUS),180.,all_angles[(i-1)%len(data)]-360., all_angles[i],0,1)

        noises = np.random.uniform(0, 0.05, (100, 100))
        piechart = Figure3.AddNoise(piechart,noises)
        for i in range(len(subcharts)):
            subcharts[i] = Figure3.AddNoise(subcharts[i],noises)

        return piechart,subcharts

# data, labels = Figure3.generate_datapoint()
# chart, subs = Figure3.data_to_barchart(data)
# subchart = np.hstack(subs)
# print(np.roll(labels, np.where(data==np.max(data))[0]))
# cv2.imshow("aaa",chart)
# cv2.imshow("bbb",subchart)
# cv2.waitKey(0)

# data, labels = Figure3.generate_datapoint()
# chart,subs = Figure3.data_to_piechart_aa(data)
# subchart = np.hstack(subs)
# print(np.roll(labels, np.where(data==np.max(data))[0]))
# cv2.imshow("aaa",chart)
# cv2.imshow("bbb",subchart)
# cv2.waitKey(0)