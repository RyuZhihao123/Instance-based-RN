import numpy as np
import cv2

class Figure4:

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
        pairs = [10. * 10. ** ((i - 1.) / 12.) for i in range(1, 11)]
        value_A = np.random.choice(pairs)
        value_B = value_A
        while value_B == value_A:
            value_B = np.random.choice(pairs)

        data = [np.round(value_A), np.round(value_B)]

        if (value_A < value_B):
            ratio = np.round(value_A) / np.round(value_B)
        else:
            ratio = np.round(value_B) / np.round(value_A)

        label = ratio

        return data, label

    @staticmethod
    def data_to_type1(data):
        barchart = np.ones((100, 100))
        barchart1 = np.ones((100,100))
        barchart2 = np.ones((100,100))

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(10, 93)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = data[1]  # fixed pos 2
        all_values[3] = np.random.randint(10, 93)
        all_values[4] = np.random.randint(10, 93)
        all_values[5] = np.random.randint(10, 93)
        all_values[6] = np.random.randint(10, 93)
        all_values[7] = np.random.randint(10, 93)
        all_values[8] = np.random.randint(10, 93)
        all_values[9] = np.random.randint(10, 93)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i * gap + i * b_width
            right_bar = start + i * gap + i * b_width + b_width

            # print left_bar, right_bar
            cv2.rectangle(barchart,(left_bar,99),(right_bar,99 - int(d)),0,1)

            if i == 1 or i == 2:
                barchart[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) +1] = 0

            if i == 1:
                cv2.rectangle(barchart1,(left_bar,99),(right_bar,99 - int(d)),0,1)
                barchart1[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) +1] = 0
            if i == 2:
                cv2.rectangle(barchart2,(left_bar,99),(right_bar,99 - int(d)),0,1)
                barchart2[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) + 1] = 0

        noises = np.random.uniform(0, 0.05, (100, 100))

        barchart = Figure4.AddNoise(barchart,noises)
        barchart1 = Figure4.AddNoise(barchart1,noises)
        barchart2 = Figure4.AddNoise(barchart2,noises)

        return barchart,barchart1,barchart2

    @staticmethod
    def data_to_type3(data):
        barchart = np.ones((100, 100))
        barchart1 = np.ones((100,100))
        barchart2 = np.ones((100,100))
        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(10, 93)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = np.random.randint(10, 93)
        all_values[3] = np.random.randint(10, 93)
        all_values[4] = np.random.randint(10, 93)
        all_values[5] = np.random.randint(10, 93)
        all_values[6] = data[1]  # fixed pos 2
        all_values[7] = np.random.randint(10, 93)
        all_values[8] = np.random.randint(10, 93)
        all_values[9] = np.random.randint(10, 93)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i * gap + i * b_width
            right_bar = start + i * gap + i * b_width + b_width
            cv2.rectangle(barchart,(left_bar,99),(right_bar,99 - int(d)),0,1)

            if i == 1 or i == 6:
                barchart[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) + 1] = 0
            if i == 1:
                cv2.rectangle(barchart1,(left_bar,99),(right_bar,99 - int(d)),0,1)
                barchart1[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) +1] = 0
            if i == 6:
                cv2.rectangle(barchart2,(left_bar,99),(right_bar,99 - int(d)),0,1)
                barchart2[94:95, int(left_bar + b_width / 2):int(left_bar + b_width / 2) + 1] = 0

        noises = np.random.uniform(0, 0.05, (100, 100))

        barchart = Figure4.AddNoise(barchart,noises)
        barchart1 = Figure4.AddNoise(barchart1,noises)
        barchart2 = Figure4.AddNoise(barchart2,noises)

        return barchart,barchart1,barchart2

    @staticmethod
    def data_to_type2(data):
        '''
        '''
        barchart = np.ones((100, 100))
        barchart1 = np.ones((100, 100))
        barchart2 = np.ones((100, 100))
        # we build the barchart to the top
        all_values = [0] * 10
        all_values[0] = data[0]  # fixed pos but max. 56
        current_max = 93 - all_values[0]
        all_values[1] = np.random.randint(8, current_max / 4.)
        all_values[2] = np.random.randint(8, current_max / 4.)
        all_values[3] = np.random.randint(8, current_max / 4.)
        all_values[4] = np.random.randint(8, current_max / 4.)
        current_max = np.sum(all_values[0:5])

        # draw left, right of the left stacked barchart
        cv2.line(barchart,(10, 99), (10, 99 - int(current_max)),0)
        cv2.line(barchart,(40, 99), (40, 99 - int(current_max)),0)
        current = 0
        for i, d in enumerate(all_values):

            cv2.line(barchart, (10, 99 - (int(d) + current)), (40,99 - (int(d) + current)),0)
            if i == 0:
                cv2.rectangle(barchart1, (10, 99 - (current)), (40, 99 - (int(d) + current)), 0)

            current += int(d)
            if i == 0:
                barchart[int(99 - int(d) / 2):int(99 - int(d) / 2) + 1, 25:26] = 0
                barchart1[int(99 - int(d) / 2):int(99 - int(d) / 2) + 1, 25:26] = 0

        all_values[5] = data[1]  # fixed pos but max. 56
        current_max = 93 - all_values[5]
        all_values[6] = np.random.randint(8, current_max / 4.)
        all_values[7] = np.random.randint(8, current_max / 4.)
        all_values[8] = np.random.randint(8, current_max / 4.)
        all_values[9] = np.random.randint(8, current_max / 4.)
        current_max = np.sum(all_values[5:])
        # draw left, right of the left stacked barchart
        cv2.line(barchart, (60, 99), (60, 99 - int(current_max)),0)
        cv2.line(barchart, (90, 99), (90, 99 - int(current_max)),0)

        current = 0
        for i, d in enumerate(all_values[5:]):

            cv2.line(barchart, (60,99 - (int(d) + current)), (90, 99 - (int(d) + current)), 0)
            if i == 0:
                cv2.rectangle(barchart2, (60,99 - (current)), (90, 99 - (int(d) + current)), 0)
            current += int(d)
            if i == 0:
                barchart[int(99 - int(d) / 2):int(99 - int(d) / 2 + 1), 75:76] = 0
                barchart2[int(99 - int(d) / 2):int(99 - int(d) / 2 + 1), 75:76] = 0
        noises = np.random.uniform(0, 0.05, (100, 100))

        barchart = Figure4.AddNoise(barchart,noises)
        barchart1 = Figure4.AddNoise(barchart1,noises)
        barchart2 = Figure4.AddNoise(barchart2,noises)
        return barchart,barchart1,barchart2

    @staticmethod
    def data_to_type4(data):
        barchart = np.ones((100, 100))
        barchart1 = np.ones((100, 100))
        barchart2 = np.ones((100, 100))
        # we build the barchart to the top
        all_values = [0] * 10
        current_max = 93 - data[0]
        all_values[0] = np.random.randint(8, current_max / 4.)
        all_values[1] = np.random.randint(8, current_max / 4.)
        all_values[2] = np.random.randint(8, current_max / 4.)
        all_values[3] = np.random.randint(8, current_max / 4.)
        below_last_sum = np.sum(all_values[0:4])
        all_values[4] = data[0]
        current_max = np.sum(all_values[0:5])
        above_last_sum = current_max

        # draw left, right of the left stacked barchart
        cv2.line(barchart, (10,99), (10, 99 - int(current_max)), 0)
        cv2.line(barchart, (40,99), (40, 99 - int(current_max)), 0)

        current = 0
        for i, d in enumerate(all_values):

            cv2.line(barchart, (10,99 - (int(d) + current)), (40, 99 - (int(d) + current)), 0)
            if i == 4:
                cv2.rectangle(barchart1, (10, 99 - (current)), (40, 99 - (int(d) + current)), 0)
            current += int(d)
            if i == 4:
                barchart[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 25:26] = 0
                barchart1[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 25:26] = 0


        below_last_sum2 = below_last_sum
        above_last_sum2 = above_last_sum

        ctr = 0
        while below_last_sum2 == below_last_sum or above_last_sum2 == above_last_sum:
            if ctr == 20:
                raise Exception()

            current_max = 93 - data[1]
            all_values[5] = np.random.randint(8, current_max / 4.)
            all_values[6] = np.random.randint(8, current_max / 4.)
            all_values[7] = np.random.randint(8, current_max / 4.)
            all_values[8] = np.random.randint(8, current_max / 4.)
            below_last_sum2 = np.sum(all_values[5:9])
            all_values[9] = data[1]

            current_max = np.sum(all_values[5:])
            above_last_sum2 = current_max

            ctr += 1  # exception counter

        # draw left, right of the left stacked barchart
        cv2.line(barchart, (60, 99), (60, 99 - int(current_max)), 0)
        cv2.line(barchart, (90, 99), (90, 99 - int(current_max)), 0)
        current = 0
        for i, d in enumerate(all_values[5:]):

            cv2.line(barchart, (60, 99 - (int(d) + current)), (90, 99 - (int(d) + current)), 0)
            if i==4:
                cv2.rectangle(barchart2, (60, 99 - (current)), (90, 99 - (int(d) + current)), 0)
            current += int(d)
            if i == 4:
                barchart[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 75:76] = 0
                barchart2[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 75:76] = 0

        noises = np.random.uniform(0, 0.05, (100, 100))

        barchart = Figure4.AddNoise(barchart,noises)
        barchart1 = Figure4.AddNoise(barchart1,noises)
        barchart2 = Figure4.AddNoise(barchart2,noises)
        return barchart,barchart1,barchart2

    @staticmethod
    def data_to_type5(data):

        barchart = np.ones((100, 100))
        barchart1 = np.ones((100, 100))
        barchart2 = np.ones((100, 100))
        all_values = [0] * 10
        current_max = 93 - data[0] - data[1]
        if current_max <= 9:
            raise Exception('Out of bounds')

        all_values[0] = np.random.randint(3, current_max / 3.)
        all_values[1] = np.random.randint(3, current_max / 3.)
        all_values[2] = np.random.randint(3, current_max / 3.)
        all_values[3] = data[0]
        all_values[4] = data[1]
        current_max = np.sum(all_values[0:5])

        # draw left, right of the left stacked barchart
        cv2.line(barchart,(10, 99), (10, 99 - int(current_max)), 0)
        cv2.line(barchart,(40, 99), (40, 99 - int(current_max)), 0)

        current = 0
        for i, d in enumerate(all_values):

            cv2.line(barchart,(10, 99 - (int(d) + current)), (40, 99 - (int(d) + current)), 0)
            if i == 3:
                cv2.rectangle(barchart1,(10, 99 - (current)),(40, 99 - (int(d) + current)), 0)
            if i == 4:
                cv2.rectangle(barchart2, (10, 99 - (current)), (40, 99 - (int(d) + current)), 0)
            current += int(d)
            if i == 3 or i == 4:
                barchart[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 25:26] = 0
            if i == 3:
                barchart1[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 25:26] = 0
            if i == 4:
                barchart2[int(99 - current + (int(d) / 2)):int(99 - current + (int(d) / 2) + 1), 25:26] = 0
        current_max = 93
        all_values[5] = np.random.randint(8, current_max / 5.)
        all_values[6] = np.random.randint(8, current_max / 5.)
        all_values[7] = np.random.randint(8, current_max / 5.)
        all_values[8] = np.random.randint(8, current_max / 5.)
        all_values[9] = np.random.randint(8, current_max / 5.)
        current_max = np.sum(all_values[5:])

        # draw left, right of the left stacked barchart
        cv2.line(barchart, (60, 99), (60, 99 - int(current_max)), 0)
        cv2.line(barchart, (90, 99), (90, 99 - int(current_max)), 0)

        current = 0
        for i, d in enumerate(all_values[5:]):
            cv2.line(barchart, (60, 99 - (int(d) + current)), (90, 99 - (int(d) + current)), 0)
            current += int(d)
        noises = np.random.uniform(0, 0.05, (100, 100))

        barchart = Figure4.AddNoise(barchart,noises)
        barchart1 = Figure4.AddNoise(barchart1,noises)
        barchart2 = Figure4.AddNoise(barchart2,noises)
        return barchart,barchart1,barchart2

    @staticmethod
    def data_to_type6(data):
        choices = ['Figure4.data_to_type1', 'Figure4.data_to_type2',
                   'Figure4.data_to_type3', 'Figure4.data_to_type4', 'Figure4.data_to_type5']

        choice = np.random.choice(choices)

        return eval(choice)(data)

#
# data,label = Figure4.generate_datapoint()
# i1,i2,i3 = Figure4.data_to_type5(data)
# print(data, label)
# cv2.imwrite('a.bmp',i1*255)
# cv2.imwrite('b.bmp',i2*255)
# cv2.imwrite('c.bmp',i3*255)
# cv2.waitKey(0)