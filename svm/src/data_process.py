# -*- coding: utf-8 -*-
# @Time: 2023/11/9 10:32
# @Author: Kellybai
# @File: data_process.py
# Have a nice day!

import numpy as np


class Transform:
    def __init__(self, args):
        self.data_name = args.data_name
        self.data_trans = args.data_trans
        self.class_type = args.class_type

    def trans_and_split(self):
        '''
        if split data, deal with data and split it
        if not split, deal with train data and test data separately
        :return:None
        '''
        if self.data_trans == 'both':
            data = np.genfromtxt('./dataset/raw_'+self.data_name+'.csv', delimiter=',')  # ,dtype=[float, float, float, float, float, float, int]
            # drop rows with Nan
            data = self.drop_na(data)
            print(np.shape(data))
            # replace the labels
            data = self.change_label(data)
            self.trans_data(data, self.data_name)
            # split data
            self.split_data()

        elif self.data_trans == 'split':
            # split data
            self.split_data()


        elif self.data_trans == 'transfer':
            train_data = np.genfromtxt('./dataset/raw_train_'+self.data_name+'.csv', delimiter=',')
            test_data = np.genfromtxt('./dataset/raw_test_'+self.data_name+'.csv', delimiter=',')
            # drop rows with Nan
            train_data = self.drop_na(train_data)
            test_data = self.drop_na(test_data)

            # replace labels
            train_data = self.change_label(train_data)
            test_data = self.change_label(test_data)
            self.trans_data(train_data, self.data_name + '_train')
            self.trans_data(test_data, self.data_name + '_test')


    def drop_na(self, data):
        '''
        drop rows with Nan
        :param data:
        :return: data
        '''
        return np.delete(data, np.where(np.isnan(data.sum(axis=1))), axis=0)

    def change_label(self, data):
        '''
        change class label into the liblinear form
        if class_type==two, change the label into [-1,1]
        if class_type==multi change the label into [1,2,3...]
        :param data:
        :return: data
        '''
        if self.class_type == 'two':
            # print(list(set(data[:, -1])))
            class_list = list(set(data[:, -1]))
            class_list.sort()
            for i in range(len(data[:, -1])):
                if data[i, -1] == class_list[0]:
                    data[i, -1] = -1
                else:
                    data[i, -1] = 1

        if self.class_type == 'multi':
            class_list = list(set(data[:, -1]))
            class_list.sort()
            for i in range(len(data[:, -1])):
                data[i, -1] = class_list.index(data[i,-1]) + 1

        return data

    def trans_data(self, data, data_name):
        '''
        tranform data form csv into liblinear form
        :return: data in liblinear form
        '''
        lines = data.shape[0]
        columns = data.shape[1]
        print(lines, columns)
        new_data = np.zeros([lines, 2 * columns - 1], dtype=list)
        i = 0
        while i < lines:
            j = 1
            new_data[i][0] = data[i][columns - 1]
            while j < columns:
                new_data[i][2 * j - 1] = j
                new_data[i][2 * j] = data[i][j - 1]
                j = j + 1
            i = i + 1

        new_data, newlines, newcolumns = new_data, lines, 2 * columns - 1

        f = open('./dataset/'+data_name+'.binary', "w+")
        i = 0
        while i < newlines:
            # if classification --> label shall be int
            f.write(str(int(new_data[i][0])))

            # if regression  --> label shall be real
            #  f.write(str(new_data[i][0]))
            j = 1
            while j < newcolumns:
                if j % 2 == 0 and new_data[i][j] != 0:
                    f.write(str(new_data[i][j]))
                if j % 2 == 1 and new_data[i][j+1] != 0:
                    f.write(" ")
                    f.write(str(new_data[i][j]) + ":")
                j = j + 1
            f.write("\n")
            i = i + 1
        f.close()

    def split_data(self):
        '''
        split data into train data and test data
        :return:None
        '''
        # read the data
        data = open('./dataset/'+self.data_name+'.binary', 'r')
        data_lines = np.array(data.readlines())
        # divide index
        ind = np.random.permutation(len(data_lines))[:int(0.3*len(data_lines))]
        # print(type(ind))
        # generate train_dataset
        delete_mask = np.zeros_like(data_lines, dtype=bool)
        delete_mask[ind] = True
        train_data = data_lines[~delete_mask]
        print('train_data_length', len(train_data))
        train_file = open('./dataset/'+self.data_name+'_train.binary', 'w')
        train_file.writelines(train_data)
        train_file.close()
        # generate test_dataset
        test_data = data_lines[ind]
        test_file = open('./dataset/'+self.data_name+'_test.binary', 'w')
        test_file.writelines(test_data)
        test_file.close()






