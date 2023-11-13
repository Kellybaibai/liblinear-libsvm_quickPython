# -*- coding: utf-8 -*-
# @Time: 2023/11/9 20:35
# @Author: Kellybai
# @File: train.py
# Have a nice day!


from liblinear.liblinearutil import *
from libsvm.svmutil import *
import time
import numpy as np


class svm_model:
    def __init__(self, args):
        self.linear_train_param = args.linear_train_param
        self.libsvm_train_param = args.libsvm_train_param
        self.data_name = args.data_name
        self.search_param = args.search_param
        self.scaling = args.scaling
        self.scale_param = 0

    def read_train_data(self):
        y, x = svm_read_problem('./dataset/' + self.data_name + '_train.binary', return_scipy=True)  # y: ndarray, x: csr_matrix
        # scale x
        if self.scaling:
            self.scale_param = csr_find_scale_param(x, lower=0)
            x = csr_scale(x, self.scale_param)
        return y, x

    def read_test_data(self):
        y, x = svm_read_problem('./dataset/' + self.data_name + '_test.binary', return_scipy=True)
        if self.scaling:
            x = csr_scale(x, self.scale_param)
        return y, x

    def training_data(self, y, scaled_x, train_type):
        # search the best c
        if self.search_param:
            param_list_linear = np.arange(0.1, 5, 0.2)
            param_list_svm = np.arange(0.5, 3.5, 0.5)
            param_dict = {'linear': param_list_linear, 'libsvm':param_list_svm}
            self.searching_param(param_dict[train_type], y, scaled_x, train_type)
        print('---------------------------training--------------------------------')

        # train libliear model
        if train_type == 'linear':
            print('Liblinear training param: ', self.linear_train_param)
            t0 = time.time()
            model = train(y, scaled_x, self.linear_train_param)
            t1 = time.time()
            print('Liblinear training time: ', t1-t0)
            save_model(self.data_name + '_' + train_type + '.model', model)

        # train libsvm model
        else:
            print('Libsvm training param: ', self.libsvm_train_param)
            t2 = time.time()
            model = svm_train(y, scaled_x, self.libsvm_train_param)
            t3 = time.time()
            print('Libsvm training time: ', t3-t2)
            svm_save_model(self.data_name + '_' + train_type + '.model', model)
        print('---------------------------end-training------------------------------')



    def searching_param(self, param_arr, y, scaled_x, train_type):
        '''
        find the best param c
        '''
        m_list = []
        if train_type == 'linear':
            for i in param_arr:
                m = train(y, scaled_x, self.linear_train_param[:-1] + str(i) +' -v 5')
                m_list.append(m)
            c_max = param_arr[m_list.index(max(m_list))]
            self.linear_train_param = self.linear_train_param[:-1] + str(c_max)
        else:
            for i in param_arr:
                m = svm_train(y, scaled_x, self.libsvm_train_param[:-1] + str(i) + ' -v 5')
                print('self.libsvm_train_param[:-1] + str(i) + -v 5', self.libsvm_train_param[:-1] + str(i) + ' -v 5')
                m_list.append(m)
            print('param_arr',param_arr)
            c_max = param_arr[m_list.index(max(m_list))]
            self.libsvm_train_param = self.libsvm_train_param[:-1] + str(c_max)
        print('m_max', max(m_list))
        print('c_max', c_max)

    def predicting(self, y, scaled_x, test_type):
        print('---------------------------predicting-----------------------------')
        if test_type == 'linear':
            m = load_model(self.data_name + '_' + test_type + '.model')
            t4 = time.time()
            predict(y, scaled_x, m)
            t5 = time.time()
            print('Liblinear predicting time: ', t5 - t4)
        else:
            m = svm_load_model(self.data_name + '_' + test_type + '.model')
            t6 = time.time()
            svm_predict(y, scaled_x, m)
            t7 = time.time()
            print('Libsvm predicting time: ', t7 - t6)
        print('---------------------------end-predicting--------------------------')










