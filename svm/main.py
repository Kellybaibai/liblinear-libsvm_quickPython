# -*- coding: utf-8 -*-
# @Time: 2023/11/9 10:32
# @Author: Kellybai
# @File: main.py
# Have a nice day!

import argparse
import numpy as np
from src.data_process import Transform
from src.download import download_data
from src.training import svm_model
import sys
import io
from datetime import datetime

def get_args():
    '''
    训练参数选择：
    options:
    -s type : set type of solver (default 1)
      for multi-class classification
         0 -- L2-regularized logistic regression (primal)
         1 -- L2-regularized L2-loss support vector classification (dual)
         2 -- L2-regularized L2-loss support vector classification (primal)
         3 -- L2-regularized L1-loss support vector classification (dual)
         4 -- support vector classification by Crammer and Singer
         5 -- L1-regularized L2-loss support vector classification
         6 -- L1-regularized logistic regression
         7 -- L2-regularized logistic regression (dual)
      for regression
        11 -- L2-regularized L2-loss support vector regression (primal)
        12 -- L2-regularized L2-loss support vector regression (dual)
        13 -- L2-regularized L1-loss support vector regression (dual)
    -c cost : set the parameter C (default 1)
    -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    -e epsilon : set tolerance of termination criterion
        -s 0 and 2
            |f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,
            where f is the primal function and pos/neg are # of
            positive/negative data (default 0.01)
        -s 11
            |f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.0001)
        -s 1, 3, 4 and 7
            Dual maximal violation <= eps; similar to libsvm (default 0.1)
        -s 5 and 6
            |f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,
            where f is the primal function (default 0.01)
        -s 12 and 13\n"
            |f'(alpha)|_1 <= eps |f'(alpha0)|,
            where f is the dual function (default 0.1)
    -B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)
    -wi weight: weights adjust the parameter C of different classes (see README for details)
    -v n: n-fold cross validation mode
    -C : find parameters (C for -s 0, 2 and C, p for -s 11)
    -q : quiet mode (no outputs)
    :return: args
    '''
    parser = argparse.ArgumentParser()
    #data config
    parser.add_argument('--data_name', type=str, default='cod_rna')
    parser.add_argument('--class_type', type=str, default='two', help='class type,options:[two, single, multi]')
    parser.add_argument('--linear_train_param', type=str, default='-s 3 -q -c 4')
    parser.add_argument('--libsvm_train_param', type=str, default='-s 0 -q -t 0 -c 4')
    parser.add_argument('--data_trans', type=str, default='no',help='transformation, options=[no,split,transfer,both]')
    parser.add_argument('--data_download', type=bool, default=False)
    parser.add_argument('--train_model', type=str, default='both', help='train liblinear or libsvm, options:[no, both, linear, libsvm]')
    parser.add_argument('--search_param', type=bool, default=True)
    parser.add_argument('--test_model', type=str, default='both', help='train liblinear or libsvm, options:[no, both, linear, libsvm]')
    parser.add_argument('--scaling', type=bool, default=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('start')
    print('---------------------------------------------------')
    # redirect output
    sys.stdout = io.StringIO()
    np.random.seed(1234)
    args = get_args()
    print('dataset_name', args.data_name)
    if args.data_download:
        download_data(args)

    if args.data_trans != 'no':
        transform = Transform(args)
        transform.trans_and_split()


    model = svm_model(args)
    if args.train_model != 'no':
        train_y, train_x = model.read_train_data()
        if args.train_model == 'both':
            model.training_data(train_y, train_x, 'linear')
            model.training_data(train_y, train_x, 'libsvm')
        elif args.train_model == 'linear':
            model.training_data(train_y, train_x, 'linear')
        else:
            model.training_data(train_y, train_x, 'libsvm')

        print('train_shape',train_y.shape[0], train_x.shape[0])

    if args.test_model != 'no':
        test_y, test_x = model.read_test_data()
        if args.test_model == 'both':
            model.predicting(test_y, test_x, 'linear')
            model.predicting(test_y, test_x, 'libsvm')
        elif args.test_model == 'linear':
            model.predicting(test_y, test_x, 'linear')
        else:
            model.predicting(test_y, test_x, 'libsvm')
        print('test_shape', test_y.shape[0],test_x.shape[0])

    # get input of console
    output = sys.stdout.getvalue()
    now = datetime.now()
    current_time = now.strftime("%H_%M_%S")
    log_file = open('./log/log_' + current_time + '.txt', "w+")
    log_file.writelines(output)
    log_file.close()

    # recover output
    sys.stdout = sys.__stdout__



