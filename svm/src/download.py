# -*- coding: utf-8 -*-
# @Time: 2023/11/9 10:31
# @Author: Kellybai
# @File: download.py
# Have a nice day!

import requests

def download_data(args):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    dataset_name = args.data_name+'/'+args.data_name+'.data'
    r = requests.get(url+dataset_name)
    with open('./dataset/raw_'+args.data_name+'.csv', 'wb') as f:
        f.write(r.content)
