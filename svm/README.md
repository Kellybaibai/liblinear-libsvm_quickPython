# liblinear-libsvm_quickPython
A repository for quickly using liblinear/libsvm in python language
# Dependency
before using this quickPython, make sure you have installed liblinear,libsvm and numpy
# Dataset
You can directly deal with the libsvm-form data, or use process data from UCI with the quickPython, and train the models afterwards.
If you use libsvm-form without any process, make sure your data is `.binary` form and the labels are {-1,+1} for two-class SVC, and {1,2,3...} for multi-class SVC.
# Param
You can use `python ./main.py --params` to quickly run the data processing, training and predicting tasks. Here is an example:
```
python ./main.py --data_name cod_rna --train_model linear --test_model linear
```
It means training a liblinear model with "cod_rna" data and testing the model.

Here are the params you can use:

--data_name: which dataset you shall use

--class_type: for SVC, class type, options:[two, single, multi]

--linear_train_param: default='-s 3 -q -c 4'

--libsvm_train_param: default='-s 0 -q -t 0 -c 4'

--data_trans: transformation, options=[no,split,transfer,both]

--data_download: whether to download data from a url

--train_model: train liblinear or libsvm, options:[no, both, linear, libsvm]

--search_param: whether to search the param -c when training

--test_model: train liblinear or libsvm, options:[no, both, linear, libsvm]

--scaling: whether to scale the data when training and testing

