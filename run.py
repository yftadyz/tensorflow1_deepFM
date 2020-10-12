# ---- TF1.14 static computation graph version ----
import joblib
import pandas as pd
import numpy as np
import os
import time
import sys
import random
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.metrics import accuracy_score
from itertools import chain
from utils import fill_na,preprocess,prepare_data,build_points_list
from model import deepFM_tf1

# ----- prepare dataset -----
dftrain, dfeval = prepare_data('titanic-train.csv','titanic-eval.csv')

batch_size=32

epoch_num=20

epoch_size=dftrain.shape[0]//batch_size

if dftrain.shape[0]%batch_size!=0:

    epoch_size+=1
    
ft_names=['sex', 'age', 'n_siblings_spouses', 'parch', 'fare','class', 'deck', 'embark_town', 'alone']


label_name='survived'

train_points=build_points_list(dftrain,ft_names,label_name)
eval_points=build_points_list(dfeval,ft_names,label_name)
        
# ----- train ------
#initalize model
parameters={}
parameters['fm_cols']=ft_names
parameters['label_name']=label_name
parameters['fm_emb_dim']=32
parameters['hidden_units']=[32,16]
parameters['dropprob']=0.3
parameters['batch_size']=batch_size
parameters['epoch_size']=epoch_size
parameters['learning_rate']=0.01

mymodel=deepFM_tf1(parameters)

mymodel.train(train_points,eval_points,20,save_path='save/titanic_deepfm.ckpt',load_path=None)

#mymodel.train(train_points,eval_points,10,save_path='save/titanic_deepfm_v2.ckpt',load_path='save/titanic_deepfm.ckpt')

#res=mymodel.predict(eval_points,load_path='save/titanic_deepfm_v2.ckpt')
