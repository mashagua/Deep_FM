# * coding:utf-8 *
# @author    :mashagua
# @File      :main.py
# @Software  :PyCharm

import gc
import pandas as pd
import pandas as pd
import tensorflow as tf
embedding_size = 5
epochs = 25
deep_layers_activation = tf.nn.relu
l2_reg = 0.1
lr = 0.01
tf.set_random_seed(2019)
train_file = r'data/train.csv'
test_file = 'data/test.csv'
IGNORE_COLS = ["click", "id"]
def FeatureDictionary(traindata,testdata,num_cols,ignore_cols):
    df=pd.concat([traindata,testdata],axis=0)
    feat_dict={}
    total_cnt=0
    for col in df.columns:
        if col in ignore_cols:
            continue
        if col in num_cols:
            feat_dict[col]=total_cnt
            total_cnt+=1
            continue
            
        unique_vals=df[col].unique()
        unique_cnt=df[col].unique()
        feat_dict[col]=dict(zip(unique_vals,range(total_cnt,total_cnt+unique_cnt)))
        
        
        

traindata = pd.read_csv(train_file)
testdata = pd.read_csv(test_file)
# feature_dict, feature_size = FeatureDictionary(traindata, testdata, num_cols=NUMERIC_COLS, ignore_cols=IGNORE_COLS)
