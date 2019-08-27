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
embedding_size = 5   #k=8
epochs = 40
deep_layers_activation = tf.nn.relu
## 模型参数
deep_layers = [32,32]

train_file = r'data/train.csv'
test_file = 'data/test.csv'
IGNORE_COLS = ["click", "id"]
NUMERIC_COLS = ["C1", "C18", "C16"]
CATEGORITAL_COLS = ['C15']

# 给每一个特征维度都进行编号。
def FeatureDictionary(dfTrain=None, dfTest=None, numeric_cols=None, ignore_cols=None):

    df = pd.concat([dfTrain, dfTest], axis=0)#合并数据，按照行合并
    feat_dict = {}
    total_cnt = 0

    for col in df.columns: #df.columns=['id', 'click', 'C1', 'C18', 'C16', 'C15', 'C17']
        if col in ignore_cols:
            continue
        # 连续特征只有一个编号
        if col in numeric_cols:
            print(col)
            feat_dict[col] = total_cnt
            print(feat_dict)

            total_cnt += 1
            continue

        # 离散特征，有多少个取值就有多少个编号
        unique_vals = df[col].unique()#离散特征的取值，unique()移除数组中重复的值
        unique_cnt = df[col].nunique()#查看有多少个不同值
        feat_dict[col] = dict(zip(unique_vals, range(total_cnt, total_cnt + unique_cnt)))
        print(feat_dict)
        total_cnt += unique_cnt
    feat_size = total_cnt
    return feat_dict, feat_size

def DataParser(feat_dict=None, df=None, has_label=False):
    assert not (df is None), "df is not set"

    dfi = df.copy()


    if has_label:
        y = df['click'].values.tolist()#values()返回字典中的所有值,tolist()将数组或者矩阵转化为列表。
        dfi.drop(['id','click'],axis=1, inplace=True)
    else:
        ids = dfi['id'].values.tolist()
        dfi.drop(['id'],axis=1, inplace=True)

    dfv = dfi.copy()


    for col in dfi.columns:
        if col in IGNORE_COLS:
            dfi.drop([col], axis=1, inplace=True)
            dfv.drop([col], axis=1, inplace=True)
            continue

        if col in NUMERIC_COLS: # 连续特征1个维度，对应1个编号，这个编号是一个定值
            dfi[col] = feat_dict[col]#找到对应索引值
        else:
            # 离散特征。不同取值对应不同的特征维度，编号也是不同的。
            dfi[col] = dfi[col].map(feat_dict[col])#在feat_dict[col]中根据dfi[col]的值找到对应的索引
            dfv[col] = 1.0


    # 取出里面的值
    Xi = dfi.values.tolist()
    Xv = dfv.values.tolist()

    del dfi, dfv
    gc.collect()

    if has_label:
        return Xi, Xv, y
    else:
        return Xi, Xv, ids

def load_data():
    traindata=pd.read_csv(train_file)
    testdata=pd.read_csv(test_file)
    feature_dict, feature_size = FeatureDictionary(traindata, testdata, numeric_cols=NUMERIC_COLS, ignore_cols=IGNORE_COLS)
    Xi_train, Xv_train, y = DataParser(feat_dict=feature_dict, df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids = DataParser(feat_dict=feature_dict, df=dfTest, has_label=False)
    field_size = len(Xi_train[0])#5
    print(field_size)
    return Xi_train, Xv_train, y,Xi_test, Xv_test, ids,field_size,feature_size

if __name__=="__main__":
    