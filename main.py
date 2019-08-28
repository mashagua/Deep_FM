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
    Xi_train, Xv_train, y = DataParser(feat_dict=feature_dict, df=traindata, has_label=True)
    Xi_test, Xv_test, ids = DataParser(feat_dict=feature_dict, df=testdata, has_label=False)
    field_size = len(Xi_train[0])
    print(field_size)
    return Xi_train, Xv_train, y,Xi_test, Xv_test, ids,field_size,feature_size

def inference(feat_index,feat_value,feature_size):
    weights={}
    weights['feature_embedding']=tf.Variable(initial_value=tf.random.normal(shape=[feature_size,embedding_size],stddev=0.1,mean=0),
           name='feature_embedding',dtype=tf.float32)
    weights['feature_bias']=tf.Variable(initial_value=tf.random.uniform(shape=[feature_size,1],maxval=1.0,minval=0.0),name='feature_bias',dtype=tf.float32)
    num_layer=len(deep_layers)
    input_size=field_size
    num_layer = len(deep_layers)#2
    input_size = field_size * embedding_size#隐藏层的输入是field_size*embedding_size
    glorot = np.sqrt(2.0 / (input_size + deep_layers[0])) # glorot_normal: stddev = sqrt(2/(fan_in + fan_out))
    weights['layer_0'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size, deep_layers[0]],mean=0,stddev=glorot),
                                 dtype=tf.float32)#第一层隐层的权重维度为（field_size*embedding_size）*deep_layers[0]
    weights['bias_0'] = tf.Variable(initial_value=tf.random_normal(shape=[1, deep_layers[0]],mean=0,stddev=glorot),
                                dtype=tf.float32)
    for i in range(1, num_layer):#一共就2层
        glorot = np.sqrt(2.0 / (deep_layers[i - 1] + deep_layers[i]))
        weights['layer_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[deep_layers[i - 1], deep_layers[i]],mean=0,stddev=glorot),
                                          dtype=tf.float32)
        weights['bias_%d' % i] = tf.Variable(initial_value=tf.random_normal(shape=[1, deep_layers[i]],mean=0,stddev=glorot),
                                         dtype=tf.float32)
 # #Output Layer
    deep_size = deep_layers[-1]#32

    fm_size = field_size + embedding_size
    input_size = fm_size + deep_size
    glorot = np.sqrt(2.0 / (input_size + 1))
    weights['concat_projection'] = tf.Variable(initial_value=tf.random_normal(shape=[input_size,1],mean=0,stddev=glorot),
                                           dtype=tf.float32)
    weights['concat_bias'] = tf.Variable(tf.constant(value=0.01), dtype=tf.float32)
##构造模型，预测是输出值
  #Embedding
    #按照feat_index（x_i）的索引（one-hot中值为1的位置）找到对应权重中相应行
    embeddings_origin = tf.nn.embedding_lookup(weights['feature_embedding'], ids=feat_index) # [None, field_size, embedding_size]

    feat_value_reshape = tf.reshape(tensor=feat_value, shape=[-1,field_size, 1]) # [-1 * field_size * 1]
  # 一维特征
    y_first_order = tf.nn.embedding_lookup(weights['feature_bias'], ids=feat_index) # [None, field_size, 1]
    w_mul_x = tf.multiply(y_first_order, feat_value_reshape) # [None, field_size, 1]
    y_first_order = tf.reduce_sum(input_tensor=w_mul_x, axis=2) # [None, field_size]
  # 交叉项组合特征
     ##multiply不是矩阵相乘，而是矩阵对应位置相乘。
    embeddings = tf.multiply(embeddings_origin, feat_value_reshape) # [None, field_size, embedding_size]
    summed_features_emb = tf.reduce_sum(input_tensor=embeddings, axis=1) # [None, embedding_size]
    summed_features_emb_square = tf.square(summed_features_emb)
    squared_features_emb = tf.square(embeddings)
    squared_features_emb_summed = tf.reduce_sum(input_tensor=squared_features_emb, axis=1) # [None, embedding_size]
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_features_emb_summed)#交叉项输出
  # Deep模块
    #在deep部分将每一个field得到的embedding值连接起来[[e_1,e_2,...e_8],[],[]]
    y_deep = tf.reshape(embeddings_origin, shape=[-1, field_size * embedding_size]) # [None, field_size * embedding_size]
    for i in range(0, len(deep_layers)):#2
        y_deep = tf.add(tf.matmul(y_deep, weights['layer_%d' % i]), weights['bias_%d' % i])
        y_deep = deep_layers_activation(y_deep)
  # 输出
    concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)#按行拼接
    out = tf.add(tf.matmul(concat_input, weights['concat_projection']), weights['concat_bias'])
    out = tf.nn.sigmoid(out)#输出值

    return out,weights


#训练
Xi_train, Xv_train, y,Xi_test, Xv_test, ids,field_size,feature_size=load_data()
feat_index = tf.placeholder(dtype=tf.int32, shape=[None, field_size], name='feat_index') # [None, field_size]
feat_value = tf.placeholder(dtype=tf.float32, shape=[None, None], name='feat_value') # [None, field_size]
label = tf.placeholder(dtype=tf.float16, shape=[None,1], name='label')
out,weights = inference(feat_index,feat_value,feature_size)
loss = tf.losses.log_loss(label, out)
if l2_reg > 0:
    loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights['concat_projection'])
    for i in range(len(deep_layers)):
        loss += tf.contrib.layers.l2_regularizer(l2_reg)(weights['layer_%d' % i])
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss)

sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())
feed_dict = {
    feat_index: Xi_train,
    feat_value: Xv_train,
    label:      np.array(y).reshape((-1,1))
}

for epoch in range(epochs):
    train_loss,opt = sess.run((loss, optimizer), feed_dict=feed_dict)
    print("epoch: {0}, train loss: {1:.6f}".format(epoch, train_loss))

#预测
dummy_y = [1] * len(Xi_test)

feed_dict_test = {
    feat_index: Xi_test,
    feat_value: Xv_test,
    label: np.array(dummy_y).reshape((-1,1))
}

pres = sess.run(out, feed_dict=feed_dict_test)

for i in range(len(pres)):
    if pres[i] > 0.5:
        pres[i] = 1
    else:
        pres[i] = 0

sub = pd.DataFrame({"id":ids, "pred":np.squeeze(pres)})
print("prediction:")
print(sub)
    

if __name__=="__main__":
    pass
    