import pandas as pd
import xgb_train as xgb
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('E:\\IsoGD\\output_DF.csv')

# 划分训练集、测试集和验证集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

# 提取特征和标签
train_features = train_data.iloc[:, 1:].values
train_labels = train_data.iloc[:, 0].values
valid_features = valid_data.iloc[:, 1:].values
valid_labels = valid_data.iloc[:, 0].values
test_features = test_data.iloc[:, 1:].values
test_labels = test_data.iloc[:, 0].values

# 构建DMatrix数据集
dtrain = xgb.DMatrix(train_features, label=train_labels)
dvalid = xgb.DMatrix(valid_features, label=valid_labels)
dtest = xgb.DMatrix(test_features, label=test_labels)

# 设置xgboost参数
params = {
    'objective': 'multi:softmax',
    'num_class': len(data['folder_name'].unique()),
    'max_depth': 6,
    'eta': 0.3,
    'gamma': 0,
    'min_child_weight': 1,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'lambda': 1,
    'alpha': 0,
    'scale_pos_weight': 1,
    'eval_metric': 'merror',
    'seed': 42
}

# 训练模型
num_rounds = 100
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, num_rounds, evals=watchlist, early_stopping_rounds=10)
# 在测试集上进行预测
pred_labels = model.predict(dtest)

# 计算准确率
accuracy = (pred_labels == test_labels).mean()
print('Test accuracy:', accuracy)