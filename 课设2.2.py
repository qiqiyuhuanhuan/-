# 导入所需的库
import os
import librosa
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# 定义音频文件的类别标签
label = ['aloe', 'burger', 'cabbage', 'candied_fruits',
         'carrots', 'chips', 'chocolate', 'drinks', 'fries',
         'grapes', 'gummies', 'ice-cream', 'jelly', 'noodles', 'pickles',
         'pizza', 'ribs', 'salmon', 'soup', 'wings']
label_dict = dict(zip(label, range(len(label))))

# 定义提取音频特征的函数
def extract_features(path, rates=(1.0,)):
    y_0, sr = librosa.load(path)  # 加载音频文件
    y_list = [librosa.effects.time_stretch(y_0, rate=rate) for rate in rates]  # 对音频进行时间拉伸
    features = []
    for y in y_list:
        mel = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128).T  # 提取MFCC特征
        features.append(np.mean(mel, axis=0))  # 计算每个音频的特征均值
    return np.array(features)

# 定义提取训练集特征的函数
def extract_features_train(parent_dir, max_file=10):
    X, Y = [], []
    for sub_dir in label:
        _, _, filenames = next(os.walk(os.path.join(parent_dir, sub_dir)))  # 遍历每个类别的文件夹
        for filename in tqdm(filenames[:max_file]):
            features = extract_features(os.path.join(parent_dir, sub_dir, filename), (0.5, 0.7, 1.0, 1.4, 2.0))  # 提取特征并进行数据增强
            for feature in features:
                X.append(feature)
                Y.append(label_dict[sub_dir])
    return [np.array(X), np.array(Y)]

# 定义提取测试集特征的函数
def extract_features_test(parent_dir):
    X = []
    _, _, filenames = next(os.walk(parent_dir))
    for filename in tqdm(filenames):
        X.append(extract_features(os.path.join(parent_dir, filename))[0])  # 提取测试集特征，不进行数据增强
    return np.array(X)

# 定义保存测试集文件名的函数
def save_name():
    _, _, filenames = next(os.walk('./test_b'))
    with open('path', 'w') as f:
        f.writelines([filename + '\n' for filename in filenames])

# 定义保存训练集和测试集特征的函数
def save_features():
    save_name()
    X, Y = extract_features_train('./train', 1000)
    print(X.shape, Y.shape)
    np.save('X.npy', X)
    np.save('Y.npy', Y)
    X_ = extract_features_test('./test_b')
    print(X_.shape)
    np.save('X_.npy', X_)

# 定义自定义的train_test_split方法
def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]
    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test

# 定义从文件加载特征的函数
def load_features():
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    X_ = np.load('X_.npy')
    return X, Y, X_

# 定义建立模型的函数
def classifier():
    model = Sequential()
    model.add(Dense(1024, input_dim=128, activation="relu"))  # 输入层
    model.add(Dropout(0.2))  # 防止过拟合
    model.add(Dense(1024, activation="relu"))  # 隐藏层
    model.add(Dense(20, activation="softmax"))  # 输出层
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# 定义将预测结果保存为csv文件的函数
def to_csv(model, X_, save_path='submit.csv'):
    predictions = model.predict(X_)
    preds = np.argmax(predictions, axis=1)
    preds = [label[x] for x in preds]
    path = []
    with open("path") as f:
        for line in f:
            path.append(line.strip())
    result = pd.DataFrame({'name': path, 'label': preds})
    result.to_csv(save_path, index=False)

# 处理数据集并保存为文件
save_features()
# 从文件中加载特征
X, Y, X_ = load_features()
# 对特征进行标准化处理
train_mean = np.mean(X, axis=0)
train_std = np.std(X, axis=0)
X = (X - train_mean) / train_std
X_ = (X_ - train_mean) / train_std
# 将类别转换为one-hot编码
Y = to_categorical(Y)
# 训练测试数据分离
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_ratio=0.1, seed=666)
# 建立模型
model = classifier()
# 使用所有数据进行训练
model.fit(X, Y, epochs=10, batch_size=7000, validation_data=(X_test, y_test))
# 将预测结果保存为csv文件
to_csv(model, X_, 'submit.csv')