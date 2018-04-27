
'''
created in 2018/4/34
@author Jie Wang
Kaggle_HousePrice_Predict

'''

import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from  sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


#step1:数据预处理，包含以下几个方面
# 将train和test合并，一起清洗；离散型变量处理（one-hot编码）；缺失值处理（填充）；连续型变量标准化（回归问题需要这一步）
def dataProcessing(trainData, testData):
    train_df = trainData
    test_df = testData

    train_y = np.log1p(train_df.pop('SalePrice')) #将label从train_df中提取出来，以进行合并train_x和test_x，
    all_df = pd.concat((train_df, test_df), axis=0) #合并，axis=0表示把test_df加到train_df后边

    all_df['MSSubClass'] = all_df['MSSubClass'].astype(str) #这个特征虽然是连续型数字，但实质上是类别离散型特征，故将其转化为string类型，当作离散型处理
    # print(all_df['MSSubClass'].value_counts())

    all_dummy_df = pd.get_dummies(all_df) #第一步：将所有离散型变量进行one-hot处理
    # print(all_dummy_df.head())

    mean_cols = all_dummy_df.mean() #求出均值，用于填充缺失值
    all_dummy_df = all_dummy_df.fillna(mean_cols) #第二步，缺失值处理


    numeric_cols = all_df.columns[all_df.dtypes != 'object'] #找到数值型数据
    # print(numeric_cols)
    #第三步：用z-score进行标准化，z=(x-μ)/σ,其中，x为原始数据，μ为平均数，σ为标准差
    # numeric_col_means = all_dummy_df.loc[:, numeric_cols].mean()
    # numeric_col_std = all_dummy_df.loc[:, numeric_cols].std()
    # all_dummy_df.loc[:, numeric_cols] = (all_dummy_df.loc[:, numeric_cols] - numeric_col_means) / (numeric_col_std)
    # print(all_dummy_df.head())
    #也可直接用sklearn进行标准化
    all_dummy_df.loc[:, numeric_cols] = StandardScaler().fit_transform(all_dummy_df.loc[:, numeric_cols])


    dummy_train_df = all_dummy_df.loc[train_df.index]
    dummy_test_df = all_dummy_df.loc[test_df.index]
    return dummy_train_df, dummy_test_df, train_y


def model_Ridge(X_train, Y_train, X_test): #用岭回归建模，岭回归可以不用考虑特征提取，可以把所有特征都放进去学习
    x_train = X_train
    y_train = Y_train
    alphas = np.logspace(-3, 2, 50)
    test_scores = []
    min_score = 1
    for alpha in alphas:#通过交叉验证，找到最优参数
        clf = Ridge(alpha)
        test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
        test_scores.append(np.mean(test_score))
        temp = np.mean(test_score)
        if temp < min_score:
            min_score = temp
            optimal_alpha = alpha
    #下面进行预测
    ridge = Ridge(alpha = optimal_alpha)
    ridge.fit(x_train, y_train)
    y_pred = np.expm1(ridge.predict(X_test))
    return  alphas, test_scores, y_pred

def summsion_csv(pred_y, Test): #kaggle提交
    submmision_df = pd.DataFrame(data = {'Id':Test.index, 'SalePrice': pred_y})
    submmision_df.to_csv("submission.csv", columns=['Id', 'SalePrice'], index=0)

if __name__ == '__main__':
    train = pd.read_csv("train.csv", index_col=0)
    test = pd.read_csv("test.csv", index_col=0)
    dummy_train_DF, dummy_test_DF, train_Y = dataProcessing(train, test)
    Alphas, test_Scores, Y_pred = model_Ridge(dummy_train_DF, train_Y, dummy_test_DF)
    print(Y_pred)
    summsion_csv(Y_pred, test)









