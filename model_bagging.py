'''
    created in 2018/5/2
    @author Jie Wang
    kaggle_HousePrice_Predict using bagging

'''

from train_model import dataProcessing
from train_model import model_Ridge
from train_model import summsion_csv
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

if __name__ == '__main__':
    #数据预处理
    train = pd.read_csv("train.csv", index_col=0)
    test = pd.read_csv("test.csv", index_col=0)
    dummy_train_df, dummy_test_df, train_y = dataProcessing(train, test)
    Alpha, Test_score, y_pred = model_Ridge(dummy_train_df, train_y, dummy_test_df)

    #下面是用bagging处理，该bagging中基学习器是岭回归，学习器的个数通过交叉验证来确定
    ridge = Ridge(alpha = Alpha)
    # params = [1, 10, 15, 20, 25, 30, 40]
    params = np.arange(1,50)
    test_scores = []
    min_score = 1
    for param in params:
        clf = BaggingRegressor(base_estimator= ridge, n_estimators= param)
        test_score = np.sqrt(-cross_val_score(clf, dummy_train_df, train_y, cv=10, scoring='neg_mean_squared_error'))
        temp = np.mean(test_score)
        if temp < min_score:
            min_score = temp
            optimal_params = param
        test_scores.append(np.mean(test_score))
    print(optimal_params)

    # plt.plot(params, test_scores)
    # plt.title("n_estimators vs CV_error")
    # plt.show()

    #用训练好的参数预测
    br = BaggingRegressor(base_estimator= ridge, n_estimators=optimal_params)
    br.fit(dummy_train_df, train_y)
    y_final = np.expm1(br.predict(dummy_test_df))
    # print(y_final)
    summsion_csv(y_final, test)


