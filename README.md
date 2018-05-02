# Kaggle_HousePricePredict v1.0
本项目是来自于kaggle上的房价预测比赛。大体流程如下：原始数据分析（用jupyter进行原始数据的分析，包括原始数据分布特点、特征相关性分析、离群点分析、缺失值分析等）、数据预处理（离散型变量one-hot处理、缺失值填充、连续型变量标准化）、建模预测（单个岭回归模型）
# Kaggle_HousePricePredict v2.0
在原有模型train_model中岭回归的基础上，使用了ensemble（集成学习）方法。主要使用了bagging和boosting两种方法，经提交测试，bagging的效果好于boosting
