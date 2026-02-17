import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split# 训练集和测试集的划分
from sklearn.preprocessing import MinMaxScaler# 数据归一化:将数据缩放到指定的范围内，通常是0到1之间,它是对数据的一种线性变换，以便于模型的训练和预测。
from sklearn.linear_model import LogisticRegression# 逻辑回归模型的搭建
from sklearn.metrics import classification_report# 分类报告:用于评估分类模型的性能，提供了精确率、召回率、F1值等指标的详细报告。

# 读取数据
dataset = pd.read_csv("data/breast_cancer_data.csv")
# print(dataset)

# 提取x
X = dataset.iloc[:, : -1]
# iloc是pandas中的一个索引器，用于基于位置进行数据选择。它接受行和列的索引参数，返回相应位置的数据。
#在这里，:表示选择所有行，:-1表示选择除了最后一列以外的所有列，即提取特征数据（第0列到倒数第二列）。
# print(X)

# 提取数据中的标签
Y = dataset['target']
# print(Y)

# 划分数据集和测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# 将数据集划分为训练集和测试集。它接受特征数据（X）和标签数据（Y），以及测试集的比例（test_size）作为参数（测试集占0.2），并返回划分后的训练集和测试集。

# 进行数据的归一化
sc = MinMaxScaler(feature_range=(0, 1))# feature_range参数指定了缩放后的范围。
x_train = sc.fit_transform(x_train)#fit用于计算训练数据的最小值和最大值，并将其存储在sc对象中；transform用于将训练数据进行缩放到指定的范围内。
x_test = sc.transform(x_test)#不fit！
#注意：在训练集上使用fit_transform方法进行拟合和转换，而在测试集上只使用transform方法进行转换，以确保测试集的数据被缩放到与训练集相同的范围内。
# print(x_train)

# 逻辑回归模型搭建
lr = LogisticRegression()
lr.fit(x_train, y_train)

# 打印模型的参数
# print('w:', lr.coef_)
# print('b', lr.intercept_)

# 利用训练好的模型进行推理测试
pre_reslut = lr.predict(x_test)#打印预测结果
# print(pre_reslut)

# 打印预测结果的概率
pre_reslut_proba = lr.predict_proba(x_test)#打印概率
# print(pre_reslut_proba)

# 获取恶性肿瘤的概率
pre_list = pre_reslut_proba[:, 1]#行索引为:所有行，列索引为1，即获取每个样本属于恶性肿瘤的概率。
# print(pre_list)

# 设置阈值
thresholds = 0.3#大于0.3的概率被认为是恶性肿瘤，小于等于0.3的概率被认为是良性肿瘤。（人为设置）

# 设置保存结果的列表
result = []#保存结果的列表，用于存储根据阈值调整后的分类结果（0/1）。
result_name = []#保存结果名称的列表，用于存储每个样本的分类名称（良性/恶性）。

for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        result.append(1)
        result_name.append('恶性')
    else:
        result.append(0)
        result_name.append('良性')

# 打印阈值调整后的结果
# print(result)
# print(result_name)

# 输出结果的精确率和召回还有f1值
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print(report)
# 精确率（Precision）是指在所有被预测为正类的样本中，实际为正类的比例；
# 召回率（Recall）是指在所有实际为正类的样本中，被正确预测为正类的比例；（恶性肿瘤recall为1.00时表示所有恶性肿瘤都被预测到了，后续不会漏判）
# F1值是精确率和召回率的调和平均数，用于综合评估模型的性能。（越大模型综合能力越好）
