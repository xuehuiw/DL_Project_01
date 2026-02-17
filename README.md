# DL_Project_01 - 乳腺癌检测逻辑回归实战

## 1. 项目结构

```text
DL_Project_01/
├── README.md                      # 项目说明文档
├── data/
│   └── breast_cancer_data.csv     # 核心数据文件：包含乳腺癌的特征和分类结果
└── scr/
    └── 逻辑回归实战--乳腺癌检测.py   # 主程序代码：用于训练模型并进行预测
```

## 2. 代码详细讲解（小白入门版）

本项目利用逻辑回归模型（Logistic Regression）来根据体检数据预测乳腺癌是良性还是恶性。你可以把这个程序想象成在**教一个学生（模型）如何看体检单（数据）来判断病人是否患病**。

### 总体流程
1.  **准备教材**：读取数据 (`pandas`)。
2.  **划分考题**：把数据分成“平时作业”（训练集）和“期末考试”（测试集）。
3.  **统一标准**：把数据缩放到 0 到 1 之间，方便比较（归一化）。
4.  **上课学习**：让模型学习数据规律（`fit`）。
5.  **参加考试**：让模型去预测测试集的结果。
6.  **查看成绩**：评估模型预测得准不准。

### 详细步骤解析

#### 第一步：搬运工具箱 (Import)
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
```
*   **作用**：这些是 Python 的第三方库。
    *   `pandas`：不仅能读 Excel/CSV，还能像操作 Excel 表格一样操作数据。
    *   `sklearn` (Scikit-learn)：机器学习的神器，里面包含了各种模型（如逻辑回归）和工具（如切分数据、评分）。

#### 第二步：读取数据
```python
dataset = pd.read_csv("data/breast_cancer_data.csv")
X = dataset.iloc[:, : -1]
Y = dataset['target']
```
*   `pd.read_csv(...)`：把 `data/breast_cancer_data.csv` 这个文件读进来变成一张表格。
*   `X` (特征)：这是体检单上的各项指标（比如肿瘤大小、半径等）。`iloc[:, :-1]` 意思是取**所有的行**，以及**除了最后一列之外的所有列**。
*   `Y` (标签)：这是结果（良性还是恶性）。代码中也就是最后一列 `target`。

#### 第三步：划分“平时作业”和“期末考试”
```python
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
```
*   `train_test_split`：这是一个极其常用的函数。
    *   它把数据切成两份。
    *   `test_size=0.2`：意思是用 **80% 的数据用来训练**（x_train, y_train），**20% 的数据用来测试**（x_test, y_test）。
    *   **为什么？** 就像学生不能看着答案考试一样，我们必须留一部分数据是模型没见过的，这样才能知道它是不是真的学会了。

#### 第四步：统一量纲 (归一化)
```python
sc = MinMaxScaler(feature_range=(0, 1))
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
```
*   `MinMaxScaler`：把所有数据压缩到 0 到 1 之间。
    *   比如“肿瘤面积”可能是 500，“平滑度”可能是 0.05。数字差异太大，模型会误以为面积更重要。归一化消除了这种偏见。
*   `fit_transform(x_train)`：**先计算**训练集的最大最小值（fit），**然后转换**（transform）。模型学会了训练集的范围。
*   `transform(x_test)`：**注意！** 这里只转换，不重新计算范围。这是为了用同样的标准去衡量测试集（就像考试打分标准要和平时一样）。

#### 第五步：建立并训练模型 (逻辑回归)
```python
lr = LogisticRegression()
lr.fit(x_train, y_train)
```
*   `LogisticRegression`：这是我们选择的“大脑”。逻辑回归主要用于**二分类问题**（是/否，良性/恶性）。
*   `lr.fit(...)`：这是**最关键的一步**。机器在“学习”。它在寻找 X（特征）和 Y（结果）之间的数学关系。

#### 第六步：利用模型预测
```python
pre_reslut = lr.predict(x_test)
pre_reslut_proba = lr.predict_proba(x_test)
pre_list = pre_reslut_proba[:, 1]
```
*   `lr.predict(x_test)`：直接给出结果。模型看一眼测试卷子（x_test），直接填答案：0（良性）或 1（恶性）。
*   `lr.predict_proba(x_test)`：给出**概率**。比如模型会说：“我觉得这个样本有 80% 的概率是恶性”。
*   `pre_list`：我们只取了“它是恶性（1）”的那一列概率值。

#### 第七步：自定义判断标准 (阈值调整)
```python
thresholds = 0.3
# ...
for i in range(len(pre_list)):
    if pre_list[i] > thresholds:
        # ... 判定为恶性
    else:
        # ... 判定为良性
```
*   通常逻辑回归默认概率 > 0.5 就是恶性。
*   但在医疗领域，我们**宁可错杀一千，不可放过一个**。所以你在这里把门槛降低到了 `0.3`。只要有 30% 的概率是恶性，我们就把它标记为恶性，提醒医生注意。这是一段很好的业务逻辑代码。

#### 第八步：批改试卷 (分类报告)
```python
report = classification_report(y_test, result, labels=[0, 1], target_names=['良性肿瘤', '恶性肿瘤'])
print(report)
```
*   `classification_report`：这是老师的评语。它对比了 `y_test`（标准答案）和 `result`（模型的答案）。
*   **结果怎么看？**
    *   **Precision (精确率)**：为了追求不出错。预测为“恶性”的里面，有多少**真的**是恶性？
    *   **Recall (召回率)**：为了追求不漏诊。所有**真的**恶性病例里，模型找出来了多少？（在医疗检测中，这个指标通常最重要，越高越好，意味着没有漏掉病人）。
    *   **F1-score**：两者的综合分数。

