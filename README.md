# 精准资助代码说明

## 数据预处理及特征工程

直接用了别人的代码，我没细看

## 文件说明

| 文件名               | 说明                                       |
| -------------------- | ------------------------------------------ |
| blend.py             | 根据投票法，融合三个模型                   |
| explain.py           | 对一个结果使用可视化的方式作出解释         |
| meachine_learning.py | 使用机器学习的模型构建分类器               |
| mlp.py               | 使用多层感知机（全连接神经网络）构建分类器 |
| readdata.py          | 读入数据集                                 |

## 运行方法

- 运行blend.py
  - train()会调用meachine_learning.py和mlp.py进行训练，test()会输出测试集上的准确率。

- 运行explain.py
  - exp为解释器，会输出一个样本预测为某一类的概率及解释

## 数据规模

训练集：1396×1151

测试集：349×1151

无标签数据集：3242（后期可以考虑一下半监督）

## 当前模型

| 模型                       | 参数(未写出的即为默认参数)                                   | 权重 |
| -------------------------- | ------------------------------------------------------------ | ---- |
| DecisionTreeClassifier     | random_state=0                                               | 2/15 |
| RandomForestClassifier     | random_state=0                                               | 2/15 |
| SVMClassifier              | C=2, gamma=10, probability=True                              | 1/15 |
| GradientBoostingClassifier | learning_rate=1.0, max_depth=1, random_state=0               | 2/15 |
| KNeighborsClassifier       | 默认                                                         | 2/15 |
| LogisticRegression         | 默认                                                         | 1/15 |
| Multi-Layer Perceptron     | MLP(<br/>  (layer1): Sequential(<br/>    (0): Linear(in_features=1151, out_features=300, bias=True)<br/>    (1): ReLU(inplace)<br/>  )<br/>  (layer2): Sequential(<br/>    (0): Linear(in_features=300, out_features=100, bias=True)<br/>    (1): ReLU(inplace)<br/>  )<br/>  (layer3): Sequential(<br/>    (0): Linear(in_features=100, out_features=4, bias=True)<br/>  )<br/>) | 5/15 |

## 测试集结果

Accuracy: 100%