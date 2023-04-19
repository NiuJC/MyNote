# Learning the Evolutionary and Multi-scale Graph Structure for Multivariate Time Series Forecasting

# 摘要

为了使图神经网络具有灵活实用的图结构，本文研究了如何对时间序列的演化和多尺度相互作用进行建模。特别地，我们首先提供了一个与扩展卷积相配合的层次图结构来捕获时间序列之间的尺度特定相关性。然后，以递归的方式构造一系列邻接矩阵来表示每一层的演化相关性。并通过统一的神经网络集成上述各部分，得到最终的预测结果。通过这种方式，我们可以同时捕获成对的相关性和时间依赖性。最后，在单步和多步预测任务上的实验证明了我们的方法比现有方法的优越性。

# 1. 介绍

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252143443.png)

# 2. 准备工作

定义1

定义2

# 3. 方法

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252143829.png)

## 3.1 总览

## 3.2 学习进化图结构

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252143621.png)

## 3.3 时间卷积模块

## 3.4 进化图卷积模块

# 4. 实验

## 4.1 数据集和参数设置

## 4.2 基线

## 4.3 主要结论

4.3.1

4.3.2

## 4.4 消融实验

## 4.5 进化图结构的研究