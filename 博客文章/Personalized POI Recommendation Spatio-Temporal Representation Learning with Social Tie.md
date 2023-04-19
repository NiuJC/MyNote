---
title: "Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie"
date: ""
updated:
tags:
- 论文阅读笔记
- POI Recommendation
categories:
keywords:
description:
cover: false
---



# Personalized POI Recommendation: Spatio-Temporal Representation Learning with Social Tie

[International Conference on Database Systems for Advanced Applications](https://link.springer.com/conference/dasfaa dasfaa)

# 1. Introduction

论文提出了一种新的统一的时空神经网络框架，称为PPR，它利用用户的签到记录和社会关系，通过联合嵌入和顺序建模推荐个性化的poi来查询用户。具体而言，PPR首先通过在异构图中联合建模用户-POI关系、顺序模式、地理影响和社会关系来学习用户和POI表示，然后使用设计的基于LSTM模型的时空神经网络对用户个性化顺序模式进行建模，实现个性化POI推荐。

## 1.1 challenge

- 上下文因素，POI推荐可能受到各种上下文因素的影响，包括社会关系影响、地理影响、时间背景等。

- 动态和个性化的偏好，用户的偏好随着时间的推移而动态变化。在不同的时间和环境下，用户可能喜欢不同的poi。

## 1.2 contribution

- 论文提出了一种新的PPR模型，用于个性化POI推荐，该模型融合了用户的签到记录和社会关系。我们综合考虑用户- poi关系、顺序模式、地理效应和社会关系，构建异构图，学习用户和poi的表征。

- 论文提出了一种时空神经网络，通过将用户、POI嵌入和POI类别串联起来，生成个性化行为序列，对用户动态的个性化偏好进行建模。

# 2. Problem Definition

- Definition1(POI): $<p, l, cat>$
- Definition2(Check-in record): $c = <u, v, t>$
- Definition3(Trajectory): 一个用户的Check-in records的序列
- Defination4(Social Ties): 定义为一个图$G_u = ( U, E_u)$

# 3. Methodology

## 3.1 Heterogeneous Graph Construction

首先采用异构图$G=(V,U,E,W)$建模用户与poi之间的多重关系。V和U分别为POI和用户，E表示三种边分别为$E_u,E_v, E_{u,v}$，$E_u$表示用户之间的关系，$E_v$表示用户的访问轨迹，$E_{u,v}$表示某用户访问某POI的次数，W表示边的权重。

#### Modeling User-POI Relation.

 ![image-20221213024600752](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130246773.png)

表示用户$u_i$访问POI $v_j$的签到频率

#### Modeling Sequential and Geographical Effect.

$Δt^u_{k, k+1}$为轨迹中两个连续签入记录之间的时间间隔。

$l^u_{k,k+1}$表示轨迹中一对签入记录的连续状态。$θ$为时间阈值

 ![image-20221213025051032](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130250054.png)

某一边$e_{i,j}∈E_v$的序列权重$w^{(seq)}_{i,j}$定义为：

 ![image-20221213025340329](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130253353.png)

就是所有用户在他们的轨迹中首先访问$v_i$，然后访问$v_j$的总次数。

地理影响：

 ![image-20221213030812428](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130308452.png)

$N(v_i)$表示邻居，$d_{i,j}$表示欧氏距离

顺序影响和地理影响结合：

 ![image-20221213030827789](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130308811.png)

#### Modeling Social Tie Strength.

两个有社会联系的用户$u_i$和$u_j$，将边权值$w_{i,j}$赋值为:
 ![image-20221213031018830](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130310860.png)

#### Densifying Graph.

在图G的基础上构造一个稠密图，将每个用户和POI视为一个节点，并通过增加高阶邻居来扩展这些低进出度节点的邻居。只考虑将二阶邻居扩展到每个节点。如果G中某个节点的离度小于预定义的阈值ρ，创建一条从节点vi到其二阶外邻居节点$v_j$的边，并按如下方式分配权重:

 ![image-20221213031419475](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130314499.png)

$d^{(o)}_k$表示出度

## 3.2 Learning Latent Representation

large-scale information network embedding

![image-20221213032030211](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130320236.png)

通过ASGD(异步随机梯度)优化和边缘采样技术最小化目标函数O

## 3.3 Modeling User Dynamic and Personalized Preference

![image-20221213032146906](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130321939.png)

为了对用户动态和个性化偏好进行建模，将用户嵌入、POI嵌入和POI类别串联起来，生成一个新的更个性化的嵌入来表示签到记录。

ht和ct分别表示t时刻LSTM的隐藏状态和细胞状态。给定用户u及其轨迹序列Tu，首先将用户嵌入、POI嵌入与用户访问过的POI类别进行拼接，得到新的嵌入序列。其次，将所有用户的新嵌入序列输入LSTM网络。在输出层，连接了一个多层感知器(MLP)。使用以下目标函数来训练模型:

![image-20221213032820778](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130328803.png)

## 3.4 Personalized POI Recommendation

![image-20221213033225658](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130332682.png)

![image-20221213033311018](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130333039.png)

最后，根据推荐分数对所有poi进行排名，并选择前k的poi作为用户u在下一个τ时间段更有可能访问的候选poi。

# 4. Experiments

## 4.1 Datasets

![image-20221213021843676](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130218764.png)

## 4.2 Evaluation Metrics

Accuracy (Acc@k)

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130334778.png" alt="image-20221213033420754" style="zoom: 80%;" />

Precision (Pre@k)

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130334188.png" alt="image-20221213033432165" style="zoom:80%;" />

Recall (Rec@k) 

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130334799.png" alt="image-20221213033450773" style="zoom:80%;" />

Normalized Discounted Cumulative Gain (NDCG@k)

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130335062.png" alt="image-20221213033503038" style="zoom:80%;" />

## 4.3 Baselines

- Rank-GeoFM
- SR-RNN
- GE
- PEU-RNN
- SAE-NAD

## 4.4 Comparation study

![image-20221213022223836](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130222871.png)

![image-20221213022237261](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130222293.png)

![image-20221213022422267](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130224303.png)

## 4.5 Ablation Study

![image-20221213022533023](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212130225053.png)

PPR-RL

PPR-Seq

PPR-Den

PPR-GRU

# 5. Conclusion

提出了一种新的用于个性化POI推荐的时空表示学习模型。结合用户- poi关系、顺序效应、地理效应和社会纽带，构建了一个异构网络。然后，利用嵌入技术来学习用户和poi的潜在表征。鉴于RNN在序列预测问题上的成功，将用户和POI的级联嵌入序列输入到时空网络中，以捕捉用户的动态和个性化偏好。



