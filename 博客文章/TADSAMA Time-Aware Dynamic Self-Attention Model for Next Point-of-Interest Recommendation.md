---
title: "TADSAM:A Time-Aware Dynamic Self-Attention Model for Next Point-of-Interest Recommendation"
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



# TADSAM:A Time-Aware Dynamic Self-Attention Model for Next Point-of-Interest Recommendation

# 1. Introduction

论文提出了一个时间感知动态自我注意模型TADSAM来预测用户未来的下一步决策活动。首先，我们使用一种扩展的自我注意机制来处理用户复杂的签到记录。其次，考虑到时间的影响，我们将用户签到记录划分到不同的时间窗口，并开发了一种个性化的权重计算方法来挖掘用户行为的时间模式。实验结果表明，该方法在稀疏签入数据的下一个POI推荐方面优于新模型

# 2. Challenge

1. 随着时间的变化，用户的历史行为表现出多样性和复杂性;

2. 由于用户轨迹数据的稀疏性，很难在相应的时间模式中捕捉到用户的偏好

# 3. Contribution

- 论文提出了一种时间感知动态自我注意网络方法。使用一个相对时间矩阵来扩展传统的自我注意机制，以探索用户签到记录之间的关系。通过这种方法，不仅可以探索用户复杂多样的行为模式，还可以了解用户兴趣的变化。

- 我们将用户的历史签到序列划分为不同的时间窗口，并设计了一种个性化的权重计算方法，以更有效地探索用户行为的时间模式。

- 我们结合POI的流行度和地理影响来提高系统的性能，克服了数据稀疏的问题，方便了系统的检索。

# 4. Method

## 4.1 Problem Formulation

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051422352.png" alt="image-20221205142233330" style="zoom: 67%;" />

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051422339.png" alt="image-20221205142257321" style="zoom:67%;" />

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051423873.png" alt="image-20221205142317854" style="zoom:67%;" />

 <img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051423133.png" alt="image-20221205142340113" style="zoom:67%;" />

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051425523.png" alt="image-20221205142553499" style="zoom: 67%;" />

## 4.2 Model Framework

![image-20221205141917269](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051419366.png)

## 4.3 Embedding Layer

嵌入层由用户的历史轨迹矩阵和相对时间矩阵组成。嵌入层主要研究用户轨迹矩阵和时间相对矩阵的潜在表示。用户签到记录的原始时间戳是离散的，因此我们将时间映射到168个维度，对应一周168个小时，这有助于了解用户访问某地的具体时间，并反映出周期性。

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060027345.png" alt="image-20221206002726323" style="zoom:50%;" />

时间相对矩阵扩展了传统的自我注意机制，有效地反映了用户轨迹序列的关系。相对较短的时间表明两个POI之间有很强的依赖性。

## 4.4 Extended Self-attention Layer

提出了一个时间感知的自注意模型来输入时间动态轨迹序列。将自我注意扩展到考虑序列中两个访问位置之间的时间间隔。

自注意层包括两个子层:多头自注意层和前馈网络层。首先，利用多头注意机制，从不同角度分析用户具有时间信息的历史轨迹，挖掘用户的各种偏好;具体实现如下:

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060029719.png" alt="image-20221206002947694" style="zoom:50%;" />

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060031737.png" alt="image-20221206003105717" style="zoom:50%;" />

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060031436.png" alt="image-20221206003143412" style="zoom: 50%;" />

使用softmax函数计算各权重系数：

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060032736.png" alt="image-20221206003250715" style="zoom:50%;" />

在时间感知注意层之后，应用前馈网络对模型进行非线性赋值，提高了模型的表达能力：

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060034928.png" alt="image-20221206003402905" style="zoom:50%;" />

在扩展的多头自注意层和前馈网络叠加之后，会出现更多的问题，如过拟合、训练过程不稳定、需要更多的训练时间等。因此，在多头注意机制和前馈网络层之后，我们增加了层归一化、残差连接和dropout技术来解决这些问题，具体表现为:

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060035107.png" alt="image-20221206003520086" style="zoom: 67%;" />

SA (Eu)表示经过dropout技术、残差链接和层归一化处理的多头自注意层的结果。同样Au是FFN处理的结果，也相当于最终的输出。输出是用户复杂的偏好矩阵，包含时间信息和来自签入序列的相关性。

## 4.5 Time Partten Layer

因为用户的日常活动分布不均匀，所以将一天平均分配是不明智的。论文通过使用用户在一天中的访问行为的概率密度来确定这段时间的跨度。CatDM采用个性化的权重计算方法，准确获取各时间窗口的利益，取得了良好的效果。因此，我们用同样的方式来探索用户时间偏好。我们设计了多个窗口来区分用户的行为，并提取一天中不同时间段的时钟影响，如图2所示。每个窗口$W^u_i$定义如下:

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060043081.png" alt="image-20221206004305058" style="zoom:67%;" />

划分12个时间窗口，让每个时间窗口的概率密度为1 12。让时间窗的等效概率密度来决定每个时间窗的时间间隔。可以把每个窗口看作是用户行为的子序列，这意味着每个时间窗口代表了用户在时间段内的短期偏好。为了捕获用户对每个时间窗口的兴趣，我们使用窗口状态来组合每个时间窗口的用户签入行为。设$|W^u_i|$表示每个时间窗口内的签入数。大小为d的各时间窗口状态Wi如下:

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060043454.png" alt="image-20221206004333435" style="zoom:67%;" />

## 4.6 Next POI Prediction

将地理因素和流行度纳入模型。首先，根据POI的距离，删除一些远离用户的POI; 在POI推荐上，用户行为服从幂律分布，我们过滤一些远离用户当前位置的POI，得到一批初始过滤的POI候选集。因为POI经常被访问，所以POI更受欢迎。因此，根据POIs的受欢迎程度，我们剔除了一些不受欢迎的POIs。一个兴趣点的受欢迎程度等于这个结果，用访问该兴趣点的用户数除以一天内访问所有兴趣点的用户数。最后，我们得到了一批新的更接近和更流行的POI候选集和他们的embedding:

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060046168.png" alt="image-20221206004635147" style="zoom:50%;" />

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060047145.png" alt="image-20221206004706126" style="zoom:50%;" />

然后过滤掉那些距离用户当前位置超过5公里且受欢迎度为0.5的poi。

得到每个用户的候选poi评分：

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060048596.png" alt="image-20221206004832577" style="zoom:80%;" />

在上述公式中，第一项表示用户对候选位置的偏好，第二项表示我们将预测的时间与候选poi之间的相关性。分数越高，用户就越有可能访问候选POI。最后，给出一个top-K推荐列表，并根据计算出的分数发送给用户。

## 4.7 Model Training

![image-20221205150628133](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051506157.png)

# 5. Experiment

## 5.1 Experimental Setup

### 5.1.1 Dataset

![image-20221206001939978](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060019004.png)

### 5.1.2 Baseline Methods

- POP
- FPMC−LR
- PRME−G
- STRNN
- TMCA
- STGN

### 5.1.3 Evaluation Metrics 

- Recall@N
- NDCG@N

## 5.2 Performance Comparison

![image-20221206001653875](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060016921.png)

## 5.3 Impact of Different Components of TADSAM

$TADSAM_{NW}$表示一个没有分割窗口的简单模型。本文利用当天用户签到行为的概率密度来分配时间窗口。

$TADSAM_{AW}$表示被分割平均的时间窗口模型，用时间间隔矩阵来捕捉用户两种登录行为之间的关系。

$TADSAM_{NT}$建模用户的整个签到序列的自注意力机制，以捕获用户的偏好。

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212060013043.png" alt="image-20221206001259972" style="zoom:150%;" />

## 5.4 Influence of Hyper-parameters

TADSAM模型中有两个关键的超参数:嵌入维度和自注意块数。

当维数超过32时，模型的性能趋于稳定；实验结果表明，增加自注意块并不能提高模型的推荐性能。这种情况可能是模型参数随着自注意块的增加而产生的过拟合问题。

![image-20221205152332651](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051523687.png)

## 5.5 Analyze the impact of filtering

<img src="https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202212051521669.png" alt="image-20221205152112629" style="zoom:80%;" />

## 5.6 Analysis of results

利用用户轨迹的时间相对矩阵扩展了传统的自我注意机制，获取了用户轨迹的时间上下文信息。为了进一步探索用户轨迹的时间模式，我们添加了不同的时间窗口来确定用户在每个时间段的偏好。通过比较模型的成分实验结果，时间窗口的加入显著提高了推荐性能。



# 6. Conclusion

首先，利用相对时间矩阵展开自我注意块，建立动态时间关系，提取用户的不同兴趣偏好;其次，将获得的偏好特征划分到不同的时间段，并使用一种巧妙的线性组合方法计算用户的偏好。最后，根据筛选后的候选POI，计算用户访问每个候选POI的概率。