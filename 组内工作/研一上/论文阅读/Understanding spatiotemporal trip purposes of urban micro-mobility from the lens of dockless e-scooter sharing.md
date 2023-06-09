
# 摘要

本文旨在通过无桩电动滑板车用户行为的视角来理解城市微移动的时空出行目的。我们首先开发了一个时空主题建模方法来推断无桩电动滑板车使用的潜在出行目的。然后，以华盛顿特区为例，我们将该模型应用于包含83002次有效用户出行、19370个POI场馆和土地利用土地覆盖数据的数据集，系统地探索城市微移动的跨时空出行目的。研究结果证实，一组未发现的100个出行主题可以作为微移动用户时空出行目的的有效代理。本文的研究结果为城市当局和无桩电动滑板车公司在未来的智慧城市中制定更可持续的城市交通规划和更有效的车辆配置提供了重要的启示。

# 1. 介绍

最近，新兴的无桩微移动服务，如电动滑板车和电动自行车，为我们提供了前所未有的数据源，以更细的时空粒度揭示和理解城市移动模式。为了解决“最后一英里”问题，微移动用户可以直接到达他们最终目的地的前门。这一事实有助于通过挖掘海量用户出行数据更好地理解微移动用户活动的潜在时空出行目的。对这种微移动数据的时空分析可以在更细的时空尺度上揭示用户活动的出行目的。

本文对无桩电动滑板车出行目的的量化建模进行了时空语义分析，即跨时空的出行主题。为此，利用潜狄利克雷分配(LDA) (Blei, Ng， & Jordan, 2003)，将从Foursquare (fourquare, 2021)收集的兴趣点(POI)数据与无坞电动滑板车OD行程相结合，以揭示用户行程的出行目的及其时空模式。具体而言，本文旨在回答以下几个研究问题:

- 基于基于位置的社交媒体数据，更确切地说是Foursquare POI数据，我们能在多大程度上理解无桩电动滑板车旅行的目的?
- 空间因素(如土地利用和土地覆盖(LULC))和时间因素(如工作日、周末、高峰时间和非高峰时间)如何形成出行目的模式?

# 2. 相关工作

本文分析包括三个步骤。首先，针对无桩式电动滑板车应用程序接口(API)中人体移动数据的提取，提出了一种自动提取用户出行轨迹的方法，并利用该方法识别用户出行轨迹的OD位置。然后，利用LDA主题模型进一步挖掘POI集群及其对应的类别，作为OD位置语义上下文的代理，发现无桩电动滑板车使用的底层Trips Topics。此外，还讨论了Trips主题的时空格局。需要注意的是，这里我们选择LDA进行主题建模，因为它不需要预定义的关键字，也不需要语法逻辑，因此LDA被认为是其他主题建模方法中最适合的。

# 3. 数据描述

## 3.1 无桩电动滑板数据

在本文中，选择华盛顿特区作为我们的研究区域。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241723664.png)

目前可用的无桩电动滑板车的快照数据集包括地理坐标、车辆ID、时间戳和电池电量。结果，我们提取了主要来自两个微移动供应商(即Bird和Jump)的83002次有效用户出行，提取了每次出行的时间戳和OD坐标(表1)。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241840066.png)

如图2所示，除了准确的OD时间戳，我们还将出行开始时间设置为最近的小时，并将其聚合到一天的小时和一周的天，以可视化出行分布。值得注意的是，由于没有个人用户的信息，所有的用户旅行实际上都是匿名的。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241840487.png)

## 3.2 POI数据

华盛顿特区城市场馆的POI数据来自foursquare——一个基于位置的社交媒体平台。Foursquare拥有世界上最大的POI数据库之一，其中包括超过1.05亿POI记录，包括各种场所及其语义信息，从主要和次要的POI类别、姓名和地址，到用户喜欢和排名。此外，来自Foursquare的一个专门的、用户友好的场所API允许用户在一定距离内搜索附近的POIs。因此，我们设置了300米的缓冲区，以确定距离电动滑板车用户出行OD位置较近的场馆。我们的主要假设是，微移动用户倾向于打开离出发地最近的电动滑板车，然后直接前往目的地地点的前门。然而，这一假设可能并不适用于每一次电动滑板车旅行。我们测试了不同的缓冲距离，发现300米为无桩电动滑板车用户的OD场地提供了合理的估计。

通过网格搜索策略，我们的POI数据集包含来自9个主要类别的19370个场所，分别是艺术与娱乐、学院与大学、食品、专业及其他场所、夜生活与场所、户外与娱乐、商店与服务、旅游与交通和住宅。然后，缓冲83002个有效用户行程的OD位置，以指定相应的POI集群。更重要的是，将电动滑板车用户出行的OD位置作为连接相关城市人类活动与POI集群所代表的地理语义信息的桥梁，使我们能够理解城市微移动的真实出行目的。

## 3.3 辅助数据

使用LULC层与OD数据叠加，并说明城市空间是如何在无桩电动滑板车用户出行的OD位置周围实际使用的。因此，我们的研究区域被划分为5个LULC类别，即**商业(例如，办公区和商业区)、住宅(例如，低、中、中、高密度住宅区)、公共和娱乐(例如，机构、联邦、准公共和娱乐)、公园和开放空间，以及工业**。

使用工作日的高峰时间、工作日的非高峰时间和周末。在华盛顿特区，工作日(星期一至星期五)早上6时至9时30分为早高峰，工作日(星期一至星期五)下午3时30分至6时30分为晚高峰。在此背景下，**将所有无桩电动滑板车出行分为4个时间段，分别为周末、工作日早高峰、工作日下午高峰和工作日非高峰。**

# 4. 方法

基于收集到的数据集，我们将每个**有效的用户行程(包括OD位置、时间戳、LULC信息和相应的POI集群)**
视为LDA主题建模的单个文档，随后由提取的LDA主题的概率分布表示。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241839986.png)

## 4.1 地理语料库的定义

- 定义1

  假设u是一个用户行程，那么我们用数学方法将用户行程的行程目的表述为一个未知函数$f(x^u)$

  $x_u$定义为一组时空语义属性${p _{(O, D)}, t _{(O, D)}, l _{(O, D)}, c _{(O, D)}}$。

  $p _{(O, D)}, t _{(O, D)}$分别为行程OD位置的地理坐标和时间戳。

  $l _{(O, D)}, c _{(O, D)}$分别为$p_{(O,D)}$附近的显性LULC类和POI场所群

- 定义2

  给定$x_u$的POI簇$c _{(O, D)}$，设M为出行目的语料库，该语料库由LDA主题建模中分别用于开始出行和前往某地的两类出行目的词$\{w_O,w_D\}$组成。接下来，我们将c~O~(即$c_O^i$)中的第$i$个POI的$\{w_O\}$定义为三元组${\{ct,od,na\}}c_O^i$，其中$ct,od,na$分别表示主要类别、起点索引(即1)或目的索引(即2)和$c_O^i$的详细类别。同样，$\{w_D\}$也可以用类似的方式定义。

- 定义3

  基于旅行目的词语料库$M$~$\{W_O, W_D\}$, 可为无桩电动滑板车用户出行u构造LDA主题建模文档du如下:

  ![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241851924.png)

  作为更好地理解这些定义的显式示例，图4解释了使用无桩电动滑板车OD行程和POI集群构建LDA主题建模文档的详细过程。

  ![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241852806.png)

  首先，以等权重的方式聚类一个POI场馆列表(即OD位置300米缓冲区内的场馆)。我们的方法侧重于揭示个体层面的微移动出行目的。因此，到访某些POI场地的概率与场地受欢迎程度的关系较小，而更多受使用者的时空活动驱动。例如，一个体育场可以在白天比夜间吸引更多的观众，从而获得大量的用户点赞。而附近的夜生活酒吧在夜间对电动滑板车用户更有吸引力。其次，我们使用POI场馆的主要和详细类别，而不是POI名称，因为这些类别使我们能够比场馆名称更好地识别场所的功能。例如，夜生活场所(如酒吧)的功能，用鸡尾酒吧的详细分类比用真正的POI名称更容易理解。

  ![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241902168.png)

  图5图形化地解释了我们的模型到主题模型的映射。显然，从下到上主要有三个关键组件，它们是主题模型中的单词、主题和文档。首先，将用户行程与POI集群进行集成，生成其OD位置的基本词，这些词对应于主题模型中的词;其次，通过LDA主题建模将Trips Topics提取为与主题对应的单词集合. 最后，将用户出行文档用主题分布表示，与主题模型中的文档对应。例如，给定一个有效的用户出行du，模型得到一个具有不同概率(如0.45,0.2,0.01)的主题分布作为其出行目的模式的代理，该主题分布可以根据不同的时空因素进一步聚合。

  ![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241902304.png)

## 4.2 潜在迪利克雷分布

LDA作为一种生成概率主题建模方法，已被广泛用于从大型语料库中查找自然组。此外，LDA不需要预定义的关键字或文档主题。与现有的基于聚类方法的研究不同，本文采用LDA方法通过结合POI数据和OD行程来揭示无桩电动滑板车出行目的的语义组，其实现基于R中的主题模型包.

图6显示了LDA的图形模型，其中每个文档被视为主题的混合，每个主题被表示为单词的混合。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241904684.png)

首先，给定K个主题，每个主题K都与一组单词概率的分布(即$ϕ_k$)相关，该分布可由一个带超参数β的狄利克雷分布估计，该分布用$Dirichlet(β)$表示。类似地，由一组词$w_{du}$组成的文档$d_u$，是通过从给定超参数α的狄利克雷分布(称为Dirichlet(α)，为$w_{du}$中的每个词分配主题的狄利克雷分布中对K个主题概率(即θdu)的分布进行抽样得到的。为了生成一个具有$θ_{du}$的词$w_{d_u^i}$，每个主题$Z_{d_u^i}$基于一个多项式分布$Multinominal(θ_{du})$从K个主题中采样，并且词$w_{d_u^i}$从另一个多项式分布$Multinominal(ϕ_{Z_{d_u^i}})$中派生。简而言之，这些过程概括为三个步骤:

1. 用$ϕ_k$~$Dirichlet(β)$确定每一个主题k的词分布$ϕ_k$

2. 用$θ_{du}$~$Dirichlet (α)$确定每个文档du的主题分布θdu

3. 对于每个文档du和du中的每个单词$w_{d_u^i}$:

   3.1 通过$Multinominal(θ_{du})$来指定一个主题$Z_{d_u^i}$

   3.2 从$Multinominal(ϕ_{Z_{d_u^i}})$中选取一个词$w_{d_u^i}$

吉布斯抽样作为马尔可夫链蒙特卡罗(MCMC)的一种形式，考虑到从大数据集中提取主题的效率，采用吉布斯抽样估计狄利克雷(α)和狄利克雷(β)的两个后发分布。另一方面，吉布斯抽样能够在数据的低维子集上模拟复杂(大多是复杂和高维)的分布，其中所有其他数据都影响每个子集。在LDA的R实现中，吉布斯抽样迭代运行了500个周期。

此外，对LDA模型的训练要求的是应当提供主题数量K。我们使用perplexity作为对不同主题数量K的模型泛化性能的度量，其中perplexity值越小表示泛化性能越好。在代数上，困惑相当于每个词的几何平均似然的非标准化逆。对于一组文件du∈[1,M]，其perplexity可计算如下：

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241923799.png)

## 4.3 时空主题

鉴于LDA模型的主题分布，我们感兴趣的是旅行目的模式如何在不同地点和不同时间段之间变化。为此，我们进一步研究了基于以下时空因素的主题分布模式: **LULC类别、工作日高峰期、工作日非高峰期和周末**。根据不同时空因素提取主题分布，计算主题子分布之间的相似度，衡量其差异。

### 4.3.1 主题分布的空间因素

基于4.2节LDA模型估计的主题分布，利用以下公式推导每个LULC类的主题分布:

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241925574.png)

$Dis.^L(T^k)$是第k个特定L类下LULC类的主题分布，$P(T_i^k|L)$是LULC L类中第k个主题下的旅行文档的第$i$个主题分布.

实际上，式(3)可以看作是不同LULC类主题分布的部分聚合，进一步揭示了无桩电动滑板车出行目的的空间格局。

### 4.3.2 主题分布的时间因素

此外，为了描述出行目的的时间模式，将主题分布进一步分解为4个时间段(周末、工作日早高峰、工作日晚高峰和工作日非高峰)。每个时间段的主题分布计算如下:

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210241926518.png)

$Dis.^W(T^k)$ 表示在特定时间段W下的第k个主题分布， $P(T_j^k|W)$表示第j个W时间段内第k个主题下的行程文档的主题分布，M为W时间内行程文档的数量

结果分别提取出周末、工作日早高峰、工作日晚高峰和工作日非高峰4个时间段的4个主题子分布。



## 4.4 相似性与假设检验

为了衡量Trips主题分布在空间和时间上的显著性差异，我们使用相似性度量和统计检验来检验它们之间(即基于空间和时间因素)的相似性(或不相似性)。
Jensen - Shannon (JS)散度，又称信息半径(IRad) ，用来衡量a和b两个主题分布之间的相似度。JS散度的计算方法如下:
$$
JS(a,b)= \frac{1}{2}[KL(a,b)+KL(a,b)]
$$
$$
KL(a,b)=\sum_{i=1}^{K} a_ilog_2\frac{a_i}{b_i}
$$
其中KL为Kullback-Leibler散度， K为主题数。

除了JS差异，我们还应用Watson的U^2^同质性检验来衡量每对主题分布之间差异的显著性水平。作为Cram´er-von Mises拟合优度检验的修正版本，Watson 's U^2^检验在统计上检验两个分布(如来自LDA的Trips Topics分布)之间的差异有多显著。更具体地说，零假设是两个分布实际上是来自同一个分布的样本。因此，拒绝零假设意味着两个主题分布显著不同。设显著性水平为0.001,0.05,0.1，则临界值分别为0.385,0.181,0.152。

# 5. 结果

## 5.1 无桩电动滑板车出行主题

在本文中，采用LDA主题模型对83002次无桩电动滑板车有效行程进行了探索，其中80%的数据用于模型训练，其余数据根据Eq.(2)定义的困惑度值进行模型评价。如图7所示，当主题数达到100时，困惑度曲线趋于相对平坦。因此，我们主观决定使用K = 100作为Trips Topics的数量。根据Steyvers和Griffiths(2007)的建议，LDA模型的另外两个超参数(α和β)分别设为0.1和0.01。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210251952063.png)

Trips Topic主要有3种类型，如关于用户出发地点的主题(如Trips Topic #38)， OD索引为1。类似地，有关于目的地的主题，如Trips Topic#97，也有关于两个O&D地点的主题(如Trips Topic#57)。为了更好地理解这些变体，出发地Trips Topic指的是那些涉及到围绕出发位置的公共POI类别的用户旅行，而对于它们的目的地位置则没有必要这样做。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210251953545.png)

Trips Topic # 38是刚从夜店酒吧出来的用户，Trips Topic#57更可能是正在旅游的用户，Trips Topic#97可能是大学生或教师

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252008876.png)

表3为单个用户行程(即用户行程#46625、#77795和#14471)提供了三个选定的主题分布示例，它们由前面提到的行程主题#38、#57和#97所主导。通过结合词汇分布和话题分布，我们可以定量地推断出个人出行的目的。例如，用户旅行#14471显然是目的地驱动的，因为目的地旅行主题#97是主题分布中的主要主题(即，以0.4574的概率)，而用户旅行#46625更有可能被归类为关注起源的旅行，因为在主题分布中，起源旅行主题#38的概率值高于旅行主题#40的概率值。结果，分别为83002次有效的无桩电动滑板车旅行计算了100个Trips Topics的独特主题分布。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252019783.png)

图8为不同Trips topic的直方图以及累积的话题概率，其中每个用户的出行根据其话题分布的最高概率值被划分到特定的Trips topic中。

## 5.2 时间主题模式

城市人口流动模式从一天中的几个小时和一周中的几天不断地以有规律的方式演变。这种时间模式也可以从图9中行程流的桑基图(例如，从出发地到目的地)中观察到，其中LULC类被用作空间因子来聚类OD位置。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252023634.png)

从视觉对比中，我们发现了几个有趣的发现:

- 在工作日高峰时段，上午从住宅区前往商业区的车流量最多，下午从商业区前往住宅区的车流量最多。
- 在非高峰时段，出行流比高峰时段更为平衡，LULC类之间的车流对称，这与中相似的研究结果相呼应
- 在周末，公众(公共及康乐)和公园(公园及休憩用地)之间的行程明显增加。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252025936.png)

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252027883.png)

我们生成了4个详细的直方图(即Trips Topics数量)，以及图10中不同时间段累计的主题概率(即前10个主题概率)。根据图10，我们选择了累计主题概率最高的Trips主题#75、#59和#97，并在表4中详细说明了它们的前10个单词分布。例如，在工作日早高峰时段，Trips Topic # 75给出的出行目的模式与用户工作地点(如办公室、科技创业公司、会议室)和咖啡店结束的电动滑板车出行有关，主要通过无桩电动滑板车服务确定用户每天早上通勤的目的地。这意味着人们倾向于使用无桩电动滑板车的目的是在工作日的早上通勤的最后一英里。然而，在Trips Topic # 75中，并没有从数据中观察到他们来自哪里。类似地，对于工作日非高峰时间，Trips Topic # 97如表2所示，演示了在大学校园(如演讲厅、宿舍和图书馆)停放电动滑板车的临时旅行目的模式，这可能是由学生和/或大学工作人员进行的。有趣的是，我们可以观察到关于观光和旅游目的的Trips Topic # 59对工作日下午高峰时间和周末模式的显著影响，在周末模式下，用户最有可能在博物馆、纪念碑和雕塑附近选择电动滑板车。这一事实表明，使用电动滑板车上下班在早高峰时段比下午高峰时段更受欢迎。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252030998.png)

在表5中我们详细阐述了不同时间主题分布的两种相似度量，即JS散度和Waston’s U2两样本检验，其中零假设指的是两个主题分布没有显著差异。如前文所述，在工作日非高峰时段和周末时段，城市微移动出行目的模式高度相似(即临界值为0.0193)，导致JS发散值最小为0.0269，零假设不被否定。反之，工作日早高峰时段的Trips Topics分布与其他时段明显不同，这可以解释为用户早通勤活动的强烈影响。

## 5.3 空间主题模式

城市土地利用本质上是异质性的。这种异质性在一定程度上受到人类流动性活动的影响。因此，本文将Trips Topics的分布进一步分解为5个LULC类的子分布。目的是探讨如何将行程目的模式与不同的LULC类相关联。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252032736.png)

在深入研究主题建模结果之前，图11展示了LULC指定的时间模式，从中我们注意到以下发现:

- 住宅和商业区是用户出行最多的区域，工作日和周末之间没有显著差异。
- 周末前往公众及康乐、公园及休憩用地及工业地区的旅客人数显著增加。
- 在工作日早高峰时段，电动滑板车的使用高峰主要出现在商业区，而这种模式在公园和开放空间很少出现。
- ![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252110080.png)

此外，图12中这些由lulc指定的Trips Topics分布使我们能够探索相应的trip目的模式。一般来说，住宅和商业区没有特别的高峰，但在公共和娱乐、公园和开放空间和工业区有显著流行的话题。这些观察告诉我们，不同的用户活动更可能涉及到住宅和商业区域，而人们使用无桩电动滑板车作为交通工具到其他三个区域，通常是出于特定的原因，如学习，健身，或观光。除了这些观察结果，我们还研究了图12中的Trips主题#35、#75、#97、#57和#73，并在表6中探索了它们的前10个单词分布。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252111637.png)

对于住宅和商业区域，旅行主题#35和#75都关于用户目的地的地点，#35涵盖了人们在日常生活中经常光顾的商店(如咖啡店、银行、洗衣房)，#75涉及写字楼和工作场所。在这里，这种模式也与前面提到的工作日早高峰对商业区的强烈影响是一致的。毫不奇怪，与大学活动相关的旅行话题#97在公共和娱乐区域占主导地位，而关于旅游景点之间的往返或直接旅行的旅行话题#57在公园和开放空间区域的累计概率最高。在工业区域，虽然用户出行很少，但trips Topic # 73成功地揭示了无桩电动滑板车用户的运动和健身活动(如去健身房、游泳池和健身工作室)。

考虑到LULC类的内在异质性，在表7中我们可以直观地观察到5个LULC类之间的高度多样性和显著性差异，其中大多数拒绝具有高显著性水平(即0.001)。然而，人们可能会注意到，在公共、娱乐和公园和开放空间之间有类似的主题分布。一个潜在的原因可能是LULC区域都由一些常见的Trips主题所主导。对这种相似性的深入理解将有助于解释，但这超出了本文的范围。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252116112.png)

# 6. 讨论

基于第5节中所揭示的时空旅行目的模式，我们可以进一步探索单个旅行主题的时空模式，例如图13中的#59和#97，我们将基于不同时间段的用户旅行的OD位置连接起来。如表4和图10所示，#59是一个与观光旅游活动相关的origin Trips Topic，主要在工作日下午高峰时段和周末时段占主导。这些发现可以通过图13中的观察得到证实，我们确定了美国国会大厦和史密森尼地铁站周围的起源地点集群，特别是在工作日下午高峰时间和周末。有趣的是，我们注意到用户目的地的高度多样性，这意味着这些电动滑板车用户以相似的旅行目的开始旅行，但最终前往市中心以外的不同目的地。因此，这也是为什么#59只涉及起源POI的单词的原因。类似地，在目的地旅行主题#97中，主要是在乔治华盛顿大学医院和主大学校园周围的目的地地点集群，在工作日非高峰时间有较大的旅行量。

![](https://nnpicture.oss-cn-hangzhou.aliyuncs.com/picture202210252119476.png)

从无桩电动滑板车用户出行推断出的出行主题和Foursquare POIs对于理解城市微移动活动的时空出行目的确实是有意义和有效的。然而，由于具体原因，仍有局限性，应在今后的工作中加以解决:

- 未揭示的出行目的模式的粒度受限于本文考虑的时空因素(如LULC类、高峰期与非高峰期、工作日与周末)。因此，我们无法实时、实时地捕捉电动滑板车用户的出行目的。一个主要原因是收集到的无桩电动滑板车数据集并没有覆盖整个行程密度相当大的时间段。此外，本文没有考虑无桩电动滑板车使用量的月度和季节性变化，而主题子分布在时间因子上的经验教训为这种分析提供了可行的方法。未来，我们将扩展我们的数据集，包括更多有效的用户旅行，以及覆盖多个季节的更长的时间周期。
- 由于现有微移动服务的可达性有限，本研究主要关注无桩电动滑板车服务的使用。我们相信，如果未来更多的相关数据集向研究人员开放，将会对城市微流动的出行目的有更全面的理解。
- 潜在的社会人口统计和城市环境因素可能给无桩电动滑板车OD数据带来偏差，本研究未考虑其影响。尤尼斯等人(2020年)对这种影响进行了早期的调查。
- 作为一个潜在的错误源，我们用来提取行程目的语义的foursquare POI数据给所提出的方法带来了一些限制。例如，虽然考虑了等权重的POI集群，但开放时间等因素可能导致与关闭的POI场馆相关的错误分类。因此，鼓励今后的工作将POI层面的时间性纳入到用户出行目的的推断中。

除了工作日早高峰通勤出行的确认发现，我们的分析发现，用户出行的动机主要是周末甚至工作日下午高峰的观光和旅游活动，而大学工作人员和学生出于学术目的的出行对华盛顿工作日非高峰的trips Topics概率分布有很强的影响。所有这些结果表明，城市特定的出行目的和时空无桩电动滑板车使用之间存在着很强的潜在关系。因此，根据大量无桩电动滑板车OD数据和基于位置的社交媒体POI数据对出行目的进行量化建模，可以通过考虑特定城市的环境来预测微移动用户行为。
