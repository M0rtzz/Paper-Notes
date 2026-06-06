---
title: >-
  [论文解读] When Do Graph Foundation Models Transfer? A Data-Centric Theory
description: >-
  [ICML2026][图学习][图基础模型] 这篇论文用 graphon 把不同大小、不同域的图放到同一连续空间里，证明图基础模型的跨域输出差异可以分解为两个有限采样误差和一个内在 graphon 域差异，并用合成与真实图实验说明图大小、结构偏移和谱位置编码稳定性共同决定迁移成败。
tags:
  - "ICML2026"
  - "图学习"
  - "图基础模型"
  - "graphon"
  - "迁移理论"
  - "位置编码"
  - "图数据策展"
---

# When Do Graph Foundation Models Transfer? A Data-Centric Theory

**会议**: ICML2026  
**arXiv**: [2605.29828](https://arxiv.org/abs/2605.29828)  
**代码**: https://github.com/zhuconv/GraphFM  
**领域**: graph_learning  
**关键词**: 图基础模型, graphon, 迁移理论, 位置编码, 图数据策展

## 一句话总结
这篇论文用 graphon 把不同大小、不同域的图放到同一连续空间里，证明图基础模型的跨域输出差异可以分解为两个有限采样误差和一个内在 graphon 域差异，并用合成与真实图实验说明图大小、结构偏移和谱位置编码稳定性共同决定迁移成败。

## 研究背景与动机
**领域现状**：图基础模型希望像语言基础模型一样，用一个预训练 backbone 迁移到多种图域，例如社交网络、推荐、生物和安全图。近期工作更多从模型侧推进，包括更大的图预训练、更复杂的 tokenization、MoE 路由、图 prompt 和适配器。

**现有痛点**：图数据的异质性远强于文本。不同域的图不仅特征和标签语义不同，节点规模、连接模式、谱结构和适合的归纳偏置也不同。因此，同一个 GFM 在某些域可以正迁移，在另一些域可能负迁移。现有方法往往给出架构或训练技巧，却缺少一个回答“两个图域本身有多难迁移”的数据中心理论。

**核心矛盾**：如果只用有限图之间的距离来判断迁移难度，距离会混入采样噪声和规模差异；如果只看模型表示距离，又会被具体 backbone 的信息损失影响。论文想分开两件事：有限图是不是足够接近其潜在生成机制，以及两个潜在图域本身是否相似。

**本文目标**：作者希望建立一种 size-independent 的图域比较方式，在固定 Lipschitz backbone 下，把跨图输出变化拆成可解释的项，并把这个分解转化为图基础模型训练和数据策展建议。

**切入角度**：graphon 是 dense graph 的连续极限对象，可以表示潜在图生成机制。把有限图嵌入为 step-graphon 后，不同大小的图就能在同一个算子空间中比较，而图域之间的差异也可以写成 graphon operator 的 relabeling-invariant discrepancy。

**核心 idea**：先把有限图与 latent graphon 的误差分离出来，再用 graphon mismatch 描述真正的域差异；对带谱位置编码的 set backbone 和 message-passing backbone，都能得到同样的“采样误差 + graphon 域差异 + 采样误差”迁移上界。

## 方法详解
论文主要是理论和受控实验结合。理论部分先定义图、graphon、两类 backbone 和两种谱位置编码，再逐步证明：有限图可以等价嵌入为 step-graphon；同一个 graphon 采样出的图越大，step-graphon 越接近原 graphon；两个图之间的模型输出差异可以由各自采样误差和两个 graphon 的内在差异控制。

### 整体框架
给定一个有限图 $G$，论文把归一化图算子 $\Delta=A/n$ 转成 step-graphon $W_\Delta$。这样，大小为 128、512 或 2048 的图都不再只是不同维度矩阵，而是 $[0,1]^2$ 上的分段函数。对于两个从 graphon $W^{(1)}$ 与 $W^{(2)}$ 采样的图 $G^{(1)}$ 与 $G^{(2)}$，模型输出差异可以按路径拆分：$G^{(1)}$ 到 $W^{(1)}$ 的采样误差，$W^{(1)}$ 到 $W^{(2)}$ 的 graphon 域差异，再从 $W^{(2)}$ 到 $G^{(2)}$ 的采样误差。

这个分解对两类常见 GFM 接口都成立。第一类是 set-based backbone，例如 DeepSets 或 Graph Transformer，它把图结构编码进 positional embedding 后再做集合读出。第二类是 message-passing backbone，例如 GCN/GIN，它同时依赖图算子和节点 token。论文用 Lipschitz 条件把 backbone 对输入 token 和图算子扰动的敏感性包进常数 $L_\theta$。

### 关键设计
1. **用 step-graphon 统一不同大小的图**:

    - 功能：让跨尺寸图比较从矩阵维度不一致的问题，变成同一函数空间中的算子距离问题。
    - 核心思路：把大小为 $n$ 的图算子 $\Delta$ 映射成 $W_\Delta(u,v)=\sum_{i,j}n\Delta_{ij}1_{P_i}(u)1_{P_j}(v)$。图上的离散邻域平均对应 step-graphon 上的积分，因此模型操作可以在连续空间里对齐。
    - 设计动机：GFM 迁移经常跨图大小，如果没有 size-independent 表示，就很难区分“图太小导致近似差”和“图域本身不同”。

2. **输出差异的三项误差分解**:

    - 功能：把迁移难度拆成有限采样误差和内在域差异。
    - 核心思路：在 Lipschitz backbone 和稳定 PE 假设下，set backbone 有 $\|f_\theta(t_{G^{(1)}})-f_\theta(t_{G^{(2)}})\|\le L_\theta C_{PE}(\epsilon_1+\epsilon_{gra}+\epsilon_2)$；message-passing 版本多了图算子扰动项，常数变成 $L_\theta(1+C_{PE})$。
    - 设计动机：这个式子直接告诉我们，提升迁移不能只扩大图规模，也不能只做域混合；采样充分性和 latent graphon mismatch 都会成为瓶颈。

3. **把谱位置编码稳定性纳入迁移上界**:

    - 功能：解释为什么 PE 维度和 eigengap 会影响 GFM 跨域泛化。
    - 核心思路：Eig-PE 的稳定常数随 $\sqrt{k}\max_{\ell\le k}1/\gamma_\ell$ 增大，小 eigengap 会放大特征向量扰动；Proj-PE 使用 top-$k$ 子空间投影，理论上对基旋转更不敏感，但仍依赖子空间 gap。
    - 设计动机：图 Transformer 和许多 GNN 都依赖谱 PE。论文把 PE 不稳定从经验调参问题变成理论项 $C_{PE}$，说明过大的 PE 维度可能提高表达力，也可能放大域 shift。

### 损失函数 / 训练策略
理论部分不提出新的训练损失。实验中，作者构造低秩 Fourier graphon 分类任务，固定总节点预算 100k，在训练尺寸 $n\in\{128,256,512,1024\}$ 间用混合参数 $\lambda$ 调整大小分布；测试包含 $n\in\{128,256,512,1024,2048\}$，其中 2048 是 OOD 尺寸。默认模型为 DeepSets + top-32 Eig-PE，也用 GIN 验证 message-passing 情形。真实图实验中，作者估计每类图的 class-specific graphon，再采样少量合成图做 graph merging augmentation。

## 实验关键数据

### 主实验
真实数据实验展示了 graphon-based augmentation 在 COLLAB、IMDB-BINARY、REDDIT-BINARY 上的效果。指标是 test error，越低越好。

| 合成图比例 | COLLAB | IMDB-BINARY | REDDIT-BINARY | 观察 |
|------------|--------|-------------|---------------|------|
| Vanilla | 0.4384 | 0.4631 | 0.4108 | 不做 augmentation |
| 1% | 0.4069 | 0.4631 | 0.4367 | COLLAB 最优，REDDIT 变差 |
| 2% | 0.4428 | 0.4631 | 0.4293 | 整体不稳定 |
| 3% | 0.4256 | 0.4508 | 0.4084 | IMDB 与 REDDIT 最优 |
| 4% | 0.4355 | 0.5369 | 0.4182 | IMDB 明显恶化 |
| 5% | 0.4753 | 0.4631 | 0.4330 | 过多 augmentation 会伤害 |

### 消融实验
论文的实验不是传统模块消融，而是围绕理论分解验证不同误差项的作用。

| 分析项 | 实验设置 | 主要现象 | 对理论的支持 |
|--------|----------|----------|--------------|
| Size shift | 固定总节点预算，增大 $\lambda$ 把训练预算转向更大图 | train-test token discrepancy 单调下降，但 test error 呈 U 形 | 只降低采样误差不够，训练尺寸覆盖不足会抬高 ID error |
| Graph merging | 每类估计 graphon 后采样 1%-5% 更大合成图 | 大 train-test size gap 时最有帮助，中间区域收益小且波动 | augmentation 需要针对缺失的 graphon/size 区域，而非越多越好 |
| Graphon shift | 固定 $\lambda=0.2$ 且 baseline test error <0.05，替换 50% 测试图为扰动 graphon | in-graphon error 基本稳定，out-of-graphon error 快速上升并主导总误差 | $\epsilon_{gra}$ 会在真正域偏移时成为主导项 |
| Eig-PE 维度 | 扫描 top-$k$ eigenvector PE | 小 $k$ 欠表达，中等 $k$ 最好，大 $k$ 因 eigengap 变小而更不稳定 | $C_{eig}\propto\sqrt{k}/\min eigengap$ 的趋势符合实验 |
| Proj-PE 维度 | 固定 readout 维度 $m=32$，扫描 spectral rank $k$ | 大 $k$ 也会恶化，低 rank 时 OOD error 有时优于 Eig-PE | Proj-PE 的理论优势受 learnable readout 和谱结构影响 |

### 关键发现
- 图越大通常越接近其 latent graphon，但训练集中只偏向大图会减少小图覆盖，导致 ID error 上升；因此 size generalization 存在覆盖和近似之间的 trade-off。
- 真实图上的 graph merging augmentation 不是单调有效。COLLAB 只需要 1% 合成图，IMDB-BINARY 和 REDDIT-BINARY 在 3% 附近更好，5% 反而可能恶化。
- Graphon shift 与 size shift 叠加时最危险。熟悉 graphon 内部的尺寸变化可以被模型处理，但一旦测试图来自扰动 graphon，OOD graphon error 会迅速主导。
- 谱 PE 的维度不能简单越大越好。增加 $k$ 提供更多结构信息，但也引入更小 eigengap 和更强 token 不稳定性。

## 亮点与洞察
- 论文把“图基础模型何时能迁移”从模型技巧问题，转成数据生成机制和有限采样近似问题，这个视角很适合指导数据策展。
- 三项误差分解很有解释力：如果两个图域的 latent graphon 差异大，单纯扩大图规模或加大模型都不一定能解决负迁移。
- PE 稳定性被明确接入迁移上界，这是图 Transformer 中非常现实的问题；很多时候 PE 维度调参其实是在表达力和稳定性之间折中。
- 真实图 augmentation 结果提醒我们，graphon 估计和采样增强应该小心控制比例，过多合成图可能把训练分布推向另一个偏置。

## 局限与展望
- 理论主要针对 dense graph / graphon 设置，对稀疏大图、异质图、动态图和带丰富节点属性的真实图还需要扩展。
- Lipschitz backbone 和谱 gap 假设有助于证明，但实际深层 GNN/Graph Transformer 的常数很难估计，也可能较松。
- 实验的合成 graphon 任务便于验证理论，但与真实 GFM 预训练语料之间仍有距离。
- Proj-PE 实验没有完全隔离理论预测，因为 readout 是可学习的，优化因素和谱结构会混在一起；未来可以用固定 projector 和特定谱 gap 结构做更干净的验证。

## 相关工作与启发
- **vs graphon 泛化理论**: 以往工作多分析特定 GNN 在 graphon 极限下的收敛或泛化；本文更关注 GFM 场景下的跨域输出 shift，并覆盖 set tokenization 与 message passing 两类接口。
- **vs MMD / Gromov-Wasserstein**: MMD 依赖选定表示空间，GW 比较有限图会混入采样噪声；本文的 graphon discrepancy 直接比较潜在生成机制，并把采样误差显式拆出来。
- **vs Graph Transformer PE 工作**: PEG、SignNet、BasisNet、SPE 等方法改进 PE 稳定性或不变性；本文不提出新 PE，而是解释 PE 稳定性如何影响跨域迁移上界。
- **启发**: 构建 GFM 训练集时，不应只追求更大规模图或更多数据集，而应估计图域覆盖、graphon mismatch 和尺寸分布，针对缺失区域做数据增强。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 GFM 迁移难度分解为 graphon mismatch 与有限采样误差，理论视角清晰。
- 实验充分度: ⭐⭐⭐⭐☆ 合成实验验证了多项理论预测，也有真实图 augmentation；若能接入大规模预训练 GFM 会更强。
- 写作质量: ⭐⭐⭐⭐☆ 理论链条完整，但符号密集，对不熟悉 graphon 与谱扰动的读者有门槛。
- 价值: ⭐⭐⭐⭐☆ 对图基础模型的数据选择、尺寸增强和 PE 调参很有指导意义，尤其适合做 GFM 数据策展和评测设计。# When Do Graph Foundation Models Transfer? A Data-Centric Theory

**会议**: ICML2026  
**arXiv**: [2605.29828](https://arxiv.org/abs/2605.29828)  
**代码**: https://github.com/zhuconv/GraphFM  
**领域**: graph_learning  
**关键词**: 图基础模型, graphon, 迁移理论, 位置编码, 图数据策展

## 一句话总结
这篇论文用 graphon 把不同大小、不同域的图放到同一连续空间里，证明图基础模型的跨域输出差异可以分解为两个有限采样误差和一个内在 graphon 域差异，并用合成与真实图实验说明图大小、结构偏移和谱位置编码稳定性共同决定迁移成败。

## 研究背景与动机
**领域现状**：图基础模型希望像语言基础模型一样，用一个预训练 backbone 迁移到多种图域，例如社交网络、推荐、生物和安全图。近期工作更多从模型侧推进，包括更大的图预训练、更复杂的 tokenization、MoE 路由、图 prompt 和适配器。

**现有痛点**：图数据的异质性远强于文本。不同域的图不仅特征和标签语义不同，节点规模、连接模式、谱结构和适合的归纳偏置也不同。因此，同一个 GFM 在某些域可以正迁移，在另一些域可能负迁移。现有方法往往给出架构或训练技巧，却缺少一个回答“两个图域本身有多难迁移”的数据中心理论。

**核心矛盾**：如果只用有限图之间的距离来判断迁移难度，距离会混入采样噪声和规模差异；如果只看模型表示距离，又会被具体 backbone 的信息损失影响。论文想分开两件事：有限图是不是足够接近其潜在生成机制，以及两个潜在图域本身是否相似。

**本文目标**：作者希望建立一种 size-independent 的图域比较方式，在固定 Lipschitz backbone 下，把跨图输出变化拆成可解释的项，并把这个分解转化为图基础模型训练和数据策展建议。

**切入角度**：graphon 是 dense graph 的连续极限对象，可以表示潜在图生成机制。把有限图嵌入为 step-graphon 后，不同大小的图就能在同一个算子空间中比较，而图域之间的差异也可以写成 graphon operator 的 relabeling-invariant discrepancy。

**核心 idea**：先把有限图与 latent graphon 的误差分离出来，再用 graphon mismatch 描述真正的域差异；对带谱位置编码的 set backbone 和 message-passing backbone，都能得到同样的“采样误差 + graphon 域差异 + 采样误差”迁移上界。

## 方法详解
论文主要是理论和受控实验结合。理论部分先定义图、graphon、两类 backbone 和两种谱位置编码，再逐步证明：有限图可以等价嵌入为 step-graphon；同一个 graphon 采样出的图越大，step-graphon 越接近原 graphon；两个图之间的模型输出差异可以由各自采样误差和两个 graphon 的内在差异控制。

### 整体框架
给定一个有限图 $G$，论文把归一化图算子 $\Delta=A/n$ 转成 step-graphon $W_\Delta$。这样，大小为 128、512 或 2048 的图都不再只是不同维度矩阵，而是 $[0,1]^2$ 上的分段函数。对于两个从 graphon $W^{(1)}$ 与 $W^{(2)}$ 采样的图 $G^{(1)}$ 与 $G^{(2)}$，模型输出差异可以按路径拆分：$G^{(1)}$ 到 $W^{(1)}$ 的采样误差，$W^{(1)}$ 到 $W^{(2)}$ 的 graphon 域差异，再从 $W^{(2)}$ 到 $G^{(2)}$ 的采样误差。

这个分解对两类常见 GFM 接口都成立。第一类是 set-based backbone，例如 DeepSets 或 Graph Transformer，它把图结构编码进 positional embedding 后再做集合读出。第二类是 message-passing backbone，例如 GCN/GIN，它同时依赖图算子和节点 token。论文用 Lipschitz 条件把 backbone 对输入 token 和图算子扰动的敏感性包进常数 $L_\theta$。

### 关键设计
1. **用 step-graphon 统一不同大小的图**:

	- 功能：让跨尺寸图比较从矩阵维度不一致的问题，变成同一函数空间中的算子距离问题。
	- 核心思路：把大小为 $n$ 的图算子 $\Delta$ 映射成 $W_\Delta(u,v)=\sum_{i,j}n\Delta_{ij}1_{P_i}(u)1_{P_j}(v)$。图上的离散邻域平均对应 step-graphon 上的积分，因此模型操作可以在连续空间里对齐。
	- 设计动机：GFM 迁移经常跨图大小，如果没有 size-independent 表示，就很难区分“图太小导致近似差”和“图域本身不同”。

2. **输出差异的三项误差分解**:

	- 功能：把迁移难度拆成有限采样误差和内在域差异。
	- 核心思路：在 Lipschitz backbone 和稳定 PE 假设下，set backbone 有 $\|f_\theta(t_{G^{(1)}})-f_\theta(t_{G^{(2)}})\|\le L_\theta C_{PE}(\epsilon_1+\epsilon_{gra}+\epsilon_2)$；message-passing 版本多了图算子扰动项，常数变成 $L_\theta(1+C_{PE})$。
	- 设计动机：这个式子直接告诉我们，提升迁移不能只扩大图规模，也不能只做域混合；采样充分性和 latent graphon mismatch 都会成为瓶颈。

3. **把谱位置编码稳定性纳入迁移上界**:

	- 功能：解释为什么 PE 维度和 eigengap 会影响 GFM 跨域泛化。
	- 核心思路：Eig-PE 的稳定常数随 $\sqrt{k}\max_{\ell\le k}1/\gamma_\ell$ 增大，小 eigengap 会放大特征向量扰动；Proj-PE 使用 top-$k$ 子空间投影，理论上对基旋转更不敏感，但仍依赖子空间 gap。
	- 设计动机：图 Transformer 和许多 GNN 都依赖谱 PE。论文把 PE 不稳定从经验调参问题变成理论项 $C_{PE}$，说明过大的 PE 维度可能提高表达力，也可能放大域 shift。

### 损失函数 / 训练策略
理论部分不提出新的训练损失。实验中，作者构造低秩 Fourier graphon 分类任务，固定总节点预算 100k，在训练尺寸 $n\in\{128,256,512,1024\}$ 间用混合参数 $\lambda$ 调整大小分布；测试包含 $n\in\{128,256,512,1024,2048\}$，其中 2048 是 OOD 尺寸。默认模型为 DeepSets + top-32 Eig-PE，也用 GIN 验证 message-passing 情形。真实图实验中，作者估计每类图的 class-specific graphon，再采样少量合成图做 graph merging augmentation。

## 实验关键数据

### 主实验
真实数据实验展示了 graphon-based augmentation 在 COLLAB、IMDB-BINARY、REDDIT-BINARY 上的效果。指标是 test error，越低越好。

| 合成图比例 | COLLAB | IMDB-BINARY | REDDIT-BINARY | 观察 |
|------------|--------|-------------|---------------|------|
| Vanilla | 0.4384 | 0.4631 | 0.4108 | 不做 augmentation |
| 1% | 0.4069 | 0.4631 | 0.4367 | COLLAB 最优，REDDIT 变差 |
| 2% | 0.4428 | 0.4631 | 0.4293 | 整体不稳定 |
| 3% | 0.4256 | 0.4508 | 0.4084 | IMDB 与 REDDIT 最优 |
| 4% | 0.4355 | 0.5369 | 0.4182 | IMDB 明显恶化 |
| 5% | 0.4753 | 0.4631 | 0.4330 | 过多 augmentation 会伤害 |

### 消融实验
论文的实验不是传统模块消融，而是围绕理论分解验证不同误差项的作用。

| 分析项 | 实验设置 | 主要现象 | 对理论的支持 |
|--------|----------|----------|--------------|
| Size shift | 固定总节点预算，增大 $\lambda$ 把训练预算转向更大图 | train-test token discrepancy 单调下降，但 test error 呈 U 形 | 只降低采样误差不够，训练尺寸覆盖不足会抬高 ID error |
| Graph merging | 每类估计 graphon 后采样 1%-5% 更大合成图 | 大 train-test size gap 时最有帮助，中间区域收益小且波动 | augmentation 需要针对缺失的 graphon/size 区域，而非越多越好 |
| Graphon shift | 固定 $\lambda=0.2$ 且 baseline test error <0.05，替换 50% 测试图为扰动 graphon | in-graphon error 基本稳定，out-of-graphon error 快速上升并主导总误差 | $\epsilon_{gra}$ 会在真正域偏移时成为主导项 |
| Eig-PE 维度 | 扫描 top-$k$ eigenvector PE | 小 $k$ 欠表达，中等 $k$ 最好，大 $k$ 因 eigengap 变小而更不稳定 | $C_{eig}\propto\sqrt{k}/\min eigengap$ 的趋势符合实验 |
| Proj-PE 维度 | 固定 readout 维度 $m=32$，扫描 spectral rank $k$ | 大 $k$ 也会恶化，低 rank 时 OOD error 有时优于 Eig-PE | Proj-PE 的理论优势受 learnable readout 和谱结构影响 |

### 关键发现
- 图越大通常越接近其 latent graphon，但训练集中只偏向大图会减少小图覆盖，导致 ID error 上升；因此 size generalization 存在覆盖和近似之间的 trade-off。
- 真实图上的 graph merging augmentation 不是单调有效。COLLAB 只需要 1% 合成图，IMDB-BINARY 和 REDDIT-BINARY 在 3% 附近更好，5% 反而可能恶化。
- Graphon shift 与 size shift 叠加时最危险。熟悉 graphon 内部的尺寸变化可以被模型处理，但一旦测试图来自扰动 graphon，OOD graphon error 会迅速主导。
- 谱 PE 的维度不能简单越大越好。增加 $k$ 提供更多结构信息，但也引入更小 eigengap 和更强 token 不稳定性。

## 亮点与洞察
- 论文把“图基础模型何时能迁移”从模型技巧问题，转成数据生成机制和有限采样近似问题，这个视角很适合指导数据策展。
- 三项误差分解很有解释力：如果两个图域的 latent graphon 差异大，单纯扩大图规模或加大模型都不一定能解决负迁移。
- PE 稳定性被明确接入迁移上界，这是图 Transformer 中非常现实的问题；很多时候 PE 维度调参其实是在表达力和稳定性之间折中。
- 真实图 augmentation 结果提醒我们，graphon 估计和采样增强应该小心控制比例，过多合成图可能把训练分布推向另一个偏置。

## 局限与展望
- 理论主要针对 dense graph / graphon 设置，对稀疏大图、异质图、动态图和带丰富节点属性的真实图还需要扩展。
- Lipschitz backbone 和谱 gap 假设有助于证明，但实际深层 GNN/Graph Transformer 的常数很难估计，也可能较松。
- 实验的合成 graphon 任务便于验证理论，但与真实 GFM 预训练语料之间仍有距离。
- Proj-PE 实验没有完全隔离理论预测，因为 readout 是可学习的，优化因素和谱结构会混在一起；未来可以用固定 projector 和特定谱 gap 结构做更干净的验证。

## 相关工作与启发
- **vs graphon 泛化理论**: 以往工作多分析特定 GNN 在 graphon 极限下的收敛或泛化；本文更关注 GFM 场景下的跨域输出 shift，并覆盖 set tokenization 与 message passing 两类接口。
- **vs MMD / Gromov-Wasserstein**: MMD 依赖选定表示空间，GW 比较有限图会混入采样噪声；本文的 graphon discrepancy 直接比较潜在生成机制，并把采样误差显式拆出来。
- **vs Graph Transformer PE 工作**: PEG、SignNet、BasisNet、SPE 等方法改进 PE 稳定性或不变性；本文不提出新 PE，而是解释 PE 稳定性如何影响跨域迁移上界。
- **启发**: 构建 GFM 训练集时，不应只追求更大规模图或更多数据集，而应估计图域覆盖、graphon mismatch 和尺寸分布，针对缺失区域做数据增强。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 GFM 迁移难度分解为 graphon mismatch 与有限采样误差，理论视角清晰。
- 实验充分度: ⭐⭐⭐⭐☆ 合成实验验证了多项理论预测，也有真实图 augmentation；若能接入大规模预训练 GFM 会更强。
- 写作质量: ⭐⭐⭐⭐☆ 理论链条完整，但符号密集，对不熟悉 graphon 与谱扰动的读者有门槛。
- 价值: ⭐⭐⭐⭐☆ 对图基础模型的数据选择、尺寸增强和 PE 调参很有指导意义，尤其适合做 GFM 数据策展和评测设计。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Structure-Centric Graph Foundation Model via Geometric Bases](structure-centric_graph_foundation_model_via_geometric_bases.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[ICML 2025\] Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](../../ICML2025/graph_learning/towards_graph_foundation_models_learning_generalities_across_graphs_via_task-tre.md)
- [\[ICML 2026\] Are Common Substructures Transferable? Riemannian Graph Foundation Model with Neural Vector Bundles](are_common_substructures_transferable_riemannian_graph_foundation_model_with_neu.md)
- [\[ICML 2026\] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models](finding_the_minimal_parameter_budget_for_implicit_reasoning_a_data_complexity_dr.md)

</div>

<!-- RELATED:END -->
