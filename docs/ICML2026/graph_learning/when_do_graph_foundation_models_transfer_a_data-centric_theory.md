---
title: >-
  [论文解读] When Do Graph Foundation Models Transfer? A Data-Centric Theory
description: >-
  [ICML2026][图学习][图基础模型] 这篇论文用 graphon 把不同大小、不同域的图放到同一连续空间里，证明图基础模型的跨域输出差异可以分解为两个有限采样误差和一个内在 graphon 域差异，并用合成与真实图实验说明图大小、结构偏移和谱位置编码稳定性共同决定迁移成败。 领域现状：图基础模型希望像语言基础模型一样…
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
**领域**: 图学习  
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
这是一篇理论加受控实验的工作，要回答的是"两个图域之间到底有多难迁移"。它不提出新架构，而是给出一套可解释的迁移上界：先把任意大小的有限图嵌入到同一连续空间，再证明两个图上的模型输出差异只由三件事决定——各自的采样充分性，以及两个域潜在生成机制本身的差距。

### 整体框架
核心做法是把有限图换成连续对象再做比较。给定一个有限图 $G$，论文把归一化图算子 $\Delta=A/n$ 转成 step-graphon $W_\Delta$，于是大小 128、512、2048 的图不再是维度不同的矩阵，而都是 $[0,1]^2$ 上的分段函数，可以直接量算子距离。在这个空间里，论文把两个图的输出差异沿"采样—域差异—采样"这条链路拆开：图越偏离自己的潜在 graphon，采样误差越大；两个 graphon 本身越不像，域差异越大。整套结论对两类主流 GFM 接口都成立——把结构编码进位置编码再做集合读出的 set backbone（DeepSets、Graph Transformer），以及同时依赖图算子和节点 token 的 message-passing backbone（GCN、GIN），二者的敏感性都用 Lipschitz 常数 $L_\theta$ 统一刻画。

### 关键设计

**1. step-graphon 嵌入：把"不同大小的图无法直接比"这件事消掉**

GFM 迁移天然跨图大小，而不同尺寸图是不同维度的矩阵，没法直接算距离，于是"图太小导致近似差"和"两个图域本身不同"这两种迁移困难会纠缠在一起分不开。论文的解法是把大小为 $n$ 的图算子 $\Delta$ 映射成 step-graphon $W_\Delta(u,v)=\sum_{i,j}n\Delta_{ij}\mathbf{1}_{P_i}(u)\mathbf{1}_{P_j}(v)$，把离散邻域平均对应成连续空间上的积分。这样所有图都被嵌入同一个函数空间，跨尺寸比较从"矩阵维度对不上"变成了"算子距离 size-independent"，后面拆解采样误差和域差异才有共同的度量基准。

**2. 三项误差分解：把迁移难度拆成可单独诊断的项**

有了统一空间，论文证明在 Lipschitz backbone 与稳定 PE 假设下，set backbone 的跨图输出差异被一条上界控制：

$$\|f_\theta(t_{G^{(1)}})-f_\theta(t_{G^{(2)}})\|\le L_\theta C_{PE}(\epsilon_1+\epsilon_{gra}+\epsilon_2)$$

其中 $\epsilon_1,\epsilon_2$ 是两个有限图各自相对其潜在 graphon 的采样误差，$\epsilon_{gra}$ 是两个 graphon 之间的内在域差异（relabeling-invariant discrepancy）；message-passing 版本因为还要承受图算子本身的扰动，常数收紧为 $L_\theta(1+C_{PE})$。这个分解之所以有用，是因为它把"提升迁移"从一句口号拆成了三个可以分别下手的瓶颈：扩大图规模只压得动 $\epsilon_1,\epsilon_2$，对 $\epsilon_{gra}$ 无能为力，所以当两个域 latent graphon 差太多时，单纯堆数据或堆模型都救不了负迁移。

**3. 把谱位置编码稳定性写进上界：解释 PE 维度为什么是双刃剑**

上界里的 $C_{PE}$ 不是常量，而是谱位置编码稳定性的显式刻画，这把"PE 维度该调多大"从经验玄学变成了有理论依据的取舍。对 Eig-PE，稳定常数随 $\sqrt{k}\,\max_{\ell\le k}1/\gamma_\ell$ 增长，一旦某个 eigengap $\gamma_\ell$ 很小，对应特征向量在域偏移下会剧烈旋转，把误差放大；Proj-PE 改用 top-$k$ 子空间投影，对基旋转更不敏感，但仍受子空间 gap 牵制。结论是 PE 维度 $k$ 越大表达力越强、但越容易踩到小 eigengap，于是表达力和跨域稳定性之间存在内在权衡。

### 一个完整示例
用论文合成实验的数字串一遍这条上界：取两个图 $G^{(1)}$（$n=128$）和 $G^{(2)}$（$n=2048$，OOD 尺寸），backbone 固定为 DeepSets + top-32 Eig-PE。若两者其实来自同一个 graphon，则 $\epsilon_{gra}=0$，差异只剩 $\epsilon_1+\epsilon_2$；此时小图 $G^{(1)}$ 的 $\epsilon_1$ 偏大、大图 $G^{(2)}$ 的 $\epsilon_2$ 很小，对应实验里"图越大越贴近 latent graphon、token discrepancy 单调下降"。但若把 $G^{(2)}$ 换成来自扰动 graphon 的图，$\epsilon_{gra}$ 突然变成主导项，上界整体抬高——这正对应实验中 out-of-graphon error 迅速主导总误差。再把 PE 从 top-32 调到很大的 $k$，$C_{PE}$ 因小 eigengap 而膨胀，会把同样的 $\epsilon_{gra}$ 进一步放大，于是大 $k$ 在 OOD 上反而更糟。三个旋钮各自对应上界里的一项，诊断和干预都能对号入座。

### 损失函数 / 训练策略
理论部分不引入新损失。实验侧构造低秩 Fourier graphon 分类任务，固定总节点预算 100k，用混合参数 $\lambda$ 在训练尺寸 $n\in\{128,256,512,1024\}$ 之间调配大小分布；测试集覆盖 $n\in\{128,256,512,1024,2048\}$，其中 2048 是 OOD 尺寸。默认 backbone 为 DeepSets + top-32 Eig-PE，并用 GIN 验证 message-passing 情形。真实图实验中，作者先按类估计 class-specific graphon，再采样少量更大的合成图做 graph merging augmentation，对应"针对缺失 size/graphon 区域补数据"的策略。

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
