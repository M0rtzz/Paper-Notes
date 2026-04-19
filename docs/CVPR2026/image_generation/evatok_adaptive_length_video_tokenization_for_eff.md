---
title: >-
  [论文解读] EVATok: 自适应长度视频Tokenization用于高效视觉自回归生成
description: >-
  [CVPR 2026][图像生成][video tokenizer] 提出EVATok框架——通过最优token分配估计+轻量路由器+自适应tokenizer训练的三步流程，让视频tokenizer按片段复杂度自适应分配token长度，在UCF-101上节省24.4%+ token同时达到SOTA生成质量。
tags:
  - CVPR 2026
  - 图像生成
  - video tokenizer
  - adaptive token
  - autoregressive generation
  - efficiency
  - VQ-VAE
---

# EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)  
**代码**: [项目页](https://silentview.github.io/EVATok/)  
**领域**: 视频生成  
**关键词**: video tokenizer, adaptive tokenization, autoregressive generation, proxy reward, Q-Former

## 一句话总结

提出四阶段框架EVATok：先用proxy tokenizer估计每个视频的最优token分配方案，再训练轻量路由器一次前向预测这些分配，最终训练自适应tokenizer按内容复杂度灵活分配token数，在UCF-101上以24.4%的token节省达到SOTA生成质量。

## 研究背景与动机

**自回归视频生成**的核心流程是先用视频tokenizer把像素压缩成离散token序列，再用AR模型在token序列上建模。token序列的长度直接决定了下游生成的计算开销——序列越长，attention的计算量平方增长。

**现有的痛点**在于：几乎所有视频tokenizer对不同视频、不同时间段都分配相同数量的token。但视频中信息密度极度不均——静态背景、重复纹理的片段信息量很少，快速运动、场景切换的片段信息密度极高。这种"一刀切"的固定分配策略在简单片段上浪费了大量token（重建质量已饱和），在复杂片段上token又不够用（欠表达导致质量下降）。

**核心矛盾**是：自适应分配需要知道"最优分配是什么"，但（1）"最优"如何定义？需要一个可量化的质量-效率权衡指标；（2）逐视频搜索最优分配计算量太大；（3）tokenizer架构需要支持可变长度输入。之前的方法如ElasticTok用阈值启发式搜索、AdapTok用mini-batch内ILP，都是局部次优的。

**EVATok的切入角度**：定义一个proxy reward指标来量化单个分配方案的质量-成本权衡，用暴力搜索找到每个视频的最优分配作为监督信号，训练一个轻量级路由器来一次性预测最优分配，从而跳过搜索阶段。核心idea：把"找最优分配"转化为"分类预测任务"，用小模型的一次前向替代昂贵的逐样本搜索。

## 方法详解

### 整体框架

EVATok分为四个阶段依次执行：Stage 1 训练proxy tokenizer（能在任意token分配下重建视频）→ Stage 2 用proxy tokenizer对100k视频暴力搜索最优分配，构建(视频, 最优分配)训练集 → Stage 3 训练轻量ViT-S路由器，将最优分配预测建模为分类任务 → Stage 4 用路由器指导，从头训练最终的自适应tokenizer。

### 关键设计

1. **Proxy Reward与最优分配定义**:

    - 功能：为每个视频的每种token分配方案量化其质量-成本权衡
    - 核心思路：定义 $R_{\text{proxy}} = w_q Q(\mathcal{E},x,a) - w_l L(a)$，其中 $Q$ 是重建质量（归一化LPIPS）、$L(a)$ 是归一化token长度、$w_q, w_l$ 是偏好权重。对每个视频遍历所有 $5^4=625$ 种候选分配，选proxy reward最大的作为最优分配 $a^*$
    - 设计动机：之前的方法缺乏对"最优"的明确定义，靠启发式搜索容易陷入局部最优。proxy reward将质量和成本统一到一个标量指标中，使最优分配变得可计算、可比较

2. **轻量级路由器（Router）**:

    - 功能：一次前向传播预测输入视频的最优token分配，替代暴力搜索
    - 核心思路：ViT-S架构（19.9M参数），将视频patchify后加[CLS] token，输出 $m^T$ 个分配类别的概率。在Stage 2构建的100k样本上用交叉熵损失训练为分类任务
    - 设计动机：暴力搜索每个视频需要625次前向，路由器将其压缩为一次前向。实验证明路由器预测接近暴力搜索的帕累托前沿，且能泛化到训练时未见的数据集

3. **Q-Former风格1D可变长度Tokenizer**:

    - 功能：支持不同时间块使用不同数量token的编码-解码架构
    - 核心思路：输入视频spatio-temporal patchify后，根据分配方案 $a=(k_1,...,k_T)$ 初始化不同数量的1D query，通过Q-Former编码层与3D embeddings交互后VQ量化产生离散token。解码端用第一个1D token初始化3D query来重建
    - 设计动机：避免了tail-token-dropping策略的两个问题——（1）被丢弃的尾部token仍然消耗计算；（2）尾部query在编码时角色模糊（不知道自己会不会被丢弃）。直接在query初始化阶段就确定长度更高效

### 损失函数 / 训练策略

总损失为 $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vqgan}} + \lambda \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{entropy}}$，其中：

- $\mathcal{L}_{\text{vqgan}}$：L1重建 + 感知损失 + GAN对抗损失 + VQ码本损失
- $\mathcal{L}_{\text{align}}$：decoder中间3D特征与预训练V-JEPA2-L特征的cosine similarity对齐，$\lambda=0.7$
- $\mathcal{L}_{\text{entropy}}$：LFQ的entropy loss促进码本使用率，$\gamma=0.02$

**进阶设计**：最终tokenizer训练（Stage 4）额外使用VideoMAE-B作为语义判别器，将其多层特征送入可训练1D CNN头做真假判别，显著提升重建和下游生成质量。

## 实验关键数据

### 主实验

| 方法 | 参数量(Tok+Gen) | 重建rFVD↓ | 生成gFVD↓ | 重建Token数 | 生成Token数 |
|------|----------------|-----------|-----------|------------|------------|
| LARP-L-Long | 173M+632M | 20 | 57 | 1024 | 1024 |
| AdapTok | 195M+633M | 36 | 67 | 1024 | 1024 |
| OmniTokenizer | 82M+650M | 42 | 191 | 1280 | 1280 |
| **EVATok** | **145M+633M** | **9.7** | **48** | **774(-24.4%)** | **756(-26.2%)** |

### 消融实验

| 配置 | rFVD↓ | Token数 | 说明 |
|------|-------|---------|------|
| 均匀分配(Proxy Tok.) | 73 | 1024 | 固定分配基线 |
| 均匀分配(Final Tok.) | 63 | 1024 | Final tokenizer更好 |
| 路由器(Proxy Tok.) | 50 | 721(-29.6%) | 路由器分配提升显著 |
| 路由器(Final Tok.) | 33 | 721(-29.6%) | 两个改进叠加效果最佳 |
| +VideoMAE判别器 | 9.2 | 721(-29.6%) | 语义判别器带来巨大提升 |

### 关键发现

- 自适应分配在相同平均token数下，质量-成本曲线全面优于固定分配：WebVid上可节省56%token、UCF上42%token达到同等rFVD
- Final tokenizer显著优于Proxy tokenizer（相同训练量），说明消除variable-length tokenizer的训练-推理gap很重要
- 路由器在训练时未见的UCF数据集上仍然接近暴力搜索的最优前沿，泛化性好
- VideoMAE语义判别器的引入使rFVD从33降至9.2，是单一最大的质量提升因子

## 亮点与洞察

- "先定义最优→暴力搜索标注→训练分类器模仿"的范式非常优雅，把一个看似连续优化的问题转化为离散分类任务，既有理论上的最优性保证又高效可行。这个设计模式（用小模型预测大模型的最优配置）在其他场景下也有很强的复用价值。
- 避免tail-token-dropping的设计选择很有洞察——在query初始化阶段就确定长度，避免了"编码了但要丢弃"的浪费和角色模糊问题。

## 局限与展望

- 候选分配空间是 $m^T$ 的指数级（本文625种），视频更长或粒度更细时搜索空间会爆炸，需要更高效的分配空间设计
- 路由器使用全局ViT-S且每个视频只预测一次，对于长视频中局部复杂度剧变的场景可能不够灵活
- 固定码本大小（8192/16384），与自适应token长度的搭配是否最优尚未探索

## 相关工作与启发

- **vs LARP/AdapTok**: 这两个方法也做自适应视频tokenization，但分配策略是启发式的（阈值搜索/mini-batch ILP），EVATok通过proxy reward给出了"最优分配"的明确定义，且路由器的泛化性更好
- **vs ElasticTok**: ElasticTok用tail-token-dropping实现变长，EVATok论证了这种方式的效率和性能问题，改用query初始化时直接确定长度

## 评分

- 新颖性: ⭐⭐⭐⭐ 四阶段框架完整，proxy reward定义优雅，路由器替代搜索的思路很好
- 实验充分度: ⭐⭐⭐⭐ 质量-成本曲线分析、消融实验、系统级比较都很全面
- 写作质量: ⭐⭐⭐⭐ 四个阶段的叙述逻辑清晰，问题定义严谨
- 价值: ⭐⭐⭐⭐ 24.4%~29.6%的token节省在视频生成的部署中有直接实用价值
---
title: >-
  [论文解读] EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation
description: >-
  [CVPR 2026][video tokenizer][adaptive token] 提出EVATok四阶段框架，通过proxy reward定义最优token分配、训练轻量路由器快速预测分配、训练自适应tokenizer消除训练-推理gap，在UCF-101上以24.4%+ token节省达到SOTA视频生成质量。
tags:
  - CVPR 2026
  - video tokenizer
  - adaptive token
  - autoregressive generation
  - efficiency
  - VQ-VAE
---

# EVATok: Adaptive Length Video Tokenization for Efficient Visual Autoregressive Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)  
**代码**: [项目页](https://silentview.github.io/EVATok/)  
**领域**: 视频生成 / 视频Tokenizer  
**关键词**: video tokenizer, adaptive token, proxy reward, autoregressive generation, Q-Former

## 一句话总结
提出EVATok四阶段框架，通过proxy reward最优化token分配估计、轻量路由器快速预测、自适应tokenizer消除训练-推理gap，在UCF-101上以24.4%+ token节省达到SOTA视频生成质量。

## 研究背景与动机

**自回归视频生成的核心瓶颈**：AR视频生成模型依赖视频tokenizer将像素压缩为离散token序列，token序列长度直接决定重建质量和下游生成的计算成本。现有tokenizer对所有时间块均匀分配固定数量token，完全不考虑内容复杂度差异。

**信息密度不均匀问题**：视频中信息密度分布极不均匀——静态/重复片段被过度分配token（质量已饱和），而动态/复杂布局片段token不足（欠表达导致重建劣化）。这在因果视频tokenizer中尤为严重，因为信息密度不仅跨样本变化，还沿时间维度变化。

**现有自适应方法的不足**：ElasticTok通过阈值搜索确定分配，但属于启发式方法，无法优化整体质量-成本权衡。AdapTok使用mini-batch ILP，分配决策依赖batch组成和固定预算约束，同样不够最优。核心缺失是：没有一个明确的"最优分配"定义和估计方法。

## 方法详解

### 整体框架
EVATok分四阶段：(1) 训练proxy tokenizer用于最优分配估计；(2) 用proxy reward搜索最优分配，构建(视频, 最优分配)数据集；(3) 训练轻量路由器进行一次前向预测最优分配；(4) 用路由器预测的分配训练最终自适应tokenizer，消除训练-推理gap。

### 关键设计

1. **Proxy Reward与最优分配定义**：
    - 功能：定义一个同时度量重建质量和token成本的标量指标，用于评价特定token分配的质量-成本权衡
    - 核心思路：$R_{\text{proxy}} = w_q Q(\mathcal{E}_{\text{proxy}}, x, a) - w_l L(a)$，其中 $Q$ 为归一化LPIPS重建质量，$L(a)$ 为归一化token长度，$w_q, w_l$ 反映用户对质量vs效率的偏好
    - 设计动机：将"最优分配"严格形式化为最大化proxy reward的分配 $a^* = \arg\max_{a \in A} R_{\text{proxy}}$，避免启发式搜索的次优性。通过遍历所有候选分配（$5^4=625$种）找到最优

2. **Q-Former式1D可变长Tokenizer架构**：
    - 功能：实现一个能根据指定token分配编解码视频的可变长度tokenizer
    - 核心思路：输入视频先patchify为3D嵌入，然后根据分配 $a=(k_1,...,k_T)$ 初始化不同数量的1D query（通过2D池化从对应时间块的3D嵌入衍生），经Q-Former编码器层编码后向量量化为离散token。解码器用每个时间块的第一个1D token初始化3D query进行重建
    - 设计动机：放弃传统tail-token-dropping策略，因为(1)被丢弃的token在编码时仍消耗计算；(2)尾部query在编码时不知道自己是否会被丢弃，角色模糊。直接在query初始化时决定token数量更高效

3. **轻量路由器与最终Tokenizer训练**：
    - 功能：用ViT-S级别（19.9M参数）路由器将逐样本brute-force搜索加速为一次前向分类
    - 核心思路：在100k WebVid视频上构建(视频, 最优分配)分类数据集训练路由器，将最优分配预测建模为 $m^T$ 类分类任务。最终tokenizer从零训练并使用路由器预测的分配，而非复用proxy tokenizer
    - 设计动机：proxy tokenizer训练时覆盖所有 $m^T$ 种分配，但推理时每个视频只用一种，存在训练-推理gap。Stage 4的最终tokenizer消除了这一gap，实验证实比直接用proxy tokenizer提升显著

### 损失函数 / 训练策略

- Tokenizer总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vqgan}} + \lambda \mathcal{L}_{\text{align}} + \gamma \mathcal{L}_{\text{entropy}}$
- $\mathcal{L}_{\text{vqgan}}$ 包含L1重建损失、感知损失、GAN对抗损失、VQ codebook损失
- $\mathcal{L}_{\text{align}}$：与V-JEPA2-L的patch级cosine similarity对齐，$\lambda=0.7$
- $\mathcal{L}_{\text{entropy}}$：codebook熵损失促进codebook利用率，$\gamma=0.02$
- 最终tokenizer可选启用VideoMAE-B语义判别器，结合表征对齐显著提升重建和下游生成质量
- Codebook大小：proxy用16384，最终tokenizer用8192（公平比较）

## 实验关键数据

### 主实验

| 方法 | Tok. Param | #rTokens | rFVD↓ | #gTokens | gFVD↓ (UCF) |
|------|-----------|----------|-------|----------|-------------|
| LARP-L-Long (632M GPT) | 173M | 1024 | 20 | 1024 | 57 |
| AdapTok | 195M | 1024 | 36 | 1024 | 67 |
| OmniTokenizer | 82.2M | 1280 | 42 | 1280 | 191 |
| **EVATok (633M GPT)** | **145M** | **774 (-24.4%)** | **9.7** | **756 (-26.2%)** | **48** |

K600帧预测：EVATok gFVD=4.0，比LARP(5.1)和AdapTok(11)更优，且生成token数少15.8%。

### 消融实验

| 配置 | rFVD↓ | gFVD↓ |
|------|-------|-------|
| 完整recipe (Uniform) | 13 | 98 |
| - VideoMAE判别器 | 65 | 155 |
| - V-JEPA2对齐 | 18 | 144 |
| - 两者都去掉 | 80 | 230 |

| 分配策略 | LPIPS↓ | rFVD↓ | #rTokens |
|---------|--------|-------|----------|
| 固定均匀 (Proxy Tok.) | 0.1178 | 73 | 1024 |
| 固定均匀 (Final Tok.) | 0.1056 | 63 | 1024 |
| Router (Proxy Tok.) | 0.1182 | 50 | 721(-29.6%) |
| Router (Final Tok.) | 0.1068 | 33 | 721(-29.6%) |

### 关键发现
- 自适应分配在相当或更好的重建质量下，可节省24-30%的token
- 最终tokenizer比proxy tokenizer性能更好（消除训练-推理gap的收益）
- 下游AR模型在自适应长度token序列上训练可获得更好的生成FVD（首次证明）
- Max-proxy-reward策略在质量-成本曲线上一致优于阈值搜索和均匀分配
- 路由器可泛化到未见数据集（WebVid训练→UCF评估）

## 亮点与洞察
- **Proxy reward概念**：将"最优分配"从模糊直觉严格形式化为可计算的优化目标，是核心理论贡献
- **训练-推理gap的显式处理**：发现并解决了可变长tokenizer的一个普遍问题——训练时覆盖所有分配但推理时只用少数
- 分配示例与人类直觉高度一致：重复/简单/静态内容被分配更少token，非重复/复杂/动态内容更多
- 语义编码器（V-JEPA2对齐 + VideoMAE判别器）对视频tokenizer的双重增强效果值得关注

## 局限与展望
- 仅在16帧128×128的短低分辨率视频上验证，扩展到更长更高分辨率视频的效率增益有待验证
- Proxy reward的权重 $w_q, w_l$ 需要人工指定，不同应用场景的最优偏好不同
- 路由器分类为625个离散类别，更精细或连续的分配可能进一步提升
- Codebook大小对最终tokenizer和proxy不一致（8192 vs 16384），可能影响公平性

## 相关工作与启发
- **LARP / AdapTok**：最直接的对比基线，分别使用更大codebook和mini-batch ILP策略
- **Q-Former架构**在视觉tokenizer中的应用（源自InstructBLIP），1D序列设计使长度调整更自然
- **V-JEPA2表征对齐**：将视频语义编码器引入tokenizer训练的趋势值得追踪
- 启发：proxy-then-distill思路可推广到其他需要自适应预算分配的场景（如图像tokenizer、audio tokenizer）

## 评分
- 新颖性: ⭐⭐⭐⭐ proxy reward定义和四阶段框架设计新颖，系统性解决了最优分配问题
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证、丰富消融、质量-成本曲线分析完善
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，四阶段逐步展开逻辑顺畅
- 价值: ⭐⭐⭐⭐ 首次证明自适应长度token序列可提升下游AR生成，对视频生成领域有实际影响
---
title: >-
  [论文解读] EVATok: 自适应长度视频Tokenization用于高效视觉自回归生成
description: >-
  [CVPR 2026][video tokenizer] 提出EVATok框架——通过最优token分配估计+轻量路由器+自适应tokenizer训练的三步流程，让视频tokenizer按片段复杂度自适应分配token长度，在UCF-101上节省24.4%+ token同时达到SOTA生成质量。
tags:
  - CVPR 2026
  - video tokenizer
  - adaptive token
  - autoregressive generation
  - efficiency
  - VQ-VAE
---

# EVATok: 自适应长度视频Tokenization用于高效视觉自回归生成

**会议**: CVPR 2026  
**arXiv**: [2603.12267](https://arxiv.org/abs/2603.12267)  
**代码**: [项目页](https://silentview.github.io/EVATok/)  
**领域**: 视频理解 / 视频生成 / 模型压缩  
**关键词**: video tokenizer, adaptive token, autoregressive generation, efficiency, VQ-VAE

## 一句话总结
提出EVATok框架——通过最优token分配估计+轻量路由器+自适应tokenizer训练的三步流程，让视频tokenizer按片段复杂度自适应分配token长度，在UCF-101上节省24.4%+ token同时达到SOTA生成质量。

## 背景与动机
自回归（AR）视频生成依赖视频tokenizer将像素压缩为离散token序列，token序列的长度直接决定下游生成的计算成本。现有视频tokenizer对所有时间块都均匀分配固定数量的token，完全不考虑内容复杂度的差异。然而视频中的信息密度分布极不均匀——静态背景、重复纹理、缓慢运动的片段包含很少的信息，而快速运动、场景切换、精细纹理的片段信息密度极高。

## 核心问题
统一token分配对简单片段浪费token（用了很多token但重建质量已经饱和），对复杂片段则token不够（欠表达导致重建变差）。如何让不同视频、不同片段获得最优的token数量分配？挑战有三：（1）"最优"如何定义？需要在重建质量和效率之间找帕累托最优（2）最优分配对每个视频都不同，逐视频优化太慢（3）tokenizer需要能处理不等长的token输入。

## 方法详解

### 整体框架
EVATok框架分三步：① 估计最优token分配 → ② 训练路由器预测分配 → ③ 训练自适应tokenizer执行分配。

### 关键设计

**1. 最优Token分配估计（Optimal Token Assignment Estimation）**
- 对每个视频的每个时间块，尝试不同的token数量，评估"质量-成本"权衡
- 用搜索或优化算法找到整个视频的最优分配方案（在总token预算下最大化整体重建质量）
- 这一步是离线的、逐视频的，计算量大但只做一次，产出的分配作为后续步骤的训练目标

**2. 轻量级路由器（Lightweight Router）**
- 训练一个小型网络，输入视频片段的视觉特征，预测该片段应分配的最优token数量
- 路由器的训练目标：模仿步骤①估计出的最优分配
- 推理时，路由器一次前向传播即可为所有片段预测token分配，无需逐帧搜索
- 路由器本身参数量极小，推理开销可忽略

**3. 自适应Tokenizer训练**
- 基于路由器预测的分配方案，训练一个能处理**不等长token序列**的视频tokenizer
- 不同时间块可以有不同数量的离散token
- 编码器/解码器架构设计支持可变长度输入

**4. 视频语义编码器集成**
- 在tokenizer训练中集成视频语义编码器（如CLIP视频特征）
- 这个高级训练recipe提升了重建的语义质量和下游AR生成的效果
- 不只是像素级重建，还保证语义级保真

### 损失函数/训练策略
- Tokenizer训练：重建损失（L1/L2 + perceptual loss）+ VQ量化损失 + 语义对齐损失
- 路由器训练：模仿最优分配的分类/回归损失
- AR生成模型：标准自回归交叉熵损失，在EVATok产出的变长token上训练

## 实验关键数据

| 数据集 | 方法 | FVD↓ | Token节省 |
|--------|------|------|-----------|
| UCF-101 | LARP (固定长度) | 基线 | 0% |
| UCF-101 | **EVATok** | **SOTA** | **≥24.4%** |
| UCF-101 | 固定长度baseline | 基线 | 0% |

### 消融实验要点
- 自适应 vs 固定分配：自适应在同等平均token数下FVD显著更低
- 路由器准确度：路由器预测与真实最优分配的一致性高（>90%），说明分配是可预测的
- 语义编码器集成：加入后FVD进一步降低，说明语义信号对token质量有帮助
- token数量的最优分布：简单片段集中在低token区间，复杂片段分散在高token区间，分布呈长尾

## 亮点 / 我学到了什么
- "先估计最优解，再训路由器模仿"的两步范式非常实用——避免了端到端训练中最优性和效率的矛盾
- 24.4%的token节省直接意味着AR生成的24.4%计算量减少，这在视频生成的实际部署中价值巨大
- 路由器>90%的预测准确率说明"片段复杂度"是一个对视觉特征高度可预测的量
- 与语义编码器集成的策略表明token质量不只是像素级概念，语义层面的信号同样重要

## 局限与展望
- 路由器本身的计算开销虽小但非零，对极端延迟敏感的场景是否可忽略？
- 最优token分配的估计依赖离线搜索，训练集之外的新视频类型是否泛化？
- 自适应长度是否会给AR生成模型带来训练不稳定（因为序列长度不固定）？
- 能否推广到图像tokenizer？图像的空间区域也有复杂度差异

## 与相关工作的对比
- vs LARP等固定长度视频tokenizer：EVATok在更少token下达到更好质量
- vs TiTok/MAGVIT等先进tokenizer：EVATok的核心贡献是自适应分配策略，可作为它们的增强
- vs TrajTok：TrajTok聚焦理解端的轨迹分组，EVATok聚焦生成端的token长度优化，互补

## 与我的研究方向的关联
- 自适应token分配的框架直接可扩展到VLM的视觉token压缩——对简单图像区域分配少token
- "路由器预测最优配置"的设计模式可复用：训练小模型预测大模型的最优超参数/配置
- 与BiGain、TrajTok等工作形成视觉token效率的完整方法族

## 评分
- 新颖性: ⭐⭐⭐⭐ — 自适应token分配不算新概念，但三步框架的系统化设计和在视频生成上的验证有价值
- 实验充分度: ⭐⭐⭐⭐ — UCF-101验证充分，但缺少更大规模/更多数据集的验证
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，三步流程一目了然
- 对我的价值: ⭐⭐⭐⭐ — 路由器+自适应分配的设计模式可直接借鉴

<!-- RELATED:START -->

## 相关论文

- [Causal Motion Diffusion Models for Autoregressive Motion Generation](causal_motion_diffusion_models_for_autoregressive_motion_generation.md)
- [Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)
- [InnoAds-Composer: Efficient Condition Composition for E-Commerce Poster Generation](innoads-composer_efficient_condition_composition_for_e-commerce_poster_generatio.md)
- [Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control](taming_video_models_for_3d_and_4d_generation_via_zero-shot_camera_control.md)
- [ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation](adapt_attention_driven_adaptive_prompt_scheduling_and_interpolating_orthogonal_c.md)

<!-- RELATED:END -->
