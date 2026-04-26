---
title: >-
  [论文解读] CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner
description: >-
  [ECCV 2024][视频理解][骨架动作识别] 提出CrossGLG框架，利用LLM生成的文本描述以"全局→局部→全局"的方式引导骨架特征学习，在单样本3D动作识别中以仅2.8%的SOTA模型参数量大幅超越对手。
tags:
  - ECCV 2024
  - 视频理解
  - 骨架动作识别
  - 单样本学习
  - 大语言模型
  - 跨模态引导
  - 全局-局部-全局
---

# CrossGLG: LLM Guides One-Shot Skeleton-Based 3D Action Recognition in a Cross-Level Manner

**会议**: ECCV 2024  
**arXiv**: [2403.10082](https://arxiv.org/abs/2403.10082)  
**代码**: 有  
**领域**: 视频理解  
**关键词**: 骨架动作识别, 单样本学习, 大语言模型, 跨模态引导, 全局-局部-全局

## 一句话总结

提出CrossGLG框架，利用LLM生成的文本描述以"全局→局部→全局"的方式引导骨架特征学习，在单样本3D动作识别中以仅2.8%的SOTA模型参数量大幅超越对手。

## 研究背景与动机

单样本骨架动作识别面临两个核心挑战：

**局部信息丢失**：现有方法只关注关节位置等低层信息，无法聚焦关键局部区域

**泛化能力弱**：缺乏高层语义信息指导，对未见过的动作类别难以泛化

人类可以仅凭少量观察就识别关键运动线索并推断整体动作。受此启发，作者提出利用大语言模型（LLM）生成的富含人类知识的文本描述来引导骨架特征学习。这是**首次将LLM文本信息引入单样本骨架动作识别**。

## 方法详解

### 整体框架

CrossGLG采用双分支架构：
- **骨架编码分支**（蓝色）：仅处理骨架序列，推理时只用此分支
- **跨模态引导分支**（绿色）：训练时利用LLM文本引导骨架特征学习

两个分支共享同一分类器，使骨架分支在训练中隐式学习到文本蕴含的高层语义。

### LLM生成知识丰富的动作描述

设计两种prompt从ChatGPT获取信息：
1. **全局动作描述prompt**：描述执行某动作时哪些关节/身体部位最重要（如"挥手"→手臂、手腕）
2. **关节运动描述prompt**：为每个关节生成细粒度的局部运动描述（如"右手：在头部上方快速左右摆动"）

只需替换prompt中的动作名称和关节列表即可适配不同数据集。

### 全局→局部引导（Global-to-Local）

**关节重要性判别模块（JID）**：
1. 骨架编码器的前 Npre 个编码块输出 fpre，沿时间维度池化得到关节整体运动特征
2. JID（两层线性+softmax）预测每个关节的重要性 kout
3. 从全局文本中提取关键关节分布 kgt（利用Stanford CoreNLP提取名词短语，映射为关节）
4. **校准损失**：L_calibrate = MSE(kout, kgt)
5. 后续编码块中，用 kout 对空间交互后的特征重加权

### 局部→全局引导（Local-to-Global）

**跨模态交互模块**（M层交互块）：
1. 文本编码器（DeBERTa-V2-XLarge）编码关节运动描述
2. 文本和骨架特征投影到公共空间
3. 每层交互块执行三种非局部交互：
    - **文本-文本自注意力**：让每个关节文本特征获取其他关节的语义上下文
    - **文本引导骨架交叉注意力**：用文本特征作为Q引导骨架特征的整合
    - **融合**：文本和骨架特征相加后经MLP处理

### 双分支训练与推理

- 训练时：两个分支输出共享MLP分类器
- 总损失：L_overall = L_s + 0.5*L_calibrate + 0.2*L_c
- 推理时：**完全不需要文本输入**，仅用骨架编码分支，额外参数仅+0.1M
- 使用分布校准方法进行单样本分类

### 损失函数 / 训练策略

- L_s：骨架分支分类交叉熵损失
- L_c：跨模态分支分类交叉熵损失
- L_calibrate：JID校准MSE损失
- 跨模态分支梯度可回传到编码块，高层语义信息通过训练过程被注入骨架编码器

## 实验关键数据

### 主实验：NTU RGB+D 120

| 方法 | 20类 | 40类 | 60类 | 80类 | 100类 | 参数(M) |
|------|------|------|------|------|-------|---------|
| APSR | 29.1 | 34.8 | 39.2 | 42.8 | 45.3 | - |
| MotionBERT | 35.5 | 54.3 | 56.5 | 52.8 | 61.0 | 60.3 |
| InfoGCN | 37.0 | 53.9 | 58.8 | 55.7 | 56.1 | 1.6 |
| InfoGCN+GAP | 35.1 | 54.8 | 50.8 | 53.2 | 59.9 | 1.6 |
| **InfoGCN+CrossGLG** | **45.3** | **56.8** | **62.1** | **61.6** | **62.6** | **1.7** |

### 即插即用效果验证

| 骨架编码器 | 20类基线→+CrossGLG | 参数开销 |
|------------|---------------------|----------|
| MotionBERT | 35.5→51.0 (+15.8) | +0.5M |
| HDGCN | 39.0→43.0 (+4.0) | +0.1M |
| InfoGCN | 37.0→45.3 (+8.3) | +0.1M |

### NTU RGB+D 60

| 方法 | 10类 | 20类 | 30类 | 40类 | 50类 |
|------|------|------|------|------|------|
| MotionBERT | 58.3 | 61.0 | 70.0 | 70.3 | 74.5 |
| InfoGCN | 51.1 | 62.1 | 65.7 | 72.1 | 72.3 |
| **InfoGCN+CrossGLG** | **57.9** | **67.1** | **70.9** | **73.4** | **75.6** |

### 消融实验

| G2L | L2G | 20类 | 60类 | 100类 |
|-----|-----|------|------|-------|
| x | x | 37.0 | 58.8 | 56.1 |
| v | x | 43.3 | 60.9 | 61.7 |
| x | v | 42.5 | 61.7 | 58.6 |
| **v** | **v** | **45.3** | **62.1** | **62.6** |

JID插入位置消融（Npre）：第5个编码块最优（太浅特征不够丰富，太深影响力不足）

### 关键发现

- G2L和L2G各自都带来显著提升，组合效果最佳
- 在NTU 120 20类设定下 InfoGCN+CrossGLG 比纯InfoGCN 提升8.3%
- 模型大小仅为MotionBERT的2.8%（1.7M vs 60.3M）
- 在Kinetics数据集上同样超越SOTA，验证了复杂场景的泛化能力

## 亮点与洞察

1. **LLM知识蒸馏到骨架模型**：训练时用LLM文本引导，推理时完全不需要文本，优雅解决了模态不对称问题
2. **即插即用设计**：对MotionBERT/HDGCN/InfoGCN等编码器只需+0.1M参数即获显著提升
3. **全局-局部-全局范式**：先全局文本聚焦关键关节（局部），再局部特征交互汇聚全局表示
4. **可视化验证**：对未见类别的动作，模型无需微调即能关注正确的关键关节

## 局限与展望

- LLM生成的文本质量依赖于prompt设计和LLM能力，不同动作的描述质量可能不均
- JID的关节重要性是静态的（每个动作类别固定），未考虑同一动作不同阶段的动态变化
- 仅在NTU和Kinetics数据集验证，未探索更具挑战性的细粒度动作识别场景
- 文本编码器（DeBERTa-V2-XLarge）虽推理时不用但训练时仍需，增加训练成本

## 相关工作与启发

- 与GAP（全监督文本引导）对比：GAP在单样本设定下甚至不如baseline，证明了CrossGLG设计在few-shot下的优越性
- APSR虽引入语义信息但引入量太少且推理时无法检测重要关节
- 双分支共享分类器的设计可推广到其他跨模态知识迁移场景

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次将LLM引入单样本骨架识别，全局-局部-全局引导机制新颖)
- 实验充分度: ⭐⭐⭐⭐ (三个数据集+多编码器验证+详细消融)
- 写作质量: ⭐⭐⭐⭐ (动机清晰，framework图直观)
- 价值: ⭐⭐⭐⭐⭐ (即插即用+极致效率，实用性极强)

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[ECCV 2024\] Efficient Few-Shot Action Recognition via Multi-Level Post-Reasoning](efficient_few-shot_action_recognition_via_multi-level_post-reasoning.md)
- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)

<!-- RELATED:END -->
