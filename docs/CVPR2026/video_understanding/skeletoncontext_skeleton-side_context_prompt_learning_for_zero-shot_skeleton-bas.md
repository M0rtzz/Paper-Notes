---
title: >-
  [论文解读] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition
description: >-
  [CVPR 2026][视频理解][零样本动作识别] 提出SkeletonContext框架，通过跨模态上下文提示模块从预训练语言模型重建骨骼数据缺失的环境和物体上下文语义，并用关键部位解耦模块增强运动关键关节的判别力，在NTU-60/120和PKU-MMD上的零样本和广义零样本设置中达到SOTA。
tags:
  - "CVPR 2026"
  - "视频理解"
  - "零样本动作识别"
  - "骨骼序列"
  - "上下文提示学习"
  - "跨模态对齐"
  - "关键部位解耦"
---

# SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition

**会议**: CVPR 2026  
**arXiv**: [2603.29692](https://arxiv.org/abs/2603.29692)  
**代码**: [https://github.com/NingWang2049/skeletoncontext](https://github.com/NingWang2049/skeletoncontext)  
**领域**: 视频理解 / 动作识别  
**关键词**: 零样本动作识别, 骨骼序列, 上下文提示学习, 跨模态对齐, 关键部位解耦

## 一句话总结

提出SkeletonContext框架，通过跨模态上下文提示模块从预训练语言模型重建骨骼数据缺失的环境和物体上下文语义，并用关键部位解耦模块增强运动关键关节的判别力，在NTU-60/120和PKU-MMD上的零样本和广义零样本设置中达到SOTA。

## 研究背景与动机

1. **领域现状**：零样本骨骼动作识别（ZSSAR）通过将骨骼特征与文本嵌入在共享空间中对齐来识别未见类别。现有方法主要关注更好的骨骼编码器、数据增强或外部知识增强。
2. **现有痛点**：骨骼序列只有关节坐标，不包含物体、环境等上下文线索。"在键盘上打字"和"在纸上写字"的骨骼运动高度相似，但缺少"键盘"和"纸"的上下文就无法区分。
3. **核心矛盾**：骨骼模态天然缺乏上下文信息，而语义描述中包含丰富的上下文，两者之间存在本质的语义鸿沟，直接对齐效果有限。
4. **本文目标**：为骨骼表示注入语言驱动的上下文语义，弥合跨模态对齐的语义鸿沟。
5. **切入角度**：用LLM生成结构化的上下文描述（环境+使用物体+目标物体），然后训练模型从骨骼运动中"重建"这些上下文，让骨骼编码器自身获得上下文感知能力。
6. **核心 idea**：让骨骼编码器通过掩码重建学会从运动模式推断上下文语义（如交互物体和环境）。

## 方法详解

### 整体框架

骨骼序列经Shift-GCN提取特征后，分别送入两个模块：(1) 跨模态上下文提示模块——通过差分关节编码器获取细粒度骨骼特征，与BERT处理的掩码上下文提示双向交叉注意力交互，重建被掩码的上下文词（环境、物体），得到上下文增强骨骼特征；(2) 关键部位解耦模块——预测关节重要性图突出运动关键关节。两个分支分别与对应的语义嵌入通过对比损失对齐。

### 关键设计

1. **跨模态上下文提示模块 (Cross-Modal Context Prompt)**:

    - 功能：让骨骼编码器获得推断上下文语义（交互物体、环境）的能力
    - 核心思路：先用LLM（ChatGPT-4）生成每类动作的结构化描述，格式为"In [环境], [身体部位] uses [物体] to [子动作] on [目标物体]"。每类生成10个描述。训练时将环境、使用物体、目标物体三个slot用[MASK]替换，骨骼特征通过双向交叉注意力与BERT的token表示交互，BERT的掩码预测头重建被掩码的上下文词。通过上下文重建损失$\mathcal{L}_{ccr}$让骨骼特征学会蕴含上下文信息
    - 设计动机：不同于SCoPLe等方法增强文本编码器，本方法直接增强骨骼编码器侧的表示，使骨骼特征自身携带上下文信息

2. **差分关节编码器 (Differential Joint Encoder)**:

    - 功能：捕捉关节间的细微差异，建模姿态特定的空间依赖
    - 核心思路：将骨骼特征池化到拓扑级，投影为query和key，计算差分拓扑表示$A^{diff} = \phi(\mathcal{T}_1(H_x^Q) - \mathcal{T}_2(H_x^K))$，即所有关节对之间的差异矩阵。然后用此差异矩阵对原特征加权聚合，得到拓扑增强嵌入$F_x^{diff}$
    - 设计动机：不同姿态的"指纹"藏在关节对之间的差异中——"弯腰"暗示桌面场景，"抬手"暗示头部交互。差分编码隐式反映上下文线索

3. **渐进式部分掩码 (Progressive Partial Masking)**:

    - 功能：作为课程学习策略，逐渐增加上下文重建难度
    - 核心思路：定义掩码比例$r_t = \min(1, t/T)$随训练步数线性增长。训练初期仅掩码少量上下文slot（如只掩码环境），使重建任务较简单；随着训练推进，掩码比例增加直到所有slot全掩码，迫使模型仅从骨骼运动和BERT语言先验推断完整上下文
    - 设计动机：结构化提示格式与BERT预训练的自然语言差异大，直接全掩码重建太难导致训练不稳定。渐进策略桥接了分布差异

### 损失函数 / 训练策略

总损失$\mathcal{L} = \mathcal{L}_{align} + \mathcal{L}_{ccr} + \mathcal{L}_{kpd}$：
- $\mathcal{L}_{align}$：对比交叉熵损失，分别对齐上下文增强骨骼特征与上下文语义嵌入、以及关键部位特征与动作语义嵌入
- $\mathcal{L}_{ccr}$：掩码上下文重建损失，监督BERT恢复被掩码的上下文词
- $\mathcal{L}_{kpd}$：关节重要性校准损失，用LLM生成的身体部位先验$K_{gt}$引导关节权重学习

推理时使用校准堆叠（calibrated stacking）缓解GZSL中的域偏移，聚合上下文分支和部位分支的预测。

## 实验关键数据

### 主实验

ZSL准确率（%）：

| 方法 | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|------|-------------|--------------|----------------|---------------|
| STAR (ACMM24) | 81.4 | 45.1 | 63.3 | 44.3 |
| Neuron (CVPR25) | 86.9 | 62.7 | 71.5 | 57.1 |
| FS-VAE (ICCV25) | 86.9 | 57.2 | 74.4 | 62.5 |
| **Ours** | **89.6** | **64.4** | 74.2 | 60.1 |

GZSL调和均值H（%）：

| 方法 | NTU-60 55/5 | NTU-60 48/12 | NTU-120 110/10 | NTU-120 96/24 |
|------|-------------|--------------|----------------|---------------|
| ScoPLe (CVPR25) | 70.8 | 57.9 | 52.2 | 52.2 |
| Neuron (CVPR25) | 71.4 | 59.1 | 63.3 | 53.6 |
| FS-VAE (ICCV25) | 75.7 | 52.1 | 63.3 | 54.7 |
| **Ours** | **77.1** | **61.1** | 63.1 | **56.1** |

### 消融实验

| DJE | SCG | PPM | KPD | NTU60-ZSL | NTU120-GZSL |
|-----|-----|-----|-----|-----------|-------------|
| ✗ | ✗ | ✗ | ✗ | 79.4 | 49.4 |
| ✓ | ✗ | ✗ | ✗ | 81.4 | 51.4 |
| ✓ | ✓ | ✗ | ✗ | 83.9 | 55.4 |
| ✓ | ✓ | ✓ | ✗ | 87.4 | 55.9 |
| ✓ | ✓ | ✓ | ✓ | **89.6** | **56.1** |

### 关键发现

- **上下文重建是主要贡献**：SCG引入带来最大跳跃（81.4→83.9 ZSL），PPM进一步稳定化提升到87.4
- 在困难相似类上（Hard Level），本方法GZSL达55.8%，比Neuron高12.0个点、比FS-VAE高5.1个点，验证上下文推断在细粒度区分中的关键作用
- 去掉$\mathcal{L}_{ccr}$（即去掉上下文重建监督）ZSL从89.6降至86.4，证实LLM上下文对跨模态对齐的必要性
- 对象相关slot（Use Object + Target Object）比环境slot贡献更大（87.0 vs 84.4），因为骨骼动作主要由手物交互定义
- 在PKU-MMD上GZSL调和均值达71.4%，比第二名Neuron高2.2个点

## 亮点与洞察

- **反向思维——增强骨骼而非文本**：之前的方法（SCoPLe、Neuron）主要增强文本编码器以更好地匹配骨骼，但SkeletonContext反过来增强骨骼表示使其携带上下文语义。这从根本上解决了信息不对称的问题
- **掩码重建作为跨模态知识转移的桥梁**：借鉴VL-BEiT等视觉-语言预训练的掩码重建思路，但创新性地用于无视觉的骨骼模态，让BERT的语言知识"流入"骨骼编码器
- **定性分析有说服力**：推理时无需任何文本输入就能从骨骼推断出"键盘"或"笔/纸"等上下文物体，直观展示了模型确实学到了运动-上下文映射

## 局限与展望

- 依赖ChatGPT-4生成描述质量和结构化模板的合理性，不同LLM可能产生不同效果
- 仅用三个slot（环境、使用物体、目标物体），未考虑细粒度的身体部位交互方式
- Shift-GCN作为骨骼编码器已非最新选择，用更强的编码器（如CTR-GCN、InfoGCN）可能进一步提升
- 在NTU-120 110/10 split上未超过FS-VAE（74.2 vs 74.4），说明在较多已见类的场景下上下文增强的边际收益可能减少

## 相关工作与启发

- **vs SCoPLe (CVPR25)**: 通过联合调整文本和骨骼提示实现数据驱动语义对齐，但未引入额外上下文信息。SkeletonContext通过重建从根本上补全了骨骼的信息缺失
- **vs Neuron (CVPR25)**: 使用多轮LLM生成的side information动态引导骨骼-语义协同，但仍在对齐层面操作。SkeletonContext直接在骨骼编码器侧注入上下文
- **vs FS-VAE (ICCV25)**: 频率-语义建模分解骨骼运动为高低频组件，是互补方向——可以将频率分解与上下文注入结合

## 评分

- 新颖性: ⭐⭐⭐⭐ 将掩码重建用于骨骼-语言跨模态上下文注入是新颖的视角
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集多个split、GZSL+ZSL、相似类实验、充分消融
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，但部分公式符号稍显冗余
- 价值: ⭐⭐⭐⭐ 对零样本骨骼动作识别有明确推动，"增强骨骼侧而非文本侧"的思路值得推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](../../ECCV2024/video_understanding/sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[AAAI 2026\] SUGAR: Learning Skeleton Representation with Visual-Motion Knowledge for Action Recognition](../../AAAI2026/video_understanding/sugar_learning_skeleton_representation_with_visual-motion_knowledge_for_action_r.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[ICML 2026\] SkelHCC: A Hyperbolic CLIP-Driven Cache Adaptation Framework for Skeleton-based One-Shot Action Recognition](../../ICML2026/video_understanding/skelhcc_a_hyperbolic_clip-driven_cache_adaptation_framework_for_skeleton-based_o.md)

</div>

<!-- RELATED:END -->
