---
title: >-
  [论文解读] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness
description: >-
  [CVPR 2025][机器人][灵巧手抓取] 本文提出 DexGrasp Anything，将三种物理约束力集成到扩散模型的训练和采样阶段，实现几乎所有开放数据集上 SOTA 的灵巧手抓取姿态生成，并构建了包含 15K+ 物体、340万+ 抓取姿态的最大规模灵巧抓取数据集。
tags:
  - "CVPR 2025"
  - "机器人"
  - "灵巧手抓取"
  - "扩散模型"
  - "物理约束"
  - "大规模数据集"
  - "通用抓取"
---

# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness

**会议**: CVPR 2025  
**arXiv**: [2503.08257](https://arxiv.org/abs/2503.08257)  
**代码**: [https://github.com/4DVLab/DexGrasp-Anything](https://github.com/4DVLab/DexGrasp-Anything)  
**领域**: 图像生成  
**关键词**: 灵巧手抓取、扩散模型、物理约束、大规模数据集、通用抓取

## 一句话总结
本文提出 DexGrasp Anything，将三种物理约束力集成到扩散模型的训练和采样阶段，实现几乎所有开放数据集上 SOTA 的灵巧手抓取姿态生成，并构建了包含 15K+ 物体、340万+ 抓取姿态的最大规模灵巧抓取数据集。

## 研究背景与动机

**领域现状**：灵巧手抓取是机器人操作核心能力。ShadowHand 有 24 个关节参数，搜索空间极大。扩散模型因能生成多样化高质量样本而成为建模抓取分布的主流选择。

**现有痛点**：现有扩散方法在训练和采样中缺乏物理约束，生成的抓取姿态经常出现手-物体穿透或接触不足。同时现有数据集规模有限、物体类别窄。

**核心矛盾**：扩散模型标准 MSE 训练目标只关注噪声预测精度，不包含物理可行性的显式监督。

**本文目标**：(1) 将物理先验嵌入扩散模型训练和采样；(2) 构建大规模高质量灵巧抓取数据集。

**切入角度**：通过 Tweedie 公式从噪声预测反推干净样本 $\hat{h}_0$ 来计算物理约束损失，在采样阶段用后验采样持续施加物理引导。

**核心 idea**：三种物理约束力（surface pulling、external-penetration repulsion、self-penetration repulsion）+ 训练和采样双重注入 + LLM 增强物体表示。

## 方法详解

### 整体框架
给定物体3D点云 $O$，训练条件扩散模型生成灵巧手抓取姿态 $h = (\theta, R, t) \in \mathbb{R}^{33}$。物体特征通过 Point Transformer 编码并融合 LLM 语义先验。

### 关键设计

1. **三种物理约束力**:

    - 功能：确保生成的抓取姿态物理可行
    - 核心思路：(a) **表面拉力（SPF）**：对手掌内表面点计算到物体最近邻距离，近距离点施加拉力使手指贴近；(b) **外穿透斥力（ERF）**：利用签名距离检测穿透并施加斥力；(c) **自穿透斥力（SRF）**：手指间成对距离低于阈值时施加斥力防止互穿。
    - 设计动机：SPF 确保"抓得住"，ERF 确保"不穿透"，SRF 确保"不打架"——覆盖灵巧抓取三个核心物理要求。

2. **物理感知训练 + 物理引导采样**:

    - 功能：将物理约束同时嵌入训练和推理
    - 核心思路：训练时从噪声预测通过 Tweedie 反推 $\hat{h}_0$，施加 $L_{PADG} = L_{simple} + \sum \alpha_i L_{PA_i}(\hat{h}_0)$。采样时用 classifier guidance 风格的后验采样，将物理约束梯度映射到 posterior mean 偏移量，并用球面高斯约束缓解估计偏差。
    - 设计动机：训练阶段提供"稀疏"的物理监督，采样阶段迭代精细化将分布引导到物理可行区域，两者互补。

3. **LLM 增强的物体表示**:

    - 功能：为点云几何特征补充语义先验
    - 核心思路：LLM 生成物体描述，BERT 编码为语义向量，与 Point Transformer 几何特征拼接后通过 cross-attention 融入扩散主干。
    - 设计动机：纯点云难区分功能相似而形状不同的物体，语义先验提供高层"怎么抓"的知识。

### 损失函数 / 训练策略
$L_{PADG} = L_{simple} + \alpha_1 L_{SPF} + \alpha_2 L_{ERF} + \alpha_3 L_{SRF}$。"Model-in-the-loop"策略：用已训练模型生成新姿态，经 IsaacGym 验证后加入数据集。

## 实验关键数据

### 主实验
5 个数据集评估（Suc.6 = 6N力下成功率）：

| 方法 | DexGraspNet | UniDexGrasp | MultiDex | RealDex | DexGRAB |
|------|-------------|-------------|----------|---------|---------|
| UniDexGrasp | 33.9 | 23.7 | 21.6 | 27.1 | 20.8 |
| GraspTTA | 18.6 | 21.0 | 30.3 | 13.3 | 14.4 |
| DexGrasp Anything | **最优** | **最优** | **最优** | **最优** | **最优** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 基础扩散 | 基线 | 无物理约束 |
| + SPF + ERF + SRF | 显著提升 | 三力联合 |
| + LLM 语义 | 额外提升 | 语义先验有益 |
| DGA 数据集 | 所有方法均提升 | 数据质量关键 |

### 关键发现
- 物理约束在训练和采样阶段都有独立贡献，组合最佳
- DGA 数据集（340万抓取）显著提升现有方法——只换数据也能提升
- "model-in-the-loop"可持续扩大数据集

## 亮点与洞察
- **物理约束注入扩散模型范式**：通过 Tweedie 映射到数据空间施加约束，可迁移到分子设计等领域。
- **数据飞轮**：借鉴 SAM 的 model-in-the-loop 扩大数据集是可复制的增长模式。
- **三力平衡设计完备且直觉清晰**。

## 局限与展望
- 仅在 ShadowHand 验证，其他灵巧手迁移性未探索
- 开环方法，不考虑执行中接触反馈
- 合成物体与真实世界的 sim-to-real gap 待解决

## 相关工作与启发
- **vs UniDexGrasp**: 本文在其物理约束基础上整合到扩散模型训练+采样
- **vs SceneDiffuser**: 通用场景交互扩散，缺乏灵巧抓取专用约束
- 数据集构建方法论可作为其他操作领域参考

## 评分
- 新颖性: ⭐⭐⭐⭐ 物理约束注入扩散训练+采样的完整方案
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集全面评估，消融详尽
- 写作质量: ⭐⭐⭐⭐ 方法清晰，公式完整
- 价值: ⭐⭐⭐⭐ 数据集和方法对灵巧抓取社区都有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ManiVideo: Generating Hand-Object Manipulation Video with Dexterous and Generalizable Grasping](manivideo_generating_hand-object_manipulation_video_with_dexterous_and_generaliz.md)
- [\[CVPR 2025\] Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References](learning_physics-based_full-body_human_reaching_and_grasping_from_brief_walking_.md)
- [\[CVPR 2025\] ZeroGrasp: Zero-Shot Shape Reconstruction Enabled Robotic Grasping](zerograsp_zero-shot_shape_reconstruction_enabled_robotic_grasping.md)
- [\[AAAI 2026\] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](../../AAAI2026/robotics/towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](universal_actions_for_enhanced_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
