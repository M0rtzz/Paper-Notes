---
title: >-
  [论文解读] Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing
description: >-
  [ECCV 2024][人体理解][模态缺失] 本文提出MMA-FAS框架解决多模态人脸反欺骗中的模态缺失问题，通过模态解耦适配器从频率分解角度分离模态不变和模态特有特征，结合LBP引导的对比损失和自适应模态组合采样策略，在所有模态缺失场景下均达到SOTA。
tags:
  - ECCV 2024
  - 人体理解
  - 模态缺失
  - 人脸反欺骗
  - 模态解耦
  - 对比学习
  - Transformer
---

# Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing

**会议**: ECCV 2024  
**论文链接**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/3248_ECCV_2024_paper.php)
**代码**: 无  
**领域**: 人脸理解 / 多模态学习 / 人脸反欺骗  
**关键词**: 模态缺失, 人脸反欺骗, 模态解耦, 对比学习, Vision Transformer

## 一句话总结

本文提出MMA-FAS框架解决多模态人脸反欺骗中的模态缺失问题，通过模态解耦适配器从频率分解角度分离模态不变和模态特有特征，结合LBP引导的对比损失和自适应模态组合采样策略，在所有模态缺失场景下均达到SOTA。

## 研究背景与动机

**领域现状**：多模态人脸反欺骗（FAS）利用RGB、深度（Depth）、红外（IR）等多种模态的信息来判别活体和欺骗攻击，因为不同模态提供了互补的线索（如深度图能有效识别平面攻击，红外能区分温度差异）。当所有模态都可用时，多模态FAS的性能显著优于单模态方法。

**现有痛点**：在实际部署中，某些模态的传感器可能不可用（设备不配备深度相机）或者暂时失效（红外传感器故障），导致模态缺失。此时基于Vision Transformer的多模态FAS方法性能会急剧下降，因为这些模型在训练时假设所有模态完整，无法优雅地处理缺失情况。现有应对模态缺失的方法主要依赖模态不变特征（modality-invariant features）——即跨模态共享的通用特征——来缓解缺失问题，但它们完全忽略了模态特有特征（modality-specific features）的价值。

**核心矛盾**：模态不变特征虽然在模态缺失时仍然可用，但它们只捕获了跨模态共享的信息，丢失了每个模态独特的判别线索（如RGB的纹理细节、深度图的几何信息、IR的热分布）。另一方面，模态特有特征在对应模态缺失时无法获取。如何在利用特有特征的判别力和应对其不可用性之间取得平衡，是核心挑战。

**本文目标** (1) 如何显式地分离模态不变特征和模态特有特征？(2) 如何在模态缺失时仍然有效利用已有模态的特有特征？(3) 如何平衡不同模态组合场景下的训练过程？

**切入角度**：作者提出从频率分解的角度来解耦模态不变和模态特有特征——不同模态共享的信息主要体现在低频结构上（如人脸轮廓），而模态特有信息主要体现在高频细节上（如纹理、噪声模式）。基于这个观察，设计轻量级的适配器进行特征分解。同时，利用LBP（Local Binary Pattern）作为纹理先验来引导对比学习，强化特征的判别力。

**核心 idea**：通过频率分解适配器显式分离模态不变/特有特征，结合LBP引导的对比损失和自适应采样策略，实现对模态缺失的鲁棒FAS。

## 方法详解

### 整体框架

MMA-FAS基于预训练的ViT构建。输入为多模态人脸图像（RGB、Depth、IR的任意组合），输出为活体/欺骗判别结果。核心架构包括：(1) ViT主干提取基础特征；(2) 模态解耦适配器（Modality-Disentangle Adapters）插入ViT各层，从频率角度分解模态不变和特有特征；(3) LBP引导的对比损失在特征空间中根据攻击类型和模态组合进行聚类；(4) 自适应模态组合采样策略动态调整训练过程中不同模态组合的采样概率。

### 关键设计

1. **模态解耦适配器（Modality-Disentangle Adapters）**:

    - 功能：在ViT的每一层中显式分离模态不变特征和模态特有特征
    - 核心思路：适配器由两个并行分支组成——不变分支和特有分支。不变分支通过低通滤波操作提取跨模态共享的低频特征（如整体结构和轮廓信息），特有分支通过高通滤波提取每个模态独有的高频细节（如纹理模式和噪声特征）。具体实现上，使用可学习的频率分解函数将适配器的中间特征分为低频和高频分量。不变特征 $f_{inv}$ 由低频分量产生，对模态身份不敏感；特有特征 $f_{spec}$ 由高频分量产生，编码模态独特的判别信息。最终特征是两者的加权组合：$f = f_{inv} + \gamma \cdot f_{spec}$，其中 $\gamma$ 根据模态是否可用进行调控
    - 设计动机：直接在ViT的隐层中进行频率分解是一种轻量级且有效的解耦方式。频率域的物理含义为解耦提供了可解释的先验——低频对应跨模态共享结构，高频对应模态独有细节。适配器的设计使得ViT主干参数可以冻结，大幅减少训练参数

2. **LBP引导的对比损失（LBP-Guided Contrastive Loss）**:

    - 功能：利用LBP纹理先验强化特征空间中的类别判别力和模态鲁棒性
    - 核心思路：该损失由两个层级组成。批次级模态掩码（batch-level modality masking）：在每个训练batch中随机mask掉某些模态的输入，强制模型在各种模态缺失场景下都能工作。样本级模态掩码（sample-level modality masking）：对batch中的每个样本独立随机决定掩码模态，增加训练多样性。在此基础上，LBP引导的对比损失计算样本间的对比关系。LBP提取的纹理特征作为先验锚点——同一攻击类型的样本应该具有相似的LBP模式。对比损失鼓励：同类别且同模态组合的样本特征聚近，不同类别的样本特征拉远，同类别但不同模态组合的样本特征通过模态不变分支保持接近
    - 设计动机：传统的对比损失只区分正负样本，不考虑模态组合的差异。LBP引导使得对比学习能同时增强模态特有和模态不变特征。LBP本身是一种经典的纹理描述符，在FAS领域有很好的先验表现，用它来引导深度特征的学习是一种有效的知识迁移

3. **自适应模态组合采样策略（Adaptively Modal Combination Sampling）**:

    - 功能：动态调整训练过程中不同模态组合场景的采样概率，平衡各场景的学习进度
    - 核心思路：在模态缺失训练中，不同模态组合的难度不同——全模态最容易，只有单个模态最难，两个模态居中。如果均匀采样，模型会过度拟合简单场景而在困难场景上欠拟合。自适应采样策略动态监测每种模态组合在验证集上的性能，对性能较差（即更困难）的组合分配更高的采样概率。具体来说，以每种组合的损失值的倒数作为权重来调整采样概率，性能差的组合获得更多训练机会。采样概率在训练过程中周期性更新，避免频繁震荡
    - 设计动机：多模态缺失场景下存在组合数爆炸问题（3个模态有7种非空组合），不同组合的学习难度差异显著。自适应采样确保模型在所有场景下都能达到良好性能，而非只在部分场景上表现好

### 损失函数 / 训练策略

总训练损失为：$L = L_{cls} + \alpha L_{LBP-con} + \beta L_{dis}$。其中 $L_{cls}$ 是分类交叉熵损失，$L_{LBP-con}$ 是LBP引导的对比损失，$L_{dis}$ 是解耦正则化损失（约束不变特征和特有特征的正交性）。训练时使用ViT预训练权重初始化主干并冻结，只训练适配器和分类头，参数量小、训练高效。

## 实验关键数据

### 主实验

| 缺失场景 | 指标 | MMA-FAS | 之前SOTA | 数据集 |
|----------|------|---------|---------|--------|
| 缺Depth | ACER↓ | 最优 | 次优 | CASIA-SURF / WMCA |
| 缺IR | ACER↓ | 最优 | 次优 | CASIA-SURF / WMCA |
| 缺RGB | ACER↓ | 最优 | 次优 | CASIA-SURF / WMCA |
| 只有RGB | ACER↓ | 最优 | 次优 | CASIA-SURF / WMCA |
| 全模态 | ACER↓ | 最优 | 次优（与VP-FAS等对比） |
| 跨数据集 | ACER↓ | 最优 | 次优 | CASIA→WMCA等 |

### 消融实验

| 配置 | ACER(avg) | 说明 |
|------|----------|------|
| Full MMA-FAS | 最优 | 完整模型 |
| w/o 模态解耦适配器 | 明显上升 | 不分离不变/特有特征 |
| w/o LBP引导对比损失 | 中等上升 | 对比学习缺少纹理先验引导 |
| w/o 自适应采样策略 | 中等上升 | 均匀采样导致困难场景欠拟合 |
| w/o 频率分解（直接分割） | 明显上升 | 频率分解优于简单的通道分割 |
| 仅不变特征 | 较大上升 | 忽略特有特征损失判别力 |
| 仅特有特征 | 较大上升 | 模态缺失时特有特征不可用 |

### 关键发现
- 模态解耦适配器的贡献最大，特别是在模态严重缺失（只有一个模态）的场景下，解耦方法显著优于不解耦的baseline
- LBP引导对比损失在跨数据集场景下贡献突出，说明LBP的纹理先验有助于提取更泛化的特征
- 自适应采样策略在不同缺失场景间的性能方差更小，证明其平衡效果
- 不变特征和特有特征缺一不可：只用不变特征丢失了模态独有的判别信息，只用特有特征在缺失时失效

## 亮点与洞察
- **频率分解做模态解耦是一个优雅的方案**：不需要额外的解耦网络，只用适配器中的低通/高通滤波就能实现。物理含义清晰——低频是跨模态共享结构，高频是模态独有细节。这个思路可以迁移到任何多模态融合任务中处理模态缺失
- **LBP先验与深度对比学习的结合**：LBP是传统方法中验证过的有效纹理描述符，将它作为对比学习的锚点/引导，是一种将传统方法先验融入深度学习的巧妙方式。这种"传统先验引导深度学习"的策略有通用性
- **自适应采样解决训练不平衡**：多模态缺失场景的组合爆炸导致的训练不平衡问题被自适应采样优雅地解决，这个trick可以直接应用于其他多模态or多任务学习场景

## 局限与展望
- 频率分解假设低频=共享、高频=特有可能过于简化，某些跨模态共享信息也可能出现在中高频段
- 目前只实验了RGB+Depth+IR三模态的组合，扩展到更多模态（如近红外NIR、热红外LWIR等）时的有效性未知
- LBP引导的对比损失计算需要额外提取LBP特征，增加了预处理开销
- 自适应采样策略需要周期性评估验证集性能来更新概率，在在线学习场景下可能不太实际
- 在无任何模态可用时（极端情况下所有传感器都失效），该方法自然无能为力

## 相关工作与启发
- **vs VP-FAS**: VP-FAS使用视觉提示（visual prompts）来处理模态缺失，通过学习模态特定的prompt来适配不同输入。MMA-FAS则通过显式的特征解耦来处理，更加可解释且效果更好
- **vs Flexible-Modal FAS (FM-FAS)**: FM-FAS通过模态无关的训练来处理缺失，但没有区分不变和特有特征。MMA-FAS的解耦策略更精细，能在缺失时保留可用特有特征的价值
- **vs ViTAF**: ViTAF在ViT上使用辅助适配器进行FAS，但不专门处理模态缺失问题。MMA-FAS的解耦适配器是专为模态缺失设计的，同时保持了ViTAF轻量化微调的优势

## 评分
- 新颖性: ⭐⭐⭐⭐ 频率分解解耦+LBP引导对比+自适应采样三者结合新颖有效
- 实验充分度: ⭐⭐⭐⭐ 涵盖所有缺失场景、消融完整、有跨数据集验证
- 写作质量: ⭐⭐⭐⭐ 问题formulation清晰，方法动机阐述充分
- 价值: ⭐⭐⭐⭐ 模态缺失是多模态学习的核心问题，本文方案具有实际部署价值

<!-- RELATED:START -->

## 相关论文

- [TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](../../ICCV2025/human_understanding/dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [Large Motion Model for Unified Multi-Modal Motion Generation](large_motion_model_for_unified_multimodal_motion_generation.md)
- [PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](../../AAAI2026/human_understanding/pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)
- [FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)

<!-- RELATED:END -->
