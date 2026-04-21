---
title: >-
  [论文解读] Synergistic Prompting for Robust Visual Recognition with Missing Modalities
description: >-
  [ICCV 2025][多模态][missing modality] 提出Synergistic Prompting（SyP）框架，通过动态适配器生成自适应缩放因子来调节基础prompt（动态prompt），并与共享跨模态特征的静态prompt协同，实现在模态缺失场景下的鲁棒视觉识别，在MM-IMDb/Food101/Hateful Memes三个数据集上全面超越DCP等SOTA。
tags:
  - ICCV 2025
  - 多模态
  - missing modality
  - 提示学习
  - CLIP
---

# Synergistic Prompting for Robust Visual Recognition with Missing Modalities

**会议**: ICCV 2025  
**arXiv**: [2507.07802](https://arxiv.org/abs/2507.07802)  
**代码**: 无  
**领域**: 多模态VLM / 缺失模态学习 / Prompt Learning  
**关键词**: missing modality, dynamic prompt, synergistic prompting, CLIP, multi-modal learning

## 一句话总结
提出Synergistic Prompting（SyP）框架，通过动态适配器生成自适应缩放因子来调节基础prompt（动态prompt），并与共享跨模态特征的静态prompt协同，实现在模态缺失场景下的鲁棒视觉识别，在MM-IMDb/Food101/Hateful Memes三个数据集上全面超越DCP等SOTA。

## 研究背景与动机

**领域现状**：多模态学习在跨模态检索、VQA等领域取得显著进展，主要依赖大规模配对数据集和预训练多模态Transformer（如CLIP）。然而现实场景中，传感器故障、隐私问题、数据采集困难等因素导致模态输入经常不完整。

**现有方法分类**：
   - **联合学习**：通过对齐潜在特征空间来建模跨模态关联，但依赖masking/补全策略，引入噪声
   - **跨模态生成**：试图从可用模态重建缺失模态，但模态异质性导致重建质量差、计算开销高
   - **Prompt方法**：利用可学习prompt适配预训练模型，参数高效但存在两个核心缺陷

**现有prompt方法的两大缺陷**：
   - **(1) 静态prompt缺乏灵活性**：所有输入使用相同的prompt嵌入，不管缺失的是哪个模态、缺失程度如何，无法适应动态的真实场景
   - **(2) 层间缺乏协同**：简单的prompt tuning无法充分利用层次化模型表示中的多模态依赖关系，关键模态缺失时性能不可靠

**核心矛盾**：静态prompt的"one-size-fits-all"设计与真实场景中模态缺失模式的高度动态性之间的不匹配。当缺失的是关键模态时，模型需要自适应地增强可用模态的权重，这是静态方法做不到的。

**本文切入点**：结合动态prompt（输入自适应）和静态prompt（保持预训练知识），形成"协同提示"策略，同时保证灵活性和稳定性。

## 方法详解

### 整体框架
SyP基于CLIP双流架构（图像编码器+文本编码器），冻结backbone参数，仅更新prompt和FC层。对每个输入，根据模态缺失类型 $m \in \{c, m_1, m_2\}$（完整、缺失图像、缺失文本），生成模态特定的协同prompt $P_m^I$（图像端）和 $P_m^T$（文本端），prepend到输入token序列中。

### 关键设计1：动态适配器（Dynamic Adapter）

动态适配器的核心是根据**当前输入的可用模态特征**计算自适应缩放因子，动态调节基础prompt的强度。

**特征拼接**：将图像特征 $X_I \in \mathbb{R}^{d_I}$ 和文本特征 $X_T \in \mathbb{R}^{d_T}$ 拼接为联合特征向量：

$$X_C = [X_I, X_T]$$

**缩放因子计算**：通过带瓶颈结构的MLP计算缩放因子：

$$S_d = \sigma\left(\text{MLP}\left(\text{ReLU}\left(\frac{1}{r}W_1 X_C + b_1\right)\right)W_2 + b_2\right)$$

其中 $r$ 为降维比率，sigmoid保证 $S_d \in [0,1]$。较大的 $S_d$ 增强prompt影响力，较小的 $S_d$ 降低影响力，从而根据各模态的相关性进行自适应调节。

**动态prompt生成**：缩放因子逐元素乘以基础prompt：

$$P_{I,D} = P_{I,B} \odot S_d, \quad P_{T,D} = P_{T,B} \odot S_d$$

当某模态缺失时，缩放因子会增加对应prompt的权重；模态完整时则降低权重，实现自适应的模态权重分配。

### 关键设计2：协同提示策略（Synergistic Prompting）

**静态prompt**：捕获图像和文本模态之间的共享特征，通过学习的线性投影函数映射到各模态空间：

$$P_{I,S} = G_I(P_S), \quad P_{T,S} = G_T(P_S)$$

**最终协同prompt**：将动态prompt和静态prompt逐元素相加：

$$P_m^I = P_{I,D} + P_{I,S}, \quad P_m^T = P_{T,D} + P_{T,S}$$

这种组合确保模型同时利用模态特定的自适应调整（动态prompt）和共享的跨模态特征（静态prompt）。

### 关键设计3：层间Prompt传播

Prompt在Transformer层间递归传播，第 $i$ 层的prompt由前一层的prompt经变换函数生成：

$$P_m^{I,R_i} = F_i(P_m^{I,R_{i-1}}) = \text{LN}(\text{FC}(\text{GeLU}(\text{FC}(P_m^{I,R_{i-1}}))))$$

这确保每层的prompt整合了前一层的学习表示和当前层的输入特征，捕获层次化的多模态特征。

### 损失函数

使用标准的任务损失函数，最终多模态prompt特征通过拼接图像和文本prompt后接FC层得到：

$$P_{\text{final}}^{(i)} = \text{FC}(P_m^{I,R_{i-1}} \| P_m^{T,R_{i-1}})$$

训练总损失为所有样本的任务损失之和：$\mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \mathcal{L}_i(P_{\text{final}}^{(i)})$

## 实验

### 数据集与设置
- **MM-IMDb**：25,959个图文电影数据，多标签类型分类，F1-Macro
- **UPMC Food-101**：101类食物，含噪声图文对，Top-1 Accuracy
- **Hateful Memes**：10,000+仇恨梗检测，AUROC
- Backbone: CLIP ViT-B/16，prompt长度 $L_p=36$，应用于 $M=6$ 层
- AdamW，lr=1e-3，weight decay=2e-2，20 epochs，batch size=32

### 主实验结果

| 数据集 | 缺失率 | 缺失模态 | DCP | SyP (本文) | 提升 |
|--------|--------|----------|-----|------------|------|
| MM-IMDb (F1) | 50% | 双模态均衡 | 52.32 | 55.02 | +2.70 |
| MM-IMDb (F1) | 70% | 双模态均衡 | 51.42 | 52.90 | +1.48 |
| MM-IMDb (F1) | 90% | 双模态均衡 | 48.04 | 49.63 | +1.59 |
| Food101 (Acc) | 50% | 双模态均衡 | 85.24 | 86.17 | +0.93 |
| Food101 (Acc) | 70% | 双模态均衡 | 81.87 | 82.45 | +0.58 |
| Food101 (Acc) | 90% | 双模态均衡 | 79.87 | 81.03 | +1.16 |
| Hateful Memes (AUROC) | 50% | 双模态均衡 | 66.02 | 68.16 | +2.14 |
| Hateful Memes (AUROC) | 70% | 双模态均衡 | 66.08 | 68.42 | +2.34 |
| Hateful Memes (AUROC) | 90% | 双模态均衡 | 66.78 | 68.93 | +2.15 |

SyP在所有数据集、所有缺失率、所有缺失模式下均超越SOTA DCP，在Hateful Memes上提升最为显著（+2~7个百分点）。

### 消融实验

| 变体 | Hateful Memes | Food101 | MM-IMDb |
|------|--------------|---------|---------|
| w/o Synergistic Prompts（仅微调分类器） | 57.35 | 71.59 | 44.63 |
| 仅动态Prompt | 66.37 | 82.90 | 51.21 |
| 仅静态Prompt | 65.62 | 83.06 | 50.34 |
| **SyP（动态+静态协同）** | **68.16** | **86.17** | **54.72** |

| 变体 | Hateful Memes | Food101 | MM-IMDb |
|------|--------------|---------|---------|
| 仅基础Prompt | 64.27 | 81.68 | 48.95 |
| 基础Prompt + 动态适配器 | 66.37 | 82.90 | 51.21 |
| 协同Prompts（无动态适配器） | 66.19 | 84.85 | 51.90 |
| **协同Prompts + 动态适配器** | **68.16** | **86.17** | **54.72** |

### 关键发现
1. 动态+静态协同比单独任一prompt策略提升3~4个百分点，证明"协同"是关键
2. 动态适配器在协同prompt基础上再带来约2个百分点的提升，证明自适应缩放的有效性
3. 在高缺失率（90%）下SyP的性能衰减明显小于baseline，展现出优越的鲁棒性
4. 数据集特性差异：MM-IMDb和Food101更依赖文本语义，文本缺失影响更大；Hateful Memes中图像缺失影响更大

## 亮点与洞察
1. **动态+静态的互补设计很巧妙**：动态prompt负责自适应调整（灵活性），静态prompt保持预训练知识基底（稳定性），二者逐元素加法组合简洁有效
2. **缩放因子的设计直觉清晰**：当模态缺失时，缺失模态输入为零张量，拼接后的联合特征会自然偏向可用模态，sigmoid MLP输出的缩放因子也会相应调整，无需显式的缺失检测机制
3. **参数效率高**：冻结CLIP backbone，仅训练prompt和FC层，添加的参数量极少

## 局限性
1. 仅在 $M=2$ 模态场景下验证，未扩展到三模态或更多模态
2. 缺失模态用零填充张量替代，更先进的缺失处理策略未被探索
3. 实验仅在分类任务上验证，生成类任务（captioning、VQA）未涉及
4. 动态适配器的降维比率 $r$ 等超参数的敏感性分析不足

## 相关工作
- **缺失模态学习**：联合学习（SMIL, ShaSpec）、跨模态生成（LEL, MTP）、prompt方法（MMP, DePT, DCP）
- **Prompt Learning**：CoOp、MaPLe、DePT、DCP等prompt tuning方法
- **多模态预训练**：CLIP、ViT等backbone

## 评分
- 新颖性：3/5（动态+静态prompt组合的idea有一定新意，但整体框架较为工程化）
- 技术深度：3/5（方法简洁但理论分析不深入）
- 实验充分度：4/5（三个数据集、多种缺失率、丰富的消融实验）
- 写作质量：3/5（较清晰但略冗长）

<!-- RELATED:START -->

## 相关论文

- [Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs](../../ECCV2024/multimodal_vlm/metaprompting_for_automating_zeroshot_visual_recognitio.md)
- [Recognition-Synergistic Scene Text Editing](../../CVPR2025/multimodal_vlm/recognition-synergistic_scene_text_editing.md)
- [Generalizable Object Re-Identification via Visual In-Context Prompting](generalizable_object_re-identification_via_visual_in-context_prompting.md)
- [MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [Beyond Heuristic Prompting: A Concept-Guided Bayesian Framework for Zero-Shot Image Recognition](../../CVPR2026/multimodal_vlm/beyond_heuristic_prompting_a_concept-guided_bayesian_framework_for_zero-shot_ima.md)

<!-- RELATED:END -->
