---
title: >-
  [论文解读] Universal Domain Adaptation for Semantic Segmentation
description: >-
  [CVPR 2025][语义分割][通用域适应] 首次提出语义分割的通用域适应（UniDA-SS）任务和 UniMAP 框架，通过域特定原型区分（DSPD）和基于目标的图像匹配（TIM）两个核心组件，在不需要预先知道类别配置的情况下实现从合成数据到真实数据的有效适应，显著超越现有 UDA-SS 方法。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "通用域适应"
  - "原型学习"
  - "图像匹配"
  - "伪标签"
---

# Universal Domain Adaptation for Semantic Segmentation

**会议**: CVPR 2025  
**arXiv**: [2505.22458](https://arxiv.org/abs/2505.22458)  
**代码**: [https://github.com/KU-VGI/UniMAP](https://github.com/KU-VGI/UniMAP)  
**领域**: 语义分割 / 域适应  
**关键词**: 通用域适应, 语义分割, 原型学习, 图像匹配, 伪标签

## 一句话总结

首次提出语义分割的通用域适应（UniDA-SS）任务和 UniMAP 框架，通过域特定原型区分（DSPD）和基于目标的图像匹配（TIM）两个核心组件，在不需要预先知道类别配置的情况下实现从合成数据到真实数据的有效适应，显著超越现有 UDA-SS 方法。

## 研究背景与动机

**领域现状**：无监督域适应语义分割（UDA-SS）旨在利用标注的合成数据（源域）在未标注的真实数据（目标域）上实现高质量分割。现有方法主要分为对抗学习和自训练两大类，其中自训练方法（如 DAFormer, HRDA, MIC）通过伪标签和域混合技术取得了显著进展。

**现有痛点**：现有 UDA-SS 方法都假设源域和目标域的类别关系是已知的，这在实际场景中并不现实。具体来说：(1) 目标域可能包含源域中没有的类（目标私有类）；(2) 源域可能包含目标域中没有的类（源域私有类）。当存在源域私有类时，模型可能错误地将这些类与目标域对齐，导致负迁移和性能严重下降。

**核心矛盾**：当类别配置未知时，自训练方法依赖的伪标签置信度会受到致命影响——源域私有类与某些通用类的特征相似度高，导致通用类的伪标签置信度下降，被错误分配为目标私有类（unknown），进而无法有效学习通用类和目标私有类。

**本文目标**：提出 UniDA-SS 任务，在不知道类别配置的前提下实现鲁棒的域适应分割，同时正确分类通用类和检测目标私有类。

**切入角度**：核心问题在于"提升通用类的置信度"。作者观察到通用类在两个域中虽然语义相同但特征存在差异，用单一原型表示会导致置信度不足。

**核心 idea**：为每个类分配两个域特定原型（源+目标），利用像素嵌入与两个原型的相对距离来区分通用类和私有类，同时通过图像匹配策略增加训练中通用类的曝光率。

## 方法详解

### 整体框架

基于标准自训练 UDA-SS 框架：学生网络 $f_\theta$ 在源域有标签数据和目标域伪标签上训练，教师网络 $g_\phi$ 通过 EMA 更新并生成目标伪标签。分类头设为 $C_s+1$（额外一个 unknown 类）。在此基础上加入 DSPD（原型学习+权重缩放）和 TIM（图像匹配策略），外加 DACS 域混合技术。

### 关键设计

1. **域特定原型区分（DSPD）**:

    - 功能：通过为每个类分配源/目标两个原型，增强目标域通用类的置信度，同时区分通用类和私有类
    - 核心思路：使用固定的 Simplex ETF 空间分配 $2C+1$ 个原型（每个类 2 个域特定原型 + 1 个 unknown 原型），保证所有原型对间等余弦相似度。训练时用三个损失约束像素嵌入与原型的关系：交叉熵损失 $\mathcal{L}_{CE}$ 拉近对应原型，像素-原型对比学习 $\mathcal{L}_{PPC}$ 在全局空间做推拉，像素-原型距离优化 $\mathcal{L}_{PPD}$ 进一步拉近距离。关键是利用像素嵌入与源/目标原型的相对距离计算权重 $w = \frac{2(d_s+1)(d_t+1)}{(d_s+1)+(d_t+1)}$（调和平均），通用类像素与两个原型距离应相近（$w$ 大），私有类只接近一个原型（$w$ 小），从而给通用类更高权重
    - 设计动机：传统方法将同一类在两个域中视为完全相同，忽略了域特定特征差异。通过域特定原型独立学习两个域的特征表示，提升目标预测的置信度

2. **基于目标的图像匹配（TIM）**:

    - 功能：在训练 batch 中优先选择包含最多通用类像素的源图像，增加通用类的学习机会
    - 核心思路：首先计算目标伪标签中每个类的像素比例 $f_c$，然后对稀有类赋予更高权重 $\hat{f_c} = \text{softmax}(\frac{1-f_c}{T})$。对每张源图像计算匹配分数 $S_s = \sum_{c \in c^*} n_c^s \hat{f_c}$（其中 $c^*$ 是源标签和目标伪标签中重叠的类），选择分数最高的源图像与目标图像配对进入训练 batch
    - 设计动机：源域私有类的存在会稀释通用类的学习比例，TIM 通过智能匹配确保每个 batch 都有尽可能多的通用类像素参与训练，同时通过类别加权缓解长尾不平衡

3. **UniDA-SS 基准设定**:

    - 功能：定义了完整的 UniDA-SS 评测协议
    - 核心思路：基于 GTA5 → IDD 和 Pascal-Context → Cityscapes 两个跨域设定，分别设计 CDA-SS、ODA-SS、PDA-SS、OPDA-SS 四种场景的评测基准。评估指标包括通用类平均 IoU（Common）、私有类（unknown）IoU（Private）和 H-score
    - 设计动机：之前只有分类任务有 UniDA 的形式定义和评测，在语义分割这个更精细粒度的任务上此前无人探索

### 损失函数 / 训练策略

总损失 = 源域分割损失 $\mathcal{L}_{seg}^s$ + 加权目标域分割损失 $w \cdot q_t \cdot \mathcal{L}_{seg}^t$ + 原型损失 $\mathcal{L}_{proto} = \mathcal{L}_{CE} + \lambda_1\mathcal{L}_{PPC} + \lambda_2\mathcal{L}_{PPD}$。使用 DACS 域混合技术。伪标签阈值 $\tau_p$ 用于将低置信度像素标记为 unknown。教师网络通过 EMA 从学生网络更新。

## 实验关键数据

### 主实验

Pascal-Context → Cityscapes (OPDA-SS):

| 方法 | Common mIoU | Private IoU | H-score |
|------|-------------|-------------|---------|
| MIC (CDA-SS SOTA) | 48.67 | 7.85 | 13.51 |
| BUS (ODA-SS) | 57.64 | 20.38 | 30.11 |
| **UniMAP (Ours)** | **60.94** | **31.27** | **41.33** |

GTA5 → IDD (OPDA-SS):

| 方法 | Common mIoU | Private IoU | H-score |
|------|-------------|-------------|---------|
| DAFormer | 52.05 | 21.07 | 29.99 |
| HRDA | 53.43 | 32.15 | 40.14 |
| **UniMAP** | **55.95** | **39.65** | **46.42** |

### 消融实验

| 配置 | Common | Private | H-score |
|------|--------|---------|---------|
| Baseline | 57.64 | 20.38 | 30.11 |
| + DSPD | 59.12 | 27.85 | 37.90 |
| + TIM | 59.78 | 25.42 | 35.68 |
| + DSPD + TIM (UniMAP) | **60.94** | **31.27** | **41.33** |

### 关键发现

- DSPD 对私有类检测贡献最大（Private IoU 从 20.38 → 27.85），说明域特定原型有效区分了通用/私有类
- TIM 主要提升通用类性能（Common 从 57.64 → 59.78），验证了增加通用类曝光率的策略
- 两者组合效果互补，H-score 从 30.11 提升到 41.33（+37%）
- 现有 CDA-SS 方法（MIC）在 PDA/OPDA 场景下性能严重下降，证明了 UniDA-SS 任务的必要性

## 亮点与洞察

- **首次定义 UniDA-SS 任务**：将分类中的 UniDA 推广到语义分割这个更具挑战性的像素级任务，贡献了完整的 benchmark，对社区具有引领意义
- **调和平均权重的巧妙设计**：利用像素嵌入与两个域特定原型距离的调和平均来区分通用/私有类，直觉简洁（通用类两个距离接近→调和平均大，私有类一远一近→调和平均小）
- **TIM 的类别加权策略**：不只是选通用类多的源图像，还对稀有通用类赋予更高权重，同时解决了图像匹配和类别不平衡两个问题

## 局限与展望

- ETF 原型是固定的，无法根据训练过程动态调整，可能限制了表达力
- TIM 依赖伪标签质量，训练初期伪标签不准确时匹配策略的效果可能受限
- 目前只在 GTA5→IDD 和 Pascal-Context→Cityscapes 两个跨域设定上验证，更多域对（如 Synthia→Cityscapes）的验证会更完整
- 未探索更复杂的类别关系（如类别间的层次/包含关系）

## 相关工作与启发

- **vs MIC (CDA-SS)**: 闭集域适应的 SOTA，但在有源域私有类时性能急剧下降，说明 CDA 假设在实际中不可靠
- **vs BUS (ODA-SS)**: 开集域适应方法，能处理目标私有类但不处理源域私有类。UniMAP 在其基础上扩展了对源域私有类的处理能力
- **vs UniDA 分类方法（UAN, UniOT）**: 分类任务的 UniDA 方法直接迁移到分割效果很差（H-score < 15），证明了像素级任务需要专门的设计
- 原型学习的思路（ProtoSeg）和 ETF 空间的使用，可以迁移到其他域适应任务（如目标检测、实例分割）

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次定义 UniDA-SS 任务有引领价值，但具体方法（原型+匹配）的新颖性一般
- 实验充分度: ⭐⭐⭐⭐ 多个场景设定、详细消融，但域对数量可以更多
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，motivation 论证充分
- 价值: ⭐⭐⭐⭐ 新任务定义+benchmark 对社区有持续影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[CVPR 2026\] RecycleLoRA: Rank-Revealing QR-Based Dual-LoRA Subspace Adaptation for Domain Generalized Semantic Segmentation](../../CVPR2026/segmentation/recyclelora_rank-revealing_qr-based_dual-lora_subspace_adaptation_for_domain_gen.md)
- [\[ICLR 2026\] Universal Multi-Domain Translation via Diffusion Routers](../../ICLR2026/segmentation/universal_multi-domain_translation_via_diffusion_routers.md)
- [\[CVPR 2025\] CrossEarth-SAR: A SAR-Centric and Billion-Scale Geospatial Foundation Model for Domain Generalizable Semantic Segmentation](crossearth-sar_a_sar-centric_and_billion-scale_geospatial_foundation_model_for_d.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)

</div>

<!-- RELATED:END -->
