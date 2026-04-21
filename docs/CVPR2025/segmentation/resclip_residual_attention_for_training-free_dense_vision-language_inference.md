---
title: >-
  [论文解读] ResCLIP: Residual Attention for Training-free Dense Vision-language Inference
description: >-
  [CVPR 2025][图像分割][开放词汇语义分割] 发现 CLIP 中间层的交叉相关自注意力具有定位属性，提出残差交叉相关自注意力（RCS）和语义反馈精炼（SFR）两个即插即用模块，显著提升 CLIP 在开放词汇语义分割中的密集推理能力。
tags:
  - CVPR 2025
  - 图像分割
  - 开放词汇语义分割
  - CLIP
  - 免训练
  - 残差注意力
  - 密集推理
---

# ResCLIP: Residual Attention for Training-free Dense Vision-language Inference

**会议**: CVPR 2025  
**arXiv**: [2411.15851](https://arxiv.org/abs/2411.15851)  
**代码**: [GitHub](https://github.com/yvhangyang/ResCLIP)  
**领域**: 语义分割  
**关键词**: 开放词汇语义分割, CLIP, 免训练, 残差注意力, 密集推理

## 一句话总结

发现 CLIP 中间层的交叉相关自注意力具有定位属性，提出残差交叉相关自注意力（RCS）和语义反馈精炼（SFR）两个即插即用模块，显著提升 CLIP 在开放词汇语义分割中的密集推理能力。

## 研究背景与动机

- CLIP 等视觉-语言模型在图像级开放词汇任务中表现出色，但在像素级密集预测任务（如语义分割）中表现不佳
- 已有研究发现 CLIP 最后一层的自注意力呈现空间不变性（spatial-invariant），即所有 patch 的注意力模式相似，失去了位置区分能力
- 现有免训练方法（如 SCLIP、ClearCLIP、NACLIP）通过将最后一层的 query-key 注意力替换为自相关注意力（query-query 或 key-key）来获得空间协变特征
- 然而这些方法忽略了交叉相关自注意力（query-key）所捕获的丰富空间对应关系
- 作者发现一个关键现象：CLIP **非最后层**的交叉相关注意力（C2SA）同样展现出定位属性和类别特异性
- 需要一种方法将中间层的定位信息引入最后一层，同时增强同类区域的语义一致性

## 方法详解

### 整体框架

ResCLIP 包含两个模块：残差交叉相关自注意力（RCS）和语义反馈精炼（SFR）。RCS 从 CLIP 中间层提取并聚合交叉相关注意力 $\mathcal{A}_c$，与最后一层的自相关注意力 $\mathcal{A}_s$ 做残差融合。SFR 利用初次分割结果作为语义反馈，进一步调整注意力得分以增强同类区域的聚焦和局部一致性。两个模块可以作为即插即用方案无缝集成到 SCLIP、ClearCLIP、NACLIP 等现有方法中。

### 关键设计

**1. 残差交叉相关自注意力（RCS）**

- **功能**：利用中间层的交叉相关注意力修复最后一层注意力的空间不变性问题
- **核心思路**：从第 $s$ 层到第 $e$ 层提取标准 query-key 注意力 $\mathcal{A}_{qk}^i$，取平均得到聚合注意力 $\mathcal{A}_c = \frac{1}{N}\sum_{i=s}^{e}\mathcal{A}_{qk}^i$。然后与最后一层的自相关注意力做加权融合 $\mathcal{A}_{rcs} = (1-\lambda_{rcs})\cdot\mathcal{A}_s + \lambda_{rcs}\cdot\mathcal{A}_c$
- **设计动机**：自相关注意力（SCSA）虽然解决了空间不变性但缺乏跨特征动态性，而中间层的交叉相关注意力保留了丰富的类别特异定位信息（可视化清楚显示中间层注意力能关注同类物体区域）

**2. 语义反馈精炼（SFR）**

- **功能**：显式增强同类区域的注意力聚焦，保持局部空间一致性
- **核心思路**：先利用 RCS 获得初步分割图 $\mathcal{M}$，构建语义掩码：若 patch $(m,n)$ 与目标 patch 属同一类则注意力保持，否则衰减。使用连通域分析区分连通与不相邻的同类区域，对不相邻区域施加基于切比雪夫距离的衰减函数 $D(p,q) = \exp(-\frac{d(p,q)}{\max(d(\cdot,\cdot))})$，最后用高斯核平滑。融合公式为 $S_r = (1-\lambda_{sfr})\cdot S_s + \lambda_{sfr}\cdot\hat{S}$
- **设计动机**：之前的邻域先验（如 NACLIP 的高斯核）是各向同性的，无法适应不同形状的物体。SFR 通过语义分割图提供自适应的类别感知先验，更精确地引导注意力

**3. 层融合策略**

- **功能**：确定从哪些中间层聚合交叉相关注意力
- **核心思路**：探索了累积聚合（从第1层到第n层）和滑窗聚合（窗口大小为4）两种策略，最优配置为滑窗聚合的第6→9层
- **设计动机**：不同层捕获不同粒度的空间对应关系，适当的层选择可以平衡局部细节和全局语义

### 损失函数 / 训练策略

- 完全免训练（training-free），不涉及任何额外训练或微调
- 仅修改 CLIP 最后一层的注意力计算方式
- 超参数 $\lambda_{rcs}=0.5$，$\lambda_{sfr}=0.7$（在 0.6-0.8 范围内稳定）
- 使用标准 ImageNet 提示词模板，不使用额外文本增强策略
- 滑窗推理：$224 \times 224$ 窗口，步幅112

## 实验关键数据

### 主实验

无背景类数据集上的 mIoU（ViT-B/16）：

| 方法 | VOC20 | Context59 | Stuff | Cityscape | ADE20k | 平均 |
|------|-------|-----------|-------|-----------|--------|------|
| SCLIP | 80.4 | 34.2 | 22.4 | 32.2 | 16.1 | 37.1 |
| +ResCLIP | **84.6** | 35.8 | 23.9 | 34.4 | 17.6 | 39.3(+2.2) |
| ClearCLIP | 80.9 | 35.9 | 23.9 | 30.0 | 16.7 | 37.5 |
| +ResCLIP | **87.1** | 36.4 | 24.3 | 34.5 | 17.8 | 40.0(+2.5) |
| NACLIP | 79.7 | 35.2 | 23.3 | 35.5 | 17.4 | 38.2 |
| +ResCLIP | **86.0** | **36.8** | **24.7** | **35.9** | **18.0** | **40.3(+2.1)** |

### 消融实验

基于 NACLIP（ViT-B/16，VOC20）：

| 配置 | RCS | SFR | mIoU | Δ |
|------|-----|-----|------|---|
| Baseline (NACLIP) | - | - | 79.7 | - |
| +RCS | ✓ | - | 85.5 | +5.8 |
| +SFR | - | ✓ | 81.5 | +1.8 |
| +RCS+SFR | ✓ | ✓ | **86.0** | **+6.3** |

### 关键发现

1. ResCLIP 在所有基线方法上均带来一致提升，在 ViT-L/14 上对 SCLIP 的提升高达 +13.1% mIoU
2. RCS 模块贡献最大（+5.8%），验证了中间层注意力的定位价值
3. 两个模块互补：单独使用 RCS +5.8%、SFR +1.8%，组合 +6.3%
4. 滑窗聚合策略（6→9层）优于累积聚合，说明靠近最后层的中间层信息更有价值
5. 超参数不敏感，$\lambda_{rcs}$ 和 $\lambda_{sfr}$ 在宽范围内性能稳定

## 亮点与洞察

- 首次发现 CLIP 中间层交叉相关注意力具有定位属性，这是一个被忽视但极有价值的信号源
- 方法设计为即插即用，与现有 SCLIP/ClearCLIP/NACLIP 等方法完全兼容
- SFR 利用分割结果反馈调整注意力是一种巧妙的自引导策略
- 在 ViT-L/14 上现有方法性能严重下降（SCLIP 下降 13.5%），ResCLIP 有效缓解了此问题

## 局限与展望

- SFR 模块依赖初始分割质量，若初始分割误差大则可能引入噪声
- 方法增加了推理时的计算开销（需遍历中间层注意力）
- 仅在2D语义分割任务上验证，可探索扩展到3D理解、目标检测等密集任务
- 未来可结合可训练的注意力调整策略进一步提升性能
- 层选择策略可探索更自适应的方案替代手动滑窗

## 相关工作与启发

- **SCLIP / ClearCLIP / NACLIP**: 替换最后层注意力为自相关版本的免训练方法，ResCLIP 与它们正交互补
- **MaskCLIP**: 最早将 CLIP 用于密集推理的工作，直接使用 value 特征做分割
- **ProxyCLIP**: 结合 SAM 等视觉基础模型的注意力，思路与 ResCLIP 的外部信息融合类似
- 启发：在基础模型中，不同层的特征/注意力可能携带不同类型的有价值信息，充分利用层间信息是提升密集预测的重要方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 中间层注意力的发现有洞察力，RCS 设计优雅
- **实验充分度**: ⭐⭐⭐⭐ — 8个数据集、3个基线方法、2种骨干网络，消融全面
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，图示直观，公式表述规范
- **价值**: ⭐⭐⭐⭐ — 即插即用的实用方案，对 CLIP 密集推理有重要推动

<!-- RELATED:START -->

## 相关论文

- [SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](../../ECCV2024/segmentation/sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [COSMOS: Cross-Modality Self-Distillation for Vision Language Pre-training](cosmos_cross-modality_self-distillation_for_vision_language_pre-training.md)
- [DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)
- [Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)

<!-- RELATED:END -->
