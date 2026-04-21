---
title: >-
  [论文解读] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models
description: >-
  [CVPR2026][多模态][双曲VLM] 提出UNCHA框架，在双曲VLM中用双曲不确定性建模部分图像对整体场景的语义代表性，通过不确定性引导的对比损失和蒸含损失增强组合性场景理解，在多个下游任务上超趇现有双曲VLM。
tags:
  - CVPR2026
  - 多模态
  - 双曲VLM
  - 不确定性建模
  - 部分-整体对齐
  - 组合性理解
  - 蒸含损失
---

# Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models

**会议**: CVPR2026  
**arXiv**: [2603.22042](https://arxiv.org/abs/2603.22042)  
**代码**: [github.com/jeeit17/UNCHA](https://github.com/jeeit17/UNCHA)  
**领域**: 多模态VLM  
**关键词**: 双曲VLM, 不确定性建模, 部分-整体对齐, 组合性理解, 蒸含损失

## 一句话总结
提出UNCHA框架，在双曲VLM中用双曲不确定性建模部分图像对整体场景的语义代表性，通过不确定性引导的对比损失和蒸含损失增强组合性场景理解，在多个下游任务上超趇现有双曲VLM。

## 研究背景与动机
VLM如CLIP在欧氏空间中难以捕捉层次关系（如部分-整体、父-子结构），且在多物体组合场景中存在偏差。

双曲VLM（如MERU、ATMG、HyCoCLIP）通过双曲空间的负曲率和指数体积增长更好地保留层次结构，但现有方法未建模不同部分对整体的不同语义代表性——包含该场景核心物体的裁剪比背景裁剪更能代表整体场景。

如果所有部分被平等对待，模型无法区分更具代表性的部分和较少代表性的部分。

## 方法详解

### 整体框架
在HyCoCLIP基础上增加不确定性建模：(1)定义双曲不确定性反映语义代表性 → (2)融入对比损失 → (3)通过蒸含损失校准不确定性。

### 关键设计

1. **双曲不确定性模型**：

    - $u(x) = \log(1 + \exp(-\|x\|_2))$
    - 利用双曲半径（距原点的测地距离）与不确定性的单调关系
    - 靠近原点=更抽象=更高不确定性；远离原点=更具体=更低不确定性
    - 更能代表整体的部分 → 更低不确定性

2. **不确定性引导的对比损失**：

    - 自适应温度: $\tau_{un,i}^I = \exp(u(i_i^{part})/2) \cdot \tau_{gl}$
    - 高不确定性的部分 → 更大温度 → 更小的对比损失贡献
    - 加入局部对比损失（部分图像与部分文本对齐）

3. **蒸含损失的不确定性校准**：

    - 分段连续蒸含损失: $L_{ent}^* = \max(0, \phi - \eta\omega) + \alpha\phi$（Leaky-ReLU式松弛）
    - 不确定性校准: $L_{ent}^{cal} = \lfloor L_{ent}^* \rfloor e^{-u(p)} + u(p) + \mathcal{H}(\tilde{u}(p))$
    - 弱蒸含关系时鼓励增大不确定性；$u(p)$防止过度不确定
    - 熵正则化$\mathcal{H}$防止不确定性均匀化坍塞

### 损失函数 / 训练策略
$L = \mathcal{L}_{con}^{un} + \lambda_{ent}\mathcal{L}_{ent}^{un}$

基于Lorentz模型的双曲空间，使用指数映射/对数映射在流形和切空间间转换。

## 实验关键数据

### 主实验

| 模型 | ImageNet | CIFAR-10 | CUB | Cars | Pets | 说明 |
|------|---------|----------|-----|------|------|------|
| CLIP (ViT-S/16) | 36.7 | 70.2 | 9.8 | 6.9 | 44.6 | 基线 |
| MERU | 35.4 | 71.2 | 11.3 | 5.2 | 42.7 | 双曲基线 |
| HyCoCLIP | 提升 | 提升 | 提升 | 提升 | 提升 | 加入Part对齐 |
| UNCHA | 最优 | 最优 | 最优 | 最优 | 最优 | 加入不确定性建模 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无不确定性引导 | 性能下降 | 平等对待所有部分不够 |
| 无熵正则化 | 嵌入空间坍塞 | 不确定性趋向均匀 |
| 不确定性 vs 相似度 | r=-0.739 | 强负相关证实建模有效 |

### 关键发现
- 不确定性与部分-整体相似度的强负相关(r=-0.739)验证了建模的有效性
- 语义上更具代表性的部分显示更低不确定性，模糊或不具代表性的裁剪显示更高不确定性
- 在零样本分类、检索、多标签分类等多个下游任务上均超趇现有双曲VLM

## 亮点与洞察
- 用双曲半径作为不确定性的代理是自然且优雅的设计
- 熵正则化防止不确定性坍塞的设计细节体现了对双曲空间特性的深入理解
- Leaky-ReLU式松弛的蒸含损失解决了压入锥体后梯度为零的问题
- 可视化分析直观展示了不确定性与语义代表性的对应关系

## 局限与展望
- 双曲空间的计算复杂度限制了向更大规模模型的扩展
- 部分图像通过随机裁剪生成，未q探索更智能的部分分割策略
- 仅在ViT-S/16和ViT-B/16上验证，更大视觉编码器待验证
- 不确定性阈值$\tau_A$的设置较为启发式

## 相关工作与启发
- MERU首先引入双曲VLM但仅建模跨模态蒸含
- HyCoCLIP扩展到模态内蒸含但未区分部分代表性
- 双曲半径作为不确定性代理的思路可推广到任何双曲表示学习场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 双曲不确定性+语义代表性建模新颖
- 实验充分度: ⭐⭐⭐⭐ 16个数据集的零样本分类+多维度评估
- 写作质量: ⭐⭐⭐⭐ 公式推导详尽，结构清晰
- 价值: ⭐⭐⭐⭐ 推进了双曲VLM的组合性理解能力

<!-- RELATED:START -->

## 相关论文

- [When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models](../../ICLR2026/multimodal_vlm/error_notebook-guided_training-free_part_retrieval_in_3d_cad_assemblies_via_visi.md)
- [AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)
- [Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)

<!-- RELATED:END -->
