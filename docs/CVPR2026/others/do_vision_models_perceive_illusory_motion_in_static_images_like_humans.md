---
title: >-
  [论文解读] Do Vision Models Perceive Illusory Motion in Static Images Like Humans?
description: >-
  [CVPR 2026][运动错觉] 本文系统评估了多种光流模型在旋转蛇等静态图像运动错觉上的表现，发现仅人类启发的Dual-Channel模型在模拟眼跳条件下能再现人类感知的旋转运动。
tags:
  - CVPR 2026
  - 运动错觉
  - 光流模型
  - 人类视觉
  - 旋转蛇错觉
  - 生物启发模型
---

# Do Vision Models Perceive Illusory Motion in Static Images Like Humans?

**会议**: CVPR 2026  
**arXiv**: [2604.09853](https://arxiv.org/abs/2604.09853)  
**代码**: 有  
**领域**: 视觉感知/计算神经科学  
**关键词**: 运动错觉, 光流模型, 人类视觉, 旋转蛇错觉, 生物启发模型

## 一句话总结

本文系统评估了多种光流模型在旋转蛇等静态图像运动错觉上的表现，发现仅人类启发的Dual-Channel模型在模拟眼跳条件下能再现人类感知的旋转运动。

## 研究背景与动机

**领域现状**：DNN在光流估计基准上已超越人类，但在鲁棒性上仍有差距。视觉运动错觉为探测人机差异提供了有力工具，但现有研究集中于动态错觉（如reverse-phi），对静态图像错觉的研究不足。

**现有痛点**：旋转蛇错觉——一种在完全静态图像中人类强烈感知到旋转运动的现象——现有光流模型能否再现尚不清楚。该错觉依赖于微妙的亮度不对称和注视性眼动。

**核心矛盾**：标准DNN光流模型在基准测试上表现优异，但其计算策略是否与人类视觉系统共享关键原理仍不明确。

**本文目标**：评估代表性DNN和人类启发运动模型再现静态图像运动错觉的能力，识别关键计算组件。

**切入角度**：使用in silico心理物理学方法，在统一实验流水线下系统性比较10种运动估计模型。

**核心idea**：双通道运动处理、眼动瞬态信号和循环整合是再现人类运动感知的关键机制。

## 方法详解

### 整体框架

(1) 生成旋转蛇错觉图像和对照图像（灰度/蓝黄/红绿三种配色）；(2) 在静态和模拟眼跳两种条件下评估10种模型；(3) 消融分析识别关键组件。

### 关键设计

1. **统一实验流水线**:

    - 功能：在控制条件下公平比较不同架构的模型
    - 核心思路：所有模型使用官方预训练权重，在相同的错觉/对照图像上评估。模拟眼跳通过平移图像产生瞬态视网膜滑移
    - 设计动机：确保差异可归因于模型架构而非训练/评估差异

2. **模拟眼跳条件**:

    - 功能：模拟人类观看旋转蛇时的生理条件
    - 核心思路：人类感知旋转蛇错觉时需要眼跳等注视性眼动提供瞬态信号。通过移动图像模拟这种视网膜滑移
    - 设计动机：心理物理学研究表明该错觉在固定凝视下显著减弱，眼动是触发错觉的关键

3. **消融分析**:

    - 功能：识别再现错觉的关键计算组件
    - 核心思路：对Dual-Channel模型进行系统消融：(1) 基于亮度的运动信号贡献；(2) 高阶颜色-特征运动信号贡献；(3) 循环注意力机制的角色
    - 设计动机：确定哪些计算原理是人类样运动感知的必要条件

### 损失函数 / 训练策略

纯推理评估，不涉及训练。所有模型使用原始预训练权重。

## 实验关键数据

### 主实验

| 模型类型 | 静态条件 | 眼跳条件 | 再现错觉 |
|---------|---------|---------|---------|
| 多尺度DNN (FlowNet等) | 无旋转流 | 无旋转流 | ✗ |
| 循环解码DNN (RAFT等) | 无旋转流 | 无旋转流 | ✗ |
| Dual-Channel (生物启发) | 弱信号 | 预期旋转运动 | ✓ |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无亮度通道 | 错觉减弱 | 亮度信号贡献显著 |
| 无颜色-特征通道 | 错觉减弱 | 高阶信号也有贡献 |
| 无循环注意力 | 错觉消失 | 整合局部线索的关键 |
| 完整Dual-Channel | 最强一致 | 所有组件协同 |

### 关键发现

- 大多数DNN光流模型完全无法在静态图像上产生与人类一致的运动流场
- Dual-Channel模型仅在模拟眼跳条件下展现预期的旋转运动，静态条件下也较弱
- 循环注意力机制是整合局部线索为全局旋转感知的关键组件

## 亮点与洞察

- **运动错觉作为模型诊断工具**：通过人类感知偏差来区分"能工作的"和"像人类一样工作的"模型
- **生物启发计算原理的验证**：双通道运动处理、眼动瞬态和循环整合是三个可迁移的设计原则
- **对鲁棒视觉系统设计的启示**：能再现人类感知偏差的模型可能在真实世界中也更鲁棒

## 局限与展望

- 仅测试了有限的运动错觉类型
- Dual-Channel模型的实际光流估计性能未与主流DNN对比
- 仅分析了零样本推理，未探索微调能否使DNN学会再现错觉

## 相关工作与启发

- **vs 标准光流基准**: 基准测试上表现好不代表与人类视觉对齐，运动错觉提供了互补的评估维度
- **vs reverse-phi研究**: reverse-phi是动态错觉，旋转蛇是静态错觉，后者对模型的要求更高

## 评分

- 新颖性: ⭐⭐⭐⭐ 静态运动错觉在计算视觉中的首次系统评估
- 实验充分度: ⭐⭐⭐⭐ 10种模型×多种条件×消融分析
- 写作质量: ⭐⭐⭐⭐ 跨学科研究组织得当
- 价值: ⭐⭐⭐ 对光流模型设计有启发但实际应用有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Next-Scale Autoregressive Models for Text-to-Motion Generation](next-scale_autoregressive_models_for_text-to-motion_generation.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](vit3_unlocking_test_time_training_in_vision.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)
- [\[NeurIPS 2025\] Brain-Like Processing Pathways Form in Models With Heterogeneous Experts](../../NeurIPS2025/others/brain-like_processing_pathways_form_in_models_with_heterogeneous_experts.md)
- [\[CVPR 2025\] CADCrafter: Generating Computer-Aided Design Models from Unconstrained Images](../../CVPR2025/others/cadcrafter_generating_computer-aided_design_models_from_unconstrained_images.md)

</div>

<!-- RELATED:END -->
