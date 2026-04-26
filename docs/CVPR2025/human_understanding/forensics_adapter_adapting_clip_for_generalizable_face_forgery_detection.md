---
title: >-
  [论文解读] Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection
description: >-
  [CVPR 2025][人体理解][人脸伪造检测] 提出 Forensics Adapter，一个仅 5.7M 参数的轻量适配器网络，与冻结 CLIP 并行学习人脸伪造的融合边界特征，通过掩码边界预测+逐块对比+样本级对比三重目标实现跨数据集的高泛化性人脸伪造检测，CDF-v1 上 AUC 达 0.914。
tags:
  - CVPR 2025
  - 人体理解
  - 人脸伪造检测
  - CLIP适配
  - 融合边界
  - 对比学习
  - 跨数据集泛化
---

# Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection

**会议**: CVPR 2025  
**arXiv**: [2411.19715](https://arxiv.org/abs/2411.19715)  
**代码**: https://github.com/OUC-VAS/ForensicsAdapter (有)  
**领域**: 人体理解 / 人脸伪造检测  
**关键词**: 人脸伪造检测, CLIP适配, 融合边界, 对比学习, 跨数据集泛化

## 一句话总结

提出 Forensics Adapter，一个仅 5.7M 参数的轻量适配器网络，与冻结 CLIP 并行学习人脸伪造的融合边界特征，通过掩码边界预测+逐块对比+样本级对比三重目标实现跨数据集的高泛化性人脸伪造检测，CDF-v1 上 AUC 达 0.914。

## 研究背景与动机

**领域现状**：深度伪造检测的核心挑战是跨域泛化——在一种伪造方法上训练的检测器面对新伪造方法时性能急剧下降。近年来研究者尝试利用 CLIP 等视觉基础模型的通用表示来提升泛化性。

**现有痛点**：直接微调 CLIP 进行伪造检测会破坏其通用表示，降低跨域泛化性。而完全冻结 CLIP 又无法学习伪造特有的细节特征（如融合边界——伪造区域与真实区域的拼接缝隙）。

**核心矛盾**：需要保留 CLIP 的通用表示（用于泛化）同时学习伪造特定的判别特征（用于检测精度），两者在参数共享时互相冲突。

**切入角度**：用独立的轻量适配器网络专门学习融合边界特征，CLIP 仅做最小修改（1×1 卷积+注意力偏置），两者互补工作。

**核心 idea**：独立适配器学融合边界 + 最小化CLIP修改 + 三重对比目标 = 高泛化跨域伪造检测。

## 方法详解

### 关键设计

1. **双流架构（适配器 + CLIP）**:

    - 功能：分离通用表示和任务特定表示
    - 核心思路：适配器流用可学习查询 token 与 CLIP 视觉 token 交互，通过交叉注意力学习融合边界图。CLIP 流在 token 上加可训练 1×1 卷积+注意力偏置做最小修改。两流特征融合后做分类
    - 设计动机：适配器参数独立（5.7M），不干扰 CLIP 的预训练表示

2. **融合边界预测 ($\mathcal{L}_1$)**:

    - 功能：让适配器显式学习伪造区域的边界特征
    - 核心思路：将伪造掩码高斯模糊得到融合边界图，用适配器的输出预测这个边界图（MSE 损失）。只在伪造样本上计算
    - 设计动机：融合边界是跨伪造方法的通用伪造痕迹——无论什么方法产生的伪造都有拼接边界

3. **逐块+样本级对比学习 ($\mathcal{L}_2 + \mathcal{L}_3$)**:

    - 功能：拉近同类 patch/样本特征，推远异类
    - 核心思路：$\mathcal{L}_2$ 在 patch 级别对比（14×14 特征图的每个 patch 与融合边界掩码确定正负对），$\mathcal{L}_3$ 在样本级别对比
    - 设计动机：patch 级对比学局部边界判别，样本级对比学全局真伪区分

### 损失函数 / 训练策略

$\mathcal{L} = 10\mathcal{L}_0 + 200\mathcal{L}_1 + 20\mathcal{L}_2 + 10\mathcal{L}_3 + 1.5\mathcal{L}_4$。$\mathcal{L}_0$ 是二分类 BCE，$\mathcal{L}_4$ 是 ForAda++ 的视觉-文本对齐损失。仅在 FaceForensics++ 上训练。

## 实验关键数据

### 主实验

跨数据集帧级 AUC：

| 数据集 | ForAda | 前SOTA | 提升 |
|--------|--------|--------|------|
| CDF-v1 | **0.914** | 0.867 | +4.7% |
| CDF-v2 | **0.900** | 0.869 | +3.1% |
| DFDC | **0.843** | 0.758 | +8.5% |
| DFD | **0.933** | 0.915 | +1.8% |

### 消融实验

| 去掉的损失 | CDF-v2 AUC |
|-----------|-----------|
| 完整 | **0.914** |
| -$\mathcal{L}_1$ (边界) | 0.904 (-1.0%) |
| -$\mathcal{L}_2$ (patch 对比) | 0.818 (-9.6%) |
| -$\mathcal{L}_3$ (样本对比) | 0.904 (-1.0%) |

### 关键发现
- **Patch 级对比至关重要**：移除后 AUC 暴跌 9.6%，局部边界判别是跨域泛化的关键
- **DFDC 提升最大**：+8.5%，说明融合边界特征在低质量数据上更具泛化性
- **ForAda++ 文本增强有限**：平均仅 +1.3%，视觉特征已捕捉了大部分信息

## 亮点与洞察
- **融合边界作为通用伪造线索**——不同方法的伪造都有拼接边界，比学习特定伪造模式（如频率伪影）更具泛化性
- **5.7M 参数的轻量适配器打败了全量微调**——说明保持 CLIP 冻结+外挂轻量模块是利用基础模型的更好方式

## 局限与展望
- 训练需要操纵掩码（真实深度伪造无标注掩码）
- 仅限5个面部区域的文本提示，可能遗漏其他伪造区域
- WildDeepfake 上 0.803 仍待提升
- 文本编码器冻结，未联合适配视觉-文本空间

## 评分
- 新颖性: ⭐⭐⭐⭐ 双流适配器+融合边界的组合简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 6个跨域数据集，帧级+视频级，详尽消融
- 写作质量: ⭐⭐⭐⭐ 实验设计清晰
- 价值: ⭐⭐⭐⭐ 为基于CLIP的伪造检测提供了高效实用方案

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Towards General Visual-Linguistic Face Forgery Detection](towards_general_visual-linguistic_face_forgery_detection.md)
- [\[NeurIPS 2025\] DevFD: Developmental Face Forgery Detection by Learning Shared and Orthogonal LoRA Subspaces](../../NeurIPS2025/human_understanding/devfd_developmental_face_forgery_detection_by_learning_shared_and_orthogonal_lor.md)
- [\[AAAI 2026\] CLIPPan: Adapting CLIP as A Supervisor for Unsupervised Pansharpening](../../AAAI2026/human_understanding/clippan_adapting_clip_as_a_supervisor_for_unsupervised_pansharpening.md)
- [\[AAAI 2026\] Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection](../../AAAI2026/human_understanding/fine-grained_dino_tuning_with_dual_supervision_for_face_forgery_detection.md)
- [\[CVPR 2025\] SGC-Net: Stratified Granular Comparison Network for Open-Vocabulary HOI Detection](sgc-net_stratified_granular_comparison_network_for_open-vocabulary_hoi_detection.md)

<!-- RELATED:END -->
