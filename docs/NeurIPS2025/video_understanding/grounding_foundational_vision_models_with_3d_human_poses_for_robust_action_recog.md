---
title: >-
  [论文解读] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition
description: >-
  [NEURIPS2025][视频理解][动作识别] 提出一种融合 V-JEPA 2 视觉上下文特征与 CoMotion 3D 骨骼姿态数据的 cross-attention 多模态架构，在标准及高遮挡动作识别基准上优于单模态基线。
tags:
  - NEURIPS2025
  - 视频理解
  - 动作识别
  - 多模态
  - 3D human pose
  - 注意力机制
  - V-JEPA 2
  - CoMotion
---

# Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition

**会议**: NEURIPS2025  
**arXiv**: [2511.05622](https://arxiv.org/abs/2511.05622)  
**代码**: [nbabey20/groundactrec](https://github.com/nbabey20/groundactrec)  
**领域**: video_understanding  
**关键词**: action recognition, multimodal fusion, 3D human pose, cross-attention, V-JEPA 2, CoMotion

## 一句话总结

提出一种融合 V-JEPA 2 视觉上下文特征与 CoMotion 3D 骨骼姿态数据的 cross-attention 多模态架构，在标准及高遮挡动作识别基准上优于单模态基线。

## 背景与动机

- 动作识别是具身 AI 的核心能力，但现有方法存在明显的模态局限性
- **RGB 视频方法**（如 V-JEPA 2）能捕获丰富的场景上下文和物体交互信息，但在遮挡严重的场景中，关键肢体被遮挡时空间推理能力下降
- **骨骼方法**（如 CoMotion）提供显式的 3D 关节坐标、对视觉噪声和遮挡具有一定鲁棒性，但缺乏环境上下文和人物-物体交互信息
- 两种范式各有盲区且恰好互补：视觉流提供"在什么环境中做什么"，骨骼流提供"身体如何配置和运动"
- 本文的核心假设：将上下文视觉理解与精确几何骨骼融合，可实现更鲁棒、空间感知更强的动作识别

## 核心问题

1. 如何将预训练视觉基础模型的高层语义特征与 3D 人体姿态的低层几何特征进行有效融合？
2. 在高遮挡场景下，融合架构能否显著超越单模态方案？
3. cross-attention 融合机制相对于早期融合（concatenation）和晚期融合（score averaging）的优势有多大？

## 方法详解

### 整体架构

模型由两个特征提取分支 + 一个 cross-attention 融合 Transformer + 分类头组成。

### 视觉特征提取

- 对每个视频均匀采样 $T=64$ 帧（TSN 采样策略：8 段 × 8 帧/段）
- 每帧经 V-JEPA 2 的 ViT-g/384 编码器处理，提取 [CLS] token 作为帧级特征
- 得到视觉特征序列 $F_V \in \mathbb{R}^{T \times 1408}$

### 骨骼特征提取

- CoMotion 对每帧预测 SMPL 参数（pose $\theta$、translation $t$、shape $\beta$）
- SMPL 层解码得到 $J=24$ 个 3D 关节坐标
- 进行 root-relative 归一化（减去骨盆坐标），消除全局位置影响
- 展平为 $D_S = 3 \times 24 = 72$ 维向量
- 同样应用 TSN 采样与视觉流对齐，得到 $F_S \in \mathbb{R}^{T \times 72}$

### 模态嵌入与位置编码

- 两个独立的线性投影层将 $F_V$ 和 $F_S$ 映射到共同维度 $D_{model}=512$
- 各自添加可学习的 [CLS] token
- 加入正弦位置编码保持时序信息

### Cross-Attention 融合 Transformer

- 共 $L=4$ 层融合层，每层包含：
    - **双向 cross-attention**：视觉流以骨骼流为 K/V 进行注意力更新，骨骼流以视觉流为 K/V 进行对称更新
    - **self-attention 精炼**：各流分别进行 self-attention 以整合融合后的特征
    - 每个子层均有残差连接 + LayerNorm
- 8 个注意力头

### 分类头

- 提取最终层两个流的 [CLS] token，拼接后通过 MLP + softmax 输出动作类别概率

## 实验关键数据

### 数据集

| 数据集 | 描述 | 规模 |
|--------|------|------|
| InHARD | 工业动作识别，16 名受试者，14 类动作，200万+帧 | 标准 train/val 划分 |
| UCF-19-Y-OCC | UCF-101 的高遮挡子集，19 类，1732 个视频 | 真实自然遮挡场景 |

### 主实验结果（InHARD）

| 模型 | Top-1 Acc (%) | Macro mAP (%) | Macro F1 (%) |
|------|-------------|--------------|-------------|
| V-JEPA 2 baseline | 80.76 | 80.93 | 76.24 |
| CoMotion baseline | 75.92 | 74.60 | 69.52 |
| Gated recursive fusion | 79.25 | 76.90 | 73.69 |
| **Fusion (cross-attn)** | **83.47** | **84.96** | **80.21** |

### 主实验结果（UCF-19-Y-OCC 高遮挡）

| 模型 | Top-1 Acc (%) | Macro mAP (%) | Macro F1 (%) |
|------|-------------|--------------|-------------|
| V-JEPA 2 baseline | 31.83 | **58.48** | 14.23 |
| CoMotion baseline | 6.20 | 8.84 | 1.72 |
| Gated recursive fusion | 29.54 | 50.07 | 11.44 |
| **Fusion (cross-attn)** | **38.62** | 54.10 | **16.30** |

### 消融实验（融合策略对比）

| 融合方法 | InHARD Acc | UCF-19-Y-OCC Acc |
|---------|-----------|-----------------|
| Early fusion (concat) | 79.52 | 33.34 |
| Late fusion (score avg) | 80.24 | 34.42 |
| **Cross-attention** | **83.47** | **38.62** |

### 训练配置

- GPU: NVIDIA A100 SXM，30 epochs
- AdamW 优化器，lr = $3 \times 10^{-4}$，weight decay = 0.05
- cosine 学习率衰减 + 5% warmup
- batch size = 128，dropout = 0.1，gradient clipping norm = 1.0
- 3 个随机种子取均值和标准差

## 亮点

1. **互补性设计思路清晰**：将 V-JEPA 2 的隐式世界模型理解与 CoMotion 的显式几何骨骼表示结合，动机自然且合理
2. **高遮挡场景显著提升**：在 UCF-19-Y-OCC 上比 V-JEPA 2 基线高 6.79% 准确率，而 CoMotion 单独使用在遮挡下几乎崩溃（6.20%），说明融合确实互补
3. **cross-attention 优于简单融合**：消融实验清楚表明 cross-attention 比 early/late fusion 更能捕获跨模态的复杂关系
4. **架构简洁可复现**：4 层融合 Transformer、共同维度 512、标准组件堆叠，实现难度适中

## 局限与展望

1. **依赖冻结的特征提取器**：V-JEPA 2 和 CoMotion 均作为固定特征提取器使用，未进行端到端微调，特征可能不是最优
2. **数据集规模偏小**：InHARD 仅 14 类工业动作，UCF-19-Y-OCC 仅 19 类 1732 视频，泛化性有待在更大规模数据集（如 Kinetics-400、AVA）上验证
3. **高遮挡场景绝对性能仍低**：UCF-19-Y-OCC 上最佳准确率仅 38.62%，说明遮挡动作识别仍是开放问题
4. **mAP 在遮挡集上未超 V-JEPA 2**：V-JEPA 2 在 UCF-19-Y-OCC 的 mAP（58.48%）仍高于融合模型（54.10%），融合并非全面优势
5. **缺少更多 SOTA 融合基线**：仅对比了 gated recursive fusion 一种融合基线，缺少 ATFusion、MMAct 等近期方法
6. **计算开销未讨论**：同时运行 ViT-g 和 CoMotion 的推理成本很高，但论文未分析延迟和计算量

## 与相关工作的对比

- **V-JEPA 2**：自监督视频预训练模型，提供强大的上下文理解但不显式建模人体姿态；本文将其作为视觉特征的骨干
- **CoMotion**：多人 3D 姿态追踪器，能通过遮挡恢复骨骼但缺乏场景上下文；本文将其作为骨骼特征的骨干
- **Gated Recursive Fusion**：门控递归融合架构，在本文实验中表现弱于 cross-attention 融合
- **ST-GCN / 骨骼 GCN 方法**：基于图卷积的骨骼动作识别，不涉及多模态融合
- **ATFusion / 其他多模态方法**：在相关工作中被提及但未作为实验基线

## 启发与关联

- 这篇工作的核心洞察——"将世界模型的隐式理解与显式几何数据结合"——可以推广到其他需要空间推理的任务（如手部操作识别、人机协作）
- cross-attention 融合是一种通用范式，可应用于视频+深度、视频+光流等其他多模态组合
- 高遮挡场景的挑战提示：仅靠 2D 视角信息不足，3D 几何先验在遮挡推理中有不可替代的价值
- 本文属 workshop paper，篇幅和实验有限，但思路有潜力发展为完整论文

## 评分

- 新颖性: 3/5 — 融合视觉基础模型与 3D 姿态的想法合理但不算全新，cross-attention 融合是标准技术
- 实验充分度: 2.5/5 — 数据集规模小，基线对比不够丰富，缺少计算开销分析
- 写作质量: 3.5/5 — 结构清晰，动机阐述充分，数学符号规范
- 价值: 3/5 — Workshop paper 级别，验证了融合互补性的假设，但需更多大规模实验支撑

<!-- RELATED:START -->

## 相关论文

- [\[ECCV 2024\] On the Utility of 3D Hand Poses for Action Recognition](../../ECCV2024/video_understanding/on_the_utility_of_3d_hand_poses_for_action_recognition.md)
- [\[ICCV 2025\] Beyond Label Semantics: Language-Guided Action Anatomy for Few-shot Action Recognition](../../ICCV2025/video_understanding/beyond_label_semantics_language-guided_action_anatomy_for_few-shot_action_recogn.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](../../ECCV2024/video_understanding/referring_atomic_video_action_recognition.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)

<!-- RELATED:END -->
