---
title: >-
  [论文解读] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models
description: >-
  [NEURIPS2025][多模态VLM][视觉语言] 提出 BridgeVLA，通过将 3D 点云投影为多视角 2D 图像并以 2D 热力图作为中间表示来对齐输入输出空间，实现了高效且有效的 3D 机器人操作学习。
tags:
  - "NEURIPS2025"
  - "多模态VLM"
  - "视觉语言"
  - "3D Manipulation"
  - "Heatmap Prediction"
  - "Robot Learning"
  - "sample efficiency"
---

# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models

**会议**: NEURIPS2025  
**arXiv**: [2506.07961](https://arxiv.org/abs/2506.07961)  
**代码**: [项目主页](https://bridgevla.github.io/)  
**领域**: 多模态VLM  
**关键词**: Vision-Language-Action, 3D Manipulation, Heatmap Prediction, Robot Learning, sample efficiency

## 一句话总结
提出 BridgeVLA，通过将 3D 点云投影为多视角 2D 图像并以 2D 热力图作为中间表示来对齐输入输出空间，实现了高效且有效的 3D 机器人操作学习。

## 背景与动机
- 利用预训练视觉语言模型（VLM）构建视觉语言动作（VLA）模型已成为学习通用机器人操作策略的主流方向
- 然而大多数 VLA 模型仅使用 2D 图像输入，需要大量数据收集；而 3D 策略虽然样本效率高，但缺少 VLM 的广泛语义知识
- 已有的 3D VLA 方法（如 3D-VLA、SpatialVLA）存在两个关键问题：
    1. 将动作转为无空间结构的 token 序列，未能利用 3D 数据的空间先验，导致样本效率低
    2. 微调时 3D 输入与预训练时 2D 图像输入之间存在分布不对齐，影响知识迁移

## 核心问题
如何设计一个 3D VLA 模型，同时继承 VLM 的语义泛化能力和 3D 策略的空间效率优势，实现输入-输出在统一 2D 空间的对齐？

## 方法详解

### 整体框架
BridgeVLA 采用两阶段训练：2D 热力图预训练 + 3D 动作微调。以 PaliGemma（SigLIP 视觉编码器 + Gemma Transformer）为 VLM 骨干。

### 阶段一：2D 热力图预训练
- **目标**：让 VLM 学会根据文本描述预测空间热力图，弥补原始 VLM 只能预测 token 序列的局限
- **数据**：使用 RoboPoint 的 120K 目标检测数据
- **热力图构建**：对每个目标物体，以其边界框中心为 $\hat{\mathbf{x}_i}$，构建截断高斯概率图：
$$H_i^{gt}(\mathbf{x}) = \begin{cases} \exp(-\|\mathbf{x}-\hat{\mathbf{x}_i}\|^2 / 2\sigma^2) & \text{if } p_i(\mathbf{x}) \geq p_{\min} \\ 0 & \text{otherwise} \end{cases}$$
- 多目标时取平均并归一化得到最终热力图 $H^{gt}$
- **解码方式**：将 VLM 输出的图像 token 按 patch 位置重排为空间特征网格，通过可学习的凸上采样模块（convex upsampling）恢复至原图分辨率
- **损失函数**：交叉熵损失监督热力图预测
- **可扩展性**：该预训练策略理论上可利用任何能转化为热力图预测的视觉语言数据集（如关键点检测、语义分割）

### 阶段二：3D 动作微调
- **输入对齐**：将 RGB-D 图像重建点云后，渲染三个正交投影视图（俯视、前视、右视）作为 VLM 输入，与预训练时的 2D 图像对齐
- **平移动作预测**：三个视图各生成一张热力图，反投影后对 3D 网格点评分，得分最高的点即为末端执行器的平移位置
- **旋转/夹持/碰撞预测**：对每个视图的输出 token 做 max-pooling 得全局特征、从热力图峰值提取局部特征，拼接后通过 MLP 预测旋转（欧拉角，每轴 72 bins）、夹持状态和碰撞标志
- **粗到精策略**：初次预测后，以预测平移位置为中心裁剪放大点云，再做第二次前向传播，用第二次结果执行
- **训练损失**：$L = L_{trans} + L_{rot} + L_{gripper} + L_{collision}$
- **关键设计**：不向 VLM 注入额外 3D 位置信息或机器人状态，最大限度保持预训练-微调的分布一致

### 核心设计理念
1. **输入对齐**：3D → 多视角 2D 投影，与 VLM 预训练输入一致
2. **输出对齐**：动作 → 2D 热力图，与输入图像共享空间结构
3. **阶段对齐**：预训练和微调都在统一 2D 空间完成

## 实验关键数据

### RLBench（18 任务，每任务 100 条专家演示）

| 方法 | 平均成功率 | 平均排名 |
|------|-----------|---------|
| RVT-2（SOTA）| 81.4% | 2.75 |
| 3D Diffuser Actor | 81.3% | 2.67 |
| **BridgeVLA** | **88.2%** | **2.03** |

- 在 Insert Peg 上大幅领先（88.0% vs 40.0%），展示高精度操作能力
- 在 Sort Shape 上优势明显（60.8% vs 35.0%）

### COLOSSEUM（12 类扰动泛化）

| 方法 | 平均成功率 | 平均排名 |
|------|-----------|---------|
| RVT-2 | 56.7% | 1.92 |
| **BridgeVLA** | **64.0%** | **1.07** |

- 14 种扰动设定中 13 项排名第一

### GemBench（4 级泛化难度）

| 方法 | 平均 | L1 | L2 | L3 | L4 |
|------|------|-----|-----|-----|-----|
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | 17.4 |
| **BridgeVLA** | **50.0** | 91.1 | **65.0** | **43.8** | 0.0 |

- L2（新物体）和 L3（铰接物体）均达 SOTA

### 真实机器人实验
- **数据效率**：仅用每任务 3 条轨迹即达 95.4% 成功率，而 π₀ 和 SpatialVLA 在 10 条数据时几乎完全失败
- 每任务 10 条轨迹时达 96.9%，超 RVT-2（90%）平均 32%
- 在 6 种泛化设定中全面超越 RVT-2，尤其在光照变化和组合泛化上优势显著

### 消融实验

| 变体 | RLBench 平均成功率 |
|------|-------------------|
| 去掉热力图（直接回归位置） | 31.4%（↓56.8%） |
| 加入 3D 位置特征 | 56.2%（↓32.0%） |
| **完整 BridgeVLA** | **88.2%** |

## 亮点
1. **输入-输出对齐范式**：首次在 3D VLA 中实现预训练与微调的输入输出统一到 2D 空间，设计优雅
2. **极强的样本效率**：每任务仅 3 条轨迹即达 95.4% 成功率，远超 π₀、SpatialVLA 等方法
3. **热力图中间表示**的优势通过消融充分验证——去掉后成功率暴跌 57 个百分点
4. **不注入额外 3D 信息**反而更好——保持与预训练分布一致比加入更多信息更重要
5. 实验非常全面：3 个仿真基准 + 真机 7 种设定 + 3 个消融实验

## 局限与展望
1. **长时序任务表现差**：GemBench L4 成功率为 0%，缺乏子任务分解能力；可引入 LLM 做任务规划
2. **遮挡问题**：Place Cups 等任务中目标关键点在所有正交视图中被遮挡，可探索动态选择投影视角
3. **类别泛化有限**：真机 Category 设定绝对成功率不高，预训练数据与机器人数据视角差异大
4. **预训练数据规模**：当前仅用 120K 检测数据，扩充至语义分割、关键点检测等任务数据有望提升泛化
5. **正交投影信息损失**：三个固定视角的正交投影可能丢失某些视角的关键信息

## 与相关工作的对比

| 维度 | BridgeVLA | SpatialVLA | π₀ | RVT-2 | 3D Diffuser Actor |
|------|-----------|------------|-----|-------|-------------------|
| 输入 | 多视角正交投影 | 2D 图像 + Ego3D 编码 | 2D 图像 | 多视角正交投影 | 3D 点云 |
| 输出 | 2D 热力图 → 3D 动作 | Token 序列 | Flow Matching | 2D 热力图 | 3D 扩散轨迹 |
| VLM 骨干 | PaliGemma | Qwen2-VL | PaliGemma | 无 | 无 |
| 数据效率 | 极高（3 条即可） | 低（50 条仍差） | 低 | 高 | 中 |
| 输入-输出对齐 | ✓ | ✗ | ✗ | ✓（部分） | ✗ |

- 相比 SpatialVLA：BridgeVLA 不向 VLM 注入 3D 位置信息，而是通过正交投影保持分布一致性，效果更好
- 相比 RVT-2：继承了其正交投影 + 热力图设计，但加入了 VLM 骨干和预训练，获得了语义泛化能力
- 相比 π₀：3D 感知 + 热力图输出大幅提升了样本效率

## 启发与关联
1. **保持预训练分布一致性比注入更多信息更重要**——这一洞察对 VLM-based 机器人策略设计有普遍意义
2. **热力图作为中间表示**的思路可推广到其他空间预测任务（抓取点预测、导航等）
3. 预训练阶段的目标检测 → 热力图任务设计巧妙，为 VLM 适配下游空间任务提供了通用范式
4. 粗到精的两次前向传播策略有效提升精度，但也带来 2× 推理开销，值得探索更高效的方案

## 评分
- 新颖性: 8/10 — 输入-输出对齐到统一 2D 空间的范式设计新颖，热力图预训练策略巧妙
- 实验充分度: 9/10 — 3 个仿真基准 + 真机 7 种设定 + 充分消融，非常全面
- 写作质量: 8/10 — 结构清晰，5 个研究问题逐一回答，逻辑性强
- 价值: 8/10 — 3 条轨迹达 95% 成功率的样本效率突破具有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](../../CVPR2025/multimodal_vlm/mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](../../CVPR2026/multimodal_vlm/pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

</div>

<!-- RELATED:END -->
