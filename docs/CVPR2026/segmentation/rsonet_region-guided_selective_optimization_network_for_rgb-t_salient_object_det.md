---
title: >-
  [论文解读] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
description: >-
  [CVPR2026][图像分割][RGB-T 显著性检测] 提出两阶段 RGB-T 显著性检测网络 RSONet：先通过区域引导阶段计算 RGB/热红外引导图与联合引导图的相似度，选出更可靠的模态；再在显著性生成阶段利用选择性优化融合双模态特征，配合密集细节增强和互信息语义模块生成高质量显著图，在三个 RGB-T 基准上取得 SOTA 性能。
tags:
  - CVPR2026
  - 图像分割
  - RGB-T 显著性检测
  - 区域引导
  - 选择性优化
  - 多模态融合
  - Transformer
  - 视觉状态空间模型
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**会议**: CVPR2026  
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)  
**代码**: 待确认  
**领域**: 语义分割 / 显著性目标检测  
**关键词**: RGB-T 显著性检测, 区域引导, 选择性优化, 多模态融合, SwinTransformer, 视觉状态空间模型

## 一句话总结

提出两阶段 RGB-T 显著性检测网络 RSONet：先通过区域引导阶段计算 RGB/热红外引导图与联合引导图的相似度，选出更可靠的模态；再在显著性生成阶段利用选择性优化融合双模态特征，配合密集细节增强和互信息语义模块生成高质量显著图，在三个 RGB-T 基准上取得 SOTA 性能。

## 研究背景与动机

**RGB 单模态局限**：复杂背景、低对比度、模糊边界等场景下，纯 RGB 方法性能下降严重，需引入辅助模态信息。

**深度信息不足**：RGB-D 中深度图在物体与背景空间相邻时难以区分，深度质量受采集设备和距离影响较大。

**热红外模态引入**：热图不受光照变化影响，可有效补充 RGB 在夜间/低光场景的缺陷，但自身也受环境温度与材料属性影响。

**双模态不一致问题**（核心动机）：RGB 与热图中显著区域分布经常不一致——有的样本 RGB 清晰但热图模糊，有的相反。直接拼接/相加/注意力融合会引入大量噪声。

**现有融合策略的局限**：加法、乘法、拼接或注意力机制隐含地假设双模态等重要，无法适应信息质量差异大的情况。

**缺乏模态选择机制**：大多数方法没有显式判断哪个模态更可靠，本文提出"区域引导 + 相似度计算"实现自适应模态主导选择。

## 方法详解

### 整体框架

RSONet 分为两个阶段：

- **区域引导阶段（Region Guidance Stage）**：三路并行编码-解码生成引导图 $\mathbf{G}^R$、$\mathbf{G}^T$、$\mathbf{G}^{RT}$，通过相似度计算选择主导模态。
- **显著性生成阶段（Saliency Generation Stage）**：基于选择结果，用 SO 模块融合双模态特征，再分别用 DDE（低层）和 MIS（高层）模块提取细节与位置信息，跨层连接输出最终显著图。

骨干网络均采用 SwinTransformer，提取 5 级多尺度特征。

### 关键模块设计

#### 1. 上下文交互模块（Context Interaction, CI）

针对不同层级特征分辨率的差异，设计三种变体：

| 变体 | 应用层级 | 卷积核大小 |
|------|---------|-----------|
| 变体一 | 低层 $\mathbf{F}_1$ | 1×1, 3×3, 5×5, 7×7 |
| 变体二 | 中层 $\mathbf{F}_{2/3}$ | 1×1, 3×3, 5×5 |
| 变体三 | 高层 $\mathbf{F}_{4/5}$ | 1×1, 3×3 |

各分支采用级联累加策略，将上一分支输出加到当前输入以打破尺度间壁垒，最终沿通道维度拼接。

#### 2. 空间感知融合模块（Spatial-aware Fusion, SF）

对 CI 输出特征做两层 3×3 卷积，然后全局最大池化 → 1×1 卷积 → Sigmoid 生成空间权重，乘加回原特征实现空间维度优化。逐层自顶向下融合，最终生成各分支引导图。

#### 3. 相似度计算与模态选择

对三个引导图分别求均值 $M^R$、$M^T$、$M^{RT}$，比较 $|M^R - M^{RT}|$ 和 $|M^T - M^{RT}|$，差值更小的对应模态信息更准确，主导后续融合。

#### 4. 选择性优化模块（Selective Optimization, SO）

- 用引导图 $\mathbf{G}^{RT}$ 对双模态特征做乘加增强
- 通道注意力抑制引导图引入的干扰
- 更可靠模态的空间注意力用于引导另一模态特征优化
- 两路优化特征相加输出

#### 5. 密集细节增强模块（Dense Detail Enhancement, DDE）

应用于低层特征（1-3 层），采用四路并行空洞卷积（膨胀率 1/3/5/7），密集连接策略使各分支共享多尺度感受野信息。每个分支后接 VSS（Visual State Space）块进一步建模空间关系，最终沿通道拼接。

#### 6. 互信息语义模块（Mutual Interaction Semantic, MIS）

应用于高层特征（4-5 层），使用膨胀率 1/2/3 的 3×3 卷积，设计三个主分支，每个主分支内部三个子分支互相交互输出，三个主分支拼接后加通道注意力抑制噪声。

### 损失函数

联合损失 = BCE + BIoU + F-measure，对 5 个尺度的显著图施加深度监督：

$$L_{total} = \frac{1}{N}\sum_{i=1}^{N}(L_{bce} + L_{iou} + L_{fm})$$

## 实验

### 数据集与设置

- **训练集**：VT5000 训练集（2500 张）
- **测试集**：VT5000 测试集（2500 张）、VT1000（1000 张）、VT821（821 张）
- 输入分辨率 384×384，RMSprop 优化器，学习率 1e-4，单张 RTX 4080 GPU

### 主要结果

| 数据集 | $\mathcal{M}$↓ | $F_\beta$↑ | $S_\alpha$↑ | $E_\xi$↑ |
|--------|------|------|------|------|
| VT5000 | **0.020** | **0.910** | **0.926** | **0.963** |
| VT1000 | **0.014** | 0.923 | 0.946 | 0.972 |
| VT821 | 0.021 | 0.883 | 0.921 | 0.946 |

对比 27 种 SOTA 方法，VT5000 上 $F_\beta$ 较 PATNet 提升 3.4%，$E_\xi$ 提升 1.2%，$S_\alpha$ 提升 1.1%。

### 模型效率

| 指标 | 值 |
|------|-----|
| 参数量 | 88M |
| FLOPs | 143.8G |
| 推理速度 | 9.4 FPS |

参数量处于中等水平（得益于三分支权重共享），但两阶段设计导致推理速度偏低。

### 消融实验

| 设置 | $\mathcal{M}$↓ | $F_\beta$↑ | $S_\alpha$↑ | $E_\xi$↑ |
|------|------|------|------|------|
| w/o SO（加法替代） | 0.0217 | 0.8883 | 0.9213 | 0.9523 |
| w/o SO（乘法替代） | 0.0208 | 0.8948 | 0.9231 | 0.9587 |
| w/o SO（拼接替代） | 0.0215 | 0.8896 | 0.9224 | 0.9558 |
| w/o SO（逐像素软门控） | 0.0203 | 0.8951 | 0.9239 | 0.9605 |
| R→T（无区域引导） | 0.0215 | 0.8898 | 0.9230 | 0.9561 |
| T→R（无区域引导） | 0.0216 | 0.8896 | 0.9233 | 0.9554 |
| w/o DDE | 0.0203 | 0.9082 | 0.9213 | 0.9631 |
| w/o MIS | 0.0203 | 0.8997 | 0.9241 | 0.9593 |
| w/o DDE & MIS | 0.0217 | 0.9053 | 0.8995 | 0.9556 |
| **完整 RSONet** | **0.0197** | **0.9071** | **0.9261** | **0.9632** |

### 骨干网络消融

SwinTransformer 远优于 ResNet-18/34/50，也优于冻结的 SAM/DINO（大模型未适配 RGB-T 域，缺少 adaptor 导致性能反降）。

### 关键发现

- 去掉 SO 模块（用简单融合替代）性能显著下降，说明区域引导+选择性融合是核心贡献。
- 固定 R→T 或 T→R 方向融合不如自适应选择，验证了模态选择的必要性。
- DDE 和 MIS 各自贡献了细节和定位信息，同时去掉两者 $S_\alpha$ 降至 0.8995。

## 亮点

- 明确建模"双模态显著区域不一致"问题，提出区域引导 + 相似度选择的模态主导策略，动机清晰。
- CI 模块根据不同层级特征分辨率使用不同大小卷积核，设计合理。
- DDE 中密集连接 + VSS 块的组合有效挖掘低层空间结构信息。
- 消融实验充分，对 SO/DDE/MIS 均有详尽的对比和可视化分析。

## 局限性

- 推理速度仅 9.4 FPS，两阶段三分支设计导致计算开销大，难以实时部署。
- 相似度计算仅基于引导图均值的全局比较，对局部区域差异不敏感。
- 实验仅在 VT5000/VT1000/VT821 三个 RGB-T 数据集上验证，缺少 RGB-D 或视频 SOD 的泛化实验。
- 当极小目标或双模态同时质量低时检测效果不佳（论文自身给出的 failure cases）。
- 冻结大模型（SAM/DINO）作为骨干效果反降，但未尝试微调或设计 adaptor。

## 相关工作

- **RGB-D SOD**：利用深度信息增强检测，代表方法 EMTrans、Fang et al. 的 Group Transformer。
- **RGB-T SOD**：MCFNet（模态互补融合）、HRTransNet（高分辨率Transformer）、WaveNet（频域视角）、Samba（纯 Mamba 框架）、SAMSOD（基于 SAM 的方法）。
- **单模态 SOD**：AttFeedback、DenseAttFluid、BilateralExtreme 等经典方法。
- 本文与 ContriNet（TPAMI25）、Samba（CVPR25）、SAMSOD（TMM26）为最新竞争方法。

## 评分

- 新颖性: ⭐⭐⭐ — 区域引导+相似度选择模态的想法有一定新意，但各子模块（CI/SF/DDE/MIS）设计较常规
- 实验充分度: ⭐⭐⭐⭐ — 27 种方法对比 + 详尽消融 + 骨干网络分析 + 可视化 + 失败案例
- 写作质量: ⭐⭐⭐ — 公式和模块描述详细，但行文较冗长，符号繁多
- 价值: ⭐⭐⭐ — 在 RGB-T SOD 子领域取得 SOTA，但实时性不足限制实际应用

<!-- RELATED:START -->

## 相关论文

- [RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [Efficient RGB-D Scene Understanding via Multi-task Adaptive Learning and Cross-dimensional Feature Guidance](efficient_rgb-d_scene_understanding_via_multi-task_adaptive_learning_and_cross-d.md)
- [ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)
- [SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](../../AAAI2026/segmentation/sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [Concept-Guided Fine-Tuning: Steering ViTs away from Spurious Correlations to Improve Robustness](concept-guided_fine-tuning_steering_vits_away_from_spurious_correlations_to_impr.md)

<!-- RELATED:END -->
