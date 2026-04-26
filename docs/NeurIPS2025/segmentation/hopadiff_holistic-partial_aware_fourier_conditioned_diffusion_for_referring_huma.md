---
title: >-
  [论文解读] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios
description: >-
  [NeurIPS 2025][图像分割][指称人体动作分割] 首次提出指称人体动作分割(RHAS)任务——通过文本描述定位多人视频中特定个体并做帧级动作分割。构建了包含133部电影、137个动作类别、33小时视频的RHAS133数据集，并提出基于全局-局部感知傅里叶条件扩散的HopaDIFF框架，在多种评估设置下显著超越现有基线。
tags:
  - NeurIPS 2025
  - 图像分割
  - 指称人体动作分割
  - 多人场景
  - 扩散模型
  - xLSTM
  - 傅里叶条件
  - RHAS
---

# HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios

**会议**: NeurIPS 2025  
**arXiv**: [2506.09650](https://arxiv.org/abs/2506.09650)  
**代码**: [https://github.com/KPeng9510/HopaDIFF](https://github.com/KPeng9510/HopaDIFF)  
**领域**: 时序动作分割 / 多人视频理解  
**关键词**: 指称人体动作分割, 多人场景, 扩散模型, xLSTM, 傅里叶条件, RHAS  

## 一句话总结
首次提出指称人体动作分割(RHAS)任务——通过文本描述定位多人视频中特定个体并做帧级动作分割。构建了包含133部电影、137个动作类别、33小时视频的RHAS133数据集，并提出基于全局-局部感知傅里叶条件扩散的HopaDIFF框架，在多种评估设置下显著超越现有基线。

## 研究背景与动机

**领域现状**：动作分割旨在将未裁剪视频按时间划分为动作片段并分类。现有方法（FACT、ActDiff、ASQuery）主要面向**单人场景**，且动作序列遵循预定义协议（如按固定菜谱做沙拉、按说明书装配工具），缺乏真实世界多人场景的随机性和复杂性。

**现有痛点**：
   - 无多人动作分割数据集——现有数据集（50Salads、Assembly101、Breakfast）均为单人、固定流程、第一人称
   - 无文本引导机制——无法指定对哪个人做分割
   - 现有动作分割方法缺乏目标感知的部分特征推理和细粒度生成控制能力

**核心idea**：定义新任务RHAS + 构建RHAS133数据集 + 提出双分支扩散模型HopaDIFF，通过全局-局部特征交互和频域条件增强可控性。

## RHAS133 数据集

### 数据集统计

| 特性 | RHAS133 | 50Salads | Assembly101 | Breakfast |
|------|---------|----------|-------------|-----------|
| 多人 | ✓ | ✗ | ✗ | ✗ |
| 文本引导 | ✓ | ✗ | ✗ | ✗ |
| 多标签 | ✓ | ✗ | ✗ | ✗ |
| 自由流程 | ✓ | ✗ | ✗ | ✗ |
| 视角 | 第三人称 | 第一人称 | 第一人称 | 第三人称 |
| 时长 | 33h | 4h | 513h | 77h |
| 动作类别 | 137 | 50 | 101 | 10 |
| 个体数 | 542 | 25 | 53 | 52 |

- 133部电影、542个标注个体、6名领域专家交叉验证
- 动作标签遵循AVA协议，从80扩展至137个细粒度类别
- 文本引导描述目标个体外貌特征（不透露动作内容）

## 方法详解

### 框架总览
HopaDIFF采用双分支扩散架构：**全局分支**（holistic）捕获全视频上下文，**局部分支**（partial）通过GroundingDINO检测目标人物后提取裁剪视频的细粒度特征。两分支共享VLM特征提取器（BLIP-2/CLIP），但使用独立编码器。

### 1. 全局-局部条件扩散
双分支特征提取：

$$\mathbf{z}_h, \mathbf{z}_p = \mathbf{E}_h(\mathbf{F}_\phi(\mathbf{v}_h, \mathbf{r})), \quad \mathbf{E}_p(\mathbf{F}_\phi(\text{G-Dino}(\mathbf{v}_h, \mathbf{r}), \mathbf{r}))$$

其中 $\mathbf{v}_h$ 为原始视频，$\mathbf{r}$ 为文本引导，$\mathbf{F}_\phi$ 为VLM特征提取器，G-Dino实现目标人物检测和裁剪。

### 2. HP-xLSTM：跨输入门控注意力xLSTM
基于mLSTM变体，在输入门(input gate)处引入双向交叉注意力(BCA)实现全局-局部特征交互：

$$\hat{\mathbf{z}}^h, \hat{\mathbf{z}}^p = \text{HP-xLSTM}(\mathbf{z}^h, \mathbf{z}^p)$$

mLSTM的矩阵记忆更新：$\boldsymbol{C}_t^m = f_t^m \boldsymbol{C}_{t-1}^m + i_t^m \boldsymbol{v}_t^m \boldsymbol{k}_t^{m\top}$

关键创新在输入门：$\tilde{i}_t^h, \tilde{i}_t^p = \text{BCA}(\boldsymbol{w}_i^{h\top} \boldsymbol{z}_t^h + b_i^h, \; \boldsymbol{w}_i^{p\top} \boldsymbol{z}_t^h + b_i^h)$

通过双向交叉注意力，让全局分支的输入门受局部特征调制，反之亦然，实现双分支信息交换。

### 3. 傅里叶频域条件
对HP-xLSTM增强后的特征施加离散傅里叶变换(DFT)提取频域信息作为额外条件：

$$\hat{\mathbf{z}}_f^h, \hat{\mathbf{z}}_f^p = \text{DFT}(\hat{\mathbf{z}}^h), \text{DFT}(\hat{\mathbf{z}}^p)$$

解码器同时接收时空特征和频域特征：

$$\mathbf{s}^h, \mathbf{s}^p = \mathbf{D}_h(\mathbf{y}_t, \hat{\mathbf{z}}_f^h, \mathbf{z}_f^h), \quad \mathbf{D}_p(\mathbf{y}_t, \hat{\mathbf{z}}_f^p, \mathbf{z}_f^p)$$

频域条件增强细粒度可控性——低频编码整体动作节奏，高频捕捉动作边界跳变。

### 4. 训练与推理
- 训练：二元交叉熵损失 + 时序边界损失（对齐去噪序列与GT的动作边界）
- 推理：从高斯噪声出发迭代去噪，最终两分支预测取平均

## 实验关键数据

### BLIP-2特征 + 随机划分

| 方法 | ACC↑ | EDIT↑ | F1@10↑ | F1@25↑ | F1@50↑ |
|------|------|-------|--------|--------|--------|
| FACT | 26.08 | 0.27 | 52.91 | 50.77 | 47.06 |
| ActDiff | 41.85 | 7.20 | 70.56 | 68.34 | 63.29 |
| LTContent | 34.23 | 0.31 | 64.70 | 63.09 | 58.50 |
| RefAtomNet | 38.01 | 0.13 | 34.01 | 31.93 | 27.62 |
| **HopaDIFF** | **62.58** | **7.75** | **87.96** | **85.50** | **79.39** |

HopaDIFF ACC提升50%（62.58 vs 41.85），F1@50提升25%（79.39 vs 63.29），大幅领先所有基线。

### BLIP-2特征 + 跨电影评估（更严格的泛化测试）

| 方法 | ACC↑ | EDIT↑ | F1@10↑ | F1@25↑ | F1@50↑ |
|------|------|-------|--------|--------|--------|
| ActDiff | 2.36 | 15.09 | 22.44 | 22.28 | 21.80 |
| LTContent | 52.52 | 0.37 | 49.35 | 47.24 | 42.55 |
| **HopaDIFF** | **59.63** | **19.37** | **90.91** | **90.33** | **89.26** |

跨电影设置下优势更大——F1@50从42.55提升至89.26，体现全局-局部双分支的强泛化能力。ActDiff泛化严重退化（ACC降至2.36%），而HopaDIFF保持稳健。

## 亮点
1. **新任务+新数据集**——首次定义RHAS任务，RHAS133是首个多人文本引导动作分割数据集
2. **HP-xLSTM**——在xLSTM输入门引入跨输入双向注意力，优雅地实现全局-局部信息交换
3. **傅里叶条件**——在时空域基础上引入频域条件，增强扩散模型对动作边界的细粒度控制
4. **跨电影泛化**——F1@50达89.26%，远超其他方法，证明双分支架构的鲁棒性

## 局限与展望
1. 依赖GroundingDINO检测目标人物——遮挡严重或面部不可见时检测可能失败
2. RHAS133仅133部电影，虽覆盖广但数量有限
3. 两阶段架构（VLM特征预提取+扩散分割）无法端到端训练
4. 推理时需同时运行GroundingDINO+VLM+双分支扩散，计算开销较大

## 启发与关联
- RHAS任务可推广到体育分析（追踪特定运动员的动作序列）、电影内容理解
- HP-xLSTM的跨输入门控设计可用于其他需要多源信息融合的长序列建模任务
- 频域条件的思路可应用到视频生成中保持时序一致性

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 新任务+新数据集+创新技术方案，贡献全面
- 实验充分度: ⭐⭐⭐⭐ 多种评估设置+两种特征提取器，消融有待加强
- 写作质量: ⭐⭐⭐⭐ 框架清晰，但公式较多需仔细对照
- 价值: ⭐⭐⭐⭐⭐ 开创RHAS方向，数据集有长期价值

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)
- [\[ECCV 2024\] You Only Learn One Query: Learning Unified Human Query for Single-Stage Multi-Person Multi-Task Human-Centric Perception](../../ECCV2024/segmentation/you_only_learn_one_query_learning_unified_human_query_for_single-stage_multi-per.md)
- [\[NeurIPS 2025\] Seg4Diff: Unveiling Open-Vocabulary Segmentation in Text-to-Image Diffusion Transformers](seg4diff_unveiling_open-vocabulary_segmentation_in_text-to-image_diffusion_trans.md)

<!-- RELATED:END -->
