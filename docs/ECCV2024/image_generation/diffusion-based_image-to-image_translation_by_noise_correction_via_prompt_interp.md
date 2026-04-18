---
title: >-
  [论文解读] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation
description: >-
  [ECCV 2024][图像生成][图像翻译] 提出PIC（Prompt Interpolation-based Correction），一种无训练的扩散模型图像翻译方法，通过渐进式prompt嵌入插值构造噪声校正项，将其与源图像噪声预测线性组合，实现结构保持的高保真图像编辑，且推理速度（18.1s）优于所有对比方法。
tags:
  - ECCV 2024
  - 图像生成
  - 图像翻译
  - 无训练方法
  - 噪声校正
  - 提示学习
  - 扩散模型编辑
---

# Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation

**会议**: ECCV 2024  
**arXiv**: [2409.08077](https://arxiv.org/abs/2409.08077)  
**代码**: 无（基于Pix2Pix-Zero代码框架实现）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 图像翻译, 无训练方法, 噪声校正, prompt插值, 扩散模型编辑

## 一句话总结

提出PIC（Prompt Interpolation-based Correction），一种无训练的扩散模型图像翻译方法，通过渐进式prompt嵌入插值构造噪声校正项，将其与源图像噪声预测线性组合，实现结构保持的高保真图像编辑，且推理速度（18.1s）优于所有对比方法。

## 研究背景与动机

**领域现状**：基于扩散模型的文本驱动图像翻译（Image-to-Image Translation）已成为热门方向。该任务目标是根据目标prompt修改源图像的局部区域，同时保留背景和整体结构。现有方法分为两类：
1. 需要微调的方法（DiffusionCLIP、Imagic等）：对预训练模型进行定制化微调，计算成本高
2. 无训练方法（Prompt-to-Prompt、Plug-and-Play、Pix2Pix-Zero等）：操纵反向过程的去噪策略

**现有痛点**：

**起点不准确**：DDIM反向过程的起点 $\mathbf{x}_T^{\text{tgt}}$ 直接设置为 $\mathbf{x}_T^{\text{src}}$，但真正的目标起点 $\mathbf{x}_T^{\text{tgt*}}$ 与之存在偏差，导致朴素的反向过程无法生成理想的目标图像

**文本嵌入突变**：从源prompt嵌入 $\mathbf{y}^{\text{src}}$ 到目标prompt嵌入 $\mathbf{y}^{\text{tgt}}$ 的突然切换，导致生成过程不稳定

**结构保持困难**：现有方法难以在编辑目标区域的同时完美保留背景和结构

**核心矛盾**：如何在不进行任何训练的情况下，补偿反向过程错误起点带来的偏差，同时精确控制编辑区域和保留区域的边界？

**切入角度**：不直接用目标prompt进行去噪，而是渐进地从源prompt插值到目标prompt，并用插值过程中的噪声预测差异构造一个校正项，引导反向过程走向正确的目标。

**核心idea**：修改后的噪声预测 = 源图像的源prompt噪声预测（保持结构）+ 校正项（选择性编辑），其中校正项通过渐进prompt插值自然地定位到需要编辑的区域。

## 方法详解

### 整体框架

PIC修改了标准DDIM反向过程的噪声预测网络，将其分解为两部分：
1. **结构保持项**：$\epsilon_\theta(\mathbf{x}_t^{\text{src}}, t, \mathbf{y}^{\text{src}})$ —— 用源latent和源prompt重建源图像
2. **噪声校正项**：$\gamma \Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t)$ —— 引导目标区域向目标domain对齐

完整的修改后噪声预测为：
$$\hat{\epsilon}_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}^{\text{tgt}}) := \epsilon_\theta(\mathbf{x}_t^{\text{src}}, t, \mathbf{y}^{\text{src}}) + \gamma \Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t)$$

噪声校正仅在前 $\tau$ 步（默认25步）的反向过程中应用，之后使用标准DDIM完成剩余去噪。

### 关键设计

1. **噪声校正项（Noise Correction Term）**：

    - **功能**：捕捉目标prompt相对于源prompt在噪声空间中的差异，仅影响需要编辑的区域
    - **核心思路**：
    $\Delta\epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t) := \epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}_t) - \epsilon_\theta(\mathbf{x}_t^{\text{tgt}}, t, \mathbf{y}^{\text{src}})$
      即同一目标latent在插值prompt和源prompt下的噪声预测差值
    - **设计动机**：当 $\mathbf{y}_t$ 与 $\mathbf{y}^{\text{src}}$ 相同时差值为零（不编辑），当 $\mathbf{y}_t$ 趋向 $\mathbf{y}^{\text{tgt}}$ 时差值集中在与目标prompt相关的区域。可视化证实校正项自动"高亮"待编辑区域，而背景趋近于零。这有效补偿了 $\mathbf{x}_T^{\text{tgt*}}$ 与 $\mathbf{x}_T^{\text{src}}$ 之间的差距

2. **渐进式Prompt插值（Progressive Prompt Interpolation）**：

    - **功能**：在反向过程中，文本嵌入从源prompt平滑过渡到目标prompt
    - **核心思路（词替换任务）**：逐token线性插值
    $\mathbf{y}_t[\ell] = \beta_t \mathbf{y}^{\text{tgt}}[\ell] + (1 - \beta_t) \mathbf{y}^{\text{src}}[\ell]$
      其中混合系数 $\beta_t = \beta + (1-\beta) \times \frac{T-t}{T}$，随去噪步数逐渐从源嵌入过渡到目标嵌入
    - **核心思路（添加短语任务）**：对新增token直接使用目标嵌入，对后续共享token做插值：
    $\mathbf{y}_t[\ell] = \begin{cases} \mathbf{y}^{\text{src}}[\ell], & \text{if } \ell < \ell_s \\ \mathbf{y}^{\text{tgt}}[\ell], & \text{if } \ell_s \leq \ell \leq \ell_f \\ \beta_t \mathbf{y}^{\text{tgt}}[\ell] + (1-\beta_t)\mathbf{y}^{\text{src}}[\ell - \ell_f + \ell_s], & \text{if } \ell > \ell_f \end{cases}$
    - **设计动机**：避免文本嵌入的突变，让模型渐进适应目标domain，在反向过程早期（低频/结构生成阶段）保持更多源信息

3. **与现有方法的集成**：

    - **功能**：将PIC的noise correction框架推广到Prompt-to-Prompt、Plug-and-Play和Pix2Pix-Zero
    - **核心思路**：对每个方法，用 $\mathbf{y}_t$（插值嵌入）替换原方法中的 $\mathbf{y}^{\text{tgt}}$，并将方法特定的noise correction包装进PIC框架
    - **设计动机**：PIC的公式化与现有方法正交，可作为即插即用的性能增强模块

### 损失函数 / 训练策略

- **无训练方法**：不涉及任何损失函数或训练过程
- 超参数设置：$\tau = 25$（校正步数），$\gamma = 1.0$，词替换任务 $\beta = 0.3$，添加短语任务 $\beta = 0.8$
- 前向/反向过程均50步，使用Stable Diffusion v1.4作为骨干
- 使用BLIP自动生成源prompt，结合classifier-free guidance

## 实验关键数据

### 主实验（与SOTA方法对比，LAION-5B数据集）

| 方法 | CS(CLIP相似度)↑ | BD(背景距离)↓ | SD(结构距离)↓ | 推理时间(s) |
|------|---------------|-------------|-------------|-----------|
| Prompt-to-Prompt | 0.302 | 0.113 | 0.040 | 31.2 |
| Plug-and-Play | 0.305 | 0.120 | 0.036 | 24.4 |
| Pix2Pix-Zero | 0.301 | 0.136 | 0.066 | 52.2 |
| **PIC (本文)** | **0.304** | **0.071** | **0.034** | **18.1** |

PIC在背景保持（BD↓37%于PtP）和结构保持（SD最优）上显著领先，同时推理速度最快。

**PIC集成到现有方法的效果**：

| 方法 | BD↓ 原始 | BD↓ +PIC | SD↓ 原始 | SD↓ +PIC |
|------|---------|---------|---------|---------|
| PtP | 0.113 | **0.069** | 0.040 | **0.023** |
| PnP | 0.120 | **0.098** | 0.036 | **0.029** |
| P2P | 0.136 | **0.066** | 0.066 | **0.015** |

PIC作为插件一致提升所有方法的指标，尤其P2P + PIC的SD从0.066降至0.015（提升77%）。

### 消融实验

| 配置 | CS↑ | BD↓ | SD↓ | 说明 |
|------|-----|-----|-----|------|
| DDIM（朴素） | 0.302 | 0.216 | 0.094 | 基线，结构严重破坏 |
| DDIM+PI（仅prompt插值） | 0.302 | 0.184 | 0.081 | 插值带来改善但不够 |
| DDIM+NC（仅噪声校正，无插值） | 0.306 | 0.081 | 0.044 | 噪声校正大幅降低BD/SD |
| **PIC（插值+校正）** | **0.304** | **0.071** | **0.034** | 两者结合达到最优 |

### 关键发现

- 噪声校正项是性能提升的主要来源（BD从0.216降至0.081），prompt插值在此基础上进一步优化（BD从0.081到0.071）
- 校正项的可视化表明它会自动随反向过程聚焦到待编辑区域，而背景接近零值
- $\gamma$ 控制编辑强度与结构保持的权衡：低$\gamma$ 保结构，高$\gamma$ 增强保真度
- 校正仅在前$\tau$步必要，后续阶段生成细节不需要校正
- PIC的推理时间仅18.1s，因为校正项有效期有限且不引入额外梯度计算

## 亮点与洞察

- **极简但有效**：核心公式（Eq.5-7）简洁优雅，仅通过两个噪声预测的差值构造校正项，不引入任何可训练参数
- **正交性**：PIC与现有方法完全正交，可即插即用地集成到其他方法中并一致提升性能
- **效率优势**：相比需要attention map注入（PtP/PnP）或梯度计算（P2P）的方法，PIC的额外开销极小
- 渐进prompt插值的想法直觉上很合理：反向过程早期处理低频全局结构，此时应更多保留源语义；后期生成细节时再全面切换到目标语义

## 局限与展望

- 在某些复杂任务（如大幅度姿态/纹理变化）上仍会失败，所有方法（包括PIC）都难以保持源图像的精细细节
- $\beta$ 对不同任务类型（词替换vs添加短语）需要手动设置不同值，自动化超参选择有改进空间
- 仅在Stable Diffusion v1.4上实验，未验证在更新的SD2.x/SDXL/SD3上的效果
- prompt插值策略假设源/目标prompt的token可以简单对齐，对更复杂的语义编辑场景可能不适用
- 缺少用户研究来补充自动化指标

## 相关工作与启发

- **Prompt-to-Prompt**：通过注入源图像的cross-attention和self-attention map来保持结构，本文方法与之互补
- **Plug-and-Play**：替换中间层的self-attention map和特征图，计算成本较高
- **Pix2Pix-Zero**：通过梯度优化对齐cross-attention map，推理时间最长（52.2s）
- **Null-text/Negative-prompt Inversion**：改进反向过程的重建精度，可与PIC正交使用
- 启发：在扩散模型编辑中，"渐进过渡"优于"突然切换"，这与扩散过程本身的渐进性质一致

## 评分
- 新颖性: ⭐⭐⭐⭐ 噪声校正项+prompt插值的组合简单但立意新颖，理论直觉清晰
- 实验充分度: ⭐⭐⭐⭐ 覆盖6个翻译任务、3个对比方法、3个集成验证和详细消融，但缺少用户研究
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导易懂，与现有方法的集成描述详细
- 价值: ⭐⭐⭐⭐ 即插即用的特性使其具有很高的实用价值，18.1s推理速度优势明显
