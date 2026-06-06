---
title: >-
  [论文解读] Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers
description: >-
  [ICCV 2025][图像生成][MM-DiT] 分析发现 MM-DiT 架构（FLUX、SD3.5）中视觉与文本 token 数量不对称导致交叉注意力被抑制、且注意力权重对时间步不敏感…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "MM-DiT"
  - "注意力机制"
  - "Temperature Scaling"
  - "FLUX"
  - "SD3.5"
  - "Text-Image Alignment"
---

# Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers

**会议**: ICCV 2025  
**arXiv**: [2506.07986](https://arxiv.org/abs/2506.07986)  
**代码**: [https://github.com/Vchitect/TACA](https://github.com/Vchitect/TACA)  
**领域**: 图像生成 / 扩散Transformer / 文图对齐  
**关键词**: MM-DiT, Cross-Attention Suppression, Temperature Scaling, FLUX, SD3.5, Text-Image Alignment

## 一句话总结
分析发现 MM-DiT 架构（FLUX、SD3.5）中视觉与文本 token 数量不对称导致交叉注意力被抑制、且注意力权重对时间步不敏感，提出 TACA（Temperature-Adjusted Cross-modal Attention）通过温度缩放和时间步自适应调整重新平衡多模态交互，结合 LoRA 微调在 T2I-CompBench 上显著提升文图对齐（空间关系+16.4%、形状+5.9%），且几乎无额外计算开销。

## 研究背景与动机

### 问题背景
多模态扩散 Transformer（MM-DiT）将文本和视觉 token 拼接为统一序列后做全注意力，是 Stable Diffusion 3/3.5、FLUX 等 SOTA 文生图模型的核心架构。然而即使最先进的 FLUX.1 Dev 仍存在严重的文图不对齐问题，如缺失物体、属性绑定错误。

### 两个核心问题

#### 问题一：交叉注意力被抑制
MM-DiT 的统一 softmax 中，视觉 token 数量远大于文本 token（如 FLUX 生成 1024×1024 时 $N_{vis}/N_{txt} = 4096/512 = 8$），导致视觉 token 对文本 token 的注意力概率被严重压低：

$$P_{\text{vis-txt}}^{(i,j)} = \frac{e^{s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{txt}} e^{s_{ik}^{vt}/\tau} + \sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}} \approx \frac{e^{s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}}$$

分母被大量视觉-视觉交互项主导，使文本引导信号被"淹没"。这与传统 cross-attention（分母只含文本 token）形成鲜明对比。

#### 问题二：时间步不敏感的 QK 权重
去噪初期（大 $t$）需要强文本引导建立全局布局，后期（小 $t$）侧重视觉细节。但 MM-DiT 的投影矩阵 $W^Q$、$W^K$ 在所有时间步共享，无法根据去噪阶段动态调整交叉注意力强度——初始步骤中 $s_{ik}^{vt}$ 并未获得相对于 $s_{ik}^{vv}$ 的更大权重。

### 关键观察
通过可视化去噪过程中 $x_0$ 的逐步预测（Fig. 3），作者发现图像的整体构图在最初几步就已确定。如果初始布局与文本不对齐，后续步骤无法修正。

## 方法详解

### 整体框架
TACA 对 MM-DiT 的注意力机制做最小修改：(1) 模态特定温度缩放放大交叉注意力；(2) 时间步依赖的分段调整；(3) LoRA 微调消除放大引入的伪影。

### 关键设计一：模态特定温度缩放
引入温度系数 $\gamma > 1$ 放大视觉-文本交互的 logits：

$$P_{\text{vis-txt}}^{(i,j)} = \frac{e^{{\color{blue}\gamma} s_{ij}^{vt}/\tau}}{\sum_{k=1}^{N_{txt}} e^{{\color{blue}\gamma} s_{ik}^{vt}/\tau} + \sum_{k=1}^{N_{vis}} e^{s_{ik}^{vv}/\tau}}$$

$\gamma$ 作为"信号增强器"，在 softmax 的竞争中提升交叉模态交互的相对权重。可视化显示随 $\gamma$ 增大，"brown backpack"、"glass mirror"等文本描述的视觉特征逐渐变得更明显。

### 关键设计二：时间步依赖的分段调整

$$\gamma(t) = \begin{cases} \gamma_0 & t \geq t_{\text{thresh}} \\ 1 & t < t_{\text{thresh}} \end{cases}$$

其中 $t_{\text{thresh}} = 970$（1000 步中的前 10%）。这与去噪动态对齐：初期（大 $t$）需要强文本引导建立构图时施加温度缩放，后期（小 $t$）聚焦视觉细节时恢复正常注意力。

### LoRA 训练消除伪影
放大的交叉注意力 logits 会改变去噪输出分布，可能引入扭曲边界或不一致纹理。通过对注意力层施加 LoRA 适配来恢复真实图像分布：

$$W' = W + \alpha \cdot BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$$

训练时采样 $t \geq t_{\text{thresh}} = 970$ 的高时间步，聚焦语义信息最显著的初始去噪阶段。训练损失：

$$\mathcal{L} = \mathbb{E}_{x_0, t \geq t_{\text{thresh}}} \left[\|v(x_t, t) - v_\theta(x_t, t, \mathcal{P}_{\text{txt}}, \gamma(t))\|_2^2\right]$$

### 实现特点
- **零新增参数**：温度缩放仅需注意力计算时的逐元素乘法操作
- **极简实现**：核心代码修改量极小
- **两个超参数**：$\gamma_0$（温度基值）和 $t_{\text{thresh}}$（时间步阈值），可通过简单消融确定

## 实验

### 实验设置
- **模型**：FLUX.1-Dev、SD3.5-Medium
- **训练数据**：LAION 10K 图文对（LLaVA 增强 caption）
- **评估**：T2I-CompBench（属性绑定、空间关系、复杂提示）
- **LoRA 配置**：$(r, \alpha) = (16, 16)$ 和 $(64, 64)$
- **超参数**：$\gamma_0 = 1.2$，$t_{\text{thresh}} = 970$

### 主实验：T2I-CompBench 对齐评估

| 模型 | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|------|--------|--------|----------|----------|-------------|----------|
| FLUX.1-Dev | 0.7678 | 0.5064 | 0.6756 | 0.2066 | 0.3035 | 0.4359 |
| **+ TACA (r=64)** | **0.7843** | **0.5362** | **0.6872** | **0.2405** | 0.3041 | **0.4494** |
| SD3.5-Medium | 0.7890 | 0.5770 | 0.7328 | 0.2087 | 0.3104 | 0.4441 |
| **+ TACA (r=64)** | **0.8074** | **0.5938** | **0.7522** | **0.2678** | 0.3106 | **0.4470** |

FLUX.1-Dev 上：空间关系 **+16.4%**（0.2066→0.2405），形状 **+5.9%**（0.5064→0.5362）。
SD3.5-Medium 上：空间关系 **+28.3%**（0.2087→0.2678），形状 **+2.9%**。

### 消融实验：温度系数 $\gamma_0$ 的影响

| $\gamma_0$ | Color↑ | Shape↑ | Spatial↑ |
|------------|--------|--------|----------|
| 1.0（无缩放）| 0.7678 | 0.5064 | 0.2066 |
| 1.1 | ~0.775 | ~0.52 | ~0.22 |
| **1.2** | **0.7843** | **0.5362** | **0.2405** |
| 1.3 | ~0.78 | ~0.53 | ~0.24 |

$\gamma_0 = 1.2$ 在文图对齐和图像质量之间取得最佳平衡。过大的 $\gamma$ 会引入伪影。

### 消融实验：时间步阈值与 LoRA 的作用
- 温度缩放仅应用于初始 10% 去噪步骤即可获得主要收益
- 仅用 TACA 不加 LoRA 会引入局部伪影
- LoRA + TACA 在保持对齐提升的同时消除伪影

### 注意力图可视化
对比显示，TACA 在初始去噪步骤中**大幅放大了视觉 token 对文本 token 的注意力**，使图像区域能更好地关注到对应的文本描述。

## 亮点与洞察
- **问题诊断精准**：准确定位 MM-DiT 文图不对齐的两个根本原因——token 数量不对称导致的交叉注意力抑制 + 时间步不敏感
- **方案极简高效**：核心只有一个温度系数 $\gamma(t)$，无新增参数，几行代码即可实现
- **跨架构有效**：在 FLUX 和 SD3.5 两种不同 MM-DiT 实现上均有效
- **深刻的机制洞察**：揭示了全注意力机制（相比传统 cross-attention）在多模态场景下的固有缺陷
- **去噪时间动态分析**：初始步骤建立构图的观察为后续工作提供了重要设计指导

## 局限性
- LoRA 微调仍需训练数据（10K 图文对）和 GPU 资源
- 分段常数 $\gamma(t)$ 可能不是最优的时间步自适应策略，平滑的连续函数可能更好
- 仅在 T2I-CompBench 上评估，缺少人类偏好评估和 FID/IS 等生成质量指标
- 温度缩放只处理了交叉注意力的量级问题，未解决注意力分配的定向性问题
- 训练数据的 caption 质量（LLaVA 生成）可能影响 LoRA 的微调效果

## 相关工作
- **扩散 Transformer 架构**：DiT（adaLN 条件化）、CrossDiT/PixArt-α（交叉注意力）、MM-DiT（统一全注意力）
- **文图对齐改进**：CLIP 引导优化、交叉注意力控制（Attend-and-Excite）、布局规划模块、反馈驱动优化
- **MM-DiT 模型**：FLUX.1、SD3/3.5、CogVideo、HunyuanVideo

## 评分
- 新颖性：⭐⭐⭐⭐ — 对 MM-DiT 注意力机制的分析新颖且深入
- 技术深度：⭐⭐⭐⭐ — 数学推导清晰，问题根因分析有说服力
- 实验充分度：⭐⭐⭐ — 定量评估指标略单一（仅 T2I-CompBench）
- 实用价值：⭐⭐⭐⭐⭐ — 极低成本可直接应用于现有 MM-DiT 模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Exploring Multimodal Diffusion Transformers for Enhanced Prompt-based Image Editing](exploring_multimodal_diffusion_transformers_for_enhanced_prompt-based_image_edit.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[ICCV 2025\] IRGPT: Understanding Real-world Infrared Image with Bi-cross-modal Curriculum on Large-scale Benchmark](irgpt_understanding_real-world_infrared_image_with_bi-cross-modal_curriculum_on_.md)
- [\[NeurIPS 2025\] Head Pursuit: Probing Attention Specialization in Multimodal Transformers](../../NeurIPS2025/image_generation/head_pursuit_probing_attention_specialization_in_multimodal.md)

</div>

<!-- RELATED:END -->
