---
title: >-
  [论文解读] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization
description: >-
  [CVPR 2026][图像到视频] 本文提出 IPRO，通过强化学习和可微分人脸身份评分器直接优化视频扩散模型，在不修改模型架构的情况下显著提升图像到视频生成中的人脸身份一致性，在 Wan 2.2 上实现了 20%-45% 的 FaceSim 提升。
tags:
  - CVPR 2026
  - 图像到视频
  - 身份保持
  - 强化学习
  - 人脸奖励
  - 扩散模型微调
---

# Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization

**会议**: CVPR 2026  
**arXiv**: [2510.14255](https://arxiv.org/abs/2510.14255)  
**代码**: https://ipro-alimama.github.io/ (项目页)  
**领域**: 扩散模型 / 视频生成  
**关键词**: 图像到视频, 身份保持, 强化学习, 人脸奖励, 扩散模型微调

## 一句话总结

本文提出 IPRO，通过强化学习和可微分人脸身份评分器直接优化视频扩散模型，在不修改模型架构的情况下显著提升图像到视频生成中的人脸身份一致性，在 Wan 2.2 上实现了 20%-45% 的 FaceSim 提升。

## 研究背景与动机

**领域现状**：图像到视频（I2V）生成已经取得了很大进展，CogVideoX、HunyuanVideo、Wan 等 Diffusion Transformer 模型能够从静态图像合成时间连贯的高质量视频。人物视频生成是 I2V 的重要应用场景。

**现有痛点**：现有 I2V 模型在生成视频时难以保持输入人像的身份一致性，尤其当人物表情变化大、动作幅度大时问题更严重。当人脸在图像中占比很小时，这个问题更加突出。随着帧数增加，误差在帧间传播导致身份逐渐退化，使生成的人物外观偏离初始帧。

**核心矛盾**：一方面，身份信息已经完全编码在第一帧中，不缺信息；另一方面，现有方法（如在模型中注入额外身份模块）存在"曝光偏差"问题——训练时基于真实中间状态，推理时却基于自己生成的状态，导致误差累积和身份漂移。而且这些架构入侵式方法本质上是单人设计，难以扩展到多人场景。

**本文目标** 能否增强通用基础 I2V 模型的身份保持能力，同时不改变架构也不损害原始能力？

**切入角度**：从强化学习的视角出发，将人脸身份评分器（ArcFace）作为奖励模型，直接通过梯度反向传播优化扩散模型参数，使其生成身份一致性更好的视频。

**核心 idea**：用 ArcFace 人脸嵌入的余弦相似度作为可微奖励信号，通过截断梯度反传微调视频扩散模型以提升身份保持。

## 方法详解

### 整体框架

IPRO 的输入是初始噪声 $x_T$ 和条件图像，通过视频扩散模型进行完整的 T 步采样后得到生成视频，然后用冻结的 VAE 解码器还原像素空间，由人脸奖励模型评分。奖励信号反向传播到扩散模型的可训练参数。整个框架包含三个核心组件：人脸奖励反馈学习、Facial Scoring Mechanism（FSM）和 KL 散度正则化。

### 关键设计

1. **人脸奖励反馈学习（Facial Reward Feedback Learning）**:

    - 功能：直接优化扩散模型使生成视频最大化人脸身份一致性
    - 核心思路：目标函数为 $J(\theta) = \mathbb{E}_{x_T \sim N(0,I)}[R_{face}(sample(\theta, x_T))]$，即最大化从随机噪声采样生成的视频的人脸奖励。为降低显存和加速优化，采用 DRaFT 截断策略，仅对最后 $K$ 步（$K=4$）反传梯度：$\nabla_\theta R_{face}^K = \sum_{t=0}^{K} \frac{\partial R_{face}}{\partial x_t} \cdot \frac{\partial x_t}{\partial \theta}$。这是因为后期去噪步骤对外观细节影响最大
    - 设计动机：与监督微调（SFT）相比，奖励反馈学习从纯噪声开始生成，训练分布与推理分布对齐，直接消除了曝光偏差。SFT 的逐帧损失无法感知小幅逐渐漂移，但整体奖励可以直接优化长时身份一致性

2. **Facial Scoring Mechanism（FSM）**:

    - 功能：提供鲁棒的多角度人脸奖励信号，防止"复制粘贴"现象
    - 核心思路：将真实视频所有帧的人脸作为特征池。对每个生成帧 $i$，计算其与所有真实帧人脸的平均余弦相似度：$s_i = \frac{1}{F}\sum_{j=1}^{F} \cos(\phi(\hat{x}_i), \phi(x_j))$，最终奖励为所有生成帧的平均分。这鼓励生成的人物在多角度下与真实人物相似，同时允许自然的表情变化
    - 设计动机：如果只与参考图像计算相似度，模型会学会让视频中的人脸严格保持第一帧表情（copy-paste），失去表情多样性。如果与时间对齐的 GT 帧计算，在 SFT 训练下信号太弱。FSM 提供了广泛且信息丰富的奖励

3. **KL 散度正则化**:

    - 功能：稳定训练，防止奖励黑客（reward hacking）
    - 核心思路：在反向采样轨迹上施加多步 KL 散度约束：$D_{KL}(p_\theta(x_{0:T}) || p_{\theta_{ref}}(x_{0:T})) = \sum_{t=1}^{K} \omega_t' \|v_\theta(x_t, t) - v_{\theta_{ref}}(x_t, t)\|^2$，惩罚优化模型与原始模型在每一步的速度预测偏差
    - 设计动机：如果只用人脸奖励优化，模型会过度迎合奖励模型，生成僵硬表情、缺乏运动的视频。KL 正则化将偏差约束在较小范围，保留模型原始的视频生成能力

### 损失函数 / 训练策略

使用 Adam 优化器，学习率 2e-5，训练 100 步，batch size 64。截断梯度步数 $K=4$，人脸奖励权重 0.1，KL 损失权重 1。对于 Wan2.2 27B-A14B，仅训练 low-noise expert 部分。使用 Wan2.2-Lightning 蒸馏版本（8 步无需 CFG）提高训练效率。训练数据从互联网收集 960p 视频，保留人脸较小的场景（最大人脸框不超 100×100 像素）。

## 实验关键数据

### 主实验

| 方法 | FaceSim↑ | SC↑ | BC↑ | AQ↑ | IQ↑ | DD↑ |
|------|----------|-----|-----|-----|-----|-----|
| In-house I2V (15B) | 0.477 | 0.977 | 0.978 | 0.664 | 0.729 | 8.93 |
| + IPRO | **0.696** (+45.9%) | 0.981 | 0.981 | 0.664 | 0.726 | 8.31 |
| Wan 2.2 5B | 0.379 | 0.942 | 0.955 | 0.648 | 0.727 | 27.79 |
| + IPRO | **0.546** (+44.1%) | 0.946 | 0.956 | 0.649 | 0.724 | 27.26 |
| Wan 2.2 A14B | 0.578 | 0.951 | 0.971 | 0.659 | 0.727 | 19.45 |
| + IPRO | **0.694** (+20.1%) | 0.954 | 0.972 | 0.661 | 0.725 | 19.17 |

**与其他方法对比（基于 Wan 2.2 A14B）**:

| 方法 | FaceSim↑ |
|------|----------|
| Wan 2.2 | 0.578 |
| MoCA† (T2V 适配) | 0.582 |
| Concat-ID† (T2V 适配) | 0.606 |
| DPO | 0.628 |
| GRPO | 0.633 |
| **IPRO (Ours)** | **0.694** |

### 消融实验

| 配置 | FaceSim↑ | Hacking↓ | 说明 |
|------|----------|----------|------|
| Wan 2.2 原始 | 0.578 | 7% | 基线 |
| w/o KL 正则 | 0.754 | **58%** | FaceSim 高但严重 hacking |
| w/o FSM | 0.739 | **52%** | 同样严重 hacking |
| **Full IPRO** | **0.694** | **10%** | 平衡了身份保持与自然运动 |

| 训练框架 | FaceSim↑ |
|----------|----------|
| SFT† | 0.639 |
| CLIP 奖励† | 0.610 |
| **IPRO (ArcFace 奖励)** | **0.694** |

### 关键发现

- KL 正则化和 FSM 是防止 reward hacking 的关键：去掉任一个都会导致 50%+ 的 hacking 率
- ArcFace 作为奖励模型明显优于 CLIP（0.694 vs 0.610），因为 ArcFace 对细粒度人脸特征的判别力更强
- 使用后期（低噪声）梯度步优于前期（高噪声）步：FaceSim 0.694 vs 0.646
- IPRO 在提升身份保持的同时基本不损害原始模型的视频质量指标

## 亮点与洞察

- **不改架构的通用性**：IPRO 是纯策略优化方法，不需要额外模块，可以直接应用到任何 I2V 基础模型上。这种"奖励驱动微调"思路通用性极强，100 步就够
- **FSM 的多视角池设计**：将 GT 视频所有帧的人脸作为参考池而非单帧/对齐帧，既避免了 copy-paste 又提供了更丰富的监督信号，是处理"保持一致但允许变化"这类矛盾需求的巧妙方案
- **KL 正则与 reward hacking 的量化分析**：用 Gemini 2.5 Pro VLM 量化评估 hacking 率是很有说服力的评估方式

## 局限与展望

- 目前仅关注面部身份保持，非面部属性（如珠宝、配饰、服装）的一致性未涉及
- 训练依赖小脸场景数据集，对大脸场景的改进可能有限
- ArcFace 本身的偏差（如对某些种族或极端角度的识别不足）可能传递到生成结果
- 未来可以设计统一的"全身身份"奖励模型，覆盖面部和非面部特征

## 相关工作与启发

- **vs MoCA / Concat-ID（T2V 身份方法适配到 I2V）**: 这些方法需要额外的身份模块，改变了模型架构。IPRO 不修改架构但效果更好，说明"正确的优化目标"比"增加模块"更重要
- **vs DPO**: DPO 只优化相对偏好排序，缺乏绝对校准；面部身份是可绝对量化的指标，因此 IPRO 的直接奖励优化更适合
- **vs GRPO**: GRPO 依赖组内响应多样性，但同一 prompt 生成的视频高度相似，导致优势估计失效

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个将人脸奖励反馈学习应用到 I2V 身份保持的工作，FSM 和 KL 正则设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基础模型验证、多种对比方法、详细消融、用户研究
- 写作质量: ⭐⭐⭐⭐ 动机清晰，消融实验逻辑性强
- 价值: ⭐⭐⭐⭐ 解决了 I2V 中重要的实用问题，方法通用可迁移

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Goal-Driven Reward by Video Diffusion Models for Reinforcement Learning](goal-driven_reward_by_video_diffusion_models_for_reinforcement_learning.md)
- [\[CVPR 2025\] Identity-Preserving Text-to-Video Generation by Frequency Decomposition](../../CVPR2025/video_generation/identity-preserving_text-to-video_generation_by_frequency_decomposition.md)
- [\[ICCV 2025\] DOLLAR: Few-Step Video Generation via Distillation and Latent Reward Optimization](../../ICCV2025/video_generation/dollar_fewstep_video_generation_via_distillation_and_latent.md)
- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](../../CVPR2025/video_generation/optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)
- [\[CVPR 2026\] Anti-I2V: Safeguarding your photos from malicious image-to-video generation](anti-i2v_safeguarding_your_photos_from_malicious_image-to-video_generation.md)

<!-- RELATED:END -->
