---
title: >-
  [论文解读] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation
description: >-
  [NeurIPS 2025][图像生成][视频到音频] 提出MGAudio，首个采用模型引导(MG)训练替代无分类器引导(CFG)的视频到音频生成框架，结合双角色音视频编码器（同时用于条件注入和特征对齐），以131M参数在VGGSound上实现SOTA（FAD=0.40），且仅用10%数据即可超越多数方法。
tags:
  - NeurIPS 2025
  - 图像生成
  - 视频到音频
  - 模型引导
  - 流匹配
  - 双角色对齐
  - 无分类器引导替代
---

# MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.24103](https://arxiv.org/abs/2510.24103)  
**代码**: [GitHub](https://github.com/pantheon5100/mgaudio)  
**领域**: 视频到音频生成 / 多模态生成  
**关键词**: 视频到音频, 模型引导, 流匹配, 双角色对齐, 无分类器引导替代

## 一句话总结

提出MGAudio，首个采用模型引导(MG)训练替代无分类器引导(CFG)的视频到音频生成框架，结合双角色音视频编码器（同时用于条件注入和特征对齐），以131M参数在VGGSound上实现SOTA（FAD=0.40），且仅用10%数据即可超越多数方法。

## 研究背景与动机

视频到音频(V2A)生成旨在为无声视频合成语义对齐、时间同步的音频。当前主流方法（Diff-Foley、FRIEREN、MDSGen、MMAudio）无论采用扩散还是流匹配架构，普遍依赖**无分类器引导(CFG)**——在训练时随机丢弃10%的条件信号来学习无条件和有条件两个目标。但CFG存在两个问题：

**多任务稀释**：同时学习有条件和无条件目标，可能分散模型容量，导致两个子任务都学不充分

**训练-推理不匹配**：训练时使用条件丢弃，推理时使用CFG缩放，两者的采样行为不一致

Vision Model-Guidance (VMG) 在图像生成领域证明了直接用模型自我引导替代CFG的可行性，但其设计针对离散类别标签，在连续视频条件的音频生成中未被探索。作者还观察到，与视觉领域不同，音频领域的模型引导训练虽然加速收敛，但推理时仍需CFG来保证最高质量——这揭示了**模态特异性行为**，可能与音频的时序敏感性有关。

此外，已有方法通常只使用CAVP的视频编码器进行条件注入，丢弃了音频编码器。作者认为CAVP音频编码器可以作为中间表示的对齐目标，类似视觉领域REPA的做法。

## 方法详解

### 整体框架

MGAudio包含三个核心组件：(1) 可扩展的基于流匹配的Transformer去噪器(FBDT)；(2) 双角色音视频编码器(DRAVE)同时进行条件注入和特征对齐；(3) 音频模型引导(AMG)训练目标替代CFG。输入视频帧通过CAVP视频编码器提取特征后聚合为全局条件向量，引导flow-based Transformer从噪声生成音频潜在表示。

### 关键设计

1. **基于流匹配的去噪Transformer (FBDT)**: 采用SiT(Scalable Interpolant Transformer)架构。输入音频转为mel频谱图 $\mathbf{X} \in \mathbb{R}^{64 \times 816}$，经AudioLDM VAE编码为潜在表示 $\mathbf{x} \in \mathbb{R}^{8 \times 16 \times 204}$，patchify后展平为token序列 $\mathbf{x}' \in \mathbb{R}^{816 \times D}$（$D=768$为Base模型维度）。视频帧经CAVP编码后通过1×1卷积聚合为全局向量 $\vec{v} \in \mathbb{R}^{1 \times 768}$，通过Adaptive LayerNorm注入。流匹配目标为：

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x}_0, \vec{v}, \epsilon} \| u_\theta(\mathbf{x}_t, \vec{v}, t) - u_t(\mathbf{x}_t | \mathbf{x}_0) \|^2$$

其中 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\epsilon$，真实流方向 $u_t = \mathbf{x}_0 - \epsilon$。

2. **双角色音视频编码器 (DRAVE)**: 利用CAVP的**两个**编码器：视频编码器提供条件信号引导去噪（通过AdaLN），音频编码器提供干净音频的参考表示用于中间层对齐。对齐损失为：

$$\mathcal{L}_{\text{align-audio}} = -\mathbb{E}_{\mathbf{x}, \epsilon, t} \left[ \frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \text{similarity}(\mathbf{G}_0^i, h_\phi(\mathbf{H}_t^i)) \right]$$

其中 $\mathbf{G}_0$ 是CAVP编码的干净音频特征，$\mathbf{H}_t$ 是去噪Transformer在时间步 $t$ 的中间潜在表示，$h_\phi$ 是MLP投影层。使用余弦相似度。这种双角色设计让CAVP编码器既参与条件注入（视频分支）又提供学习信号（音频分支），充分挖掘了预训练编码器的潜力。

3. **音频模型引导 (AMG)**: 替代CFG的训练目标，核心思想是让模型自我引导。定义模型引导目标为：

$$u' = u + w \cdot \text{sg}(u_\theta(\mathbf{x}_t, \vec{v}, t)) - u_\theta(\mathbf{x}_t, \varnothing, t)$$

$$\mathcal{L}_{\text{AMG}} = \mathbb{E}_{t, \mathbf{x}_0, \vec{v}, \epsilon} \| u_\theta(\mathbf{x}_t, \vec{v}, t) - u' \|^2$$

其中 $w$ 为引导缩放因子，$\text{sg}(\cdot)$ 为stop-gradient算子防止梯度回传导致退化，$\varnothing$ 为零向量条件。使用EMA模型计算 $u'$ 以增强稳定性。训练时以概率 $\psi$ 将条件替换为零向量。

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{FM-align}} = \mathcal{L}_{\text{AMG}} + \lambda \mathcal{L}_{\text{align-audio}}$$

$\lambda = 0.5$。训练1.1M步，batch=64，lr=1e-4，单卡A100 80GB。推理时采样50步，配合CFG=1.45。

## 实验关键数据

### 主实验

**VGGSound测试集**

| 方法 | FAD↓ | FD↓ | IS↑ | KL↓ | Align Acc↑ | 参数量 |
|------|------|-----|-----|-----|-----------|--------|
| Diff-Foley | 6.25 | 23.07 | 10.85 | 3.18 | 93.94 | 860M |
| FRIEREN | 1.38 | 12.36 | 12.12 | 2.73 | 97.25 | 157M |
| MMAudio | 0.71 | 6.97 | 11.09 | 2.07 | 92.28 | 157M |
| **MGAudio** | **0.40** | **6.16** | **12.82** | 2.76 | 95.65 | **131M** |

FAD降至0.40，相比次优MMAudio(0.71)提升43%。

**UnAV-100泛化测试（零调优）**

| 方法 | FAD↓ | FD↓ | IS↑ | Align Acc↑ |
|------|------|-----|-----|-----------|
| MMAudio | 0.93 | 8.63 | 11.37 | 85.68 |
| **MGAudio** | **0.54** | **5.40** | **13.90** | 97.54 |

跨数据集泛化能力强，Align Acc远超MMAudio。

### 消融实验

**数据效率（300k步）**

| 训练数据比例 | FAD↓ | FD↓ | IS↑ | KL↓ | Align Acc↑ |
|-------------|------|-----|-----|-----|-----------|
| 5% | 1.16 | 9.72 | 9.80 | 2.67 | 93.37 |
| 10% | **0.73** | **8.79** | 10.28 | 2.64 | 95.73 |
| 100% | 0.81 | 10.25 | 9.85 | 2.71 | 95.14 |

10%数据的FAD(0.73)竟优于100%数据(0.81)，说明VGGSound存在约15%的噪声数据。

**模型引导效果**

| 设置 | CFG | MG | 双对齐 | FAD↓ | FD↓ | Align Acc |
|------|-----|-----|--------|------|-----|-----------|
| 无引导 | ✗ | ✗ | ✗ | 2.67 | 17.23 | 78.20 |
| SiT(仅CFG) | ✓ | ✗ | ✗ | 1.52 | 16.25 | 87.07 |
| 仅MG | ✗ | ✓ | ✗ | 2.13 | 16.20 | 83.87 |
| MG+CFG | ✓ | ✓ | ✗ | 1.19 | 14.26 | 92.37 |
| MGAudio完整 | ✓ | ✓ | ✓ | **1.14** | **13.09** | **93.67** |

**对齐编码器选择**

| 编码器 | FAD↓ | Align Acc↑ | 说明 |
|--------|------|-----------|------|
| DINOv2 | 5.52 | 21.66 | 视觉编码器不适合音频对齐 |
| CLAP | 1.35 | 93.19 | 音频语言预训练编码器 |
| CAVP | **1.14** | **93.67** | 音视频对比预训练更优 |

### 关键发现

- **音频领域MG≠视觉领域VMG**：视觉领域MG单独即可达到SOTA，但音频领域MG需要配合推理时CFG才能达到最优，可能因为音频的时序结构需要额外的推理时引导
- 数据质量比数据量更重要：VGGSound约6%样本音视频对齐弱，9%含静帧或静音，仅85%有效
- 模型可扩展性良好：从S/2(34M)到XL/2(680M)，FAD从2.54持续降至0.90
- UMAP可视化显示MGAudio生成的音频分布比CFG方法更紧密地聚集在真实数据分布周围

## 亮点与洞察

- **简洁的核心思想**：用模型自身（EMA版本）的有条件/无条件预测差来自我引导，替代CFG训练中的条件丢弃
- **数据效率惊人**：10%数据超越多数100%数据训练的方法，证明AMG训练策略对条件结构的高效利用
- **双角色设计实用**：不引入新的外部模型，仅充分利用已有CAVP编码器的两个分支
- 131M参数远小于Diff-Foley(860M)和See&Hear(1099M)，但性能全面碾压

## 局限与展望

- 对人类语音、歌唱等语言学复杂音频效果不佳，缺乏语音结构和音素感知
- 视觉语义模糊时可能引入混淆（如同一画面可对应多种声音）
- 推理速度受限于VAE和迭代采样，可通过一致性模型或蒸馏加速
- 未探索更精细的时序对齐策略（目前采用全局聚合）

## 相关工作与启发

- MG训练策略可推广到其他条件生成任务（文本到音频、文本到视频等）
- 双角色编码器的思想可扩展到其他预训练模型：不仅用于条件注入，还用于中间表示监督
- 数据质量发现提示VGGSound可能需要清洗——构建干净子集可能是低成本提升的有效手段
- AMG+CFG的协同效应值得在更多模态上验证

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将模型引导从图像扩展到音频生成，双角色对齐设计简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ 5个消融维度覆盖全面，数据效率和可扩展性分析深入，UMAP可视化直观
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但related work部分较冗长
- 价值: ⭐⭐⭐⭐⭐ 以极小模型实现大幅SOTA提升，数据效率发现对领域有指导意义

<!-- RELATED:START -->

## 相关论文

- [FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](../../ACL2025/image_generation/flashaudio_rectified_flow_tta.md)
- [AvatarArtist: Open-Domain 4D Avatarization](../../CVPR2025/image_generation/avatarartist_open-domain_4d_avatarization.md)
- [Vicinity-Guided Discriminative Latent Diffusion for Privacy-Preserving Domain Adaptation](vicinity-guided_discriminative_latent_diffusion_for_privacy-preserving_domain_ad.md)
- [From Cradle to Cane: A Two-Pass Framework for High-Fidelity Lifespan Face Aging](from_cradle_to_cane_a_two-pass_framework_for_high-fidelity_lifespan_face_aging.md)
- [Knowledge Distillation Detection for Open-weights Models](knowledge_distillation_detection_for_open-weights_models.md)

<!-- RELATED:END -->
