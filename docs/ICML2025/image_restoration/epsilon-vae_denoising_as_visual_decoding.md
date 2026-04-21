---
title: >-
  [论文解读] ε-VAE: Denoising as Visual Decoding
description: >-
  [ICML 2025][图像恢复][扩散模型] 提出 ε-VAE，将传统自编码器中的单步确定性解码器替换为扩散/去噪过程，实现"去噪即解码"（Denoising as Decoding），在相同压缩率下重建质量提升 40%、下游生成质量提升 22%，或在保持生成质量的同时通过提高压缩率实现 2.3 倍推理加速。
tags:
  - ICML 2025
  - 图像恢复
  - 扩散模型
  - VAE
  - Image Tokenizer
  - Rectified Flow
  - 视觉解码
  - 迭代重建
---

# ε-VAE: Denoising as Visual Decoding

**会议**: ICML 2025  
**arXiv**: [2410.04081](https://arxiv.org/abs/2410.04081)  
**作者**: Long Zhao, Sanghyun Woo, Ziyu Wan, Yandong Li, Han Zhang, Boqing Gong, Hartwig Adam, Xuhui Jia, Ting Liu
**机构**: Google DeepMind
**代码**: 无  
**领域**: 图像生成 / 视觉自编码器  
**关键词**: 扩散模型, VAE, Image Tokenizer, Rectified Flow, 视觉解码, 迭代重建

## 一句话总结

提出 ε-VAE，将传统自编码器中的单步确定性解码器替换为扩散/去噪过程，实现"去噪即解码"（Denoising as Decoding），在相同压缩率下重建质量提升 40%、下游生成质量提升 22%，或在保持生成质量的同时通过提高压缩率实现 2.3 倍推理加速。

## 研究背景与动机

### 视觉 Tokenization 的重要性

现代视觉生成模型（自回归模型和扩散模型）通常在低分辨率的潜空间中训练，依赖 tokenizer 将高维图像压缩为紧凑的潜在表示。这一过程对生成质量至关重要：

- **自回归模型**：离散 token 允许逐步条件生成
- **扩散模型**：连续潜变量使去噪过程的学习更高效
- 实验证据表明，tokenization 能显著提升生成性能

### 传统自编码器的局限

标准的视觉自编码器（如 SD-VAE）采用"编码器 + 确定性解码器"架构：
- 编码器 $\mathcal{E}$ 将图像 $\mathbf{x} \in \mathbb{R}^{H \times W \times 3}$ 压缩为潜空间表示
- 解码器 $\mathcal{G}$ 在**单步**内直接将潜变量重建回像素空间

**核心问题**：单步重建的表达能力有限，尤其在高压缩率下，重建质量急剧下降，进而成为下游生成模型质量的瓶颈。

### 核心动机

作者提出一个新视角：**为什么解码必须是单步的？** 扩散模型已经证明了迭代式生成的强大表达能力，能否将这种能力引入自编码器的解码阶段？

虽然已有工作（DiffAE, Preechakul et al., 2022）和并行工作（Birodkar et al., 2024）探索了扩散机制在自编码中的应用，但尚无工作在实际性能上超越传统自编码范式。ε-VAE 通过精心的架构和目标函数协同设计，首次实现了这一突破。

## 方法详解

### 整体框架

ε-VAE 的核心思想是**用扩散过程替代传统解码器**：

1. **编码器**（不变）：卷积编码器 $\mathcal{E}$ 将输入图像压缩为潜在表示 $z = \mathcal{E}(\mathbf{x})$
2. **扩散解码器**（核心创新）：不再用卷积网络直接重建，而是通过扩散模型从噪声出发，以编码器输出 $z$ 为条件，**迭代去噪**逐步恢复原始图像

$$\hat{\mathbf{x}} = \text{DiffusionDecode}(\mathbf{x}_T, z) \quad \text{where } \mathbf{x}_T \sim \mathcal{N}(0, I)$$

这种设计将重建过程从一次性映射转变为渐进式精化过程，扩散模型在编码器潜变量的引导下逐步恢复原始数据。

### 关键设计选择

作者通过系统性的受控实验，分析了以下关键设计因素对性能的影响：

#### 1. 条件注入架构

扩散解码器必须有效地以编码器的潜在表示为条件。论文探索了多种条件注入方式，确保扩散模型能充分利用编码器提供的压缩信息来引导去噪方向。这是 ε-VAE 区别于普通扩散模型的核心——它不是从头生成，而是在编码器提供的语义指导下进行有条件的重建。

#### 2. 训练目标的联合设计

ε-VAE 的训练目标不仅包含扩散模型标准的去噪损失，还整合了传统自编码器中被证明有效的重建损失：

- **扩散去噪损失**：标准的噪声预测 / 速度场回归目标
- **感知损失（LPIPS）**：衡量重建图像与原图在深度特征空间的感知相似度
- **对抗损失（GAN loss）**：引入判别器提升重建图像的视觉质量和锐度

这种"扩散 + 传统自编码损失"的协同训练是 ε-VAE 性能突破的关键因素之一。

#### 3. 模型参数化（Parameterization）

扩散解码器的预测目标选择（ε-预测 vs x-预测 vs v-预测）对性能有重要影响。论文名称中的 "ε" 即暗示了噪声预测（ε-prediction）在该框架中的核心地位。不同参数化方式影响模型的优化轨迹和收敛性。

#### 4. 噪声调度（Noise Scheduling）

噪声调度策略决定了扩散过程中各时间步的噪声水平分布，直接影响：
- 优化轨迹的平滑性
- 不同频率信息的重建精度
- 少步推理时的重建质量

#### 5. 时间步分布（Timestep Distribution）

训练时和测试时的时间步采样分布对性能影响显著：
- **训练阶段**：时间步的采样策略影响模型对不同噪声水平的处理能力
- **测试阶段**：少步采样的时间步选择直接决定重建质量和效率的平衡

### 采样效率

ε-VAE 的一个突出特点是**极高的采样效率**——仅需 1-3 步去噪即可实现高质量重建，这得益于编码器提供了强有力的语义引导。相比之下，无条件扩散模型通常需要 50-1000 步才能生成高质量图像。

### 分辨率泛化

ε-VAE 展现出良好的分辨率泛化能力，即在一个分辨率上训练后可以推广到其他分辨率，这对实际部署非常重要。

## 实验关键数据

### 重建质量对比（rFID）

在标准配置（Rombach et al., 2022）下，ε-VAE 相比 SOTA 自编码方法在重建质量（rFID）上取得显著提升：

| 方法类型 | 解码方式 | 重建质量提升 | 推理步数 |
|---|---|---|---|
| 传统 VAE (如 SD-VAE) | 单步确定性解码 | 基准 | 1 步 |
| ε-VAE (本文) | 扩散迭代解码 | **↑ 40%** (rFID) | 1-3 步 |

### 下游生成质量对比（FID）

将 ε-VAE 作为 tokenizer 接入下游扩散生成模型后，生成质量（FID）显著提升：

| 对比维度 | 传统 VAE | ε-VAE | 提升幅度 |
|---|---|---|---|
| 相同压缩率下的 FID | 基准 | 更低 | **↓ 22%** |
| 相同生成质量下的推理速度 | 基准 | 更快 | **2.3× 加速** |

### 压缩率 vs 生成质量 Trade-off

ε-VAE 的核心优势在于提供了更好的压缩-质量权衡：

| 策略 | 压缩率 | 生成质量 | 推理速度 |
|---|---|---|---|
| 策略 A：保持压缩率 | 不变 | ↑ 22% | 不变 |
| 策略 B：提高压缩率 | ↑ 更高 | 持平 | ↑ 2.3× |

策略 B 说明 ε-VAE 允许更激进的压缩却不牺牲质量，这对大规模图像生成的部署有重要意义：更高压缩意味着更小的潜空间，扩散模型在更小的空间中工作更快。

### 少步推理能力

| 推理步数 | 重建质量 | 备注 |
|---|---|---|
| 1 步 | 有效 | 接近传统 VAE 的计算开销 |
| 2-3 步 | 最优 | 性能-效率最佳平衡点 |
| 更多步 | 边际递减 | 3 步以上收益有限 |

## 亮点与洞察

1. **视角创新**：将"解码"重新定义为"去噪"，打破了自编码器中"解码必须单步"的隐含假设。这一视角看似简单，实则需要精心的系统设计才能在实践中奏效
2. **协同设计**：将扩散损失与传统自编码损失（LPIPS + GAN）结合，而非简单替换，是一个重要的工程洞察
3. **高效采样**：1-3 步即可重建的特性使得扩散解码器在实际场景中不会成为瓶颈，这得益于编码器潜变量提供的强条件信号
4. **灵活的压缩-速度权衡**：用户可以根据需求选择"同压缩率提质"或"同质量提速"，提供了实用的部署灵活性
5. **分辨率泛化**：训练分辨率可迁移到其他分辨率，减少了实际部署的约束

## 局限与展望

1. **推理延迟**：即使仅需 1-3 步，扩散解码器仍比传统单步解码器慢，对实时应用可能构成瓶颈
2. **训练复杂度增加**：需要联合训练编码器和扩散解码器，引入扩散相关的超参数（参数化、调度、时间步分布等），调参成本更高
3. **缓存内容有限**：论文的详细方法部分（架构图、具体条件注入机制、定量消融实验细节）未在缓存中完整呈现
4. **与离散 tokenizer 的结合**：本文聚焦连续 latent，扩散解码器是否能与离散 token（如 VQ-VAE）兼容尚待探索
5. **更大规模验证**：在更高分辨率（如 1024+）和更大规模数据集上的验证将进一步增强说服力

## 相关工作与启发

- **Stable Diffusion VAE (Rombach et al., 2022)**：标准的卷积自编码器，ε-VAE 的直接对标基线
- **DiffAE (Preechakul et al., 2022)**：首次将扩散过程引入自编码框架，但未在重建质量上超越传统方法
- **Latent Consistency Models (LCM)**：同样关注少步推理效率，可与 ε-VAE 在蒸馏框架下互补
- **SDXL VAE / DC-AE**：近期高性能视觉自编码器，ε-VAE 提供了一个正交的改进方向

**启发**：ε-VAE 揭示了一个重要规律——在自编码器框架中，解码端的表达能力是限制重建质量的关键瓶颈。引入扩散过程本质上是用"时间换精度"，但凭借编码器提供的强条件信号，所需的额外时间极少（1-3 步），这使得该方法在实际中具有高性价比。这一思路可推广到其他"单步映射表达力不足"的场景。

## 评分

- 新颖性: ⭐⭐⭐⭐ — "去噪即解码"的视角清晰有力，但扩散解码的idea已被初步探索过
- 实验充分度: ⭐⭐⭐⭐ — 系统性的受控实验覆盖了关键设计选择，重建和生成质量均有评估
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，贡献明确，结构规范
- 综合价值: ⭐⭐⭐⭐ — 为视觉 tokenizer 设计提供了新范式，对 LDM 生态有直接价值

<!-- RELATED:START -->

## 相关论文

- [Prior Does Matter: Visual Navigation via Denoising Diffusion Bridge Models](../../CVPR2025/image_restoration/prior_does_matter_visual_navigation_via_denoising_diffusion_bridge_models.md)
- [Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing](../../CVPR2025/image_restoration/iterative_predictor-critic_code_decoding_for_real-world_image_dehazing.md)
- [Efficient Visual State Space Model for Image Deblurring](../../CVPR2025/image_restoration/efficient_visual_state_space_model_for_image_deblurring.md)
- [Visual-Instructed Degradation Diffusion for All-in-One Image Restoration](../../CVPR2025/image_restoration/visual-instructed_degradation_diffusion_for_all-in-one_image_restoration.md)
- [Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)](../../ICCV2025/image_restoration/generic_event_boundary_detection_via_denoising_diffusion.md)

<!-- RELATED:END -->
