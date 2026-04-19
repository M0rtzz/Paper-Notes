---
title: >-
  [论文解读] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion
description: >-
  [ICCV 2025][图像生成][动作风格化] 提出 StyleMotif，一个单分支运动潜在扩散框架，通过风格-内容交叉归一化机制统一内容生成与多模态（文本/图片/视频/音频/运动）风格注入，相比 SMooDi 双分支设计减少 43.9% 可训练参数并提速 22.5%，同时在风格识别准确率（SRA）上提升 5.23%。
tags:
  - ICCV 2025
  - 图像生成
  - 动作风格化
  - 多模态
  - 扩散模型
  - 风格-内容融合
  - 运动生成
---

# StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion

**会议**: ICCV 2025  
**arXiv**: 2503.21775  
**代码**: https://stylemotif.github.io  
**领域**: image_generation / 动作风格化  
**关键词**: 动作风格化, 多模态, 扩散模型, 风格-内容融合, 运动生成

## 一句话总结

提出 StyleMotif，一个单分支运动潜在扩散框架，通过风格-内容交叉归一化机制统一内容生成与多模态（文本/图片/视频/音频/运动）风格注入，相比 SMooDi 双分支设计减少 43.9% 可训练参数并提速 22.5%，同时在风格识别准确率（SRA）上提升 5.23%。

## 研究背景与动机

**为什么现有方法不够？** 人体运动的质量由两个维度决定：**内容**（walking, jumping 等动作类型）和**风格**（jubilant, aggressive 等情感/个性表达）。现有方法存在以下问题：

**Text-to-Motion 方法**（MDM、MLD）：擅长生成多样内容，但忽略了"如何"执行动作的风格细节。简单地添加独立风格迁移模块会增加复杂度和累积误差

**动作风格迁移方法**（Aberman et al.、Motion Puzzle）：有效解耦内容与风格用于小规模任务，但当需要风格化大量不同内容动作时管线臃肿。且当输入动作本身是合成的或带噪声时，迁移质量会退化

**SMooDi（最新代表）**：在预训练 MLD 上增加双分支 ControlNet 式风格适配器 + 分类器风格引导，但 (a) 双分支设计增加模型复杂度和训练开销；(b) 仅支持运动序列作为风格输入

**核心限制**：现有方法的双分支设计（主生成网络 + 风格控制网络）需要维护额外参数 $\theta_s$ 和零初始化线性层 $\theta_{z_i}$，限制了并行效率。且风格输入仅限于运动序列这单一模态。

## 方法详解

### 整体框架

StyleMotif 构建于预训练的 Motion Latent Diffusion（MLD）之上，采用**单分支**设计。核心包含三个模块：

1. **风格编码器预训练**：结合 HumanML3D（内容知识）和 100STYLE（风格知识）的 VAE 编码器
2. **风格-内容交叉融合**：通过统计变换将风格特征注入扩散过程
3. **多模态对齐**：利用 ImageBind 实现跨模态风格条件

### 关键设计 1：风格编码器预训练

风格编码器源自 MLD 的 VAE 编码器，分两阶段预训练：

1. 先在 HumanML3D（14,616 条动作序列 + 44,970 条文本描述）上预训练，学习内容动作的特征表示
2. 再在 100STYLE（45,303 条风格动作）上微调，以变分自编码方式对齐内容与风格数据分布

训练后丢弃解码器，仅保留编码器作为风格编码器。这种双数据集预训练策略使编码器既理解内容结构，又能捕捉风格差异。

### 关键设计 2：风格-内容交叉归一化（Style-Content Cross Normalization）

核心创新在于用**统计变换**替代额外网络参数。给定第 $i$ 个 MLD block 的内容特征 $\mathcal{F}_c^i$ 和风格特征 $\mathcal{F}_s$：

**Step 1**：计算内容特征的均值和方差：

$$\mu_c = \frac{1}{D}\sum_{j=1}^{D}\mathcal{F}_c^{i,j}, \quad \sigma_c^2 = \frac{1}{D}\sum_{j=1}^{D}(\mathcal{F}_c^{i,j} - \mu_c)^2$$

**Step 2**：用内容统计量归一化风格特征：

$$\widetilde{\mathcal{F}}_{s,c} = \frac{\mathcal{F}_s - \mu_c}{\sqrt{\sigma_c^2 + \eta}}$$

**Step 3**：将归一化后的风格特征加回内容特征：

$$\mathcal{F}^i_{out} = \mathcal{F}_c^i + \gamma \cdot \widetilde{\mathcal{F}}_{s,c}$$

其中 $\gamma = 0.6$ 为最佳缩放比例。关键约束：融合仅在第 $m$ 个 block 之后执行一次，避免过度扭曲内容。

**与 SMooDi 对比**：SMooDi 需要在每个 block 通过零初始化线性层 $\mathcal{Z}(\cdot)$ 注入风格（需维护 $\theta_s$ 副本），而 StyleMotif 仅用无参数的统计变换实现等效功能。

### 关键设计 3：多模态对齐

通过 ImageBind 的统一多模态特征空间实现跨模态风格条件：

1. 冻结 ImageBind 的文本编码器，添加轻量投影层对齐维度
2. 使用对比学习损失在运动-文本对（来自 100STYLE）上对齐特征空间：

$$\mathcal{L}_{align} = -\frac{1}{2}\sum_{(i,j)} \log\frac{\exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^j / \tau_0)}{\sum_k \exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^k / \tau_0)} + \log\frac{\exp(\mathcal{F}_t^i \cdot \mathcal{F}_s^j / \tau_0)}{\sum_k \exp(\mathcal{F}_t^k \cdot \mathcal{F}_s^j / \tau_0)}$$

3. 推理时，任何模态（图像/视频/音频）输入到 ImageBind 提取特征后，检索最相似的运动风格特征用于风格化

### 训练策略

- 仅训练风格编码器，冻结 MLD 其他参数
- 使用 AdamW 优化器，学习率 $10^{-5}$
- 结合 classifier-free 和 classifier-based 混合引导策略

## 实验关键数据

### 主实验：运动引导风格化（Table 1）

| 方法 | SRA ↑ | FID ↓ | MM Dist ↓ | R-Precision ↑ | Diversity | Foot Skate ↓ |
|------|-------|-------|-----------|---------------|-----------|-------------|
| MLD + Aberman | 54.37 | 3.309 | 5.983 | 0.406 | 8.816 | 0.347 |
| MLD + Motion Puzzle | 63.77 | 6.127 | 6.467 | 0.290 | 6.476 | 0.185 |
| SMooDi | 72.42 | 1.609 | 4.477 | 0.571 | 9.235 | 0.124 |
| **StyleMotif** | **77.65** | **1.551** | **4.354** | **0.586** | 7.567 | **0.097** |

**关键发现**：StyleMotif 在 SRA 上比 SMooDi 提升 5.23%（72.42→77.65），同时 FID 更低（1.551 vs 1.609），Foot Skate Ratio 也更低（0.097 vs 0.124），表明生成动作更真实。

### 消融实验：风格编码器预训练策略（Table 3 上半）

| 预训练数据 | SRA ↑ | FID ↓ | MM Dist ↓ | R-Precision ↑ | Foot Skate ↓ |
|-----------|-------|-------|-----------|---------------|-------------|
| 仅 100STYLE | 76.73 | 1.788 | 4.349 | 0.571 | 0.101 |
| 仅 HumanML3D | 76.58 | 1.635 | 4.458 | 0.572 | 0.109 |
| **两者结合** | **77.65** | **1.551** | **4.354** | **0.586** | **0.097** |

**关键发现**：双数据集预训练在所有指标上均优于单数据集，验证了内容知识+风格知识联合学习的必要性。

### 运动风格迁移（Table 2）

| 方法 | SRA ↑ | FID ↓ | Foot Skate ↓ |
|------|-------|-------|-------------|
| MLD + Aberman | 61.01 | 3.892 | 0.338 |
| SMooDi | 65.15 | 1.582 | 0.095 |
| **StyleMotif** | **68.81** | **1.375** | **0.094** |

### 文本引导风格化

| 方法 | SRA ↑ | FID ↓ |
|------|-------|-------|
| MLD + ChatGPT | 4.82 | 0.614 |
| **StyleMotif** | **56.71** | **0.603** |

文本引导下 SRA 从 4.82% 猛增至 56.71%，验证多模态对齐的有效性。

### 效率对比（Table 4）

| 方法 | 总参数 | 可训练参数 | 推理时间 |
|------|--------|-----------|---------|
| SMooDi | 468M | 13.9M | 4.0s |
| **StyleMotif** | **462M** | **7.8M (-43.9%)** | **3.1s (-22.5%)** |

## 亮点与洞察

1. **无参数风格注入**：用统计归一化替代零初始化线性层，消除了多余可训练参数，是一个优雅的设计选择
2. **单分支 > 双分支**：打破了"风格控制需要额外网络分支"的范式，证明精心设计的特征融合可以在更简洁的架构中达到更好效果
3. **多模态涌现能力**：通过对齐运动编码器与 ImageBind，获得了图像/视频/音频引导的风格化能力，无需为每种模态单独训练
4. **风格插值**：支持多个风格文本的加权混合，产生平滑的风格过渡效果

## 局限性

1. **风格数据有限**：100STYLE 数据集规模制约了模型的风格泛化能力，难以覆盖所有可能的运动风格
2. **单标签最优悖论**：消融表明简短的单风格标签（如"Old"）比详细描述效果更好，提示当前对齐方法可能对复杂风格描述的理解不足
3. **运动质量受限于基础模型**：依赖 MLD 作为基础模型，其本身的内容生成质量上限制约了最终输出
4. **缩放比例 γ 需要手动调节**：γ=0.6 是通过消融实验得出的固定值，不同任务可能需要不同的最优值

## 相关工作与启发

- **SMooDi**：ControlNet 式双分支设计的代表，为风格化运动生成提供了基线
- **ImageBind**：统一多模态嵌入空间的基础模型，本文展示了其在运动领域的应用潜力
- **AdaIN（图像风格迁移）**：交叉归一化的灵感来源，将图像领域的统计风格迁移思想迁移到动作领域
- **启发**：统计归一化作为无参数特征融合机制，可推广到其他生成任务中的条件注入（如 3D 生成、视频生成中的风格控制）

## 评分 ⭐⭐⭐⭐

单分支设计理念清晰，交叉归一化策略既简洁又有效。多模态扩展是亮点，展示了涌现能力。消融实验覆盖全面（预训练策略、缩放比例、文本表达方式）。文本引导风格化的巨大提升（4.82→56.71 SRA）令人印象深刻。

<!-- RELATED:START -->

## 相关论文

- [Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)
- [Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [End-to-End Multi-Modal Diffusion Mamba](end-to-end_multi-modal_diffusion_mamba.md)
- [CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [Rethinking Cross-Modal Interaction in Multimodal Diffusion Transformers](rethinking_cross-modal_interaction_in_multimodal_diffusion_transformers.md)

<!-- RELATED:END -->
