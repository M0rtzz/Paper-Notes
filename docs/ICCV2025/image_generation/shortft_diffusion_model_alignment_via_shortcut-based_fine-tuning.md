---
title: >-
  [论文解读] ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning
description: >-
  [ICCV 2025][图像生成][扩散模型对齐] 提出 ShortFT，利用轨迹保持少步扩散模型构建去噪捷径（shortcut），将原本冗长的去噪链大幅缩短，从而实现完整的端到端奖励梯度反向传播，高效且有效地将扩散模型与奖励函数对齐。
tags:
  - ICCV 2025
  - 图像生成
  - 扩散模型对齐
  - 奖励微调
  - 去噪捷径
  - 轨迹保持蒸馏
  - 时间步感知 LoRA
  - 反向传播
---

# ShortFT: Diffusion Model Alignment via Shortcut-based Fine-Tuning

**会议**: ICCV 2025  
**arXiv**: [2507.22604](https://arxiv.org/abs/2507.22604)  
**代码**: [https://xiefan-guo.github.io/shortft](https://xiefan-guo.github.io/shortft)  
**领域**: image_generation  
**关键词**: 扩散模型对齐, 奖励微调, 去噪捷径, 轨迹保持蒸馏, 时间步感知 LoRA, 反向传播

## 一句话总结

提出 ShortFT，利用轨迹保持少步扩散模型构建去噪捷径（shortcut），将原本冗长的去噪链大幅缩短，从而实现完整的端到端奖励梯度反向传播，高效且有效地将扩散模型与奖励函数对齐。

## 研究背景与动机

扩散模型在文本到图像生成中表现卓越，但其最大似然训练目标与下游目标（美学、安全、文图一致性）之间存在冲突，因此需要通过奖励函数对模型进行对齐（alignment）。

现有对齐方法主要分三类：

**强化学习方法**（RLHF）：如 DDPO、DPOK 等，将去噪过程建模为 MDP 并用 RL 优化。但梯度方差大、效率低、对多样化 prompt 适应性差。

**反向传播方法（截断后半段）**：如 DRaFT-K、ReFL，只在去噪链后半段做反向传播。缺陷是忽略了早期阶段的直接监督，导致文图对齐次优。

**反向传播方法（部分梯度截断）**：如 AlignProp、DRTune，在去噪链中禁用部分梯度（只保留 $\alpha_t \mathbf{x}_t$ 的梯度，截断 $\beta_t \epsilon_\theta(\mathbf{x}_t, t)$ 的梯度），配合 gradient checkpointing 传播到早期步。但引入梯度偏差，优化不稳定，且计算代价高。

**核心矛盾**：标准 DDIM 需要约 50 步去噪，对应约 50 层的反向传播链，导致显存爆炸和梯度爆炸。已有方案要么截断链（信息丢失），要么截断梯度（引入偏差），都是妥协方案。

**本文洞察**：既然问题出在去噪链太长，那就从根本上缩短它——利用轨迹保持蒸馏得到的少步扩散模型（如 Hyper-SD 4-step）构建"去噪捷径"，将 50 步链缩短为若干步，从而实现完整梯度反向传播。

## 方法详解

### 整体框架

ShortFT 框架包含三个核心组件：

1. **去噪捷径（Denoising Shortcut）**：利用 Hyper-SD（4-step）在去噪链中跳过大量中间步
2. **时间步感知 LoRA（Timestep-aware LoRA）**：为不同时间段分配独立 LoRA 参数
3. **渐进式训练策略（Progressive Training）**：分阶段训练消除训练-推理偏差

优化目标与 DRaFT、AlignProp 等一致：

$$J(\theta) = \mathbb{E}_{\mathbf{c}, \mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{1})} \left[ \mathcal{R}\left(\text{Sample}(\theta, \mathbf{c}, \mathbf{x}_T), \mathbf{c}\right) \right]$$

通过梯度上升最大化可微奖励函数 $\mathcal{R}$。关键区别在于 Sample 过程使用了缩短后的去噪链。

### 关键设计一：去噪捷径

轨迹保持少步模型（如 Hyper-SD）通过蒸馏学会了在保持原始去噪轨迹的前提下跳过多个步骤。相较于简单的单步 DDIM 去噪（输出模糊、缺乏结构），Shortcut 输出与原始 SD 1.5 高度一致（HPS v2 偏差更小）。

具体实现中，将 50 步去噪链分为 $k=4$ 个段，LoRA 对应的时间步设置为 {761, 501, 261, 1}。去噪捷径分别在以下区间执行：

- 时间步 741 → 501
- 时间步 481 → 261
- 时间步 241 → 1

原本约 50 步的链被压缩为约 7-8 步，反向传播链大幅缩短。

### 关键设计二：时间步感知 LoRA

已有研究（eDiff-I）揭示了扩散模型去噪过程中的时间动态：早期阶段主要依赖文本 prompt 引导采样，后期逐渐依赖视觉特征去噪。因此全时间步共享 LoRA 参数并不合理。

ShortFT 的做法：

- 将去噪链分为 $k$ 段
- 除第一段外，每段最后一个时间步分配独立 LoRA
- 第一段保持全时间步共享 LoRA（与 DRaFT 一致）
- 后续段中连续无 LoRA 的时间步区间支持去噪捷径跳过
- 采用逐步叠加策略：LoRA $i$ 在 LoRA $i-1$ 基础上新增一个 LoRA 分支
- LoRA rank 设为 128，应用于 UNet 的前馈层和注意力层

**关键优势**：推理阶段不增加计算成本（只在对应时间步使用对应 LoRA），但训练阶段等效增加了模型容量，加速收敛。

### 关键设计三：渐进式训练策略

少步模型引入的捷径不可避免地存在误差（特别是细节上与原始模型不完全一致），直接使用会产生训练-推理偏差。

解决方案——分 $k$ 阶段渐进训练：

- **第 $i$ 阶段**：优化 LoRA $i$ 到 LoRA $k$ 的权重
- 第 $i$ 段及之前的去噪使用原始去噪链
- 第 $i$ 段之后引入去噪捷径
- 同时采用截断反向传播技术

**推理阶段**：不使用任何捷径，恢复原始去噪链生成最终图像。

### 损失函数

训练目标就是最大化奖励函数（梯度上升），没有额外重建损失。实验中使用的奖励函数包括：

- **HPS v2**：衡量人类对图像的偏好
- **PickScore**：基于用户选择的偏好模型
- **Symmetry**：鼓励图像具有水平对称特征
- **Combined reward**：PickScore × 10 + HPS v2 × 2 + Aesthetic × 0.05
- **Compressibility** 等其他奖励函数

正则化方面，不同于 DRTune 使用 CLIPScore，本文将 HPS v2 和 PickScore 按 1:10 比例混合作为联合正则项。

## 实验关键数据

### 主实验（Table 1：相同计算预算下客观评估）

| 方法 | HPS v2 ↑ | PickScore ↑ | Symmetry ↓ |
|------|----------|-------------|------------|
| SD 1.5 | 26.91 | 20.46 | 0.853 |
| DRaFT-LV | 33.13 | 23.35 | 0.418 |
| DRTune | 32.79 | 23.22 | 0.207 |
| **ShortFT** | **33.88** | **24.16** | **0.138** |

所有方法在 2×A800 上训练 6 小时，ShortFT 在三个指标上全面领先。

### 训练 10k 步结果

完整训练（10k 步使用 HPS v2 奖励）在 HPDv2 上取得 HPS v2 = **35.97**，超过 DRaFT-LV 的报告分数。

### 微调 SD vs. Hyper-SD（Table 2）

| 方法 | HPS v2 ↑ |
|------|----------|
| 微调 Hyper-SD | 32.92 |
| 微调 SD 1.5 (ShortFT) | **35.97** |

验证了微调基础模型优于微调蒸馏后的少步模型——蒸馏过程导致的性能退化和容量损失使得直接微调少步模型次优。

### 消融实验（Table 3）

| 配置 | HPS v2 ↑ | PickScore ↑ | Symmetry ↓ |
|------|----------|-------------|------------|
| w/o T-LoRA | 33.46 | 23.82 | 0.187 |
| w/o P-Training | 33.27 | 23.97 | 0.146 |
| **ShortFT** | **33.88** | **24.16** | **0.138** |

- 移除时间步感知 LoRA：HPS v2 降 0.42，PickScore 降 0.34
- 移除渐进训练：HPS v2 降 0.61，且生成图像出现局部不连贯细节（如不平滑的头发）

### 用户研究

11 名志愿者（5 名图像处理专家 + 6 名非专业）+ GPT-4V 辅助评估，ShortFT 在与 DRaFT-LV、DRTune 的对比中均获得多数票支持。

### 关键发现

1. ShortFT 不需要 gradient checkpointing，显存效率更高
2. 得益于短去噪链，学习速度比 DRTune（当前最高效方法）更快
3. 方法是架构无关的：在 UNet（SD 1.5）和 Transformer（SD 3）上均有效
4. 泛化性良好：在 HPDv2 上微调后仍能有效处理 Sora 的复杂 wild prompts
5. 适用于多种奖励函数：HPS v2、PickScore、Symmetry、Compressibility、Combined reward

## 亮点与洞察

1. **视角新颖**：不在"如何截断梯度"上修修补补，而是从根本上解决去噪链过长的问题。利用少步蒸馏模型作为捷径，是一个简洁而有效的 idea。
2. **完整梯度传播**：与 AlignProp/DRTune 的部分梯度截断不同，ShortFT 实现了真正的完整端到端反向传播，避免了梯度偏差。
3. **时间步感知 LoRA 设计精巧**：利用扩散去噪过程中不同阶段的语义差异，分配独立 LoRA 增加容量但不增加推理成本。
4. **训练-推理一致性**：渐进式训练策略正视捷径引入的误差并系统性地消除偏差，而不是忽略它。
5. **极强的实用性**：不需要 gradient checkpointing、显存友好、架构无关、奖励函数通用。

## 局限性

1. **依赖轨迹保持蒸馏模型**：方法前提是存在高质量的轨迹保持少步模型（如 Hyper-SD），如果基础模型没有对应的蒸馏版本则无法直接应用。
2. **仅验证了中等规模模型**：主要实验在 SD 1.5 上完成，SD 3 只展示了定性结果，缺乏在 SDXL 或更大模型上的系统验证。
3. **捷径引入的近似误差**：虽然通过渐进训练缓解，但捷径输出与原始模型之间的偏差在细节层面仍然存在，可能在某些精细任务上有影响。
4. **奖励函数本身的局限**：方法效果上限受限于奖励模型的质量，如果奖励模型有偏见则对齐方向也会有偏。
5. **缺少与 DPO 类方法的直接对比**：仅与反向传播类方法比较，没有与 Diffusion-DPO 等无奖励模型方法做定量对比。

## 相关工作与启发

- **DRaFT**（Clark et al., ICLR 2024）：截断反向传播只在后半段去噪链做优化，ShortFT 使用捷径避免了信息丢失。
- **DRTune**（Wu et al., ECCV 2024）：部分梯度截断 + gradient checkpointing，虽然能传播到早期但引入梯度偏差。ShortFT 无需 gradient checkpointing 且实现完整梯度。
- **AlignProp**（Prabhudesai et al., 2023）：同样截断部分梯度，ShortFT 提供了更根本的解决方案。
- **Hyper-SD**（Ren et al., 2024）：轨迹保持蒸馏方法，是 ShortFT 的关键基础设施。
- **eDiff-I**（Balaji et al., 2022）：揭示去噪过程中的时间动态，启发了时间步感知 LoRA 设计。
- **思路启发**：这种"借助蒸馏模型简化训练流程"的思路可以迁移到视频扩散模型对齐、3D 生成对齐等场景。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | 利用蒸馏模型作为训练捷径的视角新颖且实用 |
| 技术质量 | ⭐⭐⭐⭐ | 三个组件互补设计合理，消融充分 |
| 实验充分度 | ⭐⭐⭐⭐ | 多奖励函数、多架构验证，有用户研究 |
| 写作清晰度 | ⭐⭐⭐⭐⭐ | 图示精良，动机和方法阐述非常清晰 |
| 实用价值 | ⭐⭐⭐⭐ | 无需 gradient checkpointing，效率优势明显 |
| **总分** | **⭐⭐⭐⭐** | 扎实的工作，解决了扩散模型对齐中的核心效率瓶颈 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Memory-Efficient Fine-Tuning for Quantized Diffusion Model](../../ECCV2024/image_generation/memory-efficient_fine-tuning_for_quantized_diffusion_model.md)
- [TaxaDiffusion: Progressively Trained Diffusion Model for Fine-Grained Species Generation](taxadiffusion_progressively_trained_diffusion_model_for_fine-grained_species_gen.md)
- [Personalized Preference Fine-tuning of Diffusion Models](../../CVPR2025/image_generation/personalized_preference_fine-tuning_of_diffusion_models.md)
- [FreeMorph: Tuning-Free Generalized Image Morphing with Diffusion Model](freemorph_tuning-free_generalized_image_morphing_with_diffusion_model.md)
- [Focus-N-Fix: Region-Aware Fine-Tuning for Text-to-Image Generation](../../CVPR2025/image_generation/focus-n-fix_region-aware_fine-tuning_for_text-to-image_generation.md)

<!-- RELATED:END -->
