---
title: >-
  [论文解读] Randomized Autoregressive Visual Generation
description: >-
  [ICCV2025][图像生成][自回归图像生成] 提出 Randomized AutoRegressive modeling (RAR)：在标准自回归训练中以随机排列输入序列并逐步退火回光栅扫描顺序，使模型学习双向上下文，在 ImageNet-256 上以 FID 1.48 刷新自回归图像生成 SOTA，同时保持与语言模型框架的完全兼容。
tags:
  - ICCV2025
  - 图像生成
  - 自回归图像生成
  - 随机排列
  - 双向上下文
  - 退火策略
  - ImageNet
---

# Randomized Autoregressive Visual Generation

**会议**: ICCV2025  
**arXiv**: [2411.00776](https://arxiv.org/abs/2411.00776)  
**代码**: [bytedance/1d-tokenizer](https://github.com/bytedance/1d-tokenizer)  
**领域**: image_generation  
**关键词**: 自回归图像生成, 随机排列, 双向上下文, 退火策略, ImageNet

## 一句话总结

提出 Randomized AutoRegressive modeling (RAR)：在标准自回归训练中以随机排列输入序列并逐步退火回光栅扫描顺序，使模型学习双向上下文，在 ImageNet-256 上以 FID 1.48 刷新自回归图像生成 SOTA，同时保持与语言模型框架的完全兼容。

## 研究背景与动机

### 自回归模型在视觉生成中的困境

自回归（AR）模型是大语言模型（GPT、Llama、Gemini）的核心框架，在 NLP 中取得了巨大成功。在视觉生成领域，AR 模型（如 LlamaGen、Open-MAGVIT2）也展现出竞争力，但与扩散模型和 masked transformer 相比仍有差距。根本原因在于：

- **单向上下文限制**：标准 AR 模型使用因果注意力，每个 token 只能看到前面的 token，无法利用双向上下文
- **视觉信号的本质差异**：文本具有天然的从左到右顺序，但图像没有固定的 token 排列顺序；视觉信号更低层次、冗余更多，双向建模更为关键
- **顺序偏差**：采用光栅扫描（raster scan）虽是主流，但引入了方向性偏差，限制了模型学习其他方向依赖的能力

### 已有解决方案的不足

- **VAR**：将 next-token 预测改为 next-scale 预测，虽引入了尺度内双向注意力，但偏离了标准 AR 范式
- **MAR**：将 MaskGIT 框架泛化为自回归定义，自然引入双向注意力，但同样不兼容传统语言模型框架
- 这些方法虽有效，但 **破坏了与语言模型的兼容性**，不利于未来多模态统一模型的构建

### RAR 的核心思路

能否在 **完全保留** AR 模型架构和训练范式的前提下，增强双向上下文学习能力？RAR 的回答是：通过随机排列输入序列训练，让每个 token 在各种可能的上下文中都被预测，从而学到双向表示——同时通过退火策略最终回归到光栅扫描顺序，获得最佳生成质量。

## 方法详解

### 1. 标准自回归建模回顾

给定离散 token 序列 $\mathbf{x} = [x_1, x_2, \cdots, x_T]$，标准 AR 模型的目标是最大化：

$$p_\theta(\mathbf{x}) = \prod_{t=1}^{T} p_\theta(x_t | x_1, \cdots, x_{t-1})$$

每个 token 仅依赖前面的 token，上下文建模是单向的。

### 2. 随机排列目标（Permutation Objective）

RAR 的核心改进是最大化所有可能分解顺序上的期望似然：

$$p_\theta(\mathbf{x}) = \mathbb{E}_{\tau \sim \mathcal{S}_T} \left[ \prod_{t=1}^{T} p_\theta(x_{\tau_t} | x_{\tau_{<t}}) \right]$$

其中 $\mathcal{S}_T$ 是索引序列 $[1,2,\cdots,T]$ 的全部排列集合，$\tau$ 是随机采样的一个排列。由于模型参数 $\theta$ 在所有排列中共享，每个 token 都会在训练过程中被暴露在各种可能的上下文中，从而学到双向依赖关系。

**关键区别**：与 BERT/MaskGIT 使用 mask token 不同，RAR 遵循排列目标（permuted objective）方法，以自回归方式在所有可能的分解顺序上训练。

### 3. 目标感知位置编码（Target-aware Positional Embedding）

随机排列训练带来一个问题：标准位置编码在某些场景下会失效。

**问题示例**：考虑两个排列 $\tau_a = [1,2,\cdots,T-1,T]$ 和 $\tau_b = [1,2,\cdots,T,T-1]$（仅最后两个位置交换）。在预测倒数第二个 token 时，两种排列的输入特征完全相同，但预测目标不同——模型无法区分该预测哪个 token。

**解决方案**：引入一组额外的目标感知位置编码 $\mathbf{p}_{ta} = [p_1, p_2, \cdots, p_T]$，将下一个要预测的 token 的位置编码加到当前 token 上：

$$\hat{\mathbf{x}}_\tau = [x_{\tau_1} + p_{\tau_2},\ x_{\tau_2} + p_{\tau_3},\ \cdots,\ x_{\tau_{T-1}} + p_{\tau_T},\ x_{\tau_T}]$$

这样每次预测都知道目标 token 的位置索引，消除了排列带来的歧义。训练结束后，由于模型已退火到光栅顺序，两组位置编码可以合并为一组，推理时不增加参数或计算量。

### 4. 随机性退火策略（Randomness Annealing）

完全使用随机排列训练存在两个问题：
- 排列空间巨大（256 个 token 就有 $256! > 10^{506}$ 种排列），模型可能花费大量能力学习应对不同排列而非提高生成质量
- 实验表明光栅扫描仍是最优生成顺序

因此 RAR 引入退火参数 $r$，控制使用随机排列与光栅顺序的概率：

$$r = \begin{cases} 1.0, & \text{if } epoch < start \\ 0.0, & \text{if } epoch > end \\ 1.0 - \frac{epoch - start}{end - start}, & \text{otherwise} \end{cases}$$

- 训练初期 $r=1$，完全随机排列，充分学习双向表示
- $r$ 线性衰减至 0，模型逐渐过渡到光栅扫描顺序
- 训练结束时完全使用光栅顺序，与标准 AR 推理一致

最优设置：总训练 400 epoch，$start=200$，$end=300$（前 200 epoch 纯随机，200-300 逐步退火，最后 100 epoch 纯光栅）。

## 实验关键数据

### 实验设置

| 配置 | 详情 |
|------|------|
| VQ Tokenizer | MaskGIT-VQGAN，下采样 16×，codebook 大小 1024 |
| 输入分辨率 | 256×256 → 256 个离散 token |
| 数据集 | ImageNet-1K 训练集（128 万张图） |
| 训练 | AdamW，batch 2048，400 epoch（250k steps） |
| 学习率 | 线性 warmup 至 4e-4（100 epoch），余弦衰减至 1e-5 |
| 模型规模 | RAR-B (261M) / RAR-L (461M) / RAR-XL (955M) / RAR-XXL (1.5B) |

### ImageNet-256 主要结果（FID↓）

| 方法 | 类型 | 参数量 | FID |
|------|------|--------|-----|
| DiT-XL/2 | Diffusion | 675M | 2.27 |
| MDTv2-XL/2 | Diffusion | 676M | 1.58 |
| MaskBit | Masked Trans. | 305M | 1.52 |
| MAR-H | Masked AR | 943M | 1.55 |
| VAR-d30-re | Scale AR | 2.0B | 1.73 |
| LlamaGen-3B-384 | AR | 3.1B | 2.18 |
| Open-MAGVIT2-XL | AR | 1.5B | 2.33 |
| **RAR-B** | **AR** | **261M** | **1.95** |
| **RAR-L** | **AR** | **461M** | **1.70** |
| **RAR-XL** | **AR** | **955M** | **1.50** |
| **RAR-XXL** | **AR** | **1.5B** | **1.48** |

### 核心发现

- **RAR-B (261M) 即超越 LlamaGen-3B (3.1B)**：参数量减少 91%，FID 1.95 vs 2.18
- **RAR-XXL (1.5B) FID 1.48**：首次让语言模型风格的 AR 生成器超越最佳扩散模型和 masked transformer
- **退火策略至关重要**：纯光栅 FID 3.08，纯随机 FID 3.01，最优退火 (200,300) FID 2.18
- **推理速度优势**：RAR-XL 生成 8.3 img/s，比 MaskBit (0.7) 快 11.9×，比 MAR-H (0.3) 快 27.7×（得益于 KV-cache 兼容）
- **扫描顺序消融**：6 种扫描顺序（光栅、螺旋、Z 曲线等）均表现良好，但光栅扫描仍最优（FID 2.18 vs 次优 z-curve 2.29）

## 亮点与洞察

1. **极简但强效的改进**：RAR 不改变模型架构，仅在训练时随机排列输入序列并退火，就能带来 FID 从 3.08 到 1.48 的巨幅提升
2. **保持语言模型兼容性**：与 VAR、MAR 不同，RAR 完整保留 next-token prediction 和因果注意力，可直接复用 LLM 的优化技术（KV-cache、vLLM 等）
3. **退火 > 纯随机/纯光栅**：实验清楚地表明，双向上下文学习和固定扫描顺序各有优势，退火策略巧妙地结合了二者
4. **目标感知位置编码设计精巧**：通过加入下一 token 的位置信息解决排列歧义问题，且推理时零开销（合并编码）
5. **参数效率出色**：261M 的 RAR-B 即超越 1.5B+ 的竞争方法，说明双向表示学习比单纯增大模型更有效
6. **与 XLNet 的联系**：排列目标源自 NLP 中的 XLNet，RAR 将其成功移植到视觉生成，验证了跨领域方法迁移的价值

## 局限与展望

1. **无法完全实现全局上下文**：生成时总有 token 先于其他 token 生成，无法真正看到全部上下文；论文提到 resampling/refinement 可能有帮助但未探索
2. **Tokenizer 限制**：使用的 MaskGIT-VQGAN tokenizer 较老（codebook 1024），若换用更强的 tokenizer（如 TiTok、MAGVIT-v2）可能进一步提升
3. **仅验证 ImageNet 256×256**：未在更高分辨率或其他数据集上验证
4. **退火超参数敏感性**：start 和 end epoch 的选择需要仔细调优，不同训练时长下最优区间可能不同
5. **排列空间采样效率**：256! 的排列空间远未被充分探索，是否有更高效的采样策略（如课程学习式排列）值得研究
6. **未探索文本-图像统一训练**：虽强调语言模型兼容性，但未展示与文本联合训练的实验

## 相关工作与启发

- **XLNet** [Yang+ 2019]：RAR 的排列目标直接源自 XLNet 在 NLP 中的排列语言模型，但加入了退火策略和目标感知位置编码以适应视觉任务
- **LlamaGen** [Sun+ 2024]：将 Llama 架构直接用于图像生成的 AR 方法，RAR 在相同框架下通过训练策略创新大幅超越
- **MAR** [Li+ 2024]：也尝试了随机顺序 AR 框架，但发现仅替换为随机顺序提升有限，RAR 的退火策略是关键突破点
- **VAR** [Tian+ 2024]：提出 next-scale prediction 实现尺度内双向注意力，但牺牲了语言模型兼容性
- **MaskGIT/MaskBit**：基于 mask prediction 的非自回归方法，生成质量优秀但推理速度慢且不兼容 KV-cache

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation](../../NeurIPS2025/image_generation/infinitystar_unified_spacetime_autoregressive_modeling_for_v.md)
- [Grouped Speculative Decoding for Autoregressive Image Generation](grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [Continuous Visual Autoregressive Generation via Score Maximization](../../ICML2025/image_generation/continuous_visual_autoregressive_generation_via_score_maximization.md)
- [Watermarking Autoregressive Image Generation](../../NeurIPS2025/image_generation/watermarking_autoregressive_image_generation.md)

<!-- RELATED:END -->
