---
title: >-
  [论文解读] DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching
description: >-
  [CVPR2026][视频扩散模型加速] 提出 DisCa，首次将**可学习特征缓存**与**步骤蒸馏**相结合，通过轻量级神经预测器替代手工缓存策略，并设计 Restricted MeanFlow 稳定大规模视频模型蒸馏，在 HunyuanVideo 上实现 11.8× 加速且几乎无质量损失。
tags:
  - CVPR2026
  - 视频扩散模型加速
  - 特征缓存
  - 步骤蒸馏
  - MeanFlow
  - 可学习预测器
  - GAN
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching

**会议**: CVPR2026  
**arXiv**: [2602.05449](https://arxiv.org/abs/2602.05449)  
**代码**: 即将开源  
**领域**: 图像生成 / 视频生成加速  
**关键词**: 视频扩散模型加速, 特征缓存, 步骤蒸馏, MeanFlow, 可学习预测器, GAN训练

## 一句话总结

提出 DisCa，首次将**可学习特征缓存**与**步骤蒸馏**相结合，通过轻量级神经预测器替代手工缓存策略，并设计 Restricted MeanFlow 稳定大规模视频模型蒸馏，在 HunyuanVideo 上实现 11.8× 加速且几乎无质量损失。

## 背景与动机

1. **扩散模型计算开销巨大**：当前 SOTA 视频扩散模型（如 HunyuanVideo）需要数十步迭代去噪，单次生成 5 秒 704×704 视频耗时超过 1155 秒，部署成本极高。
2. **特征缓存虽免训练但存在天花板**：传统 training-free 特征缓存方法（如直接复用/插值）在高压缩比下不可避免地丢失语义和细节信息，语义分数下降可达 13-27%。
3. **步骤蒸馏在视频生成中表现不佳**：MeanFlow 等蒸馏方法在图像生成中效果出色，但直接应用于大规模视频模型时，JVP 运算的数值误差导致训练发散和严重的生成伪影。
4. **缓存与蒸馏不兼容**：蒸馏后模型的采样步骤稀疏，相邻步之间的特征差异显著增大，传统缓存方法（依赖步间相似性）在蒸馏模型上直接失效。
5. **手工预测函数能力有限**：TaylorSeer 等方法用 Taylor 展开预测特征演化，但这种基于特定先验假设的手工函数难以充分捕捉高维特征的复杂演化趋势。
6. **显存效率问题**：TaylorSeer、FORA 等方法需要为每层维护多阶缓存张量，在高分辨率长视频场景下 VRAM 消耗超过 120GB，即使 4 路序列并行也难以承受。

## 方法详解

### 整体框架

DisCa 包含两个核心模块：**Restricted MeanFlow**（稳定蒸馏）+ **Learnable Feature Caching**（可学习缓存）。整体流程为：先对 HunyuanVideo 做 CFG 蒸馏（去除 classifier-free guidance 的双次推理），再用 Restricted MeanFlow 完成步骤蒸馏压缩采样步数，最后训练轻量级预测器实现特征缓存加速。

### Restricted MeanFlow

- **问题**：原始 MeanFlow 设计目标为一步蒸馏，平均速度采样区间 $\mathcal{I} = (t-r) \in [0,1]$，对大规模视频模型而言过于激进，高压缩比部分导致训练不稳定。
- **方案**：引入限制因子 $\mathcal{R} \in (0,1)$，将采样区间裁剪为 $\mathcal{I} \in [0, \mathcal{R}]$，直接剪枝 MeanFlow 中过于激进的高压缩场景。
- **效果**：$\mathcal{R}=0.2$ 时，20 步语义分数比原始 MeanFlow 提升 5.7%，10 步提升 12.0%，有效消除畸变和伪影。

### 轻量级可学习预测器

- **推理**：每 $N$ 步中，第 1 步用完整的大模型 $\mathcal{M}$ 计算并初始化/刷新缓存 $\mathcal{C}$，后续 $N-1$ 步用轻量预测器 $\mathcal{P}$ 基于缓存进行快速预测。
- **架构**：预测器由少量 DiT Block 堆叠而成，参数量始终 < 大模型的 4%。
- **缓存设计**：仅维护**单个张量**作为全局缓存（无需逐层多阶缓存），大幅节省显存。

### 训练策略与损失函数

- **MSE 损失**：以大模型输出为 ground truth 监督预测器：$\mathcal{L}(\theta_p) = \mathbb{E}\|\mathcal{M}(x_{t'}, r', t') - \mathcal{P}(\mathcal{C}, x_{t'}, r', t')\|_2^2$
- **GAN 对抗训练**：引入基于谱归一化的多尺度判别器 + Hinge Loss，用大模型作为特征提取器 $\mathcal{F}$，在感知特征空间进行对抗训练，弥补高频细节和语义结构损失。
- **判别器损失** $\mathcal{L}_\mathcal{D}$：最大化真假样本的区分度。
- **预测器损失** $\mathcal{L}_\mathcal{P}$：MSE + $\lambda$ · 对抗损失（$\lambda=1.0$）。

## 实验关键数据

### 主实验：HunyuanVideo 上的加速对比（VBench 评测）

| 方法 | 加速比 | 峰值VRAM | 语义分↑ | 质量分↑ | 总分↑ |
|------|--------|----------|---------|---------|-------|
| 原始 50步 (无CFG蒸馏) | 1.0× | 99.2GB | 73.5 | 81.5 | 79.9 |
| TeaCache (l=0.4) | 9.22× | 97.7GB | 62.1 (-15.5%) | 78.7 | 75.4 |
| TaylorSeer (N=6,O=1) | 6.96× | 130.7GB | 63.7 (-13.3%) | 79.9 | 76.7 |
| Restricted MeanFlow 9步 | 10.7× | 97.2GB | 67.8 (-7.8%) | 81.0 | 78.4 |
| **DisCa (R=0.2, N=3)** | **8.84×** | **97.6GB** | **70.3 (-4.4%)** | **81.8** | **79.5** |
| **DisCa (R=0.2, N=4)** | **11.8×** | **97.6GB** | **69.3 (-5.7%)** | **81.1** | **78.8** |

- 在 11.8× 加速下，总分仅下降 1.4%，语义下降 5.7%，质量几乎不降（-0.5%）。
- 对比 TaylorSeer 在 6.96× 加速下语义丢失 13.3%，DisCa 在更高加速比下表现远优。
- DisCa 显存 97.6GB 显著低于 TaylorSeer 的 130.7GB 和 FORA 的 124.6GB。

### 消融实验

| 配置 | 语义分 | 质量分 | 总分 |
|------|--------|--------|------|
| 完整 DisCa | 69.3 | 81.1 | 78.7 |
| 去掉 Restricted MeanFlow | 65.2 (-5.9%) | 80.3 | 77.3 |
| 去掉 Learnable Predictor（用传统缓存） | 67.3 (-2.9%) | 80.5 | 77.9 |
| 去掉 GAN 对抗训练 | 68.5 (-1.2%) | 81.0 | 78.5 |

- Restricted MeanFlow 贡献最大（去掉后语义降 5.9%），稳定蒸馏是质量保障的基础。
- 可学习预测器相比传统缓存提升语义 2.9%，GAN 训练进一步补偿 1.2% 语义。

## 亮点

- **首创"可学习缓存 + 蒸馏兼容"范式**：用数据驱动的轻量神经网络替代手工缓存公式，开辟了 feature caching 的新方向。
- **Restricted MeanFlow 简洁有效**：仅通过一个限制因子 $\mathcal{R}$ 裁剪激进压缩场景，即可显著稳定大规模视频模型的蒸馏过程。
- **显存高效**：仅维护单张量全局缓存，预测器 < 4% 模型参数，在实际高分辨长视频场景下具备部署优势。
- **11.8× 加速近乎无损**：在 HunyuanVideo 上实现目前最高加速比，且总分仅降 1.4%，大幅超越所有对比方法。

## 局限与展望

- **仅在 HunyuanVideo 上验证**：未在其他视频扩散模型（如 CogVideoX、Wan 等）上做实验，泛化性有待验证。
- **预测器需额外训练**：虽参数量小，但仍需在目标模型上做训练（MSE + GAN），不再是纯 training-free 方案。
- **固定 N 的缓存模式**：N 值（每隔多少步刷新缓存）在推理时固定，缺乏自适应动态调整机制。
- **GAN 训练稳定性**：对抗训练天然存在不稳定风险，论文中虽展示了平稳的 loss 曲线，但在其他模型/数据上的稳健性未知。
- **Restricted MeanFlow 的 $\mathcal{R}$ 需手动调参**：不同模型/步数下最优 $\mathcal{R}$ 可能不同，缺乏自动选择策略。

## 相关工作对比

| 类别 | 代表方法 | 特点 | DisCa 优势 |
|------|----------|------|-----------|
| 直接缓存复用 | Δ-DiT, PAB, FORA | 免训练，直接复用特征 | 高压缩比下完全崩溃（语义降 20%+） |
| 自适应缓存 | TeaCache, AdaCache | 按时间步自适应决策 | 仍受限于手工策略，高压缩损失大 |
| 缓存+预测 | TaylorSeer | Taylor 展开预测特征 | 手工函数上限有限，且显存高（130GB） |
| 步骤蒸馏 | MeanFlow, Shortcut | 压缩采样步数 | 原始 MeanFlow 在视频模型上不稳定 |
| **DisCa** | 本文 | 可学习缓存 + 限制蒸馏 | 首次兼容两类方法，11.8× 近无损 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次提出可学习缓存与蒸馏兼容方案，Restricted MeanFlow 虽简单但有效
- 实验充分度: ⭐⭐⭐⭐ — VBench 多维度评测 + 消融完整，但仅单模型验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、图表丰富，数学推导呈现清楚
- 价值: ⭐⭐⭐⭐⭐ — 为视频扩散模型加速提供了实用性极强的新范式，11.8× 加速具有显著工业价值

<!-- RELATED:START -->

## 相关论文

- [PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [PerformRecast: Expression and Head Pose Disentanglement for Portrait Video Editing](performrecast_expression_and_head_pose_disentanglement_for_portrait_video_editin.md)
- [LinVideo: A Post-Training Framework towards O(n) Attention in Efficient Video Generation](linvideo_a_post-training_framework_towards_on_attention_in_efficient_video_gener.md)
- [The Devil is in the Details: Enhancing Video Virtual Try-On via Keyframe-Driven Details Injection](the_devil_is_in_the_details_enhancing_video_virtual_try-on_via_keyframe-driven_d.md)
- [I'm a Map! Interpretable Motion-Attentive Maps: Spatio-Temporally Localizing Concepts in Video Diffusion Transformers](interpretable_motion-attentive_maps_spatio-temporally_localizing_concepts_in_vid.md)

<!-- RELATED:END -->
