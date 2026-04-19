---
title: >-
  [论文解读] Memory-Efficient Fine-Tuning for Quantized Diffusion Model
description: >-
  [ECCV 2024][图像生成] 提出 TuneQDM，首个面向量化扩散模型的内存高效微调方法，通过多通道量化缩放更新和时间步感知缩放策略，在 4-bit 量化模型上实现了接近全精度模型的个性化生成效果。
tags:
  - ECCV 2024
  - 图像生成
---

# Memory-Efficient Fine-Tuning for Quantized Diffusion Model

**会议**: ECCV 2024  
**arXiv**: [2401.04339](https://arxiv.org/abs/2401.04339)  
**领域**: 图像生成

## 一句话总结

提出 TuneQDM，首个面向量化扩散模型的内存高效微调方法，通过多通道量化缩放更新和时间步感知缩放策略，在 4-bit 量化模型上实现了接近全精度模型的个性化生成效果。

## 研究背景与动机

随着扩散模型规模不断增长（SDXL 26亿参数、DALL-E 2 55亿参数），直接微调全精度模型需要巨大的内存和计算资源。模型量化（如 Q-Diffusion）可将参数压缩至 4-bit/8-bit，但量化后的模型能否直接进行下游任务微调是一个完全未被探索的问题。

作者构建了一个基线方案（Q-Diffusion 量化 + PEQA 微调 + DreamBooth 个性化），发现存在两个核心问题：

**通道间模式丢失 (P1)**: 全精度微调时权重更新呈现明显的通道间模式（inter-channel pattern），但基线仅更新通道内缩放因子，无法捕捉这些模式

**时间步感知能力不足 (P2)**: 扩散模型在不同去噪步骤中扮演不同角色（内容特征 vs 粗粒度特征 vs 噪声清除），但基线缺乏区分不同时间步的能力，导致要么主体保真度高但忽略文本提示，要么两者都无法满足

## 方法详解

### 整体框架

TuneQDM 在冻结量化整数权重的前提下，仅训练量化缩放参数，包含两个核心技术：
1. 多通道缩放更新 (Multi-Channel-wise Scale Update, MCSU)
2. 时间步感知缩放更新策略 (Timestep-Aware Scale Update, TAS)

### 关键设计

**1. 多通道缩放更新 (MCSU)**

标准量化的权重恢复为 $\hat{W}_f = s \cdot (W_q - z)$，PEQA 基线仅更新 per-channel 输出缩放因子 $s_{out} \in \mathbb{R}^m$。

MCSU 额外引入输入方向的缩放因子 $s_{in} \in \mathbb{R}^n$，将量化权重表示为可分离函数：

$$W_{tuned} = (s_{out} + \Delta s_{out}) \cdot (W_q^* - z^*) \cdot (s_{in} + \Delta s_{in})$$

仅需 (m+n) 个可训练参数（vs 全矩阵 m×n），就能捕捉到通道间的权重更新模式。

**2. 时间步感知缩放更新 (TAS)**

将总时间步 T 均匀分为 n 段，为每段独立维护一组量化缩放参数：

$$S_n = \{s_1, s_2, ..., s_n\}, \quad \mathcal{I}_n = \{I_i | (\frac{i \times T}{n}, \frac{(i+1) \times T}{n})\}$$

训练时根据时间步选择对应专家的缩放参数更新；推理时整数权重固定不变，仅切换缩放参数，内存开销极小。

**3. 整体流程**

1. 从 Q-Diffusion 量化检查点初始化权重和量化参数
2. 为每层初始化 $s_{in} \sim \mathcal{N}(1, 0.01)$
3. 冻结 $W_q$，仅训练 $s_{out}$ 和 $s_{in}$
4. 训练时根据时间步选择对应专家进行更新

### 损失函数

使用 DreamBooth 的标准扩散训练损失进行个性化微调。

## 实验关键数据

### 主实验

**单主体生成**

| 方法 | 位宽(W) | 模型大小 | 参数量 | DINO-I ↑ | CLIP-I ↑ | CLIP-T ↑ |
|------|---------|---------|--------|----------|----------|----------|
| 全精度 | 32 | 3.20GB | 859M | 0.431 | 0.746 | 0.316 |
| Baseline | 4 | 0.40GB | 0.33M | 0.519 | 0.787 | 0.313 |
| **TuneQDM** | **4** | **0.40GB** | **0.62M** | **0.551 (+6.2%)** | **0.802 (+1.9%)** | **0.306** |
| Baseline | 8 | 0.80GB | 0.33M | 0.581 | 0.824 | 0.300 |
| **TuneQDM** | **8** | **0.80GB** | **0.62M** | **0.578** | **0.816** | **0.307 (+2.3%)** |

4-bit 设置下 TuneQDM 的 DINO-I 提升 6.2%，用 8 倍压缩模型达到接近全精度的效果。

**无条件生成 (CIFAR-10)**

| 方法 | 位宽 | 参数量 | IS ↑ | FID ↓ |
|------|------|--------|------|-------|
| 全精度 | 32 | 35.8M | 9.00 | 4.53 |
| QLoRA (r=32) | 8 | 8.64M | 9.03 | 4.30 |
| QLoRA (r=2) | 8 | 0.57M | 9.03 | 4.15 |
| Baseline | 8 | 0.03M | 8.96 | 4.39 |
| **TuneQDM** | **8** | **0.13M** | **9.17** | **3.80** |

TuneQDM 以 0.13M 参数超越 QLoRA(r=32) 的 8.64M 参数，FID 从 4.30 降至 3.80。

### 消融实验

| 配置 | 位宽 | MCSU | TAS 专家数 | IS ↑ | FID ↓ |
|------|------|------|-----------|------|-------|
| 全精度 | 32 | - | - | 9.00 | 4.53 |
| Baseline | 8 | ✗ | 1 | 8.96 | 4.39 |
| + TAS only | 8 | ✗ | 2 | 9.19 | 4.24 |
| + MCSU only | 8 | ✓ | 1 | 8.97 | 4.33 |
| **TuneQDM** | **8** | **✓** | **2** | **9.17** | **3.80** |
| TuneQDM | 8 | ✓ | 4 | 9.02 | 4.15 |

MCSU + 2 专家 TAS 是最优配置；4 专家反而过拟合导致性能下降。

### 关键发现

- 全精度微调的权重变化率图显示清晰的通道间模式，基线完全丢失这些模式
- 2 个时间步专家是最优选择，与 P2 weighting 等研究吻合
- 仅存储缩放参数（~3MB/数据集），复用量化检查点即可切换不同任务
- 用户研究表明 TuneQDM 在提示保真度上显著优于基线
- 多主体生成相比全精度模型仍有性能差距，是当前方法的瓶颈

## 亮点与洞察

1. **开创性问题定义**: 首次研究量化扩散模型的微调问题，对产业落地有重要意义
2. **可分离缩放因数的洞察深刻**: 通过可视化权重变化比率发现通道间模式，并用外积结构 $s_{out} \cdot W_q \cdot s_{in}$ 优雅地解决
3. **极致参数效率**: 0.62M 可训练参数 vs 全模型 859M，压缩比超过 1000 倍
4. **即插即用**: 量化一次，缩放参数切换即可适配不同下游任务，无需重复量化

## 局限性

- 仅在 Stable Diffusion v1.5 上验证，SDXL 等更大模型的效果未知
- 多主体生成中仍有明显性能退化
- 仅支持 PTQ 量化后的微调，未探索 QAT 场景
- 4-bit 设置下 CLIP-T（文本保真度）略有下降，存在主体-提示的权衡

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次定义和解决量化扩散模型微调问题，问题设定有开创性
- **技术深度**: ⭐⭐⭐⭐ — 可分离缩放设计优雅，时间步感知策略有理论支撑
- **实验充分度**: ⭐⭐⭐⭐ — 单/多主体 + 无条件生成 + 用户研究 + 消融
- **写作质量**: ⭐⭐⭐⭐ — 动机分析（权重变化可视化）非常直观

<!-- RELATED:START -->

## 相关论文

- [MixDQ: Memory-Efficient Few-Step Text-to-Image Diffusion Models with Metric-Decoupled Mixed Precision Quantization](mixdq_memory-efficient_few-step_text-to-image_diffusion_models_with_metric-decou.md)
- [SMooDi: Stylized Motion Diffusion Model](smoodi_stylized_motion_diffusion_model.md)
- [LEGO: Learning EGOcentric Action Frame Generation via Visual Instruction Tuning](lego_learning_egocentric_action_frame_generation_via_visual_instruction_tuning.md)
- [ZigMa: A DiT-style Zigzag Mamba Diffusion Model](zigma_a_dit-style_zigzag_mamba_diffusion_model.md)
- [ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)

<!-- RELATED:END -->
