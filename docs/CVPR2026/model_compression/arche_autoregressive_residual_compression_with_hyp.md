---
title: >-
  [论文解读] ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation
description: >-
  [CVPR 2026][模型压缩][自回归熵模型] 在全卷积架构内统一层级超先验、Masked PixelCNN 空间自回归、通道条件建模和 SE 通道激励，不依赖 Transformer 或循环组件…
tags:
  - "CVPR 2026"
  - "模型压缩"
  - "自回归熵模型"
  - "超先验"
  - "Squeeze-and-Excitation"
  - "残差预测"
  - "率失真优化"
---

# ARCHE: Autoregressive Residual Compression with Hyperprior and Excitation

**会议**: CVPR 2026  
**arXiv**: [2603.10188](https://arxiv.org/abs/2603.10188)  
**代码**: [GitHub](https://github.com/sof-il/ARCHE)  
**领域**: 学习式图像压缩  
**关键词**: 自回归熵模型, 超先验, Squeeze-and-Excitation, 残差预测, 率失真优化  

## 一句话总结

在全卷积架构内统一层级超先验、Masked PixelCNN 空间自回归、通道条件建模和 SE 通道激励，不依赖 Transformer 或循环组件，以 95M 参数和 222ms 解码时间实现相对 Ballé 基线 48% BD-Rate 降低并超越 VVC Intra 5.6%。

## 研究背景与动机

**领域现状**：端到端学习式图像压缩已超越传统编解码器（JPEG/JPEG2000），通过联合优化分析变换、量化和熵模型实现更优的率失真权衡。近年来 Transformer/注意力架构和混合熵模型不断刷新性能上限。

**现有痛点**：(1) Transformer/注意力型压缩模型计算量大、推理慢，部署困难；(2) ConvLSTM 上下文模型需跨大区域维护隐状态，严格串行解码延迟高；(3) 纯通道自回归（Minnen & Singh）丢失空间局部相关性，纯空间自回归有解码瓶颈。

**核心矛盾**：建模精度与计算效率之间的权衡——更复杂的模型能更好地估计潜在表示的概率分布，但推理成本和参数量也随之暴涨。

**本文目标** 在不依赖 Transformer 或循环组件的前提下，通过纯卷积架构达到 SOTA 率失真效率，同时保持参数量和推理速度可控。

**切入角度**：将层级超先验、掩码空间自回归、通道条件建模和 SE 激励四种互补的概率建模组件统一到单一 VAE 框架中，各组件分工协作而非互相替代。

**核心 idea**：与其追求更大更深的模型，不如在卷积框架内更精细地建模全局/空间/通道三个维度的依赖关系。

## 方法详解

### 整体框架

ARCHE 基于 VAE 框架：分析变换 $g_a$ 将输入映射为潜在表示 $y$，合成变换 $g_s$ 从量化表示 $\hat{y}$ 重建图像。核心在于熵模型的层级设计：超先验提供全局统计 → Masked PixelCNN 上下文精化局部概率 → 通道条件捕获跨通道依赖 → SE 激励自适应加权通道 → LRP 修正量化误差。潜在表示 $y$ 沿通道维被分为 10 个切片顺序解码，每个切片有独立的条件变换和 LRP 子模块。

### 关键设计

1. **自回归超先验 + Masked PixelCNN 上下文**
    - 超分析变换 $h_a$ 将 $y$ 映射为侧信息 $z$，量化后传输；超合成变换 $h_s$ 从 $\hat{z}$ 重建条件先验参数（均值 $\mu$ 和尺度 $\sigma$）
    - 空间自回归先验用 Masked PixelCNN 在光栅扫描顺序下建模 $p(\hat{y}_i|\hat{y}_{<i}, \hat{z})$，Type A 掩码排除中心及之后位置确保因果性，Type B 包含中心位置
    - 多层掩码卷积堆叠配 sigmoid 非线性扩展感受野，相比 ConvLSTM 训练更稳定、推理可部分并行化
    - 超先验与上下文特征拼接后经参数网络（逐点卷积+小卷积核+非线性激活）产出最终高斯参数
2. **通道条件 + SE 激励**
    - 解码第 $c$ 通道时利用前 $c-1$ 通道的特征做轻量卷积建模，将联合概率 $p(\hat{y}_{i,c}|\hat{y}_{<i,c}, \hat{y}_{<c}, \hat{z})$ 的依赖空间从纯空间扩展到空间+通道
    - 切片变换内嵌 Squeeze-and-Excitation 块：global avg pooling 生成通道描述子 → FC(缩减比 16) → ReLU → FC → sigmoid 门控，自适应放大信息量大的通道、抑制冗余通道
    - 跨通道依赖通常比空间依赖更平滑，因此通道条件模块可保持轻量而不影响效果
3. **潜在残差预测 (LRP)**
    - 对量化后每切片预测修正：$\hat{y}'_m = \hat{y}_m + \lambda_{LRP} \cdot \text{softsign}(r_m)$
    - softsign 替代 tanh 提供更平滑梯度和有界输出，$\lambda_{LRP}$ 为可学习缩放因子
    - 显式补偿量化噪声，超先验和上下文模型无法完全消除的残差误差

### 损失函数 / 训练策略

$L = R + \lambda D$，其中 $R$ 为交叉熵码率（含 $z$ 的先验贡献和 $y|z$ 的条件贡献），$D$ 为 MSE。CLIC 数据集训练，随机 256×256 裁剪归一化至 [0,1]。8 组 $\lambda \in \{0.001, 0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1\}$ 覆盖不同码率点。Adam $lr=10^{-4}$，400 epochs，batch=8。训练时量化用均匀加性噪声近似以保持梯度可传播。潜在深度 320，10 切片，超先验深度 192，SE 缩减比 16。TensorFlow 2.11 + TFC 实现，RTX 3080 训练和测试。

## 实验关键数据

### 主实验

| 方法 | BD-Rate vs Ballé (Kodak) | BD-Rate vs VVC (Kodak) | 参数量 | 解码时间 |
|------|-------------------------|----------------------|--------|---------|
| Minnen et al. | -8.00% | +90.61% | 95.8M | 591ms |
| Minnen & Singh | -16.28% | +63.55% | 121.7M | 249ms |
| WeConvene | -6.92% | +92.47% | — | — |
| Iliopoulou et al. | -24.22% | +30.19% | 124.3M | 265ms |
| **ARCHE** | **-48.01%** | **-5.61%** | **95.4M** | **222ms** |

Tecnick 数据集：ARCHE -44.89% vs Ballé，-10.28% vs VVC Intra，趋势一致。

### 消融实验

| 变体 | 效果 |
|------|------|
| 去掉全部 AR 组件 | 退化为纯超先验模型，性能损失最大 |
| 去掉 Masked Context Model | 显著劣化，空间上下文对局部概率估计至关重要 |
| 去掉 SE | 低比特率下中等下降，通道加权对细粒度结构保留重要 |
| 切片数 2→10 | BD-Rate 增益从 ~5% 升至 >11%，再增加边际递减 |
| GMM 替代单高斯 | 无显著提升，说明条件建模已充分捕获潜在统计 |
| Checkerboard 替代 PixelCNN | 训练快 58% 但率失真劣化（尤其低比特率），推理反慢 15% |

### 关键发现

- 各组件贡献互补而非冗余，同时去除造成累积性能下降
- 10 切片为最佳平衡点：进一步切分增益边际递减但计算开销线性增长
- ARCHE 在 Kodak 和 Tecnick 两个数据集上均一致超越 VVC Intra

## 亮点与洞察

- 纯卷积架构超越 VVC Intra 且参数/速度优于多数学习式方法，有力证明精心设计的 CNN 仍有竞争力
- 各组件在概率框架内协同工作（消融证实），体现"分工建模"优于"单一复杂模块"的设计哲学
- 视觉对比在低比特率下展现更锐利边缘和更自然色彩过渡，纹理细节保留优于 VVC
- 从 LSTM 上下文升级到 Masked PixelCNN 的设计选择带来训练稳定性和速度的双重提升

## 局限与展望

- 222ms 解码时间对实时视频仍显不足，可探索 block-wise 半并行解码策略
- 仅优化 MSE，引入感知指标（LPIPS/DISTS 等）可进一步提升视觉真实感
- 未探索任务导向压缩（压缩后直接用于分类/分割等下游任务）
- 更高分辨率图像上的扩展性和内存效率未验证
- 仅在自然图像数据集上评估，医学/遥感等领域的泛化性未知

## 相关工作与启发

- **vs Iliopoulou et al. [2025] 前作**：ARCHE 用 Masked PixelCNN 替换 LSTM 上下文 + 增加 SE 激励，BD-Rate 再降 ~24pp，参数减少 29M，解码快 43ms
- **vs Minnen et al. [2018]**：在联合 AR + 超先验基础上加入通道条件/SE/LRP，BD-Rate 多降 40pp，解码从 591ms → 222ms
- **vs WeConvene [ECCV24]**：小波域方案性能较弱（-6.92% vs Ballé），ARCHE 的空域联合建模更有效
- **设计哲学启发**："不追求更大模型而追求更好的依赖建模"，多层级先验（全局/空间/通道）的互补组合方式可迁移到其他概率建模任务

## 评分

- 新颖性: ⭐⭐⭐ 各组件已有前人工作，贡献在于精心融合与工程优化
- 实验充分度: ⭐⭐⭐⭐⭐ 双数据集 + 6 基线 + 完整消融 + 视觉对比 + 计算分析 + 附录变体分析
- 写作质量: ⭐⭐⭐⭐ 方法推导详尽，表格图表丰富，附录透明
- 价值: ⭐⭐⭐ 证明精心设计的 CNN 压缩仍有竞争力，对实际部署有参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](rdvq_differentiable_vq_image_compression.md)
- [\[CVPR 2026\] On the Robustness of Diffusion-Based Image Compression to Bit-Flip Errors](on_the_robustness_of_diffusion-based_image_compression_to_bit-flip_errors.md)
- [\[CVPR 2026\] UniComp: Rethinking Video Compression Through Informational Uniqueness](unicomp_rethinking_video_compression_through_informational_uniqueness.md)
- [\[CVPR 2026\] Generative Video Compression with One-Dimensional Latent Representation](generative_video_compression_with_one-dimensional_latent_representation.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)

</div>

<!-- RELATED:END -->
