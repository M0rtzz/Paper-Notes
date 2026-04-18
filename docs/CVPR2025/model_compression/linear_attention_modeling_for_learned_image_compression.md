---
title: >-
  [论文解读] LALIC: Linear Attention Modeling for Learned Image Compression
description: >-
  [CVPR 2025][learned image compression] 首次将 RWKV 线性注意力引入学习图像压缩，提出 Bi-RWKV 变换块和 RWKV-SCCTX 熵模型，以线性复杂度实现全局感受野，BD-rate 超越 VTM-9.1 达 15.26%。
tags:
  - CVPR 2025
  - learned image compression
  - linear attention
  - RWKV
  - entropy modeling
  - transform coding
---

# LALIC: Linear Attention Modeling for Learned Image Compression

**会议**: CVPR 2025  
**arXiv**: [2502.05741](https://arxiv.org/abs/2502.05741)  
**代码**: [sjtu-medialab/RwkvCompress](https://github.com/sjtu-medialab/RwkvCompress)  
**领域**: model_compression  
**关键词**: learned image compression, linear attention, RWKV, Bi-RWKV, entropy modeling, rate-distortion

## 一句话总结

首次将 RWKV 线性注意力机制引入学习图像压缩，设计 Bi-RWKV 变换块实现线性复杂度的全局感受野特征提取，配合 RWKV 时空通道上下文熵模型，以较低复杂度超越 VTM-9.1 达 15.26% BD-rate。

## 研究背景与动机

**领域现状**: 学习图像压缩（LIC）已超越传统编解码器（JPEG、VVC），主要依赖非线性变换网络和可学习熵模型。Transformer 已成为主流骨干，但其二次方复杂度限制了高分辨率图像处理。

**现有痛点**: Swin-Transformer 方案需要窗口分割策略来近似全局注意力，局限了感受野；Mamba 等线性注意力模型已用于 NLP 但在图像压缩中探索不足。每一点编码增益都伴随着显著的复杂度增长。

**核心矛盾**: 高效的全局依赖建模 vs. 可接受的计算复杂度。

**本文切入角度**: 利用 RWKV 的线性注意力特性实现真正的全局感受野，同时保持线性计算复杂度。

**核心 idea**: 用 Bi-RWKV（双向 WKV 注意力 + Omni-Shift 局部卷积）替代 Transformer/CNN 作为变换和熵模型的基础模块。

## 方法详解

### 整体框架

遵循标准的非线性变换编码框架：
1. 分析变换 $g_a$ 将图像 $x$ 编码为隐表示 $y$
2. 超先验编码器 $h_a$ 提取超隐表示 $z$
3. RWKV-SCCTX 熵模型估计 $y$ 的条件高斯分布参数
4. 综合变换 $g_s$ 从量化隐表示 $\hat{y}$ 重建图像 $\hat{x}$
5. 所有变换网络中使用 Bi-RWKV 块替代传统 Transformer 块

### 关键设计

**1. Bi-RWKV 变换块**
- **功能**: 作为 $g_a$、$g_s$、$h_a$、$h_s$ 中的基础特征提取模块，每个块包含 Spatial Mix 和 Channel Mix 两个分支。
- **核心思路**: Spatial Mix 通过 BiWKV 注意力捕获全局空间依赖（线性复杂度），Channel Mix 通过 squared ReLU 隐式构建 MLP 实现通道融合；使用 Omni-Shift（重参数化 5×5 深度卷积）捕获 2D 局部上下文。
- **设计动机**: BiWKV 引入通道级衰减参数 $w$ 和当前 token 增强参数 $u$，根据距离自动平衡局部与全局依赖；ERF 可视化证实 RWKV 块实现了真正的全局感受野（优于 TCM 的窗口模式和 FAT 的局部增强模式）。

**2. RWKV 时空通道上下文熵模型（RWKV-SCCTX）**
- **功能**: 联合建模隐表示 $y$ 在空间和通道维度上的冗余。
- **核心思路**: 空间维度采用 checkerboard 掩码卷积分 anchor/non-anchor 两组；通道维度将 320 通道分为 5 个 chunk（16, 16, 32, 64, 剩余），用 Bi-RWKV 块建模已解码 chunk 与当前 chunk 的通道上下文；通道上下文使用不含 Omni-Shift 的 Channel Mix 模块保持 1×1 感受野以满足因果解码。
- **设计动机**: 前几个 chunk 通道数少但被后续 chunk 频繁引用，承载大部分关键信息；RWKV 的全局建模能力使其在通道上下文建模中优于纯卷积方案。

**3. BiWKV 注意力机制**
- **功能**: 在 AFT（Attention-Free Transformer）的 KV 线性注意力基础上，添加双向距离衰减和当前 token 增强。
- **核心思路**: $wkv_t = \frac{\sum_{i \neq t} e^{-(|t-i|-1)/T \cdot w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i \neq t} e^{-(|t-i|-1)/T \cdot w + k_i} + e^{u+k_t}}$，最终输出经 sigmoid 门控的 receptance 调制。
- **设计动机**: 双向处理适应 2D 图像的非因果特性；距离衰减让近邻 token 获得更高权重，兼顾局部精度和全局建模。

### 损失函数 / 训练策略

- 率失真损失: $L = \lambda \|x - \hat{x}\|^2 + R(\hat{z}) + R(\hat{y})$
- $\lambda \in \{0.0025, 0.0035, 0.0067, 0.0130, 0.0250, 0.0483\}$（MSE 优化）
- Adam 优化器，初始学习率 $10^{-4}$，衰减到 $10^{-5}$，512×512 crop 微调
- 训练集: OpenImages 前 400K 图像，RTX 4090 GPU

## 实验关键数据

### 主实验——BD-rate（PSNR，锚定 VTM-9.1）

| 方法 | 解码时间(s) | FLOPs(G) | Params(M) | Kodak | CLIC | Tecnick |
|---|---|---|---|---|---|---|
| ELIC | 0.120 | 332 | 33.3 | -7.02% | -1.19% | -7.64% |
| MambaVC | 0.222 | 393 | 47.9 | -9.73% | - | - |
| TCM-large | 0.151 | 701 | 75.9 | -11.73% | -9.41% | -10.93% |
| FAT | >10.0 | 245 | 69.8 | -14.56% | -10.79% | -14.40% |
| MLIC++ | 0.268 | 443 | 83.3 | -15.02% | -14.45% | -17.21% |
| **LALIC（本文）** | **0.150** | **286** | **63.2** | **-15.26%** | **-15.41%** | **-17.63%** |

### 消融实验

| 配置 | FLOPs(G) | Params(M) | BD-rate |
|---|---|---|---|
| 2,2,2,2 + Conv SCCTX | 164 | 27.6 | 0.00% |
| 2,4,6,6 + Conv SCCTX | 239 | 42.6 | -1.68% |
| 2,4,6,6 + Conv Plus SCCTX | 304 | 62.1 | -2.74% |
| 2,4,6,6 + **RWKV SCCTX** | 286 | 63.2 | **-3.50%** |

注意力机制消融：

| 注意力 | ΔFLOPs(G) | Loss |
|---|---|---|
| AFT | 0.60 | 0.5657 |
| AFT + Shift | 4.91 | 0.5604 |
| BiWKV + Shift | 6.80 | **0.5551** |

### 关键发现

1. **LALIC 以最低参数量在 >10% 节省的方法中实现最快解码**：150ms 解码时间 + 63.2M 参数。
2. **RWKV-SCCTX 优于 Conv Plus SCCTX**：在几乎相同参数量下 BD-rate 额外降低 0.76%，且 FLOPs 更低。
3. **高分辨率优势明显**：在 CLIC（2K）和 Tecnick（1K）上的增益大于 Kodak（768×512），验证了全局建模对高分辨率的价值。
4. **ERF 分析表明 RWKV 实现了真正的全局感受野**，且降低了隐表示的局部相关性。

## 亮点与洞察

- 首个将 RWKV 线性注意力成功应用于学习图像压缩的工作
- ERF 可视化直观展示了 RWKV vs Transformer vs CNN 的感受野差异
- 熵模型中巧妙地移除 Omni-Shift 以保持因果性，体现了对解码过程的理解
- 编解码延迟在 SOTA 方法中具有竞争力，实际可部署性强

## 局限与展望

- 编码时间（274ms）相对较长，可探索更轻量的分析变换
- 仅探索了 RWKV 一种线性注意力，未与 Mamba、RetNet 等做系统对比
- Channel Mix 中的 squared ReLU 可考虑替换为其他激活函数
- 未探索 RWKV 在视频压缩中的应用潜力
- 端到端 MS-SSIM 优化的结果未做深入分析

## 相关工作与启发

- MambaVC 首先将 SSM 引入图像压缩，走线性注意力路线；LALIC 表明 RWKV 在该任务上更优
- TCM 和 FAT 分别代表窗口 Transformer 和频率感知 Transformer 路线，RWKV 以更低复杂度超越两者
- 启发：线性注意力在低级视觉任务中的优势可能推广到超分辨率、去噪等像素级任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次 RWKV + 图像压缩，Bi-RWKV 块和 RWKV-SCCTX 设计合理
- **实验充分度**: ⭐⭐⭐⭐ 三个数据集 + 详细消融 + ERF/相关性可视化
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，可视化丰富
- **价值**: ⭐⭐⭐⭐ 在效率-性能权衡上取得了新的 SOTA，具有实际部署价值
