---
title: >-
  [论文解读] EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration
description: >-
  [ICCV 2025][图像配准] 提出 EDFFDNet，采用指数衰减自由形变 (EDFFD) 替代传统 B-spline FFD 和 TPS 进行图像配准，配合自适应稀疏运动聚合器 (ASMA) 和渐进式相关策略，在 UDIS-D 数据集上以减少 70.5% 参数、32.6% 显存的代价实现 +0.5dB PSNR 提升。
tags:
  - ICCV 2025
  - 图像配准
  - 自由形变
  - 指数衰减基函数
  - 稀疏运动聚合
  - 无监督学习
---

# EDFFDNet: Towards Accurate and Efficient Unsupervised Multi-Grid Image Registration

**会议**: ICCV 2025  
**arXiv**: [2509.07662](https://arxiv.org/abs/2509.07662)  
**代码**: 无  
**领域**: 其他（图像配准）  
**关键词**: 图像配准, 自由形变, 指数衰减基函数, 稀疏运动聚合, 无监督学习

## 一句话总结

提出 EDFFDNet，采用指数衰减自由形变 (EDFFD) 替代传统 B-spline FFD 和 TPS 进行图像配准，配合自适应稀疏运动聚合器 (ASMA) 和渐进式相关策略，在 UDIS-D 数据集上以减少 70.5% 参数、32.6% 显存的代价实现 +0.5dB PSNR 提升。

## 研究背景与动机

### 问题定义

图像配准是计算机视觉中的基础任务，需要建立不同条件下拍摄的图像之间的空间对应关系。在实际场景中，图像往往包含深度差异（depth disparities），即场景中存在多个平面或前后景物体，这使得单一单应性变换 (homography) 无法准确对齐所有区域。

### 已有方法的不足

**单一 homography 的局限性**：受平面假设约束，在非平面场景中本质上无法实现精确对齐

**多网格 homography (Multi-grid)** 方法仅是 homography 的扩展，仍无法根本解决 homography 的表达能力限制

**TPS (薄板样条)** 构建全局光滑变形场，缺乏局部支撑，在需要显著局部变形的场景中表现不佳

**B-spline FFD** 虽然具有更好的局部性，但三次 B 样条基函数计算开销大、分段计算阻碍 GPU 并行、基乘积运算需要两维度独立计算

**MLP 运动聚合器**参数量大：UDIS++ 使用的 MLP 聚合器有 68.9M 参数，部署受限

**全局相关计算**在局部细化阶段引入远距离干扰，降低精度

### 核心动机

需要一种既有 B-spline FFD 的局部性优势、又计算高效且 GPU 友好的变形模型，同时需要轻量级的运动聚合方案和合理的相关计算策略。

**核心 idea**：用指数衰减函数替代三次 B 样条作为 FFD 基函数，结合分组线性层实现稀疏运动聚合，并采用粗到细的渐进相关策略。

## 方法详解

### 整体框架

EDFFDNet 包含三个主模块：(1) 多尺度特征提取器 (MFE)，基于 ResNet50 提取目标图像和参考图像的 4/8/16 倍下采样特征；(2) 全局 homography 估计模块，使用全局相关进行初始对齐；(3) 局部细化模块（1-2 个阶段），使用 EDFFD 模型进行局部变形估计。最终变形场由全局 homography + 各阶段残差位移叠加得到。

### 关键设计

#### 1. **指数衰减自由形变 (EDFFD)**

- **功能**：用指数衰减函数替代三次 B 样条基函数来计算控制点对空间点的影响
- **核心公式**：

$$\mathbf{x}' = \mathbf{x} + \sum_{m=0}^{M_i}\sum_{n=0}^{N_i} \Delta\mathbf{p}_{m,n} \exp(-r_{m,n}/(\theta\eta))$$

其中 $r_{m,n} = \|\mathbf{x} - \mathbf{p}_{m,n}\|_2$ 是欧几里得距离，$\theta$ 控制衰减速率，$\eta$ 是网格间距。

对比传统 B-spline FFD 的逐段三次多项式：

$$\beta^3(u) = \begin{cases} \frac{2}{3} - |u|^2 + \frac{|u|^3}{2}, & 0 \leq |u| \leq 1 \\ \frac{(2-|u|)^3}{6}, & 1 \leq |u| < 2 \end{cases}$$

- **设计动机**：
    - **简化影响度量**：直接用欧几里得距离替代两维度分别计算基乘积
    - **计算效率高**：指数函数提供 $C^\infty$ 光滑性，计算开销低于三次多项式，GPU 硬件优化的超越函数单元进一步加速
    - **并行兼容**：非分段特性实现全空间并行计算，避免 B 样条条件分支阻碍 GPU 利用率
    - **局部性保持**：指数函数自然随距离显著衰减
    - 实验中 $\theta = 0.75$ 为最优平衡点

#### 2. **自适应稀疏运动聚合器 (ASMA)**

- **功能**：替代 MLP 进行运动参数聚合，将密集交互转换为稀疏交互
- **核心结构**：两层分组线性层 (GLL) + 一个线性层
    - GLL 将输入特征 $\mathbf{F}_c \in \mathbb{R}^{C_c}$ 分为 $N_g$ 组
    - 每组独立经过线性变换：$\mathbf{F}'_{g,k} = \mathbf{W}_k(\mathbf{F}_{g,k}) + \mathbf{b}_k$
    - 拼接后经 ReLU，最终线性层自适应融合并输出运动参数
- **设计动机**：受深度可分离卷积启发，分组线性层将密集连接变为稀疏连接，$N_g = 8$ 时参数减少 66.6%（23.0M vs 68.9M），精度反而提升（PSNR 25.93 vs 25.87）

#### 3. **渐进式相关策略**

- **功能**：全局 homography 阶段使用全局相关计算；局部细化阶段使用局部相关
- **全局相关**：patch-to-patch 相关，提取 $K \times K$ 密集 patch 作为卷积核

$$\mathbf{C}^g_{(x_r,y_r,x_t,y_t)} = \sum_{i,j} \frac{\langle \mathbf{F}^{(d)}_{r,(x_r+i,y_r+j)}, \mathbf{F}^{(d)}_{t,(x_t+i,y_t+j)} \rangle}{\|\mathbf{F}^{(d)}_{r,(x_r+i,y_r+j)}\| \|\mathbf{F}^{(d)}_{t,(x_t+i,y_t+j)}\|}$$

- **局部相关**：在以 $\mathbf{p}'$ 为中心、半径为 $r$ 的局部区域内计算相关
- **设计动机**：全局 homography 需要大搜索范围应对低重叠场景，但在局部细化时全局相关引入远距离干扰。渐进策略在全局阶段提供充足搜索范围，在细化阶段聚焦局部，PSNR 提升 0.39 dB 且推理时间减少 29.8%

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{content}} + \omega \mathcal{L}_{\text{shape}}$$

- **内容对齐损失**：双向 L1 损失（前向 + 反向 warp），含全局和各局部阶段项，权重 $\lambda_0 = 1, \lambda_1 = 1.3, \lambda_2 = 1.7$
- **形状保持损失**：网格内约束（限制边长度）+ 网格间约束（鼓励非重叠区域的相邻边共线），$\omega = 10$
- **两阶段训练**：先训练全局 homography 模块 10 epoch，再联合训练全局+局部模块 100 epoch
- Adam 优化器，lr=$10^{-4}$，batch size 4

## 实验关键数据

### 主实验

UDIS-D 数据集 warp 精度对比：

| 方法 | PSNR↑ (Easy) | PSNR↑ (Hard) | PSNR↑ (Avg) | SSIM↑ (Avg) |
|------|-------------|-------------|-------------|-------------|
| SIFT+RANSAC | 27.75 | 18.46 | 22.98 | 0.758 |
| APAP | 27.01 | 19.54 | 23.00 | 0.773 |
| ELA | 29.87 | 19.68 | 24.47 | 0.821 |
| UDIS | 27.84 | 20.70 | 23.80 | 0.793 |
| MGDH | 29.52 | 21.20 | 24.89 | 0.817 |
| UDIS++ | 30.19 | 21.57 | 25.43 | 0.838 |
| **EDFFDNet** | **30.63** | **22.15** | **25.93** | **0.852** |
| **EDFFDNet-2** | **31.09** | **22.79** | **26.49** | **0.868** |

计算开销对比：

| 方法 | PSNR | 参数量(M) | 显存(GB) | 总时间(ms) |
|------|------|----------|---------|-----------|
| UDIS | 23.80 | 188.8 | 7.1 | 66.7 |
| MGDH | 24.89 | 16.4 | 5.3 | 90.3 |
| UDIS++ | 25.43 | 78.0 | 4.6 | 65.8 |
| **EDFFDNet** | **25.93** | **23.0** | **3.1** | **43.6** |
| **EDFFDNet-2** | **26.49** | **34.5** | **4.3** | **55.1** |

### 消融实验

**变形模型对比**：

| 模型 | PSNR | SSIM | Warp时间(ms) | 显存(GB) |
|------|------|------|------------|---------|
| TPS | 25.49 | 0.838 | 30.5 | 3.3 |
| B-spline FFD | 25.95 | 0.850 | 39.1 | 4.7 |
| **EDFFD** | **25.93** | **0.852** | **20.6** | **3.1** |

**运动聚合对比**（$N_g = 8$）：

| 方法 | 参数量(M) | PSNR | SSIM |
|------|----------|------|------|
| MLP | 68.9 | 25.87 | 0.850 |
| MLP (压缩4×) | 27.1 | 25.76 | 0.845 |
| **ASMA** | **23.0** | **25.93** | **0.852** |

### 关键发现

- **EDFFD ≈ B-spline FFD 精度，但 warp 快 47.3%、省 34% 显存**：指数衰减函数在保持局部性的同时大幅降低计算开销
- **ASMA 参数少 66.6% 但精度更高**：稀疏交互在运动聚合中比密集交互更有效，即使将 MLP 压缩到相近参数量 ASMA 仍然更优
- **局部性因子 $\theta$ 的平衡效应**：$\theta = 0.25$ 过小导致指数缓慢衰减影响范围过大，$\theta > 1.0$ 过大导致影响范围过小，最优为 $\theta = 0.75$
- **零样本跨数据集泛化强**：在 ScanNet 上 EDFFDNet-2 PSNR 24.32（vs UDIS++ 21.79），在 ETH3D 上 21.47（vs 19.41），优势显著
- **远快于传统方法**：处理 1500×2000 图像仅 0.078s（vs APAP 159.6s、LPC 2114.9s）

## 亮点与洞察

1. **基函数设计的工程洞察**：B-spline 的三个计算瓶颈（高阶多项式、基乘积、分段计算）被指数函数一次性解决，体现了对 GPU 计算特性的深刻理解
2. **稀疏优于密集的反直觉结论**：ASMA 的分组稀疏交互不仅减少参数，还提升精度，表明运动聚合中過度的全局交互反而引入噪声
3. **渐进式相关的合理性**：全局搜索用于粗对齐、局部搜索用于精细化的设计遵循了由粗到细的经典范式
4. **额外局部细化阶段的高效性**：EDFFDNet → EDFFDNet-2 仅增加 11.5ms warp 时间即获得 +0.56 dB PSNR 提升

## 局限与展望

1. **仅在 UDIS-D 上训练和主要评估**：虽然有跨数据集零样本测试，但训练数据的多样性有限
2. **未处理动态场景**：仅针对静态场景的配准，未考虑运动物体
3. **指数衰减函数的单一性**：未探索其他可能的衰减函数（如高斯、多项式衰减）
4. **控制点网格为均匀分布**：未探索自适应控制点分布以更好处理不同区域的变形需求
5. **两阶段训练**：全局和局部模块非端到端联合训练可能限制整体性能

## 相关工作与启发

- UDIS++ 的 TPS + MLP 方案是最直接的对比方法，EDFFDNet 在各方面均实现超越
- FFD 模型在医学图像配准中广泛使用，本文将其引入自然图像配准并改进基函数
- 分组线性层的思路来自深度可分离卷积（MobileNet），在运动聚合这一新场景中验证了其有效性

## 评分

- **新颖性**: ⭐⭐⭐⭐ — EDFFD 和 ASMA 的设计简洁但有效，工程创新与问题理解紧密结合
- **实验充分度**: ⭐⭐⭐⭐⭐ — 消融覆盖变形模型/聚合器/相关策略三个维度 + 跨数据集评估 + 计算效率对比 + 与传统方法速度对比
- **写作质量**: ⭐⭐⭐⭐ — 公式推导清晰，设计动机明确，对比分析充分
- **价值**: ⭐⭐⭐⭐ — 在图像配准领域实现了精度和效率的双重提升，实用价值高

<!-- RELATED:START -->

## 相关论文

- [Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)
- [Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](../../ICLR2026/others/completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)
- [GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning](../../ACL2025/others/getreason_enhancing_image_context_extraction_through_hierarchical_multi-agent_re.md)

<!-- RELATED:END -->
