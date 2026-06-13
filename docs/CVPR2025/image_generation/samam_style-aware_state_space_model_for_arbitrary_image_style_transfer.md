---
title: >-
  [论文解读] SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer
description: >-
  [CVPR 2025][图像生成][风格迁移] 提出 SaMam，首个基于 Mamba 状态空间模型的任意图像风格迁移框架，通过风格感知 S7 块从风格嵌入预测 SSM 权重参数，配合锯齿形扫描和局部增强机制，在变换质量和效率之间取得最佳平衡。 - 全局有效感受野对风格迁移至关重要：(1) 大感受野能更好地捕获风格模式…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "风格迁移"
  - "状态空间模型"
  - "Mamba"
  - "全局感受野"
  - "高效推理"
---

# SaMam: Style-aware State Space Model for Arbitrary Image Style Transfer

**会议**: CVPR 2025  
**arXiv**: [2503.15934](https://arxiv.org/abs/2503.15934)  
**代码**: 无  
**领域**: 图像生成 / 风格迁移  
**关键词**: 风格迁移, 状态空间模型, Mamba, 全局感受野, 高效推理

## 一句话总结

提出 SaMam，首个基于 Mamba 状态空间模型的任意图像风格迁移框架，通过风格感知 S7 块从风格嵌入预测 SSM 权重参数，配合锯齿形扫描和局部增强机制，在变换质量和效率之间取得最佳平衡。

## 研究背景与动机

- 全局有效感受野对风格迁移至关重要：(1) 大感受野能更好地捕获风格模式；(2) 更多像素参与锚点像素的风格变换
- CNN 方法通过堆叠卷积层扩大感受野，计算开销大；Transformer 方法虽获得全局感受野但计算复杂度为二次方
- 扩散模型虽生成质量高，但需要大量迭代步骤，效率本质上未解决
- 在风格迁移任务中，全局感受野与计算效率的矛盾一直未被根本解决
- Mamba 状态空间模型以线性复杂度实现长距离依赖建模，为解决该矛盾提供了可能
- 但现有 SSM 存在局部像素遗忘（1D 展平导致空间邻近像素在序列中距离远）、通道冗余和空间不连续性问题
- 标准 SSM 的参数 $\mathbf{A}$、$\mathbf{D}$ 来自固定嵌入空间，无法根据不同风格动态调整

## 方法详解

### 整体框架

SaMam 由风格 Mamba 编码器、内容 Mamba 编码器和风格感知 Mamba 解码器组成。编码器将内容图像 $\mathbf{I_c}$ 和风格图像 $\mathbf{I_s}$ 编码为内容特征 $\mathbf{E_c}$ 和风格嵌入 $\mathbf{E_s}$。风格嵌入作为条件信息适配解码器参数，最终生成风格化图像 $\mathbf{I_{cs}}$。编码器和解码器均基于 VMamba 的 SS2D 块构建，并增加局部增强和锯齿形扫描改进。

### 关键设计

**1. 风格感知 S7 块 (Style-aware S6 Block)**
- **功能**: 将风格信息注入 SSM 的状态更新过程，使模型能根据不同风格动态调整行为
- **核心思路**: 与标准 S6 块不同，S7 块从风格嵌入 $\mathbf{E_s}$ 预测 SSM 的关键参数 $\mathbf{A}$ 和 $\mathbf{D}$：$\mathbf{A}, \mathbf{D} = \text{Embedder}(\mathbf{E_s})$。$\mathbf{A}$ 经离散化后展开为全局卷积核，$\mathbf{D}$ 作为通道级缩放因子。两者的风格依赖性使 SSM 在隐状态更新中同时考虑内容和风格
- **设计动机**: (1) 标准 S6 块仅基于内容更新隐状态，忽略了风格的影响；(2) $\mathbf{A}$ 通过离散化具有选择性能力，从风格嵌入预测可实现风格感知的选择性；(3) 风格依赖的全局卷积核在保持并行计算效率的同时实现风格适配

**2. 锯齿形扫描 (Zigzag Scan)**
- **功能**: 保持 2D 图像 token 序列的空间和语义连续性
- **核心思路**: 从 4 个顶点出发，以锯齿形（Z字形）路径遍历图像，而非逐行或逐列的直线扫描。首个顺时针列（或行）作为起始扫描线。这确保相邻行/列之间的 token 在序列中保持邻近
- **设计动机**: 传统行扫描在换行时产生空间不连续，导致 SSM 的衰减参数 $\bar{\mathbf{A}}$ 在相邻 token 间产生突变，造成语义不连续和不自然的风格化纹理。锯齿形扫描消除了换行跳跃，保持平滑的衰减过渡

**3. 风格感知模块组 (SAIN + SConv + SCM)**
- **功能**: 在多个层面将风格信息融入内容特征处理
- **核心思路**: (1) SAIN（风格感知实例归一化）：从 $\mathbf{E_s}$ 预测均值 $\gamma$ 和方差 $\beta$ 进行特征级归一化，传递全局风格属性；(2) SConv（风格感知卷积）：从 $\mathbf{E_s}$ 生成深度卷积核 $K \in \mathbb{R}^{C \times 1 \times k_w \times k_h}$，保留风格图像的局部几何结构；(3) SCM（风格感知通道调制）：从 $\mathbf{E_s}$ 生成 sigmoid 调制系数 $v \in \mathbb{R}^C$，进行通道级特征适配。SAIN 和 SCM 的嵌入器初始化为输出零向量，使 SAVSSM 初始化为恒等函数
- **设计动机**: 风格迁移需要在全局属性（色调、对比度）和局部结构（笔触、纹理）两个层面进行风格注入。三个模块分别覆盖实例归一化（全局）、深度卷积（局部空间）和通道调制（特征选择）

### 损失函数

采用风格迁移的标准训练损失，包括内容损失（$\mathcal{L}_c$，保持内容结构）和风格损失（$\mathcal{L}_s$，匹配 Gram 矩阵统计），以及感知损失。

## 实验关键数据

### 主实验：定量比较

| 方法 | LPIPS↓ | FID↓ | ArtFID↓ | 类型 |
|------|--------|------|---------|------|
| AesPA | 0.405 | 20.24 | 29.84 | CNN |
| S2WAT | 0.426 | 23.43 | 34.83 | Transformer |
| StyleID | 0.480 | 24.49 | 37.73 | Diffusion |
| **SaMam** | **0.388** | **17.95** | **26.31** | **Mamba** |

*SaMam 在所有三个关键指标上全面领先所有类型的方法*

### 效率对比

| 方法 | 推理时间(ms) | MACs(G) |
|------|-------------|---------|
| StyTr2 (Transformer) | ~150 | ~80 |
| AesPA (CNN) | ~50 | ~40 |
| **SaMam** | **~35** | **~25** |

*SaMam 在推理速度和计算量上取得最优效率*

### 关键发现

- Mamba 架构以线性复杂度实现了优于 CNN 和 Transformer 方法的风格迁移质量
- 锯齿形扫描相比直线扫描有效减少了风格化纹理中的不自然伪影
- 局部增强模块（LoE）弥补了 SSM 展平操作导致的局部信息损失
- SAIN（实例归一化）比标准层归一化更适合风格迁移任务
- S7 块的风格感知参数预测比固定参数+后融合的方式更有效

## 亮点与洞察

1. **SSM 参数的风格依赖化**: 将 $\mathbf{A}$ 和 $\mathbf{D}$ 从固定参数变为风格条件参数，巧妙地将风格信息注入状态更新的核心机制
2. **空间连续性的系统解决**: 锯齿形扫描从根本上解决了 SSM 用于 2D 图像时的空间不连续问题
3. **效率-质量最优平衡**: 首次证明 Mamba 架构在风格迁移中的线性复杂度优势

## 局限与展望

- 仍需四方向扫描，导致计算量为单方向的 4 倍
- 对极端风格（如高度抽象的艺术作品）的泛化能力待探索
- 风格嵌入器的设计较为简单，更复杂的风格建模可能进一步提升质量
- Mamba 模型在视觉任务中的训练稳定性仍需关注

## 相关工作与启发

- 与 StyTr2 等 Transformer 方法相比，SaMam 以线性复杂度实现了更好的质量
- 风格感知参数预测的思路（S7 块）可推广到其他条件生成任务中的 SSM 应用
- 锯齿形扫描策略对所有图像级 SSM 应用都有参考价值

## 评分

⭐⭐⭐⭐ — 首次系统性地将 Mamba 应用于风格迁移，S7 块设计优雅，锯齿形扫描有效解决空间连续性问题。实验结果在质量和效率上均令人信服，定量指标全面领先。但在更多样的风格数据上的泛化能力有待进一步验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HSI: A Holistic Style Injector for Arbitrary Style Transfer](hsi_a_holistic_style_injector_for_arbitrary_style_transfer.md)
- [\[CVPR 2025\] OmniStyle: Filtering High Quality Style Transfer Data at Scale](omnistyle_filtering_high_quality_style_transfer_data_at_scale.md)
- [\[CVPR 2025\] StyleStudio: Text-Driven Style Transfer with Selective Control of Style Elements](stylestudio_text-driven_style_transfer_with_selective_control_of_style_elements.md)
- [\[CVPR 2025\] SCSA: A Plug-and-Play Semantic Continuous-Sparse Attention for Arbitrary Semantic Style Transfer](scsa_a_plug-and-play_semantic_continuous-sparse_attention_for_arbitrary_semantic.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](../../ICCV2025/image_generation/domain_generalizable_portrait_style_transfer.md)

</div>

<!-- RELATED:END -->
