---
title: >-
  [论文解读] Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation
description: >-
  [ICCV 2025][图像生成][缝纫版型生成] 提出 SewingLDM，一个多模态条件潜空间扩散模型，通过扩展缝纫版型表示和两阶段训练策略，实现在文本、草图、体型条件控制下合复杂缝纫版型，并可无缝集成到 CG 仿真管线。 缝纫版型（Sewing Pattern）是工业界广泛使用的服装表示方式，因其兼容物理仿真和动画而在…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "缝纫版型生成"
  - "潜空间扩散模型"
  - "多模态条件"
  - "体型感知"
  - "CG管线"
---

# Multimodal Latent Diffusion Model for Complex Sewing Pattern Generation

**会议**: ICCV 2025  
**arXiv**: [2412.14453](https://arxiv.org/abs/2412.14453)  
**代码**: [项目页面](https://shengqiliu1.github.io/SewingLDM)  
**领域**: 扩散模型/服装生成  
**关键词**: 缝纫版型生成, 潜空间扩散模型, 多模态条件, 体型感知, CG管线

## 一句话总结

提出 SewingLDM，一个多模态条件潜空间扩散模型，通过扩展缝纫版型表示和两阶段训练策略，实现在文本、草图、体型条件控制下合复杂缝纫版型，并可无缝集成到 CG 仿真管线。

## 研究背景与动机

缝纫版型（Sewing Pattern）是工业界广泛使用的服装表示方式，因其兼容物理仿真和动画而在 CG 管线中具有天然优势。然而现有的缝纫版型生成方法面临以下挑战：

**复杂版型表示不足**：现有方法（如 NeuralTailor）仅支持直线和二次曲线两种边类型，无法表示现代服装设计中常见的三次曲线、圆弧线等复杂几何，也缺乏对附着约束（如领口、腰带的防滑约束）的建模。

**缺乏精细控制**：DressCode 等方法虽然支持文本生成，但在处理复杂服装描述（如一字肩礼服、方领衬衫）时失败率高，生成能力受限。参数化方法（如 GarmentCodeData）虽然支持复杂控制，但需要预定义模板和专业知识。

**忽略体型适配**：大多数方法仅在标准体型上训练，无法生成适合不同体型的定制服装。当把生成的版型穿戴到不同体型上时，会出现布料穿模、滑落等问题。

**3D 网格方法不兼容 CG 管线**：Wonder3D、RichDreamer 等 3D 网格生成方法可以生成视觉上美观的服装，但其闭合表面网格无法与现代 CG 生产流程集成，且穿戴时会出现严重的穿模现象。

## 方法详解

### 整体框架

SewingLDM 包含三个核心组件：
1. **扩展版型表示**：将每条边的特征从原始的低维扩展到 29 维，覆盖四种边类型、附着约束、缝合反转标志等
2. **紧凑潜空间压缩**：通过自编码器将高维版型表示压缩到有界紧凑的潜空间
3. **多模态条件扩散模型**：基于 DiT 架构，通过两阶段训练注入文本、草图、体型条件

### 关键设计

1. **扩展缝纫版型表示**：在原始表示基础上增加：三次曲线控制点 $C^b_{i,j} \in \mathbb{R}^4$、圆弧线参数 $C^r_{i,j} \in \mathbb{R}^3$、边类型标志 $E^t_{i,j,k}$（2-bit 表示 4 种边类型）、附着类型标志 $A_{i,j,k}$（3-bit 表示领口/腰带等约束）、缝合方向反转标志（防止仿真时缝合交叉）。每条边最终表示为 29 维向量：

    $E^f_{i,j} = V_{i,j} \oplus C_{i,j} \oplus C^b_{i,j} \oplus C^r_{i,j} \oplus S_{i,j} \oplus R_i \oplus T_i \oplus E^t_{i,j} \oplus E^m_{i,j} \oplus A_{i,j} \oplus M'_{i,j}$

   所有版型通过零填充统一到固定尺寸 $(max(N_p) \times max(N_i), 29)$。

2. **紧凑潜空间压缩**：训练自编码器将版型 $F$ 压缩到有界潜空间 $[-1, 1]$。编码后通过量化 $\hat{z} = \frac{round(n \times tanh(z))}{n}$ 使每个维度均匀分布在 $\{-1, -0.5, 0, 0.5, 1\}$（$n=2$），利于扩散模型学习分布。训练损失包括重建 MSE 损失 $\mathcal{L}_{rec}$、面板完整性损失 $\mathcal{L}_{panel}$、缝合精度损失 $\mathcal{L}_{stitch}$ 和新增的二元交叉熵损失 $\mathcal{L}_{BCE}$：

    $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{panel} + \lambda_3 \mathcal{L}_{stitch} + \lambda_4 \mathcal{L}_{BCE}$

3. **两阶段多模态条件注入**：

    - **第一阶段**：仅在文本条件下训练潜空间扩散模型（DiT 架构 + T5 tokenizer），使用 IDDPM 损失，建立基础生成能力
    - **第二阶段**：注入草图和体型条件。通过嵌入器提取草图和体型特征并拼接，送入轻量 Transformer 层融合（因为草图会随体型变化），然后通过统计归一化将融合特征 $\bm{F}_{bs}$ 对齐到潜变量特征 $\bm{F}_z$ 的分布：

    $\hat{\bm{F}}_z = \frac{(\bm{F}_{bs} - \bm{\mu}_{bs}) \times \bm{\sigma}_{bs}}{\bm{\sigma}_z + \epsilon} + \bm{\mu}_z + \bm{F}_z$

   仅微调注意力模块的输出层，保持对文本引导的响应。草图/文本条件以 25% 概率置零以支持单条件生成。

### 损失函数 / 训练策略

- 自编码器：$\lambda_1=5, \lambda_2=1, \lambda_3=1, \lambda_4=1$，训练 12 小时
- 第一阶段文本引导 LDM：IDDPM 损失，训练 2 天
- 第二阶段多模态条件：额外 10 小时收敛
- 数据集：GarmentCode 数据集 120,000 个版型，GPT-4 辅助标注文本，PiDiNet 提取草图

## 实验关键数据

### 主实验

**定量对比（生成效率、体型适配度、用户评价）**：

| 方法 | 运行时间↓ | 衣体距离↓(cm) | 用户评分↑ |
|------|---------|-------------|---------|
| RichDreamer | ~4 hours | 6.19 | 1.89 |
| Wonder3D | ~4 mins | 6.54 | 1.88 |
| Sewformer | ~3 mins | 5.45 | 2.10 |
| DressCode | ~3 mins | 3.69 | 3.56 |
| **SewingLDM (Ours)** | **~3 mins** | **2.20** | **4.60** |

**重建精度对比**：

| 方法 | Panel L2↓ | Panel Acc↑ | Edge Acc↑ | Stitch Acc↑ | 失败率↓ |
|------|----------|-----------|----------|------------|--------|
| SewFormer* | 12.3 | 79.4 | 44.7 | 2.8 | 4.3% |
| AE (Ours) | 0.64 | 99.8 | 88.5 | 90.8 | 0 |
| SewingLDM | 3.13 | 97.8 | 82.7 | 84.2 | 0 |

### 消融实验

**潜空间压缩形状消融**：

| 压缩形状 | 重建 | 生成 | 衣体距离↓ | Codebook使用率 |
|---------|------|------|---------|-------------|
| 无压缩 | ✓ | ✗ | - | - |
| 256×32, n=32 | ✓ | ✗ | - | 0% |
| 256×8, n=2 | ✓ | ✓ | 2.87 | 91% |
| **256×6, n=2** | **✓** | **✓** | **2.20** | **100%** |
| 256×4, n=2 | ✗ | - | - | - |

**多模态条件注入位置消融**：在浅层（block 0 后）注入效果最佳，深层注入会导致服装关键部件（袖子、腰带）丢失；同时微调 self-attention 和 cross-attention 的输出层效果优于单独微调任一。

### 关键发现

- SewingLDM 的衣体距离仅 2.20cm，远优于 3D 网格方法（>6cm）和其他版型方法
- 用户评价中本方法以 4.60/5.0 分大幅领先
- 紧凑潜空间（256×6, n=2）实现了 100% codebook 使用率，这是生成质量的关键
- 自编码器实现了厘米级精度重建（Panel L2=0.64cm），达到工业标准

## 亮点与洞察

- **首次实现体型感知的缝纫版型生成**：通过将体型作为条件融入扩散模型，生成的服装可直接适配不同体型，无需手动调整
- **CG 管线兼容**：生成的版型可直接用于物理仿真和动画，解决了 3D 网格方法的穿模问题
- **紧凑量化潜空间设计巧妙**：5 值均匀量化（$\{-1,-0.5,0,0.5,1\}$）既保证了重建精度，又使扩散模型易于学习
- **两阶段训练策略务实**：先建立文本基础能力，再微调注入额外模态，避免多模态干扰

## 局限与展望

- 无法处理拉链、口袋等特殊设计细节
- 复杂草图（如婚纱）的对齐存在困难
- 当前数据集来自 GarmentCodeData，覆盖的设计风格有限
- 未来可探索更全面的日常服装表示和更多条件输入

## 相关工作与启发

- NeuralTailor 的版型向量化方法是本文的基础，但其表示能力限制了复杂服装的生成
- DiT 架构的可扩展性使其适合不同规模的版型生成
- 本文的潜空间压缩策略（有界量化 + 高使用率）对其他结构化数据的生成任务有借鉴意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将 LDM 应用于缝纫版型是新颖的跨领域尝试，体型感知是重要创新
- **实验充分度**: ⭐⭐⭐⭐ 定量定性对比充分，包含用户研究和详尽消融
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 对数字服装设计行业有重要的实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space](motionstreamer_streaming_motion_generation_via_diffusion-based_autoregressive_mo.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](../../CVPR2025/image_generation/zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)
- [\[CVPR 2025\] WeGen: A Unified Model for Interactive Multimodal Generation as We Chat](../../CVPR2025/image_generation/wegen_a_unified_model_for_interactive_multimodal_generation_as_we_chat.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)

</div>

<!-- RELATED:END -->
