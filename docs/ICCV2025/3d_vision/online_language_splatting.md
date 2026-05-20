---
title: >-
  [论文解读] Online Language Splatting
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 首个在 3DGS-SLAM 系统中实现**在线、近实时、开放词汇**语言建图的框架，通过高分辨率 CLIP 嵌入、两阶段在线自编码器压缩和颜色-语言解耦优化三项创新…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "SLAM"
  - "开放词汇"
  - "语言特征嵌入"
  - "CLIP"
  - "实时语义建图"
---

# Online Language Splatting

**会议**: ICCV 2025  
**arXiv**: [2503.09447](https://arxiv.org/abs/2503.09447)  
**代码**: [https://saimouli.github.io/onlineLang](https://saimouli.github.io/onlineLang)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, SLAM, 开放词汇, 语言特征嵌入, CLIP, 实时语义建图

## 一句话总结

首个在 3DGS-SLAM 系统中实现**在线、近实时、开放词汇**语言建图的框架，通过高分辨率 CLIP 嵌入、两阶段在线自编码器压缩和颜色-语言解耦优化三项创新，在精度超越离线 SOTA 的同时实现 40×–200× 的效率提升。

## 研究背景与动机

将语言特征嵌入 3D 场景表示是实现人机交互的关键能力。现有 Lang-GS 方法（如 LangSplat、Feature3DGS、LEGaussian）依赖 SAM+CLIP 进行**逐帧离线预处理**，每帧需数分钟生成像素级语言特征，严重限制了实际应用。

现实场景中的许多任务（服务机器人进入新环境、AR 系统即时交互）都要求**即时场景理解**。虽然已有 SLAM-GS 方法（MonoGS、SplaTAM 等）可以近实时地建立几何和纹理地图，但它们**不包含语言特征**。另一类使用预标注语义地图的方法则受限于闭合词汇表，缺乏开放词汇的灵活性。

核心矛盾在于：如何在保持开放词汇能力的同时，将高维语言特征**高效地**融入 3D 高斯表示？该问题涉及三个子挑战：

**实时高分辨率 CLIP 嵌入**：离线 SAM+CLIP 是运行时瓶颈

**在线场景下的开放词汇压缩**：在线方法无法在测试场景上预训练压缩器，存在域差距

**颜色-语言联合优化冲突**：两种模态偏好不同的高斯参数，联合优化导致两者性能均下降

## 方法详解

### 整体框架

系统基于 MonoGS 的 SLAM 框架，以 3D 高斯作为唯一建图元素。训练阶段包含三个核心模块：

- **高分辨率 CLIP 嵌入模块**：从 RGB 图像实时生成高分辨率语言特征图
- **两阶段 CLIP 压缩模块**：将 768 维 CLIP 特征压缩至 15 维
- **颜色-语言解耦优化**：分离 RGB 和语言的梯度路径

推理阶段，渲染出的低维语言图通过两阶段解码器重建完整 CLIP 特征，支持开放词汇查询定位目标物体。

### 关键设计 1：高分辨率 CLIP 嵌入

取代离线多遍 SAM+CLIP 流程，采用 ConvNeXt-L 像素级 CLIP 编码器生成粗粒度嵌入图（24×24×768），再通过轻量级**超分辨率解码器（SRD）**上采样到 192×192×768：

- SRD 利用编码器第 1、2 层的中间特征，通过两个卷积上采样块逐步增强分辨率
- SRD 在 COCO / Omnidata 数据集上以监督方式训练，标签由离线 SAM+CLIP 生成
- 训练损失：$\mathcal{L} = 0.8 \cdot \mathcal{L}_{\text{cosine}} + \mathcal{L}_{\text{L1}} + 0.01 \cdot \mathcal{L}_{\text{TV}}$
- 整个模块（CLIP 编码器 + SRD）在 RTX-3090 上仅需 **18ms/帧**，占用 1.6GB 显存，其中 SRD 仅 2ms
- 高分辨率特征图有效提升了小物体和远距离物体的定位精度，减少特征溢出（feature bleeding）

### 关键设计 2：两阶段在线 CLIP 压缩

768 维 CLIP 向量直接使用代价过高，需要有效压缩：

**第一阶段——通用压缩器**：
- 8 层 MLP 自编码器，在多样化数据集（COCO）上预训练
- 768 维 → 32 维，利用语言嵌入的内在冗余
- 维度不宜太低，否则跨域泛化时开放词汇能力受损

**第二阶段——在线学习自编码器（OLAE）**：
- 2 层 MLP，32 维 → **15 维**
- 基于"单一场景的数据方差可被更少维度捕捉"的观察
- 初始化：200 次迭代（6ms/iter），之后每帧更新 1 次
- 每次迭代额外引入 2 个随机关键帧，防止灾难性遗忘
- 关键优势：即使超越了在域微调的单一自编码器，因为能自适应当前场景的主导数据分布

### 关键设计 3：颜色-语言解耦优化

实验发现联合优化 RGB 和语言通道时，两者共享高斯参数（$\alpha, \mu, \Sigma$）导致性能下降。原因：语言特征偏好较大尺度和不同旋转（语义区域均匀），而颜色需要精细纹理。

解决方案——为颜色和语言分别维护**独立的旋转 $R$、尺度 $S$ 和不透明度 $\alpha$**，但共享位置 $\mu$（避免高斯重复）：

$$C = \sum_{i} c_i \alpha_i^c \prod_{j<i}(1 - \alpha_j^c), \quad F = \sum_{i} f_i \alpha_i^f \prod_{j<i}(1 - \alpha_j^f)$$

梯度路径完全分离：颜色梯度只更新 $\alpha^c, R^c, S^c$，语言梯度只更新 $\alpha^f, R^f, S^f$。位置 $\mu$ 仅通过颜色模式反传，相机位姿估计也仅用颜色模式。额外约束 $|S_i^f - S_{i\bot}^c|$ 防止语言模式学到过度偏斜的尺度。

### 损失函数

- **SLAM 跟踪/建图**：$\mathcal{L} = \lambda|C^r - C^{gt}| + (1-\lambda)|D^r - D^{gt}|$ + 尺度各向同性正则
- **语言特征**：L1 损失对齐压缩后的语言地图
- **SRD 训练**：cosine + L1 + TV 损失
- **解耦约束**：语言尺度正则 $|S^f - \text{sg}(S^c)|$

## 实验关键数据

### 主实验：与 Lang-GS SOTA 对比（Replica 数据集）

| 方法 | 在线 | SRD | OLAE | mIoU ↑ | Loc ↑ | 时间/帧 |
|------|------|-----|------|--------|-------|---------|
| LangSplat | ✗ | - | - | 0.417 | 0.720 | 2.8 min |
| Feature3DGS | ✗ | - | - | 0.359 | 0.755 | 2.3 min |
| LEGaussian | ✗ | - | - | 0.245 | 0.682 | 32 s |
| **Ours (Omni, full)** | ✓ | ✓ | ✓ | **0.487** | **0.826** | **0.8 s** |

### TUM RGB-D 数据集

| 方法 | Scene1 mIoU | Scene1 Loc | Scene2 mIoU | Scene2 Loc | 时间/帧 |
|------|-------------|------------|-------------|------------|---------|
| LangSplat | 0.646 | 0.850 | 0.538 | 0.783 | 2.1 min |
| **Ours** | 0.599 | **0.917** | 0.535 | 0.791 | **0.6 s** |

### SLAM-GS 评估（Replica）

| 方法 | 语言 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | ATE(cm) ↓ |
|------|------|--------|--------|---------|-----------|
| SplaTAM | ✗ | 33.39 | 0.968 | 0.101 | 0.392 |
| MonoGS | ✗ | 35.72 | 0.950 | 0.075 | 0.420 |
| **Ours** | ✓ | **35.81** | 0.950 | **0.072** | 0.397 |

加入语言建图后，渲染质量和定位精度与 MonoGS 基线持平甚至略优。

### 消融实验

| 配置 | mIoU ↑ | Loc ↑ | PSNR ↑ | ATE(cm) ↓ |
|------|--------|-------|--------|-----------|
| 联合优化 (Joint) | 0.323 | 0.633 | 31.23 | 0.796 |
| **解耦优化 (Disent.)** | 0.402 | 0.622 | **35.89** | **0.325** |

解耦优化大幅提升渲染质量（+4.66 dB PSNR）和跟踪精度（ATE 降低 59%），mIoU 也显著改善。

### 关键发现

- **SRD 的作用**：高分辨率特征图显著提升 mIoU 和 Loc，尤其对小物体和远距离物体
- **OLAE 的泛化**：超越在域微调的单一自编码器，在未见场景上更稳定
- **效率**：整个网络模块仅 21ms/帧（CLIP 编码 15ms + SRD 2ms + 压缩 6ms），瓶颈在 MonoGS 基线；集成到 Hi-SLAM 后可达 7.05 FPS
- **3D 定位**：平均 CD/EMD 优于 LangSplat（0.38/0.97 vs 0.43/5.63）

## 亮点与洞察

1. **问题定义精准**：首次明确提出在线开放词汇 3D 语言建图问题，识别出三个核心子挑战并逐一解决
2. **两阶段压缩设计巧妙**：通用压缩器保持跨域泛化能力，在线压缩器自适应当前场景——类似于"全局特征 + 局部适应"的思路，在 768→32→15 的压缩链中兼顾了效率和表达力
3. **解耦优化的洞察深刻**：通过可视化发现语言和颜色偏好不同的高斯参数，提出共享位置、分离其他参数的设计，是一个简单但有效的工程创新
4. **在线方法超越离线**：反直觉地，受益于高分辨率 CLIP 嵌入和场景自适应压缩，在线方法在精度上全面超越离线方法
5. **即插即用**：框架可以集成到不同的 SLAM-GS 后端（MonoGS、Hi-SLAM），具有良好的通用性

## 局限性

1. **TUM RGB-D 上优势不显著**：运动模糊和低图像质量下在线跟踪困难，离线方法凭借 30k 次全局优化仍有优势
2. **整体速度受 SLAM 基线限制**：网络模块仅 21ms/帧，但整体 0.6-0.8s/帧的速度瓶颈在 MonoGS
3. **SRD 分辨率有限**：从 24×24 上采样到 192×192（8× 倍率），对于更高分辨率输入可能不足
4. **3D 定位在部分类别上不稳定**：如 rug 和 lamp 的 EMD 差距较大
5. **OLAE 初始化开销**：需要 200 次迭代的初始训练，在快速移动场景中可能来不及

## 相关工作与启发

- **SLAM-GS**：MonoGS、SplaTAM、RTG-SLAM 提供了高效的在线 3D 建图基础设施
- **离线 Lang-GS**：LangSplat 首创将 CLIP 特征嵌入 3DGS，但依赖 SAM+CLIP 离线预处理（~168s/帧）
- **高效 CLIP 编码**：ConvNeXt 像素级编码器 + FeatUp 风格的上采样思路，可推广到其他视觉-语言任务
- **启发**：两阶段压缩方案（通用 + 场景自适应）有潜力用于其他需要在线嵌入高维特征的任务；解耦多模态优化的思路同样适用于 3DGS 中其他模态（如音频、触觉）的集成

## 评分

| 维度 | 分数 (1-10) | 评价 |
|------|:-----------:|------|
| 新颖性 | 8 | 首个在线开放词汇 3D 语言建图，问题定义有开创性 |
| 技术深度 | 7 | 三个模块各自合理，解耦优化和两阶段压缩有洞察力 |
| 实验充分度 | 8 | Replica + TUM 双数据集，2D/3D 定位 + SLAM 评估 + 消融 |
| 写作质量 | 8 | 结构清晰，图表丰富，动机阐述到位 |
| 实用价值 | 8 | 直接面向机器人和 AR 场景，在线 + 开放词汇的组合实用性强 |
| **综合** | **8** | 一篇完成度很高的系统性工作，在一个重要问题上给出了全面解决方案 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ICML 2025\] LaGa: Tackling View-Dependent Semantics in 3D Language Gaussian Splatting](../../ICML2025/3d_vision/tackling_view-dependent_semantics_in_3d_language_gaussian_splatting.md)
- [\[CVPR 2026\] ReLaGS: Relational Language Gaussian Splatting](../../CVPR2026/3d_vision/relags_relational_language_gaussian_splatting.md)
- [\[ICCV 2025\] AutoOcc: Automatic Open-Ended Semantic Occupancy Annotation via Vision-Language Guided Gaussian Splatting](autoocc_automatic_openended_semantic_occupancy_annotation_vi.md)
- [\[ICCV 2025\] 3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)

</div>

<!-- RELATED:END -->
