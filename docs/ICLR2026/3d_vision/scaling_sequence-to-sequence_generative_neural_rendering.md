---
title: >-
  [论文解读] Scaling Sequence-to-Sequence Generative Neural Rendering
description: >-
  [ICLR 2026][3D视觉][Neural Rendering] 提出 Kaleido，一系列将 3D 视为视频特殊子域的生成模型，通过序列到序列的图像合成范式和 Masked Autoregressive 框架实现无需显式 3D 表示的新视角合成，首次在多视角设置下匹配逐场景优化方法的质量。
tags:
  - ICLR 2026
  - 3D视觉
  - Neural Rendering
  - View Synthesis
  - Transformer
  - Masked Autoregression
  - Video-3D Unification
---

# Scaling Sequence-to-Sequence Generative Neural Rendering

**会议**: ICLR 2026  
**arXiv**: [2510.04236](https://arxiv.org/abs/2510.04236)  
**代码**: [Project Page](https://shikun.io/projects/kaleido)  
**领域**: 3D Vision  
**关键词**: Neural Rendering, Novel View Synthesis, Rectified Flow Transformer, Masked Autoregression, Unified Positional Encoding, Video-3D Unification

## 一句话总结

提出 Kaleido，一系列将 3D 视为视频特殊子域的 decoder-only rectified flow transformer 生成模型，通过统一位置编码（Unified Positional Encoding）、掩码自回归框架和视频预训练策略，实现无需任何显式 3D 表示的 "any-to-any" 6-DoF 新视角合成，**首次在多视角设置下匹配逐场景优化方法（InstantNGP）的渲染质量**，并将分辨率从 512/576px 提升至 1024px。

## 研究背景与动机

新视角合成（Novel View Synthesis, NVS）是 3D 视觉的核心任务，给定若干参考视图生成任意目标视角图像。现有方法存在明确的技术瓶颈：

| 方法范式 | 代表工作 | 核心局限 |
|---------|---------|---------|
| 逐场景优化 | 3DGS, NeRF, InstantNGP | 需要大量视图 + 逐场景数分钟级优化，无法零样本泛化 |
| 前馈重建 | PixelNeRF, LRM | 依赖显式 3D 表示（点云/三平面），泛化受限 |
| 扩散生成 | Zero123++, SV3D, SEVA | 基于 U-Net 架构，分辨率限于 512/576px，难以扩展；多数需要 SDS 二阶段精修 |
| 视频控制生成 | MotionCtrl, CameraCtrl, GEN3C | 本质是 4D 时序预测，受限于单参考帧/固定轨迹，无法处理 "any-to-any" 空间查询 |

**核心洞察**：3D 可以被看作视频的一个特殊子域——两者本质上都是图像序列，区别只在于帧间的相机变换是否已知。但直接微调视频模型行不通：视频模型依赖时序 VAE，假设帧间高时序相关性，该假设在稀疏视图 3D 任务中不成立。

**关键动机**：
- 带相机标签的 3D 数据极为稀缺，而视频数据规模大数个量级
- 前代最强通用 NVS 模型 SEVA 基于 U-Net 架构，扩展性差，限于 576px
- 需要一个可扩展的纯 Transformer 架构，能从视频中学习空间先验（"visual commonsense"），然后迁移到 3D

## 方法详解

### 整体框架

Kaleido 将新视角合成完全表述为 **序列到序列的图像合成任务**：给定 $N$ 张参考视图及其 6-DoF 相机位姿 $\{(I_i, P_i)\}_{i=1}^{N}$，和 $M$ 个目标位姿 $\{P_j\}_{j=1}^{M}$，直接生成 $M$ 张目标视图 $\{I_j\}_{j=1}^{M}$。整个过程使用单一的 **decoder-only rectified flow transformer**，不使用任何显式 3D 表示（无点云、无 NeRF、无 3DGS、无深度估计）。

模型架构核心要素：
- **Tokenization**：每张图像通过 VAE 编码为 latent tokens
- **位姿条件**：通过统一位置编码（Unified Positional Encoding）将相机位姿信息注入到 token 序列中
- **掩码策略**：参考视图 tokens 为 unmasked（条件），目标视图 tokens 为 masked（待生成）
- **生成过程**：用 rectified flow 对 masked tokens 进行去噪生成

### 关键设计

**1. 统一位置编码（Unified Positional Encoding）**

这是使单一 Transformer 同时处理视频和 3D 数据的关键架构创新，**无需任何额外可训练参数**：

- 对于 **视频数据**：使用 RoPE 编码帧的时序位置 $t$ 和空间位置 $(h, w)$
- 对于 **3D 数据**：使用 RoPE 编码 Plücker 射线坐标 $(\mathbf{o}, \mathbf{d})$（由相机内外参计算），其中 $\mathbf{o}$ 为射线原点，$\mathbf{d}$ 为射线方向
- 统一设计允许视频的时序一致性先验自然迁移为 3D 的空间一致性，且架构在视频和 3D 训练间 **零修改切换**

**2. 掩码自回归框架（Masked Autoregressive）**

支持灵活的 "any-to-any" 推理模式：
- 任意 $N$ 张参考视图 → 任意 $M$ 张目标视图（$N, M$ 训练和推理时均可变）
- 使用 causal masking 区分条件视图（clean tokens）和目标视图（noisy tokens）
- 自回归迭代：已生成的高质量视图可作为新的条件加入，逐步扩展覆盖范围
- 支持极端推理：从 12 视图训练长度外推到 480 帧（40× 训练长度）的自回归生成

**3. 缩放瓶颈的系统性解决**

论文通过大量消融实验发现并解决了阻碍纯 seq2seq 模型扩展的非显而易见的瓶颈：
- **激活值溢出问题**（Sec 2.2.2）：大模型训练中出现的 massive activation overflow，通过特定激活函数选择解决
- **次优 SNR 采样器**（Sec 2.2.3）：标准扩散模型的信噪比采样策略对 3D 渲染任务不最优，提出 noise-biased sampling 改善训练
- 这些 "scaling recipe" 发现本身是核心贡献，使得简单纯 Transformer 设计首次在生成式 NVS 中可行

### 损失函数与训练策略

**两阶段训练范式**：

1. **视频预训练阶段**：在大规模视频数据上训练 rectified flow 匹配损失 $\mathcal{L} = \mathbb{E}_{t}\left[\|v_\theta(x_t, t) - (x_1 - x_0)\|^2\right]$，学习时空一致性先验
2. **3D 微调阶段**：在带相机标签的 3D 数据集上微调，统一位置编码自动切换到 Plücker 射线模式，学习精确的几何对应关系

关键训练策略：
- Rectified Flow 比标准 DDPM 具有更好的训练稳定性和采样效率
- 支持 Classifier-Free Guidance（CFG）增强生成质量
- Noise-biased sampling 策略优化 3D 生成的 SNR 分布
- 从视频到 3D 的迁移不需要任何架构修改

## 实验关键数据

### 主实验：NVS 基准对比

| 设置 | 对比方法 | Kaleido 表现 | 关键指标 |
|------|---------|-------------|---------|
| 少视图 (1-3 views) | Zero123++, SV3D, SEVA, EscherNet | 零样本大幅超越所有生成方法 | PSNR 显著领先 |
| 多视图 (>10 views) | InstantNGP (逐场景优化) | **首次匹配**优化方法质量 | PSNR 可比 |
| 单视图 3D 重建 | 前代所有方法 | SOTA | CD=1.83, VIoU=70% |
| 分辨率 | SEVA (576px), CAT3D (512px) | **首个 1024px** 生成式渲染模型 | 分辨率翻倍 |

### 对比视频控制生成模型（附录 H，rebuttal 新增）

| 方法 | 类型 | 相机精度 ($R_{err}$↓ / $T_{err}$↓) | 视觉质量 (LPIPS↓) |
|------|------|------|------|
| Wonderland (CVPR 2025) | 视频→3DGS 管线 | 较差 | 可比 |
| ViewCrafter | 视频生成 | 较差 | 较差 |
| VD3D | 视频扩散 | 较差 | 较差 |
| MotionCtrl (SIGGRAPH 2024) | 视频控制 | 较差 | 较差 |
| **Kaleido** | Seq2Seq NVS | **最优** | **最优或可比** |

在 DL3DV 和 Tanks & Temples 上，Kaleido 在相机精度上显著优于所有视频基线模型。

### 消融实验

| 消融配置 | 影响 | 说明 |
|---------|------|------|
| 去除视频预训练 | 空间一致性显著恶化 | 验证 "3D ≈ 视频子域" 假设 |
| 标准位置编码 vs 统一位置编码 | 性能下降 | Unified PE 是视频→3D 迁移的关键 |
| Encoder-decoder vs Decoder-only | 性能下降 | Decoder-only 更适合变长序列任务 |
| 标准 SNR 采样 vs Noise-biased 采样 | 性能下降 | SNR 分布优化对 3D 渲染至关重要 |
| 未修复激活值溢出 | 训练不稳定 | 大模型缩放的关键瓶颈 |
| 减小模型规模 | 持续下降 | 3D 几何理解需要足够模型容量 |

### 关键发现

1. **3D ≈ 视频的特殊子域**：视频预训练显著提升 3D 渲染质量，时序一致性有效迁移为空间一致性
2. **缩放规律明确**：模型规模增大持续改善渲染质量，纯 Transformer 在 NVS 上存在明确 scaling law
3. **隐式几何理解**：无显式 3D 表示的模型仍学到了有意义的几何（CD=1.83 的 3D 重建质量证明）
4. **视频模型不可简单替代**：对比实验表明，视频模型的时序 VAE 假设在稀疏 3D 任务中失效，GEN3C 等方法在大视角变化时因 "空缓存" 问题崩溃

## 亮点与洞察

- **范式突破**：首次证明纯 2D 序列模型可以在渲染质量上匹配 per-scene optimization（如 InstantNGP），且无需任何 SDS 二阶段精修
- **架构创新**：统一位置编码使得单一架构零修改处理视频和 3D，是一个优雅且实用的设计
- **数据规模杠杆**：通过视频预训练巧妙弥补 3D 数据稀缺性，且迁移效果显著——这为其他数据稀缺的 3D 任务提供了范式参考
- **Scaling Recipe**：系统性发现并解决了 seq2seq 3D 渲染的缩放瓶颈（激活溢出、SNR 采样），这些经验具有广泛参考价值
- **从 512px 到 1024px**：首个超越 U-Net 限制、达到 1024px 分辨率的生成式渲染模型

## 局限性与改进方向

1. **推理效率**：大模型 + 长序列的推理成本显著，零样本推理虽快于 per-scene 优化，但仍远非实时
2. **几何精度上限**：隐式 3D 理解在需要亚像素级精确几何的应用（如 AR/VR）中可能不足
3. **相机参数依赖**：仍需已知的目标视角 6-DoF 相机参数作为条件
4. **长序列退化**：480 帧极端自回归生成仍有 artifact，超长轨迹的一致性有待提升
5. **训练成本高昂**：视频预训练 + 3D 微调的整体训练开销巨大
6. **无 4D 能力**：当前设计针对静态场景，不处理动态场景重建

## 相关工作与启发

- **NeRF / 3DGS**：逐场景优化方法，提供质量上界参考；Kaleido 首次在多视角设置下匹配 InstantNGP
- **SEVA (ICCV 2025)**：基于 U-Net 的通用 NVS 模型，Kaleido 在可扩展性和分辨率上全面超越
- **CAT3D / ReconFusion / ZeroNVS**：需要 SDS 二阶段精修的生成方法；Kaleido 单阶段即达更高质量
- **GEN3C (NVIDIA)**：基于显式深度重投影的视频模型，大视角变化时因空缓存崩溃；Kaleido 的隐式先验更鲁棒
- **Wonderland (CVPR 2025)**：视频→3DGS 管线，Kaleido 在相机精度上显著胜出
- **启发**：将专业领域任务统一为序列到序列并利用大规模邻域数据预训练的思路，可推广到 3D 编辑、动态场景重建、机器人操作等数据稀缺场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （"3D-as-Video" 统一范式 + 统一位置编码 + scaling recipe 三重创新）
- 实验充分度: ⭐⭐⭐⭐ （多基准全面评估，消融充分；但初版缺少与视频模型对比，rebuttal 后补充）
- 写作质量: ⭐⭐⭐⭐ （概念清晰，系统性强；rebuttal 中的区分论述比正文更到位）
- 价值: ⭐⭐⭐⭐⭐ （开辟生成式神经渲染新方向，首次匹配优化方法质量，scaling 经验具有广泛指导意义）
---
title: >-
  [论文解读] Scaling Sequence-to-Sequence Generative Neural Rendering
description: >-
  [ICLR 2026][3D视觉][Neural Rendering] 提出 Kaleido，一系列将 3D 视为视频特殊子域的生成模型，通过序列到序列的图像合成范式和 Masked Autoregressive 框架实现无需显式 3D 表示的新视角合成，首次在多视角设置下匹配逐场景优化方法的质量。
tags:
  - ICLR 2026
  - 3D视觉
  - Neural Rendering
  - View Synthesis
  - Transformer
  - Masked Autoregression
  - Video-3D Unification
---

# Scaling Sequence-to-Sequence Generative Neural Rendering

**会议**: ICLR 2026  
**arXiv**: [2510.04236](https://arxiv.org/abs/2510.04236)  
**代码**: 有（Project Page）  
**领域**: 3D Vision  
**关键词**: Neural Rendering, View Synthesis, Rectified Flow Transformer, Masked Autoregression, Video-3D Unification

## 一句话总结

提出 Kaleido，一系列将 3D 视为视频特殊子域的生成模型，通过序列到序列的图像合成范式和 Masked Autoregressive 框架实现无需显式 3D 表示的新视角合成，首次在多视角设置下匹配逐场景优化方法的质量。

## 研究背景与动机

新视角合成（Novel View Synthesis）是 3D 视觉的核心任务，目标是给定若干参考视图，生成任意目标视角的图像。

现有方法的局限：

**逐场景优化方法**（如 3DGS、NeRF）：需要大量输入视图和逐场景优化，计算成本高

**前馈推理方法**（如 PixelNeRF、LRM）：依赖显式 3D 表示（点云、三平面等），泛化能力受限

**生成方法**（如 Zero123++、SV3D）：采用扩散模型进行视角合成，但在少视图设置下质量不稳定

**3D 数据稀缺**：带相机标签的 3D 数据集规模远小于视频数据集

核心洞察：**3D 可以被看作视频的一个特殊子域**——两者本质上都是图像序列，区别只在于帧间的相机变换是否已知。这意味着可以用统一的序列到序列框架同时处理 3D 和视频。

## 方法详解

### 整体框架

Kaleido 将新视角合成完全表述为**序列到序列的图像合成任务**，使用单一的 decoder-only rectified flow transformer，无需任何显式 3D 表示（无点云、无NeRF、无3DGS）。

### 关键设计

1. **无显式 3D 表示的生成式视角合成**

    - 完全抛弃传统的 3D 中间表示（如体素、三平面、高斯等）
    - 直接从参考视图图像序列生成目标视图图像序列
    - 核心信念：足够大的模型可以隐式学习 3D 几何理解
    - 设计动机：绕过 3D 表示的瓶颈，充分利用 2D 数据的规模优势

2. **Masked Autoregressive 框架**

    - 支持任意数量参考视图到任意数量目标视图的生成
    - 使用掩码机制区分条件视图（已知）和目标视图（待生成）
    - 自回归生成允许灵活的推理策略：
        - 少视图→多视图：从 1-2 张参考图生成 N 张新视图
        - 逐步细化：已生成的视图可作为新的条件输入
    - 支持 6-DoF 目标视角条件

3. **Decoder-Only Rectified Flow Transformer**

    - 统一框架同时建模 3D 和视频
    - Rectified Flow 比标准扩散模型有更好的训练稳定性和采样效率
    - Decoder-only 架构使得模型可以自然处理变长序列
    - 无需架构修改即可在不同任务间切换

4. **视频预训练策略**

    - 利用大规模视频数据进行预训练，然后在 3D 数据上微调
    - 视频数据提供：
        - 时序一致性先验 → 有助于多视角空间一致性
        - 运动理解 → 类似相机运动的几何理解
    - 关键优势：显著减少对稀缺 3D 数据的依赖
    - 无需任何架构改动即可完成从视频到 3D 的迁移

### 损失函数 / 训练策略

- 采用 Rectified Flow 的匹配损失进行训练
- 两阶段训练：先视频预训练获取时空一致性先验，再 3D 数据微调学习精确的几何对应
- 支持 Classifier-Free Guidance 增强生成质量

## 实验关键数据

### 主实验

在多个视角合成基准上评估（如 CO3D、RealEstate10K、ACID 等）：

| 设置 | 方法类别 | Kaleido 表现 |
|------|---------|-------------|
| 少视图 (1-3 views) | vs 其他生成方法 | 显著超越 |
| 多视图 (>10 views) | vs 逐场景优化方法 | **首次匹配** |
| 零样本 | vs 前馈方法 | 大幅领先 |

核心结果：
- **少视图设置**：零样本性能大幅超越其他生成方法（如 Zero123++、SV3D）
- **多视图设置**：首次在生成方法中达到逐场景优化方法（如 3DGS）的质量
- 同时支持物体级和场景级渲染

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无视频预训练 | 质量下降 | 空间一致性显著恶化 |
| 固定参考视图数 | 灵活性下降 | Masked AR 的优势消失 |
| 换用encoder-decoder | 性能下降 | decoder-only更适合此任务 |
| 减小模型规模 | 性能下降 | 3D理解需要足够的模型容量 |

### 关键发现

1. **3D ≈ 视频的特殊子域**：这一假设被实验充分验证，视频预训练显著提升 3D 质量
2. **规模效应明显**：模型规模的增大持续改善渲染质量
3. **Masked AR 的灵活性**：统一处理任意数量输入/输出视图的能力是关键优势
4. **隐式几何理解**：模型确实学到了有意义的 3D 几何，尽管没有显式 3D 表示

## 亮点与洞察

- **视频-3D统一范式的突破**：将看似不同的任务（视频生成和3D渲染）统一到同一框架
- **数据规模杠杆**：巧妙利用丰富的视频数据弥补 3D 数据的稀缺性
- **无显式 3D 表示的生成**：证明了纯 2D 序列模型可以隐式掌握 3D 几何
- **首次匹配优化方法**：在多视图设置下的生成质量首次达到逐场景优化方法水平
- **Scaling Law**：系统研究了序列到序列生成神经渲染的缩放规律

## 局限与展望

1. **计算成本较高**：大模型 + 长序列的推理成本显著
2. **几何精度限制**：隐式 3D 理解可能在需要精确几何的应用中不足
3. **相机参数依赖**：仍需要目标视角的相机参数作为条件
4. **对参考视图的依赖**：极端视角变化时生成质量可能下降
5. **训练成本**：视频预训练 + 3D 微调的整体训练成本较高
6. **实时应用困难**：当前的自回归生成速度难以满足实时渲染需求

## 相关工作与启发

- **NeRF / 3DGS 系列**：逐场景优化的3D表示，提供质量上界参考
- **Zero123++ / SV3D**：基于扩散的3D生成方法，作为生成方法类别的baseline
- **LRM / Instant3D**：前馈式3D重建方法
- **Sora / Video Generation**：视频生成的进展为3D渲染提供了预训练基础
- **启发**：这种"将专业领域任务统一为序列-to-序列"的思路可以推广到其他3D任务（如3D编辑、动态场景重建）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （3D-as-Video 统一范式，纯序列模型做3D渲染的新颖度很高）
- 实验充分度: ⭐⭐⭐⭐⭐ （多基准、多设置的全面评估，首次匹配优化方法）
- 写作质量: ⭐⭐⭐⭐ （概念表述清晰，实验设置严谨）
- 价值: ⭐⭐⭐⭐⭐ （开辟了生成式神经渲染的新方向，缩放规律具有指导意义）

<!-- RELATED:START -->

## 相关论文

- [LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](../../CVPR2026/3d_vision/longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [LONG3R: Long Sequence Streaming 3D Reconstruction](../../ICCV2025/3d_vision/long3r_long_sequence_streaming_3d_reconstruction.md)
- [AeroDGS: Physically Consistent Dynamic Gaussian Splatting for Single-Sequence Aerial 4D Reconstruction](../../CVPR2026/3d_vision/aerodgs_physically_consistent_dynamic_gaussian_splatting_for_single-sequence_aer.md)
- [RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence](../../AAAI2026/3d_vision/radarllm_empowering_large_language_models_to_understand_human_motion_from_millim.md)
- [Scaling View Synthesis Transformers (SVSM)](../../CVPR2026/3d_vision/scaling_view_synthesis_transformers.md)

<!-- RELATED:END -->
