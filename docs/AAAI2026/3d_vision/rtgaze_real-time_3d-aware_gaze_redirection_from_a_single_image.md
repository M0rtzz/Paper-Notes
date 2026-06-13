---
title: >-
  [论文解读] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image
description: >-
  [AAAI 2026][3D视觉][视线重定向] 提出 RTGaze，一个实时 3D 感知视线重定向方法，通过混合频率特征编码器 + 视线注入模块 + 3D 面部几何先验蒸馏，从单张图像实现 61ms/帧的高质量视线重定向，比前 SOTA 3D 方法（GazeNeRF）快 800 倍。 领域现状 视线重定向（Gaze Red…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "视线重定向"
  - "3D感知"
  - "NeRF"
  - "知识蒸馏"
  - "实时推理"
---

# RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image

**会议**: AAAI 2026  
**arXiv**: [2511.11289](https://arxiv.org/abs/2511.11289)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 视线重定向, 3D感知, NeRF, 知识蒸馏, 实时推理

## 一句话总结

提出 RTGaze，一个实时 3D 感知视线重定向方法，通过混合频率特征编码器 + 视线注入模块 + 3D 面部几何先验蒸馏，从单张图像实现 61ms/帧的高质量视线重定向，比前 SOTA 3D 方法（GazeNeRF）快 800 倍。

## 研究背景与动机

### 领域现状

视线重定向（Gaze Redirection）是指在保持身份不变的前提下，生成眼球注视方向可控的面部图像。在 VR/AR、数字人、CG 电影制作等领域有广泛应用（如视频会议中的眼神接触校正）。

现有方法分为两大类：
- **2D 方法**（ST-ED、FAZE、ReDirTrans）：通过 GAN/VAE/encoder-decoder 直接生成目标图像。速度快但**缺乏 3D 一致性**，在大角度头部姿态下效果差，无法建模视线重定向的 3D 本质。
- **3D 方法**（EyeNeRF、GazeNeRF、HeadNeRF）：基于 NeRF 构建 3D 面部表征，天然具有 3D 一致性。但推理时需要**GAN inversion 过程**（~60秒），完全无法实时。

### 现有痛点

核心矛盾：**3D 一致性与实时性能之间的权衡**。

- GazeNeRF 推理约 60 秒/帧（编码 60s + 渲染 0.06s），编码阶段需要对 input image 做 inversion 优化可学习的 latent codes，这是速度瓶颈
- 2D 方法虽快但在大姿态角下出现伪影、身份保持差
- 没有方法能同时兼顾 3D 一致性、图像质量和实时性

### 切入角度

核心 idea：**用前馈网络（feedforward）替代 GAN inversion**。直接从单张图像 + 视线标签编码出 triplane 表征，跳过耗时的优化过程。同时从预训练 3D 肖像生成器中蒸馏面部几何先验，弥补单图 3D 推理的约束不足。

## 方法详解

### 整体框架

输入：单张面部图像 $\mathbf{I}$ + 目标视线方向 $\mathbf{g} \in \mathbb{R}^2$（pitch 和 yaw）
输出：视线重定向后的面部图像 $\hat{\mathbf{I}}$

流程：
1. 混合图像编码器提取高频/低频特征
2. 视线注入模块将视线 prompt 融入高频特征
3. Triplane 解码器生成 3D 面部表征
4. 神经渲染生成最终图像

$$f = \mathcal{F}(\mathbf{I}, \mathbf{g}), \quad \mathbf{T} = \mathcal{G}(f), \quad \hat{I} = \mathcal{N}(\mathbf{T}, \mathbf{c})$$

### 关键设计

#### 1. 混合频率面部特征编码器

**功能**：从输入图像中分离提取高频（外观细节）和低频（全局几何）特征。

**核心思路**：
- **低频编码器 $\mathcal{F}_l$**：使用 ImageNet 预训练的 DeepLabV3 提取全局语义信息 → Vision Transformer 编码器精炼全局特征 → 得到低频特征 $z_l$
- **高频编码器 $\mathcal{F}_h$**：CNN 提取精细外观细节（纹理、毛发等）→ 得到高频特征 $z_h$

**设计动机**：视线重定向主要影响眼部外观（高频变化），全局面部几何（低频）相对稳定。分离编码允许视线 prompt 精确注入到对的特征层。

#### 2. 视线 Prompt 注入模块（Gaze Prompt Injection）

**功能**：将目标视线方向注入到面部表征中。

**核心思路**：
- 将视线 prompt $\mathbf{g}$（pitch + yaw）通过 MLP 嵌入，保证长度与高频特征一致
- 使用 **cross-attention** 注入：高频特征作为 query，视线嵌入同时作为 key 和 value
- 注入后的高频特征与低频特征融合，得到最终视线可控面部表征

**关键决策**：消融实验证明，视线 prompt 必须注入**高频特征**而非低频特征。注入低频特征时 FID 恶化到 67.298（vs 38.346），视线误差从 9.047° 飙升到 18.973°。这验证了"视线改变主要是外观级别变化"的假设。

#### 3. 面部几何先验蒸馏（Face Geometric Prior Distillation）

**功能**：从预训练 3D 肖像生成模型中蒸馏面部几何知识，增强单图 3D 推理的质量。

**核心思路**：
- 使用预训练的 3D 肖像生成器（Trevithick et al. 2023）作为 teacher
- 输入与目标同身份、同视线的正脸图像给 teacher，得到 teacher 的 triplane 特征
- 从 teacher 和 student 模型分别渲染深度图 $\mathbf{D}^t$ 和 $\mathbf{D}^s$
- 施加 L1 深度蒸馏损失：

$$\mathcal{L}_{\mathcal{D}} = \|\mathbf{D}^t - \mathbf{D}^s\|_1$$

**设计动机**：
- 单图到 3D 本身是欠约束问题，缺乏足够约束做准确的 3D 重建
- Teacher 模型虽不是为视线重定向设计，但其面部几何知识（深度结构）可迁移
- 仅蒸馏深度而非颜色，因为 teacher 的合成图像与目标图像的外观不一定一致

### 损失函数 / 训练策略

**总损失**：

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\mathcal{R}} + \beta \cdot \mathcal{L}_{\mathcal{D}} + \gamma \cdot \mathcal{L}_{\mathcal{P}}$$

其中 $\alpha = 1, \beta = 1, \gamma = 0.8$。

- **Mask-Guided 重建损失** $\mathcal{L}_{\mathcal{R}}$：分为面部区域和眼部区域，眼部权重 $\alpha_2 = 2$（面部 $\alpha_1 = 1$），用 L1 范数
- **蒸馏损失** $\mathcal{L}_{\mathcal{D}}$：深度图 L1 损失
- **感知损失** $\mathcal{L}_{\mathcal{P}}$：VGG16 多层特征的 L1 距离

**训练细节**：
- AdamW 优化器，编码和渲染部分学习率均 $10^{-5}$
- Batch size 4，50 epochs
- ETH-XGaze 训练集：10 frames/subject × 18 views/frame × 80 subjects
- 2 × NVIDIA A100 (40GB)，约 18 小时
- 输入分辨率 512×512，按 EG3D 方式处理

## 实验关键数据

### 主实验

**ETH-XGaze 数据集——图像质量与推理速度**：

| 方法 | 3D | FID↓ | PSNR↑ | LPIPS↓ | SSIM↑ | 总时间↓ |
|------|-----|------|-------|--------|-------|--------|
| ST-ED | × | 115.020 | 17.530 | 0.300 | **0.726** | - |
| HeadNeRF | ✓ | 69.487 | 15.298 | 0.294 | 0.720 | 60.058s |
| GazeNeRF | ✓ | 81.816 | 15.453 | 0.291 | **0.733** | 60.060s |
| **RTGaze** | **✓** | **38.346** | **19.007** | **0.262** | 0.715 | **0.061s** |

- FID 大幅领先（38.3 vs 次优 69.5），PSNR 和 LPIPS 均最优
- **推理速度 0.061s，比 GazeNeRF 快约 1000 倍**（60s → 0.061s）
- SSIM 略低于 GazeNeRF（0.715 vs 0.733），但其他所有指标全面领先

**跨数据集泛化评估**（视线/头部误差 + 身份保持度）：

| 数据集 | 方法 | LPIPS↓ | ID↑ | Gaze↓ | Head↓ |
|--------|------|--------|-----|-------|-------|
| ColumbiaGaze | GazeNeRF | 0.352 | 23.157 | 9.464 | 3.811 |
| ColumbiaGaze | **RTGaze** | **0.249** | **61.765** | **7.625** | **3.326** |
| MPIIFaceGaze | GazeNeRF | 0.272 | 30.981 | 14.933 | 7.118 |
| MPIIFaceGaze | **RTGaze** | **0.251** | **46.098** | **9.409** | **6.444** |

在 ColumbiaGaze 和 MPIIFaceGaze 上所有指标全面领先，体现了强跨数据集泛化能力。特别是身份保持度（ID）大幅提升（从 23→62, 31→46），说明前馈方式比 inversion 更好地保持身份。

### 消融实验

**视线注入位置消融**：

| 注入目标 | FID↓ | ID↑ | Gaze↓ | Head↓ |
|---------|------|-----|-------|-------|
| 低频特征 | 67.298 | 38.517 | 18.973 | 5.409 |
| **高频特征** | **38.346** | **60.708** | **9.047** | **3.631** |

注入低频特征效果断崖式下降：视线误差翻倍（9→19°），FID 近翻倍，身份保持度骤降。

**损失函数消融**：

| 配置 | FID↓ | ID↑ | Gaze↓ | Head↓ |
|------|------|-----|-------|-------|
| 仅 $\mathcal{L}_\mathcal{R}$ | 101.053 | 47.251 | 9.332 | 4.208 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P}$ | 54.682 | 52.518 | 10.911 | 3.700 |
| $\mathcal{L}_\mathcal{R} + \mathcal{L}_\mathcal{P} + \mathcal{L}_\mathcal{D}$ | **38.346** | **60.708** | **9.047** | **3.631** |

感知损失将 FID 从 101→55，蒸馏损失进一步降至 38，同时身份保持度从 47→61。3D 先验蒸馏的贡献非常显著。

### 关键发现

1. 前馈方案完全取代 GAN inversion 是可行的——不仅快 ~1000 倍，图像质量还更好（FID 38 vs 82）
2. 身份保持度的巨大提升说明 inversion 过程本身可能是身份信息丢失的原因
3. 3D 几何先验蒸馏是提升质量的关键——即使 teacher 模型不是为视线重定向设计的
4. 视线变化本质上是高频外观变化，低频几何特征不应被直接修改

## 亮点与洞察

1. **速度突破性**：在 3D 感知方法中首次实现实时（61ms），使 3D 视线重定向实际可部署（视频会议、VR 等）
2. **高频/低频解耦**：对视线重定向任务的深刻理解——视线改变眼部纹理（高频），不改变面部骨架（低频）——引导了正确的架构设计
3. **蒸馏策略精巧**：只蒸馏深度不蒸馏颜色，避免了 teacher 和 student 在外观上不一致的问题
4. **Cross-attention 注入**：用 cross-attention 而非简单拼接/加法来注入视线 prompt，允许更灵活的特征交互

## 局限与展望

1. **SSIM 略低**：在 ETH-XGaze 上 SSIM（0.715）低于 GazeNeRF（0.733），说明像素级对齐仍有提升空间
2. **训练依赖正脸图像**：蒸馏过程需要与目标同身份同视线的正脸图像，限制了训练数据的使用方式
3. **分辨率限制**：512×512 输入可能不足以捕捉极精细的眼部细节（如虹膜纹理）
4. **未处理极端姿态**：虽然 3D 方法天然更好，但文中未展示极端侧脸/大角度下的表现
5. **代码未开源**（截至目前），限制了可复现性

## 相关工作与启发

- **EG3D** (CVPR 2022)：triplane 表征的开创性工作，RTGaze 的 triplane 解码器基于此
- **Trevithick et al. 2023**：3D 肖像生成器，作为 teacher 提供几何先验蒸馏
- **Zero-1-to-3** 等单图到 3D 方法的发展将持续降低 3D 面部建模的门槛
- 类似的"前馈替代优化"思路可推广到其他 NeRF-based 应用（如表情编辑、风格迁移等）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 前馈替代 inversion + 高/低频解耦 + 蒸馏策略，组合创新有效
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集、效率对比、多维度指标评估、消融全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，量化结果丰富，但部分实现细节稍显不足
- 价值: ⭐⭐⭐⭐⭐ — 实时 3D 视线重定向具有重要实际应用价值，速度突破有里程碑意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GazeGaussian: High-Fidelity Gaze Redirection with 3D Gaussian Splatting](../../ICCV2025/3d_vision/gazegaussian_high-fidelity_gaze_redirection_with_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Generalized Geometry Encoding Volume for Real-time Stereo Matching](generalized_geometry_encoding_volume_for_real-time_stereo_matching.md)
- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](../../CVPR2026/3d_vision/human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[AAAI 2026\] Real-Time 3D Object Detection with Inference-Aligned Learning](real-time_3d_object_detection_with_inference-aligned_learning.md)

</div>

<!-- RELATED:END -->
