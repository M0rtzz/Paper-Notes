---
title: >-
  [论文解读] Video Color Grading via Look-Up Table Generation
description: >-
  [ICCV 2025][图像生成][视频调色] 提出基于扩散模型显式生成 LUT 的视频调色框架：通过 GS-Extractor 提取参考场景的高层风格特征，用 L-Diffuser 生成色彩查找表（LUT），一次生成即可无损应用于全部视频帧，并支持文本 prompt 进行亮度/对比度等细粒度调整。
tags:
  - ICCV 2025
  - 图像生成
  - 视频调色
  - LUT生成
  - 扩散模型
  - 色彩分级
  - 参考图像
  - 风格迁移
---

# Video Color Grading via Look-Up Table Generation

**会议**: ICCV 2025  
**arXiv**: [2508.00548](https://arxiv.org/abs/2508.00548)  
**代码**: [GitHub](https://github.com/seunghyuns98/VideoColorGrading)  
**领域**: 图像生成 / 视频编辑  
**关键词**: 视频调色, LUT生成, 扩散模型, 色彩分级, 参考图像, 风格迁移

## 一句话总结

提出基于扩散模型显式生成 LUT 的视频调色框架：通过 GS-Extractor 提取参考场景的高层风格特征，用 L-Diffuser 生成色彩查找表（LUT），一次生成即可无损应用于全部视频帧，并支持文本 prompt 进行亮度/对比度等细粒度调整。

## 研究背景与动机

- **核心问题**：视频调色（Color Grading）是电影后期制作的核心环节，目的是调整色彩/对比度以传达特定艺术意图和情绪氛围（如《寄生虫》中用冷暖色调隐喻阶级差异）。但这一过程需要深厚的色彩理论和艺术直觉，长期由专业调色师把控。
- **任务区分**：调色不同于色彩迁移/风格迁移
    - 色彩迁移：匹配颜色分布（低层特征）
    - 风格迁移：迁移纹理+颜色（可能破坏内容结构）
    - **调色**：传递氛围、情绪、基调等**高层主观特征**，同时严格保持内容细节
- **现有方法的不足**：
    - 色彩迁移方法（统计对齐、直方图匹配）只处理低层特征，无法传递"氛围感"
    - 风格迁移方法（WCT2、CAP-VSTNet 等）在潜空间操作，不可避免地丢失高频细节
    - 逐帧处理导致时间不一致（闪烁）
    - 现有 LUT 预测方法（NLUT、AdaCM）对未见风格泛化差、色彩失真
- **关键洞察**：跟随专业调色师的工作流——选关键帧、匹配风格、生成 LUT、应用全帧、微调润色——用显式 LUT 保证时间一致性和零内容损失

## 方法详解

### 整体框架（三步流水线）

1. **关键帧选择**：用 CLIP 图像编码器找出输入视频与参考视频中语义最相似的帧对
2. **LUT 生成**：GS-Extractor 提取参考风格特征、L-Diffuser 扩散模型生成 LUT、应用于全部帧
3. **用户偏好微调**：通过文本 prompt 进行对比度、亮度、色调等低层调整

### 关键帧选择

对输入帧序列 $I_{1:M}$ 和参考帧序列 $I'_{1:N}$，找使 CLIP 余弦相似度最大的帧对：

$$(\hat{m}, \hat{n}) = \arg\max_{m,n} \frac{f_m \cdot f'_n}{\|f_m\| \|f'_n\|}$$

每秒采样 1 帧，因为短时间内语义稳定。

### GS-Extractor（Grading Style 提取器）

- 采用 ReferenceNet 架构：两个结构相同的 U-Net
    - **GS-Extractor U-Net**：从参考帧提取高层主观特征（氛围/情绪/基调）
    - **Image Editor U-Net**：以提取特征为条件，通过空间注意力将输入帧编辑为风格匹配的目标
- 训练数据构造：从同一电影中取帧对，对其中一个应用 LUT 产生风格差异，让网络学习提取和还原这种差异
- 去噪目标：

$$\ell = \mathbb{E}_{z_k, z_0^{I_{\hat{m}}}, G(z_0^{I'_{\hat{n}}}), \epsilon, k} \left[\|\epsilon - \epsilon_\theta(z_k^{I'_{\hat{m}}}, z_0^{I_{\hat{m}}}, G(z_0^{I'_{\hat{n}}}), k)\|_2^2\right]$$

### L-Diffuser（LUT 扩散生成器）

**核心创新**：不在像素/潜空间做风格迁移，而是**直接生成 LUT**。

- 冻结 GS-Extractor，训练 L-Diffuser 从随机噪声生成 LUT
- 生成的是与恒等 LUT 的差值 $\Delta L \in \mathbb{R}^{16 \times 16 \times 16 \times 3}$，reshape 为 $64 \times 64 \times 3$ 匹配图像潜变量维度
- **条件向量**：参考帧与输入帧的 GS-Extractor 特征之差，作为从输入到参考的"风格梯度"：

$$C = G(z_0^{I'_{\hat{n}}}) - G(z_0^{I_{\hat{m}}})$$

- 扩散去噪目标：

$$\ell = \mathbb{E}_{\Delta L'_k, L^I, C, \epsilon, k} \left[\|\epsilon - \epsilon_\theta(\Delta L'_k, L^I, C, k)\|_2^2\right]$$

- 最终 LUT = 恒等 LUT + L-Diffuser 输出的 $\Delta L$

### 为什么选 LUT？

- **时间一致性**：同一 LUT 应用于所有帧，零闪烁
- **零内容损失**：LUT 是确定性的颜色映射，不修改空间结构
- **推理速度快**：LUT 查表操作近乎瞬时
- **可解释可编辑**：生成的 LUT 可以被专业调色师进一步手动调整

### 损失函数

- GS-Extractor 训练：扩散去噪损失（公式 3）
- L-Diffuser 训练：LUT 差值的扩散去噪损失（公式 5）
- 隐式约束：GS-Extractor 的高层特征对齐确保调色捕获氛围而非简单色彩分布匹配

## 实验关键数据

### 主实验：Condensed Movie 数据集

| 方法 | PSNR | SSIM | LPIPS | 推理时间(s) |
|------|:---:|:---:|:---:|:---:|
| WCT2 | 19.97 | 0.749 | 0.303 | 130.08 |
| PhotoNAS | 17.08 | 0.652 | 0.394 | 54.48 |
| Deep Preset | 20.70 | 0.743 | 0.322 | 25.64 |
| HistoGAN | 18.36 | 0.689 | 0.360 | 50.47 |
| CCPL | 17.13 | 0.620 | 0.395 | 38.55 |
| CAP-VSTNet | 20.55 | 0.764 | 0.303 | 46.97 |
| NLUT | 21.41 | 0.740 | 0.303 | 21.24 |
| **Ours** | **24.55** | **0.845** | **0.146** | **12.10** |

在所有指标上全面领先：PSNR 超第二名 3.14dB，LPIPS 降低一半以上，推理速度最快（12.1s vs 最快基线 21.2s）。

### Adobe5K 数据集

| 方法 | PSNR | SSIM | LPIPS |
|------|:---:|:---:|:---:|
| CAP-VSTNet | 16.33 | 0.786 | 0.327 |
| NLUT | 16.76 | 0.730 | 0.392 |
| **Ours** | **18.78** | **0.797** | - |

跨数据集泛化表现同样优异。

### 用户研究

- 大规模用户研究验证了定量评估难以捕获的主观审美质量
- 本方法在所有参考风格条件下获得最高用户偏好评分

### 关键发现

- 直接生成 LUT 比在潜空间做色彩变换更有效，因为 LUT 是色彩映射的原生表示
- GS-Extractor 的特征差值作为条件向量，能准确捕获"从当前风格到目标风格"的变换方向
- BRISQUE（无参考图像质量指标）和模糊度指标表明本方法最好地保持了原始视频质量

## 亮点与洞察

1. **任务定义的精准区分**：明确将"调色"和"色彩迁移/风格迁移"区分开——调色是传递高层审美意图，而非简单的颜色分布匹配
2. **LUT 作为生成目标**：用扩散模型直接生成 LUT 是一个巧妙的设计——将连续的像素级变换问题转化为离散的色彩映射学习，天然解决时间一致性和内容保持两大难题
3. **"风格梯度"条件设计**：参考与输入的特征差值作为条件向量，概念简洁且物理含义清晰
4. **贴合行业工作流**：框架设计模仿专业调色师的工作流程（选帧、匹配、调整、润色），实用性极强
5. **文本微调接口**：生成调色后支持文本 prompt 控制对比度/亮度等，扩展了用户控制维度

## 局限性

- LUT 分辨率为 $16^3$，对极端色彩变换可能产生色带（banding）效应
- 依赖 CLIP 关键帧选择——当输入和参考视频语义差异大时，帧匹配可能失败
- 数据集构造依赖已有 LUT 和电影数据，对非电影场景（如无人机航拍、医学影像）的泛化性未验证
- GS-Extractor 基于 SD U-Net 架构，模型体量不小
- 单一 LUT 应用于全部帧——对于需要帧级精细调色的场景（如时间变化的光照）适应性有限

## 相关工作

- **参考图像色调调整**：WCT2、PhotoNAS、HistoGAN、CAP-VSTNet 等
- **视频色彩一致性**：光流约束、对比学习时序一致性（CCPL）、UniST 联合训练
- **确定性色彩映射**：双边网格、MLP 变换、LUT 预测（NLUT、AdaCM）
- **LUT 学习**：3D LUT 预测、LUT 压缩与自适应权重
- **ReferenceNet 架构**：用于参考图像特征提取的双 U-Net 设计

## 评分

| 维度 | 分数 (1-5) |
|------|:---:|
| 创新性 | 4 |
| 理论深度 | 3.5 |
| 实验充分性 | 4.5 |
| 写作质量 | 4 |
| 实用价值 | 4.5 |
| **总评** | **4.1** |

将扩散模型用于显式 LUT 生成是本文最精彩的设计选择，巧妙地用确定性映射解决了视频编辑中时间一致性和内容保持两个核心问题。实验全面，用户研究充分，具有很高的工业应用价值。

<!-- RELATED:START -->

## 相关论文

- [SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer](sa-lut_spatial_adaptive_4d_look-up_table_for_photorealistic_style_transfer.md)
- [SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [AnyPortal: Zero-Shot Consistent Video Background Replacement](anyportal_zero-shot_consistent_video_background_replacement.md)
- [GCC: Generative Color Constancy via Diffusing a Color Checker](../../CVPR2025/image_generation/gcc_generative_color_constancy_via_diffusing_a_color_checker.md)

<!-- RELATED:END -->
