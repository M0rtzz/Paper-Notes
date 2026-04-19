---
title: >-
  [论文解读] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution
description: >-
  [CVPR 2026][图像恢复][超分辨率] 提出基于标定的 RAW 域退化建模框架，通过为多款智能手机相机精确标定 SR 模糊核与传感器噪声模型，将公开 sRGB 图像"反处理"为逼真的 LR RAW 数据用于训练，在相机特定和跨相机盲超分辨率场景中均显著超越基于通用退化池的基线方法。
tags:
  - CVPR 2026
  - 图像恢复
  - 超分辨率
  - RAW domain
  - degradation modeling
  - smartphone camera
  - blur kernel calibration
  - sensor noise
---

# RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2603.12493](https://arxiv.org/abs/2603.12493)  
**作者**: Ali Mosleh, Faraz Ali, Fengjia Zhang, Stavros Tsogkas, Junyong Lee, Alex Levinshtein, Michael S. Brown (Samsung AI Center-Toronto)
**领域**: image_restoration  
**关键词**: super-resolution, RAW domain, degradation modeling, smartphone camera, blur kernel calibration, sensor noise

## 一句话总结

提出基于标定的 RAW 域退化建模框架，通过为多款智能手机相机精确标定 SR 模糊核与传感器噪声模型，将公开 sRGB 图像"反处理"为逼真的 LR RAW 数据用于训练，在相机特定和跨相机盲超分辨率场景中均显著超越基于通用退化池的基线方法。

## 研究背景与动机

智能手机数字变焦依赖于基于学习的超分辨率 (SR) 模型，这些模型直接在 RAW 传感器图像上操作。然而，获取高质量的配对训练数据面临根本挑战：

- **真实数据采集困难**：使用不同焦距或不同品质相机拍摄同一场景来获取 HR-LR 对需要精确对齐与静态场景，费时费力
- **合成数据的域差距**：常用的 bicubic 下采样忽略了真实光学系统的复杂退化，导致训练后的模型在真实手机图像上泛化差
- **通用退化池的局限**：Real-ESRGAN 等方法随机采样各向同性/异性高斯核和噪声，但这些手工选择的参数无法准确代表目标手机相机的真实 PSF
- **sRGB 域退化的问题**：大多数现有方法在 sRGB 域施加退化，但模糊和噪声实际发生在相机流水线的更早阶段（RAW 域），此处场景辐射与传感器响应仍保持线性关系。sRGB 域的非线性处理引入额外域差距

核心洞察：**有原则的、精心设计的退化建模**能显著提升真实世界 SR 性能。与其依赖通用先验，不如通过标定获取设备特定的退化参数。

## 方法详解

### 整体框架

论文提出一个从标定到数据合成再到 SR 训练的完整流水线：
1. 对多款智能手机相机标定 SR 模糊核和传感器噪声模型
2. 使用标定结果将公开 sRGB 图像"反处理"为 LR RAW 图像
3. 用合成的 HR-LR 对训练 RAW-to-RGB SR 模型

LR 图像生成公式：

$$\mathbf{y} = \mathcal{M}((\mathbf{K}\mathbf{x})_{\downarrow s}) + \mathbf{n}$$

其中 $\mathbf{x}$ 为潜在 HR 图像，$\mathbf{K}$ 为 SR 模糊核，$\mathcal{M}$ 为 Bayer CFA 马赛克算子，$\mathbf{n}$ 为传感器噪声，$s$ 为下采样因子。

### SR 模糊核标定 (Sec 3.1)

SR 核结合了镜头 PSF 与传感器离散化算子，受镜头特性、传感器尺寸和微透镜阵列影响。论文通过**显示器-相机标定原型**直接在马赛克 RAW 域建模 SR 核：

1. **显示随机模式**：在校准显示器上显示 $B=20$ 个随机结构模式，用目标相机 RAW 拍摄（每个模式取 100 帧 burst 平均去噪）
2. **辐射与几何对齐**：使用灰度码结构化模式建立传感器-显示器的稠密对应，标定透视变换 $\mathcal{W}$；使用灰/色块标定显示器非线性响应 $\mathcal{D}^{-1}$
3. **联合优化**：将 FOV 分为 128×128 patch，对每个 patch 联合优化对齐参数 $\mathbf{H}$ 与 RGB 三通道 SR 核 $\{\hat{\mathbf{K}}_r, \hat{\mathbf{K}}_g, \hat{\mathbf{K}}_b\}$，使用 $\ell_1$ 损失 + ADAM 优化器
4. **无需启发式先验**：充足的 LR-HR 配对数据使问题 well-posed，无需显式施加非负性、能量守恒、稀疏性等约束

### 传感器噪声标定 (Sec 3.2)

采用异方差高斯 (HG) 噪声模型：$\mathbf{n}_i \sim \mathcal{N}(0, \beta_{\kappa,c}^1 \mathbf{y}_i + \beta_{\kappa,c}^2)$

- 对每款相机，在 7 个 ISO 级别 $\{50, 100, 200, 400, 800, 1600, 3200\}$ 分别标定
- 对 Bayer CFA 的四个颜色通道（$r, g_1, g_2, b$）分别建模
- 通过对标定 ISO 级别的参数拟合二次曲线，实现未标定 ISO 值的插值
- 使用色卡均匀区域的 burst 采样提高标定精度

### 训练数据合成 (Sec 3.3)

四步流程：
1. 将线性 RGB 图像像素值调整到目标传感器范围（考虑黑电平/白电平）
2. 从核池中随机选择 RGB 模糊核，应用后在目标 SR 尺度下采样
3. 按传感器 CFA 模式进行马赛克化
4. 反转白平衡增益后，按随机采样的 ISO 级别添加合成噪声

### 网络架构

- 使用 RRDBNet 架构进行 RAW-to-RGB SR
- 输入 RAW 转为 4 通道 GBRG 图像（2×2 block 堆叠），末端加上采样层
- 仅使用 $\ell_1$ 损失，不用 GAN 和感知损失以避免幻觉伪影

## 实验关键数据

### 实验设置

- **9 款手机相机**：Pixel 9 Pro Tele/Main, Pixel 6 Main, S24U Tele 2, S23U Tele 1/Main, S23+ Tele/Main, Mi 11 Main
- **两种评估方式**：相机特定 SR（非盲）+ 跨相机 SR（盲，测试相机退化不参与训练）
- **指标**：PSNR, SSIM（参考指标）；MTF50, MTF25（无参考，刻画细节恢复能力）
- **基线**：Bicubic, KernelGAN, MANet, Real-ESRGAN, Degradation-Transfer, RAWSR, BSRAW

### Table 1: 相机特定 SR 定量结果（4× SR）

| 方法 | S23U Tele 1 PSNR/SSIM | S24U Tele 2 PSNR/SSIM | Pixel 9 Pro Main PSNR/SSIM | Pixel 9 Pro Tele PSNR/SSIM |
|---|---|---|---|---|
| Bicubic | 32.01 / 0.952 | 33.42 / 0.935 | 30.88 / 0.874 | 30.96 / 0.868 |
| KernelGAN | 33.12 / 0.950 | 33.48 / 0.937 | 31.10 / 0.856 | 32.66 / 0.883 |
| MANet | 33.42 / 0.959 | 33.47 / 0.935 | 33.14 / 0.889 | 32.93 / 0.886 |
| Real-ESRGAN | 32.92 / 0.941 | 33.03 / 0.949 | 33.38 / 0.888 | 33.21 / 0.889 |
| Degradation-Transfer | 33.47 / 0.954 | 33.01 / 0.948 | 32.98 / 0.885 | 32.76 / 0.878 |
| **Ours (Camera-specific)** | **33.59 / 0.961** | **33.53 / 0.956** | **33.47 / 0.901** | **33.32 / 0.889** |

MTF50 方面差距更显著：S24U Tele 2 上本方法 MTF50=2.00（即 SR 后 50% 对比度的空间频率提升 200%），而 MANet 仅 0.29（29%）。

### Table 2: 跨相机 SR 定量结果（测试相机退化不在训练集中）

| 相机 | 方法 | MTF50 | MTF25 | PSNR | SSIM |
|---|---|---|---|---|---|
| Pixel 6 Main | Real-ESRGAN | 0.38 | 0.50 | 31.75 | 0.884 |
| | BSRAW | 0.97 | 0.82 | 28.79 | 0.713 |
| | RAWSR | 0.96 | 0.90 | 28.99 | 0.790 |
| | **Ours (Cross-camera)** | **1.19** | **0.86** | **32.41** | **0.905** |
| Mi 11 Main | Real-ESRGAN | 0.33 | 0.40 | 35.08 | 0.914 |
| | BSRAW | 0.34 | 0.43 | 33.65 | 0.817 |
| | RAWSR | 0.37 | 0.53 | 33.76 | 0.850 |
| | **Ours (Cross-camera)** | **0.89** | **0.74** | **35.72** | **0.938** |

在跨相机盲 SR 场景中优势更明显：Pixel 6 Main 上 PSNR 较 Real-ESRGAN 提升 **0.66 dB**，SSIM 提升 **0.021**；MTF50 从 0.38 提升至 1.19（3× 改善）。

## 亮点与洞察

- **标定驱动 vs 通用退化池**：首次系统性地证明了精心标定的设备特定退化模型优于大量随机采样的通用退化，在所有指标上一致占优
- **RAW 域建模的必要性**：在 RAW 域（场景辐射与传感器响应线性相关）进行退化建模比 sRGB 域更准确，减少了非线性处理带来的域差距
- **跨设备迁移能力**：不同智能手机的退化特征存在相似性（通过 t-SNE 可视化验证），用 7 款相机的标定退化训练的模型能在未见设备上取得最优性能
- **MTF 评估指标**：引入基于 Siemens 星靶的 MTF50/MTF25 作为无参考评估指标，比传统 PSNR/SSIM 更好地反映细节恢复质量
- **工程完整性**：提供了从灰度码对齐、显示器-传感器辐射标定、到核优化的完整技术方案，且公开标定数据

## 局限性

- **标定设备依赖**：核标定需要专业显示器-相机拍摄装置，过程较为繁琐，难以快速扩展到新设备
- **场景限制**：结论仅适用于智能手机且在良好光照条件下，未验证对 DSLR 相机或低光场景的适用性
- **空间不变假设**：虽然在 FOV 上分 patch 标定核，但每个 patch 内仍假设核为空间不变的
- **网络架构局限**：仅使用 RRDBNet 架构验证，未探索 Transformer 等更强架构是否能进一步放大退化建模的收益
- **噪声模型简化**：HG 噪声模型未考虑行噪声、固定模式噪声等更复杂的传感器噪声成分

## 相关工作

| 方向 | 代表方法 | 本文区别 |
|---|---|---|
| RAW 域 SR | RAWSR, BSRAW, Zhou et al. | 不从通用退化池采样，而是精确标定每台设备 |
| 隐式退化建模 | Bulat et al., CinCGAN, DSGAN | GAN 隐式学习退化；本文显式标定更可控可解释 |
| 显式退化建模 | Real-ESRGAN, KernelGAN, MANet | 通用参数化核 + sRGB 域操作；本文设备特定 + RAW 域 |
| 相机 PSF 标定 | Degradation-Transfer, Diamond et al. | 仅标定 1× PSF，不适合高倍 SR；本文直接标定 4× SR 核 |
| Unprocessing | Brooks et al., Graphics2RAW | 反处理流程但退化参数不精确；本文用标定参数生成更逼真 RAW |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将相机标定的严谨方法引入 SR 退化建模，系统性地证明"标定优于随机"有明确方法论贡献
- 实验充分度: ⭐⭐⭐⭐⭐ — 9 款相机、6 个基线、参考+无参考双指标、相机特定+跨相机双场景、t-SNE 退化分布可视化，非常全面
- 写作质量: ⭐⭐⭐⭐ — 技术细节严谨，标定流程文档化完善，补充材料详尽
- 价值: ⭐⭐⭐⭐ — 在手机计算摄影 SR 领域有直接工程价值，公开标定数据可促进后续研究

<!-- RELATED:START -->

## 相关论文

- [Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [Spatially-Variant Degradation Model for Dataset-free Super-resolution](../../ECCV2024/image_restoration/spatially-variant_degradation_model_for_dataset-free_super-resolution.md)
- [Audio Super-Resolution with Latent Bridge Models](../../NeurIPS2025/image_restoration/audio_super-resolution_with_latent_bridge_models.md)
- [Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)

<!-- RELATED:END -->
