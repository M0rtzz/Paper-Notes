---
title: >-
  [论文解读] LaRa: Efficient Large-Baseline Radiance Fields
description: >-
  [ECCV 2024][3D视觉][大基线重建] 提出LaRa前馈重建模型，通过**高斯体积（Gaussian Volume）**表示和**分组注意力层（Group Attention Layer）**统一局部与全局推理，仅需4张图像即可从大基线视角重建360°辐射场，且仅用**4×A100训练2天**即可超越LGM等费时方法。
tags:
  - ECCV 2024
  - 3D视觉
  - 大基线重建
  - 前馈辐射场
  - 高斯体素
  - 分组注意力
  - 2D Gaussian Splatting
---

# LaRa: Efficient Large-Baseline Radiance Fields

**会议**: ECCV 2024  
**arXiv**: [2407.04699](https://arxiv.org/abs/2407.04699)  
**代码**: https://apchenstu.github.io/LaRa/ (有)  
**领域**: 3D视觉  
**关键词**: 大基线重建, 前馈辐射场, 高斯体素, 分组注意力, 2D Gaussian Splatting

## 一句话总结

提出LaRa前馈重建模型，通过**高斯体积（Gaussian Volume）**表示和**分组注意力层（Group Attention Layer）**统一局部与全局推理，仅需4张图像即可从大基线视角重建360°辐射场，且仅用**4×A100训练2天**即可超越LGM等费时方法。

## 研究背景与动机

**领域现状**: 神经辐射场在逐场景优化和小基线设置下已取得优秀效果，但在前馈（feed-forward）大基线重建方面仍面临挑战。

**现有痛点**: 
   - 基于特征匹配的方法（MVSNeRF, MuRF）依赖图像重叠区域，无法处理大基线
   - 基于全局注意力的方法（LGM, GRM）忽略了3D重建的局部性，需32×A100级GPU资源
   - 缺乏3D归纳偏置导致重建模糊

**核心矛盾**: 全局注意力能建模长程依赖但计算昂贵且忽视局部几何约束 vs. 局部匹配高效但无法处理大视角差

**本文目标**: 用有限计算资源（学术级GPU）实现高质量大基线前馈3D重建

**切入角度**: 将体素分成局部组进行组内交叉注意力（模拟局部匹配），再用3D CNN传播组间信息（实现全局协调）

**核心 idea**: 用分组注意力在transformer中统一局部特征匹配与全局信息传播，以高效的方式实现大基线辐射场重建。

## 方法详解

### 整体框架

给定 $M=4$ 张图像及相机参数，LaRa通过三步输出Gaussian Volume：

1. **特征提取**: DINO编码器提取2D特征，通过Plücker射线注入相机信息，反投影到3D特征体积 $\mathbf{V}_f$
2. **体积变换器**: 分组注意力层逐步更新可学习嵌入体积 $\mathbf{V}_e$，输出高斯体积 $\mathbf{V}_{\mathcal{G}}$
3. **粗-精解码**: 从体素特征解码2D高斯参数，经高效光栅化渲染高分辨率图像

$$\mathbf{V}_{\mathcal{G}} = \{\mathcal{G}_i^k\}_{k=1}^K = \mathbf{f}(\mathbf{v}; \mathbf{I}, \boldsymbol{\pi})$$

### 关键设计

1. **高斯体积表示（Gaussian Volume）**: 每个体素存储 $K=2$ 个2D高斯基元，每个基元包含不透明度 $\alpha$、切向量 $\mathbf{t}$、缩放 $\mathbf{S}$、球谐系数和位移偏移 $\Delta \in [-1,1]^3$。基元位置为 $\mathbf{p}_i^k = \mathbf{v}_i + r \cdot \Delta_i^k$，其中 $r = 1/32$ 为最大位移范围。设计动机是将无序点集预测问题结构化为体素内的局部偏移预测，降低学习难度。同时使用2D Gaussian Splatting（而非3DGS）以便进行表面正则化和网格提取。

2. **分组注意力层（Group Attention Layer）**: 将体积展开为 $G=16$ 个局部组，仅在组内执行交叉注意力，然后用3D CNN在组间传播信息。核心公式：

$$\dot{\mathbf{V}}_e^{g,j} = \text{GroupCrossAttn}(\text{LN}(\mathbf{V}_e^{g,j}), \mathbf{V}_f^g) + \mathbf{V}_e^{g,j}$$

$$\ddot{\mathbf{V}}_e^{g,j} = \text{MLP}(\text{LN}(\dot{\mathbf{V}}_e^{g,j})) + \dot{\mathbf{V}}_e^{g,j}$$

$$\mathbf{V}_e^{j+1} = \text{3DCNN}(\text{LN}(\ddot{\mathbf{V}}_e^j)) + \ddot{\mathbf{V}}_e^j$$

三个子层各带残差连接，共12层堆叠。不同组在batch维度并行处理，大幅提升训练效率。关键洞察：$G=1$（全局注意力）需22天训练30 epochs，而 $G=16$ 仅需2天且效果更好。

3. **粗-精解码（Coarse-Fine Decoding）**: 

    - **粗模块**: 轻量MLP将体素特征解码为2D高斯参数，渲染得到RGB/深度/透明度图
    - **精模块**: 将高斯基元中心投影到粗渲染结果和原始图像上采样特征，使用位移特征 $|\hat{\mathbf{D}}_{\mathbf{p}} - z_{\mathbf{p}}|$（渲染深度 vs 基元深度之差）实现遮挡感知推理，通过交叉注意力+MLP预测残差球谐系数：

$$\text{SH}_{i,k}^{\text{fine}} = \text{SH}_{i,k}^{\text{coarse}} + \text{SH}_{i,k}^{\text{residuals}}$$

设计动机：DINO编码器和注意力层会丢失高频纹理信息，精模块通过直接查询原始图像特征来弥补。

4. **Plücker射线调制**: 使用Plücker射线（相机位置与射线方向的叉积）而非外参/内参矩阵来编码相机信息，通过AdaLN注入2D特征。优势是参数化独立于物体尺度、相机位置和焦距，增强泛化性。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{\text{MSE}}(\mathcal{I}, \hat{\mathcal{I}}) + \mathcal{L}_{\text{SSIM}}(\mathcal{I}, \hat{\mathcal{I}}) + \mathcal{L}_{\text{Reg}}$$

正则化项（15 epoch后启用）：

$$\mathcal{L}_{\text{Reg}} = \gamma_d \mathcal{L}_d + \gamma_n \mathcal{L}_n$$

- $\mathcal{L}_d = \sum_{i,j} \omega_i \omega_j |z_i - z_j|$：蒸馏损失，集中射线权重到表面（$\gamma_d=1000$）
- $\mathcal{L}_n = \sum_i \omega_i(1 - \mathbf{n}_i^\top \mathbf{N})$：法线一致性损失（$\gamma_n=0.2$）

训练：AdamW, lr=$2\times10^{-4}$, 余弦退火，50 epochs（每epoch 50K iter），4×A100-40G。125M可训练参数。Objaverse数据集264K场景，K-means选4个输入视角，8个视角用于监督。

## 实验关键数据

### 主实验 - 新视角合成（4输入视角）

| 方法 | Gobjaverse PSNR↑ | GSO PSNR↑ | Co3D PSNR↑ | Gobjaverse LPIPS↓ |
|------|------------------|-----------|------------|-------------------|
| MVSNeRF | 14.48 | 15.21 | 12.94 | 0.1856 |
| MuRF | 14.05 | 12.89 | 11.60 | 0.3018 |
| LGM (32×A100-80G) | 19.67 | 23.67 | 13.81 | 0.1576 |
| **Ours-fast (2天, 4×A100-40G)** | **25.30** | **26.79** | **21.56** | **0.1027** |
| **Ours (3.5天)** | **26.14** | **27.65** | **21.64** | **0.0932** |

### 消融实验 - 分组数与模块效果

| 设计变体 | Gobjaverse PSNR↑ | GSO PSNR↑ | 几何精度(0.01)↑ |
|---------|------------------|-----------|----------------|
| G=4（大组/少分组） | 22.27 | 23.06 | 31.0% |
| G=8 | 23.80 | 25.30 | 42.8% |
| w/o $\mathcal{L}_{\text{Reg}}$ | 26.16 | 27.71 | 45.6% |
| 仅粗模块 | 25.06 | 26.28 | 52.2% |
| **Full model (G=16)** | **25.30** | **26.79** | **52.2%** |

### 关键发现

- 在所有数据集上以巨大优势超越所有基线：Gobjaverse上PSNR 26.14 vs LGM的19.67（+32.9%）
- 计算效率极高：4×A100-40G (2天) vs LGM的32×A100-80G (GPU时数差32倍)
- 在真实数据Co3D上泛化良好（21.64），而LGM因依赖固定相机-物体距离仅有13.81
- 分组注意力 $G=16$ vs $G=4$ 显著提升（+3 PSNR），因为局部attention更符合3D匹配本质
- 去掉正则化后渲染指标反而提升，但几何质量下降且出现浮体
- 粗-精解码带来约+0.5 PSNR的纹理细节提升

## 亮点与洞察

- 分组注意力层是关键贡献：将3D重建的局部匹配本质融入transformer设计，比暴力全局注意力更高效且更好
- Plücker射线调制使模型对场景尺度和焦距变化鲁棒，这是在Co3D等无约束数据上泛化的关键
- 粗-精解码中的位移特征 $|\hat{D} - z|$ 巧妙解决了遮挡推理问题
- 体素内偏移取代绝对坐标预测是高效的设计，将无序点集生成转化为结构化的回归问题
- 训练资源需求极低（学术可复现），是对LGM等工业级方法的有力回应

## 局限与展望

- 体积分辨率固定为 $64^3$，可能限制大场景/细节的表示能力
- 仅展示了bounded物体重建，未验证unbounded/室外场景
- 每个体素仅2个高斯基元，对薄结构和半透明物体可能不足
- 虽然训练高效，但体积transformer在推理时的显存占用仍受分辨率限制
- 依赖DINO特征作为图像编码器，对不在DINO训练分布内的风格图像可能效果退化

## 相关工作与启发

- **MVSNeRF**: 代表性的cost volume + 体渲染方法，限于小基线
- **LGM / GRM**: 同期工作，全局transformer生成3DGS，效果好但资源消耗巨大
- **2D Gaussian Splatting**: 本文的渲染基元，比3DGS更利于表面建模和网格提取
- **启发**: 分组注意力策略可泛化到其他3D任务（如点云分割、3D检测中的体积transformer）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 分组注意力+高斯体积的组合在大基线重建中效果显著
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集 + 零样本泛化 + 网格提取 + 详细消融
- **写作质量**: ⭐⭐⭐⭐ — 框架图清晰，各组件逻辑连贯
- **价值**: ⭐⭐⭐⭐⭐ — 效果强、训练经济、可复现性高，对前馈3D重建方向极具参考价值

<!-- RELATED:START -->

## 相关论文

- [GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)
- [G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream](benerf_neural_radiance_fields_from_a_single_blurry_image_and_event_stream.md)
- [Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [SlotLifter: Slot-guided Feature Lifting for Learning Object-centric Radiance Fields](slotlifter_slot-guided_feature_lifting_for_learning_object-centric_radiance_fiel.md)

<!-- RELATED:END -->
