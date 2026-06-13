---
title: >-
  [论文解读] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data
description: >-
  [ICCV 2025][3D视觉][合成数据] 证明通过高保真**程序化合成数据**即可训练出精度媲美基础模型（如 Sapiens-2B）的以人为中心的稠密预测模型，仅需 **30 万合成图像**、**0.3B 参数**、训练成本不到同级方案的 1/16，在深度估计、表面法线估计、软前景分割三项任务上实现 SOTA 或近 SOTA 性能。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "合成数据"
  - "深度估计"
  - "表面法线估计"
  - "前景分割"
  - "DPT"
  - "高保真标注"
---

# DAViD: Data-efficient and Accurate Vision Models from Synthetic Data

**会议**: ICCV 2025  
**arXiv**: 2507.15365  
**代码**: [项目主页](https://aka.ms/DAViD)  
**领域**: 3D视觉  
**关键词**: 合成数据, 深度估计, 表面法线估计, 前景分割, DPT, 高保真标注  
**作者**: Fatemeh Saleh, Sadegh Aliakbarian, Charlie Hewitt 等 (Microsoft, Cambridge)

## 一句话总结

证明通过高保真**程序化合成数据**即可训练出精度媲美基础模型（如 Sapiens-2B）的以人为中心的稠密预测模型，仅需 **30 万合成图像**、**0.3B 参数**、训练成本不到同级方案的 1/16，在深度估计、表面法线估计、软前景分割三项任务上实现 SOTA 或近 SOTA 性能。

## 研究背景与动机

以人为中心的计算机视觉在深度估计、法线估计、前景分割等稠密预测任务上的标注获取极为困难：

**人工标注不可行**：逐像素的深度/法线标注对人类来说几乎不可能

**实验室采集受限**：复杂相机阵列或专用传感器获取的标注基于光度测量或噪声传感器，保真度有限

**实验室多样性不足**：受限于拍摄环境，难以覆盖真正的 in-the-wild 场景

**现有方法计算成本高**：Sapiens 预训练最大变体需要 1024 块 A100 训练 18 天，DepthPro 使用多阶段混合训练

关键观察：现有扫描式合成数据（如 THuman）的网格质量有限，尤其在头发、眼睛、手指等细节区域存在严重瑕疵。而**程序化合成数据**可以同时提供高保真度和完美标注。

本文的核心主张：**数据质量 >> 数据数量 + 模型大小**——用高质量合成数据训练简单模型即可匹敌大规模基础模型。

## 方法详解

### SynthHuman 数据集

基于 Hewitt et al. 的程序化数据生成管线，包含：
- **30 万张图像**，分辨率 384×512
- 面部、上半身、全身场景各占 1/3
- 每张图像附带深度、表面法线、软前景掩码真值
- 数据多样性覆盖姿态、环境、光照、外观
- 渲染时间：300 台 M60 GPU 机器 72 小时（等价 4 块 A100 约 2 周）

与扫描式合成数据（THuman, RenderPeople）对比：SynthHuman 在头发丝、眼镜、衣物褶皱等细节上具有显著更高的标注保真度，且无扫描伪影。

### 模型架构

采用统一架构处理三项任务，基于 DPT (Dense Prediction Transformer) 改进：

**编码器 (Encoder)**：ViT 骨干，使用 $\text{Read}_{proj}$ 读出操作：
$$e^l = \text{mlp}(\text{cat}(\texttt{CLS}^l, t_i^l))$$

**缩放器 (Resizer)**：轻量级全卷积图像编码器，在原始分辨率上提取特征，避免高分辨率输入时 ViT 的二次复杂度开销。ViT 编码器固定接收 384×384 输入。

**解码器 (Decoder)**：融合三路输入——上一解码块输出 $d$、编码器特征 $e$、缩放器特征 $r$：
$$d_{\text{int}}^l = \text{RConv}(d^{l-1} + \text{Interp}(\text{RConv}(e^l)))$$
$$d^l = \text{Conv}([r^l, \text{Interp}(d_{\text{int}}^l)])$$

**卷积头**：不同任务输出通道数不同（深度 1、分割 1、法线 3）。

Resizer 的设计使得推理时编码器计算量恒定，高分辨率信息由轻量卷积处理，远优于增加 ViT token 数的方案。

### 损失函数

**软前景分割**：$\mathcal{L}_\alpha = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{L1} + \mathcal{L}_{\text{dice}} + \omega_{\text{lap}} \mathcal{L}_{\text{lap}}$

**表面法线估计**：$\mathcal{L}_\eta = 1 - \eta \cdot \hat{\eta}$（余弦相似度，仅在前景区域计算）

**深度估计**：$\mathcal{L}_d = \mathcal{L}_{\text{MSE}}(s\hat{d}+t, d) + \omega_{\text{grad}} \mathcal{L}_{\text{grad}}(s\hat{d}+t, d)$
（shift-and-scale-invariant + 梯度监督，仅前景区域）

## 实验关键数据

### 主实验：深度估计

| 方法 | GFLOPs | 参数量 | Goliath-Face RMSE↓ | Goliath-Full RMSE↓ | Hi4D RMSE↓ | 平均 AbsRel↓ |
|------|--------|--------|-------------------|-------------------|-----------|-------------|
| MiDaS-DPT_L | - | 0.34B | 0.224 | 0.973 | 0.148 | 0.027 |
| DepthAnythingV2-L | 1827 | 0.34B | 0.229 | 1.039 | 0.130 | 0.025 |
| Sapiens-0.3B | 1242 | 0.34B | 0.179 | 0.690 | 0.116 | 0.021 |
| Sapiens-2B | 8709 | 2.16B | 0.158 | 0.266 | 0.095 | 0.015 |
| DepthPro | 4370 | 0.50B | 0.295 | 0.723 | 0.084 | 0.016 |
| **Ours-Base** | **344** | **0.12B** | **0.142** | **0.376** | **0.085** | **0.014** |
| **Ours-Large** | **663** | **0.34B** | **0.140** | **0.334** | **0.072** | **0.012** |

- Ours-Large 以 **663 GFLOPs** 达到与 Sapiens-2B（8709 GFLOPs）相当的精度，计算量仅为 **1/13**
- 0.12B 的 Base 模型已超过 0.34B 的 DepthAnythingV2-L 和 Sapiens-0.3B
- 约 **48 FPS** on A100

### 表面法线估计

| 方法 | Goliath-Face Mean↓ | Goliath-Full Mean↓ | Hi4D Mean↓ |
|------|-------------------|-------------------|-----------|
| Sapiens-0.3B | 18.86° | 15.72° | 15.04° |
| Sapiens-2B | 16.04° | 11.49° | 12.14° |
| **Ours-Large** | **17.15°** | **14.60°** | **15.37°** |

- 0.34B 模型性能超过同尺寸 Sapiens-0.3B
- 作者指出 Goliath/Hi4D 的真值标注本身较粗糙（口腔内部、衣物褶皱等细节缺失），模型预测反而捕获了更多细节

### 软前景分割

| 方法 | PhotoMatte85 SAD↓ | PhotoMatte85 MSE↓ | PPM-100 SAD↓ |
|------|------------------|------------------|-------------|
| MODNet | 13.94 | 0.003 | 104.35 |
| P3M-Net | 20.05 | 0.007 | 142.74 |
| **Ours** | **5.85** | **0.0009** | **78.17** |

### 消融实验

| 消融维度 | 变量 | Goliath RMSE↓ | Hi4D RMSE↓ |
|---------|------|--------------|-----------|
| **数据源** | THuman2.0 | 0.495 | 0.137 |
| | RenderPeople | 0.278 | 0.076 |
| | **SynthHuman** | **0.253** | **0.072** |
| **数据量** | 60K | 0.324 | 0.101 |
| | 150K | 0.305 | 0.085 |
| | **300K** | **0.278** | **0.085** |
| **模型尺寸** | ViT-Small | 0.310 | 0.089 |
| | ViT-Base | 0.278 | 0.085 |
| | **ViT-Large** | **0.253** | **0.072** |

关键发现：
- **数据质量差异巨大**：SynthHuman 相比 THuman2.0，Goliath RMSE 从 0.495 降至 0.253（48% 下降）
- 数据量从 60K→300K 有持续增益
- 多任务训练在某些指标（如 PPM-100 分割 SAD 66.08 vs 78.17）甚至优于单任务

## 亮点与洞察

1. **数据为王的实证**：在同等模型架构下，高保真合成数据可以弥补数据量和模型尺寸的不足——0.12B 模型超越 0.34B 的通用基础模型
2. **统一架构三任务**：仅改变输出通道数和损失函数，无需任务特定设计
3. **Resizer 模块的效率设计**：固定 ViT 输入尺寸 + 轻量卷积处理原始分辨率，避免 ViT token 增长的二次开销
4. **合成数据的隐含优势**：数据溯源、使用权、用户同意有强保障，且可通过程序控制数据多样性来应对公平性问题
5. **真值标注质量的反思**：论文发现 Goliath/Hi4D 等真实数据集的标注反而比模型预测更粗糙，暗示当前评估可能存在天花板效应

## 局限性

1. **仅限以人为中心场景**：模型和数据集专门设计用于人体相关任务，不适用于通用视觉
2. **依赖高质量合成管线**：程序化数据生成管线本身需要大量艺术创建的资产（配饰、衣物、环境）
3. **评估上的天花板效应**：当预测质量超过真值时，指标可能低估实际性能
4. **缺乏与最新方法的对比**：如 Depth Anything V2 的蒸馏策略未被充分讨论

## 相关工作与启发

- **与 Sapiens 的对比**：Sapiens 使用 3 亿真实图像自监督预训练 + 50 万合成图像微调，本文仅用 30 万合成图像直接训练，简化了流程且性能相当
- **与 DepthPro 的差异**：DepthPro 使用多阶段混合训练（真实+合成），本文证明纯合成数据即可
- **对合成数据方向的启示**：程序化数据生成比扫描式合成更有前景——保真度、多样性、标注质量同时占优
- **David vs Goliath 隐喻**：小数据集+小模型 vs 大基础模型，名字选择本身就传递了核心信息

## 评分

- **新颖性**: ⭐⭐⭐ — 核心贡献在数据和工程层面，模型架构为 DPT 改进，方法论创新度一般
- **实验**: ⭐⭐⭐⭐⭐ — 三任务全面评估，消融详尽（数据源/数据量/模型尺寸/多任务），FLOPs 对比清晰
- **写作**: ⭐⭐⭐⭐ — 论述清晰，Figure 2/4 的真值对比直观有力
- **价值**: ⭐⭐⭐⭐ — 对"数据中心AI"范式有直接实证支持，开源数据集和模型增强了可复现性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multi-view_diffusion_model_with_synthetic_data.md)
- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[ICCV 2025\] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](vit-split_unleashing_the_power_of_vision_foundation_models_via_efficient_splitti.md)
- [\[ICCV 2025\] Adversarial Exploitation of Data Diversity Improves Visual Localization](adversarial_exploitation_of_data_diversity_improves_visual_localization.md)

</div>

<!-- RELATED:END -->
