---
title: >-
  [论文解读] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion
description: >-
  [CVPR 2025][自动驾驶][恶劣天气LiDAR生成] 本文提出 WeatherGen，首个统一的多样恶劣天气 LiDAR 数据扩散生成框架，通过 Spider Mamba 生成器保持 LiDAR 物理结构、对比学习控制器实现天气可控生成，在数据保真度和下游检测性能上均显著超越基于物理模拟的方法。
tags:
  - CVPR 2025
  - 自动驾驶
  - 恶劣天气LiDAR生成
  - 扩散模型
  - Mamba
  - 对比学习
  - 数据增强
---

# WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion

**会议**: CVPR 2025  
**arXiv**: [2504.13561](https://arxiv.org/abs/2504.13561)  
**代码**: [https://github.com/wuyang98/weathergen](https://github.com/wuyang98/weathergen)  
**领域**: 自动驾驶 / 点云生成  
**关键词**: 恶劣天气LiDAR生成, 扩散模型, Mamba, 对比学习, 数据增强

## 一句话总结

本文提出 WeatherGen，首个统一的多样恶劣天气 LiDAR 数据扩散生成框架，通过 Spider Mamba 生成器保持 LiDAR 物理结构、对比学习控制器实现天气可控生成，在数据保真度和下游检测性能上均显著超越基于物理模拟的方法。

## 研究背景与动机

**领域现状**：自动驾驶在晴天场景下的 3D 感知已取得显著进展，但恶劣天气（雪、雾、雨）下的 LiDAR 数据极为稀缺。激光在恶劣天气中会发生散射和衍射，产生噪声点和点丢失，严重影响感知模型的可靠性。

**现有痛点**：(1) 现有模拟方法（FSRL、LSS、LISA）只能用单一物理模型模拟单一天气，无法在统一框架下处理多种天气。(2) 由于光学传播过程的复杂性和传感器参数的不完全知识，这些物理模拟方法生成的数据保真度有限，与真实数据存在明显域偏移。(3) LiDAR 数据本身的稀疏性和环状结构对生成模型提出了独特挑战。

**核心矛盾**：恶劣天气下的 LiDAR 数据需求量大但采集成本高昂，现有非学习式的模拟方法无法提供高保真数据来满足生成模型的训练需求。

**本文目标**：构建一个统一的、高保真的多天气 LiDAR 数据生成框架，能够以低成本生成可促进下游任务性能的恶劣天气数据。

**切入角度**：(1) 用可学习的 map-based 数据生产器提供预训练数据（解决训练数据不足问题）；(2) 设计适配 LiDAR 物理结构的 Mamba 生成器（解决保真度问题）；(3) 用 CLIP 监督的对比学习控制器提供判别性天气控制信号（解决统一生成的可控性问题）。

**核心 idea**：将 LiDAR 数据投影为 range map，利用 DDPM 扩散框架 + 专门为 LiDAR 环状结构设计的 spider mamba scan 进行去噪生成，结合对比学习实现天气可控的统一生成。

## 方法详解

### 整体框架

WeatherGen 包含三个核心组件：(1) Map-based Data Producer (MDP)——将晴天 LiDAR range map 通过可学习掩码转换为恶劣天气 range map，提供预训练数据；(2) Spider Mamba Generator (SMG)——基于 DDPM 的 U-Net 去噪网络，用 spider mamba scan 替代传统卷积/自注意力来建模 LiDAR 特征；(3) Contrastive Learning-based Controller (CLC)——使用天气编码器 + CLIP 文本编码器的对比学习生成判别性天气控制信号。训练采用预训练 + 微调策略，推理时仅需 CLC 和 SMG。

### 关键设计

1. **Map-based Data Producer (MDP)**:

    - 功能：为预训练阶段提供大量高质量的多天气 LiDAR 数据
    - 核心思路：将晴天 LiDAR 数据转换为 range map $\mathcal{R}_c$ 后，通过 map-to-map 方式生成天气数据 $\mathcal{R}_w$。生成过程包括三个部分：(a) 基于距离阈值 $r_w$ 的 Bernoulli 掩码 $\mathcal{M}_e$ 模拟激光远距离衰减和随机丢点；(b) 随机噪声 $\mathcal{R}_n$ 模拟天气引起的噪声点；(c) 可学习掩码 $\mathcal{M}_d$ 实现与真实数据分布的自适应对齐。$r_w$ 越小代表天气越恶劣。不同天气有不同的距离衰减策略（雪：远近都有丢点；雾：主要远距衰减；雨：湿地面导致远处丢点）
    - 设计动机：与纯物理模拟器不同，MDP 引入了可学习掩码，使其能从数据中学习真实分布特征。这是"预训练+微调"策略的关键——先用大量 MDP 数据预训练，再用少量真实恶劣天气数据微调

2. **Spider Mamba Generator (SMG)**:

    - 功能：在扩散去噪过程中建模 LiDAR 特征交互，保持 LiDAR 数据的物理结构
    - 核心思路：与图像 Mamba 在 patch 级别扫描不同，spider mamba 在像素级别沿 LiDAR 的光束圆环（range map 的行）和中心射线（range map 的列）进行扫描。这就像蜘蛛在网上捕猎——沿网的经线和纬线移动。技术实现上，将特征 $\mathcal{F}$ 按通道拆分为 4 部分节省计算，每部分各自经过线性投影和 $L$ 层 Mamba 处理，最后拼接还原。U-Net 结构的编解码器各包含卷积块和 spider mamba 块。后续的 Latent Feature Aligner (LFA) 通过 KL 散度对齐生成数据和真实数据在潜空间中的天气模式分布
    - 设计动机：range map 的行对应 LiDAR 的光束圆环、列对应中心射线，沿行/列扫描正好符合 LiDAR 成像过程。相比之下，卷积只能局部建模，自注意力无序连接所有点会破坏 LiDAR 物理结构。在 patch 级别操作的 Vision Mamba 在稀疏户外场景中会丢失点云的几何意义

3. **Contrastive Learning-based Controller (CLC)**:

    - 功能：生成具有紧凑判别性语义知识的天气控制信号，引导模型在不同天气条件下进行判别性生成
    - 核心思路：包含天气编码器 $\mathcal{W}$ 和冻结 CLIP 文本编码器 $\mathcal{C}$。天气编码器从 range map 提取天气嵌入 $\textbf{w}$，CLIP 编码四种预设天气文本提示得到锚点嵌入 $\bm{c}_i$。通过信息瓶颈的对比学习目标优化：最小化天气嵌入与不相关天气锚点的互信息 $I(\textbf{w}, \textbf{c}_{i\neq j})$，最大化与对应天气锚点的互信息 $I(\textbf{w}, \textbf{c}_{i=j})$。控制信号 $\textbf{w}$ 与时间步嵌入拼接后融入 SMG
    - 设计动机：在统一生成框架中，需要紧凑且判别性强的控制信号来区分不同天气。直接使用天气标签或简单 one-hot 编码缺乏语义信息。CLIP 的语言监督为天气控制信号提供了有结构化的语义锚点，对比学习进一步压缩和聚焦相关信息

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{SMG} + \mathcal{L}_{LFA} + \mathcal{L}_{CLC}$，其中 $\mathcal{L}_{SMG}$ 为标准扩散去噪的 MSE 损失，$\mathcal{L}_{LFA}$ 为生成数据与真实数据潜变量分布的 KL 散度，$\mathcal{L}_{CLC}$ 为对比学习的信息瓶颈损失。训练分两阶段：先用 MDP 生成的数据预训练全部模块，再用 Seeing Through Fog 的真实恶劣天气数据微调除 MDP 外的所有参数。

## 实验关键数据

### 主实验

KITTI-360 无条件生成性能：

| 方法 | FPD↓ | FRD↓ | MMD×10⁻⁴↓ | JSD×10⁻²↓ |
|------|------|------|-----------|-----------|
| LiDARGen | 90.29 | 579.39 | 7.39 | 7.38 |
| R2DM | 6.24 | 149.66 | 1.91 | 3.05 |
| Text2LiDAR | 4.81 | 164.16 | 0.49 | 2.01 |
| **WeatherGen** | 6.15 | **138.62** | **0.39** | **1.99** |

Seeing Through Fog 天气条件生成性能（vs 物理模拟方法）：

| 方法 | FPD↓ | FRD×10¹↓ | MMD×10⁻⁴↓ | JSD×10⁻¹↓ |
|------|------|---------|-----------|-----------|
| LSS (Snow) | 106.37 | 142.17 | 3.59 | 2.11 |
| FSRL (Fog) | 319.32 | 210.51 | 8.56 | 3.69 |
| LISA (Rain) | 301.11 | 145.23 | 4.30 | 3.23 |
| **WG (Snow)** | **59.28** | **124.17** | **1.71** | **0.77** |
| **WG (Fog)** | 314.14 | 196.89 | 8.08 | **2.66** |
| **WG (Rain)** | **86.40** | **127.06** | 4.15 | **0.93** |

### 消融实验

在 Seeing Through Fog "heavy snow" 测试集上：

| 配置 | FPD↓ | 说明 |
|------|------|------|
| w/o MDP (不用可学习掩码) | 性能下降 | 固定参数模拟不够灵活 |
| w/o SMG (替换为U-Net) | 性能下降 | 雪景噪声多，无spider mamba不能保持LiDAR物理结构 |
| w/o CLC (仅用天气编码器) | 全面下降 | 缺乏判别性控制信号导致天气生成不精确 |
| w/o LFA | 性能下降 | 无法对齐真实数据分布 |
| Full model | **最佳** | 所有组件互相补充 |

### 关键发现

- WeatherGen 在雪和雨的生成保真度上显著超越物理模拟方法（FPD 分别降低 44%/71%），证明学习式方法在捕获复杂光学传播方面远优于手工物理模型
- 仅用 7.4% 的 WeatherGen 生成数据替换晴天训练集中的对应量，就能在 3D 目标检测的 21 个指标中 18 个取得最佳，证明生成数据的实用价值
- Spider mamba scan 的物理结构保持能力在可视化中体现明显——生成物体轮廓更清晰、小目标更完整
- 对比学习控制器的缺失导致全面性能下降，说明判别性天气控制信号对于统一生成框架至关重要

## 亮点与洞察

- **首个统一天气 LiDAR 生成框架**：从"每种天气一个模拟器"到"一个框架覆盖所有天气"的范式转变，显著降低了数据获取成本
- **Spider Mamba 的端到端设计**：沿光束圆环和中心射线扫描完美适配 LiDAR 成像物理过程。这一思路可推广到任何具有特定成像几何的传感器数据（如雷达、声纳）
- **预训练+微调的数据工程策略**：用大量 MDP 模拟数据预训练、小量真实数据微调，巧妙解决了恶劣天气真实数据稀缺的问题
- **CLIP 语言监督的控制信号**：将天气语义锚定在 CLIP 的文本空间中，为控制信号提供了结构化的语义基础。这一策略可推广到其他需要属性可控生成的场景

## 局限与展望

- 目前仅支持雪、雾、雨三种天气，未考虑沙尘暴、冰雹等极端情况
- MDP 的物理模型较为简化，更精细的光学传播建模可能进一步提升数据质量
- 生成数据的标注依赖 LabelCloud 手动标注，限制了大规模数据集的构建效率
- 未探讨时序一致性——当前逐帧独立生成，连续帧之间的天气特征可能不一致
- Seeing Through Fog 数据集规模较小（200-3000帧/天气），限制了微调效果的上限

## 相关工作与启发

- **vs FSRL/LSS/LISA**: 这些方法各自为一种天气设计独立的物理模拟器，保真度受限于不完整的物理模型。WeatherGen 通过学习真实数据分布，在统一框架中生成更高保真的多天气数据
- **vs R2DM/LiDM/Text2LiDAR**: 这些扩散式 LiDAR 生成方法只关注晴天场景，且使用标准卷积/自注意力破坏了 LiDAR 物理结构。WeatherGen 的 spider mamba 设计更适合 LiDAR 数据
- **vs Vision Mamba**: Vision Mamba 在 patch 级别扫描适合图像，但在稀疏 LiDAR 数据中 patch 级操作会丢失几何意义。Spider mamba 的点级沿行/列扫描更契合 LiDAR 特性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个统一天气LiDAR生成框架，spider mamba设计精妙
- 实验充分度: ⭐⭐⭐⭐ 无条件+条件生成+下游检测验证+消融全面，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，但部分技术细节放在附录
- 价值: ⭐⭐⭐⭐⭐ 直接解决自动驾驶恶劣天气数据稀缺的实际痛点，具有很高的工程应用价值

<!-- RELATED:START -->

## 相关论文

- [CAWM-Mamba: A Unified Model for Infrared-Visible Image Fusion and Compound Adverse Weather Restoration](cawm-mamba_a_unified_model_for_infrared-visible_image_fusion_and_compound_advers.md)
- [PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization](superpc_a_single_diffusion_model_for_point_cloud_completion_upsampling_denoising.md)

<!-- RELATED:END -->
