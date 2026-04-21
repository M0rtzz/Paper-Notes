---
title: >-
  [论文解读] SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic
description: >-
  [ECCV 2024][自动驾驶][Transformer] SLEDGE 提出了首个基于生成模型的驾驶仿真器，通过 Raster-to-Vector 自编码器将驾驶场景编码为栅格化潜在图（RLM），再利用 Diffusion Transformer 生成高质量的车道图和交通参与者，实现了比 nuPlan 少 500 倍存储（<4GB）的仿真环境，同时支持 500m 长路线测试，暴露了 SOTA 规划器 PDM-Closed 超过 40% 的失败率。
tags:
  - ECCV 2024
  - 自动驾驶
  - Transformer
  - 驾驶仿真
  - 车道图生成
  - 潜在扩散模型
  - 运动规划
---

# SLEDGE: Synthesizing Driving Environments with Generative Models and Rule-Based Traffic

**会议**: ECCV 2024  
**arXiv**: [2403.17933](https://arxiv.org/abs/2403.17933)  
**代码**: [GitHub](https://github.com/autonomousvision/sledge)  
**领域**: 自动驾驶  
**关键词**: 扩散 Transformer, 驾驶仿真, 车道图生成, 潜在扩散模型, 运动规划

## 一句话总结

SLEDGE 提出了首个基于生成模型的驾驶仿真器，通过 Raster-to-Vector 自编码器将驾驶场景编码为栅格化潜在图（RLM），再利用 Diffusion Transformer 生成高质量的车道图和交通参与者，实现了比 nuPlan 少 500 倍存储（<4GB）的仿真环境，同时支持 500m 长路线测试，暴露了 SOTA 规划器 PDM-Closed 超过 40% 的失败率。

## 研究背景与动机

**领域现状**: 数据驱动的驾驶仿真器（如 nuPlan、Waymax）是评估自动驾驶规划算法的关键工具。这些仿真器通过重放（replay）真实驾驶日志中的抽象 BEV 表示（车道、交通灯、动态/静态目标）来初始化仿真环境。

**现有痛点**:
   - **存储需求巨大**: nuPlan 包含 1300 小时的驾驶日志，需要超过 2TB 的存储空间，严重提高了研究门槛
   - **路线受限**: 基于日志重放的仿真器只能在短时间（约 15 秒）和有限路线内进行测试，因为一旦规划器偏离录制路线，仿真环境的覆盖就无法保证
   - **可控性差**: 无法灵活调整交通密度、路线难度等参数来进行更全面的压力测试

**核心矛盾**: 生成模型在图像合成领域取得巨大成功，但驾驶场景的抽象表示（变长向量集合、拓扑连通性、几何精度要求）与图像的均匀网格结构截然不同，导致现代生成模型无法直接应用

**本文目标** 如何用生成模型合成仿真可用的驾驶场景（包括车道图、交通参与者），替代依赖海量日志的重放方式

**切入角度**: 设计一种统一的栅格化表示，将所有场景实体（车道、交通灯、车辆、行人、障碍物）编码到固定大小的 2D 潜在空间中，使得标准的 Diffusion Transformer 可以直接生成

**核心 idea**: 用 Raster-to-Vector 自编码器将变长向量化的驾驶场景映射为固定大小的栅格化潜在图（RLM），然后用 DiT 进行潜在扩散生成

## 方法详解

### 整体框架

SLEDGE 的生成流程分为两大阶段：(1) 训练 Raster-to-Vector Autoencoder（RVAE）学习场景的编解码；(2) 在冻结编码器产生的 RLM 上训练 Diffusion Transformer（DiT）。推理时，DiT 从噪声中生成 RLM，再由 RVAE 的解码器恢复向量化的场景实体，最后初始化基于规则的交通仿真。

### 关键设计

1. **nuPlan 向量表示（Scene State $\mathcal{S}$）**:

    - 做什么: 定义仿真所需的完整场景状态
    - 核心思路: 场景由多种实体组成：
        - 车道 $\mathbf{L} \in \mathbb{R}^{20 \times 2}$：20 个 BEV 点的折线，加邻接矩阵 $\mathbf{A} \in \mathbb{R}^{N \times N}$ 编码拓扑连通性
        - 交通灯：红灯 $\mathcal{R}$ 和绿灯 $\mathcal{G}$，与车道同格式的 $20 \times 2$ 折线
        - 交通参与者：行人 $\mathcal{P}$、车辆 $\mathcal{V}$、静态障碍物 $\mathcal{O}$，用 2D 中心、朝向、尺寸、速度描述
        - 自车速度 $\mathbf{v} \in \mathbb{R}^2$
        - 完整场景状态: $\mathcal{S} = \{\mathcal{M}, \mathcal{R}, \mathcal{G}, \mathcal{P}, \mathcal{V}, \mathcal{O}, \mathbf{v}\}$
    - 设计动机: 这种向量化表示是仿真器的标准输入格式，但实体数量可变、拓扑约束复杂，直接建模困难

2. **栅格化状态图（RSI）与 Raster-to-Vector 自编码器（RVAE）**:

    - 做什么: 将变长向量场景统一编码为固定大小的 2D 栅格化潜在图（RLM）
    - 核心思路:
        - **栅格化函数** $\rho: \mathcal{S} \rightarrow \mathbf{I}$：将场景编码为 12 通道的栅格化状态图 $\mathbf{I} \in \mathbb{R}^{W \times H \times 12}$，每种实体占 2 个通道。折线用方向向量 $\Delta = [dx, dy]$ 编码，动态目标用 2D 速度填充，静态障碍物用朝向向量填充
        - **栅格编码器** $\pi$：ResNet-50 将 RSI 下采样为紧凑的 RLM $\mathbf{M} = \pi(\mathbf{I})$，形状为 $8 \times 8 \times 64$
        - **通道分组**: RLM 的通道分为两组 $C = C_L + C_A$，车道组 $8 \times 8 \times 32$ 和交通参与者组 $8 \times 8 \times 32$
        - **向量解码器** $\phi$：基于 DETR 范式的 Transformer 解码器，用 $1 \times 1$ 空间 token 化的 RLM 作为 key/value，配合可学习的实体 query 解码出折线坐标、边界框属性和存在概率 $p \in [0,1]$
        - **通道分组掩码**: 在交叉注意力中实施二值掩码——车道 query 只能 attend 车道 token，其他 query 只能 attend 交通参与者 token。这使得在已知车道的条件下可以单独生成交通参与者
    - 设计动机: 统一不同实体类型为固定大小的 2D 表示，既兼容主流扩散模型架构，又通过通道分组支持条件生成

3. **Diffusion Transformer（DiT）**:

    - 做什么: 在 RLM 潜在空间中学习数据分布，生成新的驾驶场景
    - 核心思路:
        - **训练**: 采用 DDPM 算法，对每个场景的 RLM $\mathbf{M}$ 加噪 $\hat{\mathbf{M}} = \mathbf{M} + \sigma \boldsymbol{\mathcal{E}}$，DiT 预测噪声 $\delta(\hat{\mathbf{M}}; \mathbf{c}, \sigma)$，条件向量 $\mathbf{c}$ 为城市的 one-hot 标签（区分美国右行 vs 新加坡左行），使用 AdaLN-Zero 机制注入条件信息，优化 L2 重建损失
        - **推理**: 从噪声 $\hat{\mathbf{M}} \sim \mathcal{N}(0, \sigma_{\max}^2 \mathbf{I})$ 开始迭代去噪，解码后保留存在概率 $> \tau$ 的实体，重叠边界框保留最高概率者
        - **邻接矩阵恢复**: 通过匹配端点距离 < 1.5m 且朝向差 < 60° 的车道来提取拓扑连通性
        - **条件生成（Inpainting）**: 利用扩散模型天然的 inpainting 能力实现两个任务：(a) 已知车道条件下生成交通参与者——编码车道 token，去噪交通参与者 token；(b) 路线外推——沿路线迭代采样新位姿，将上一 tile 的 RSI 仿射变换到新位姿，已知区域作为条件补全未知区域
    - 设计动机: DiT 架构简洁、可扩展、无下/上采样操作，天然兼容任意空间分辨率的 RLM；inpainting 机制使得场景可无限延伸

4. **SLEDGE 仿真环境**:

    - 做什么: 利用生成的场景初始化反应式仿真
    - 核心思路:
        - **Hard Routes**: 从车道图中提取多条有效路线，选择转弯次数最多的作为「困难」路线
        - **Hard Traffic**: 对同一路线生成多个交通配置，选择交通参与者最多的作为「困难」交通
        - **行为仿真**: 非自车车辆投影到最近车道中心线，沿中心线行驶，纵向控制使用 Intelligent Driver Model（IDM）；行人保持匀速直线；交通灯每 15 秒切换
        - **仿真半径**: 仅仿真距自车 $\alpha = 64m$ 内的交通参与者，远处参与者保持静止，支持 500m 长路线（150 秒）的可扩展仿真
    - 设计动机: 通过动态仿真半径突破传统仿真器的路线长度限制，hard routes/traffic 提供更具挑战性的评估

### 损失函数 / 训练策略

- **RVAE 训练**:
    - 重建损失: Hungarian 匹配后计算所有属性的 L1 误差
    - 存在损失: 二值交叉熵，判断 query 是否匹配到真实实体
    - KL 散度损失: 正则化 RLM 的潜在分布
- **DiT 训练**:
    - L2 噪声重建损失: $\|\boldsymbol{\mathcal{E}} - \delta(\hat{\mathbf{M}}; \mathbf{c}, \sigma)\|_2^2$
    - 噪声尺度 $\sigma$ 从对数正态分布采样
- **模型规模**: DiT-L（138M 参数）和 DiT-XL（487M 参数），patch size $1 \times 1$
- **数据**: nuPlan 数据集，450k 训练帧 + 50k 验证帧，四座城市，64m × 64m FOV

## 实验关键数据

### 主实验：车道图重建质量

| 表示方法 | 固定大小 | 通道分组 | 大小(KB) | GEO F1↑ | TOPO F1↑ | TOPO Chamfer↓ |
|---------|---------|---------|---------|---------|---------|---------------|
| RSI | ✓ | ✓ | 524.3 | 0.933 | 0.851 | 64.824 |
| RLM (无掩码) | ✓ | ✗ | 16.0 | 0.981 | 0.945 | 20.096 |
| RLM (有掩码) | ✓ | ✓ | 8.0 | 0.980 | 0.944 | 20.624 |
| Vector (上界) | ✗ | ✓ | 4.8 | 0.997 | 0.990 | 4.174 |

### 主实验：车道图生成质量

| 方法 | 表示 | 路线长度↑ | Precision(RVEnc)↑ | Recall(RVEnc)↑ | Reach↓ | Convenience↓ |
|------|------|----------|-------------------|----------------|--------|-------------|
| VAE | RSI | 2.68±3.66 | 0.00 | 0.16 | 2.86 | 13.06 |
| HDMapGen | Vector | 28.17±14.81 | 7.48 | 12.45 | 2.49 | 18.10 |
| DiT-L | RSI | 24.78±10.38 | 19.20 | 5.94 | 1.90 | 3.95 |
| DiT-L | RLM | 32.51±9.93 | **63.99** | **61.60** | 0.88 | 3.10 |
| DiT-XL | RLM | **35.37±10.28** | **78.07** | **72.63** | **0.20** | **0.47** |

### 仿真实验：PDM-Closed 规划器失败率

| 任务 | 路线长度 | 路线/交通难度 | 转弯数 | 车辆数 | 失败率(PFR) |
|------|---------|-------------|--------|--------|------------|
| Replay | 100m | - | 0.89 | 57.40 | 0.06 |
| Lane→Agent | 100m | Easy/Easy | 0.89 | 44.61 | 0.07 |
| Lane→Agent | 500m | Hard/Hard | 4.20 | 170.87 | **0.44** |
| Lane&Agent | 100m | Easy/Easy | 0.61 | 27.30 | 0.22 |
| Lane&Agent | 500m | Hard/Hard | 3.82 | 169.66 | **0.49** |

### 关键发现

- RLM 表示仅 8KB 即可达到接近上界的重建质量（F1=0.980），比 524KB 的 RSI 表示在拓扑指标上大幅领先
- 通道分组掩码对重建质量几乎无影响，却能支持条件生成
- DiT-XL 在所有生成指标上大幅超越其他方法，性能随计算量显著提升，但对数据量不敏感（说明多样性比数量重要）
- 500m 长路线仿真暴露了 PDM-Closed 的致命弱点：无法变道和超车，这些在现有 15 秒短仿真中不易发现
- 困难路线+密集交通下，SOTA 规划器的失败率从 6% 飙升至 49%

## 亮点与洞察

1. **极致压缩**: 仅需 <4GB 即可完整设置仿真环境，相比 nuPlan 的 2TB 压缩了近 500 倍，大幅降低研究门槛
2. **统一表示设计精巧**: RSI 的 12 通道编码方案（方向向量编码折线、速度向量编码动态目标）自然而紧凑，channel group masking 巧妙实现条件/联合生成的灵活切换
3. **Inpainting 路线外推**: 利用扩散模型天然的 inpainting 能力实现场景无限延伸，无需额外训练，思路优雅
4. **评估价值**: 不仅是生成工具，更揭示了当前规划算法的盲区——长距离驾驶和复杂交通场景下的脆弱性

## 局限与展望

1. **FOV 和仿真半径较小**: 64m × 64m 的 FOV 和 64m 的仿真半径限制了高速场景的适用性
2. **车道表示过于简化**: 仅使用中心线，假设恒定车道宽度，缺少车道边界、标线等细节
3. **交通行为简单**: IDM 和匀速行人模型过于理想化，缺乏真实的交互行为
4. **评估不够充分**: 缺少在强化学习等下游任务中的验证
5. **计算开销高**: 扩散模型的推理成本较高，可考虑应用一致性蒸馏等加速技术

## 相关工作与启发

- **Scenario Diffusion**: 最接近的先驱工作，用潜在扩散+光栅解码器生成车辆，但不支持车道图生成和长距离仿真
- **HDMapGen**: 自回归逐节点生成车道图，质量和可扩展性不如本文的并行生成方案
- **DriveSceneGen**: 并发工作，在图像空间扩散生成车道和车辆，但启发式更多、效率更低
- **启发**: RLM 这种"向量→栅格→向量"的编码-解码范式可推广到其他需要生成结构化场景的任务

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个完整的生成式驾驶仿真器，RVAE+DiT 的架构设计具有原创性
- **实验充分度**: ⭐⭐⭐⭐ 系统的表示对比、生成质量评估、scaling 分析、下游仿真验证，指标体系完善
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义清晰，方法展开条理分明，图表精美直观
- **实用价值**: ⭐⭐⭐⭐⭐ 500 倍存储压缩极具实用意义，开源代码，显著降低研究门槛

<!-- RELATED:START -->

## 相关论文

- [OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multimodal_3d_occupancy_prediction_for_aut.md)
- [Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [SimWorld-Robotics: Synthesizing Photorealistic and Dynamic Urban Environments for Multimodal Robot Navigation and Collaboration](../../NeurIPS2025/autonomous_driving/simworld-robotics_synthesizing_photorealistic_and_dynamic_urban_environments_for.md)
- [ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](../../ICCV2025/autonomous_driving/recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)
- [DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)

<!-- RELATED:END -->
