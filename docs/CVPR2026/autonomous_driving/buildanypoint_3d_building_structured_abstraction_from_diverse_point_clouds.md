---
title: >-
  [论文解读] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds
description: >-
  [CVPR 2026][自动驾驶][建筑抽象重建] 提出BuildAnyPoint，通过**松耦合级联扩散Transformer(Loca-DiT)**实现从多样分布的点云（机载LiDAR、SfM、稀疏噪声点云）到结构化3D建筑Mesh的统一重建——先用分层潜在扩散恢复底层点云分布，再用自回归Transformer生成紧凑多边形Mesh。
tags:
  - CVPR 2026
  - 自动驾驶
  - 建筑抽象重建
  - 点云补全
  - 潜在扩散
  - 自回归Mesh生成
  - 级联生成框架
---

# BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds

**会议**: CVPR 2026  
**arXiv**: [2602.23645](https://arxiv.org/abs/2602.23645)  
**代码**: [项目页](https://ai4city-hkust.github.io/BuildAnyPoint/) (有)  
**领域**: Autonomous Driving / 3D Vision / 城市重建  
**关键词**: 建筑抽象重建, 点云补全, 潜在扩散, 自回归Mesh生成, 级联生成框架  

## 一句话总结

提出BuildAnyPoint，通过**松耦合级联扩散Transformer(Loca-DiT)**实现从多样分布的点云（机载LiDAR、SfM、稀疏噪声点云）到结构化3D建筑Mesh的统一重建——先用分层潜在扩散恢复底层点云分布，再用自回归Transformer生成紧凑多边形Mesh。

## 研究背景与动机

**领域现状**：从城市点云恢复轻量3D建筑模型是数字孪生、导航、灾害仿真等应用的关键需求。现有方法包括基于优化的（平面检测+组装）和基于学习的方案，但通常**只能处理特定分布的点云**。

**现有痛点**：
   - Point2Building：首创直接从点云自回归生成Mesh，但单步自回归经常产生**几何歧义**和mesh-点云错位
   - ArcPro：引入建筑语法中间表示降低歧义，但受限于**预定义几何体**（如柱体拉伸），无法处理倾斜屋顶等复杂结构，且假设每个模块有相对完整的局部点云

**核心矛盾**：如何既保持对任意点云分布的**泛化性**，又确保生成Mesh的**结构一致性和几何精度**？直接将异构点云输入自回归Mesh生成器效果差，因为这些生成器需要高质量、干净、完整的点云。

**本文目标**：构建首个**通用框架**，从任意分布的点云（LiDAR、SfM、极稀疏噪声）恢复结构化建筑抽象Mesh。

**切入角度**：利用**显式3D生成先验**约束解空间——与其直接从异构点云生成Mesh，不如先恢复背后的均匀密集点云分布，再交给已有的高质量Mesh生成器。

**核心idea**：松耦合级联 = 分层潜在扩散（恢复分布）+ 自回归Transformer（生成Mesh），通过一系列潜在空间转换渐进弥合非结构化点云→结构化Mesh的模态鸿沟。

## 方法详解

### 整体框架

Loca-DiT（图3）学习条件分布 $p_\text{BAP}(\mathcal{M} | \mathcal{P}_{in})$，分解为两个阶段：

1. **几何补全阶段**（潜在扩散）：$p(\mathcal{P}_{out} | \mathcal{P}_{in})$ — 从稀疏/噪声点云恢复均匀密集的完整点云
2. **结构化Mesh生成阶段**（自回归Transformer）：$p(\mathcal{M} | \mathcal{P}_{out})$ — 从恢复的点云自回归生成Mesh token序列

### 关键设计

#### 1. 三级潜在空间转换

- **功能**：设计三种潜在空间渐进弥合点云→Mesh的表示鸿沟
- **核心思路**：
    - **稠密潜在网格 $\mathcal{G}_d$**：对GT点云低分辨率体素化后用稀疏VAE编码，在瓶颈层**密化**稀疏网格为稠密网格——为解码器提供完整空间上下文，能"雕刻"出未占用区域
    - **稀疏潜在网格 $\mathcal{G}_s$**：高分辨率体素化后用稀疏VAE编码——精细化几何细节
    - **序列化token $\mathcal{T}_P$**：用预训练点云编码器将恢复的点云编码为固定长度token序列——与目标Mesh token序列 $\mathcal{T}_M$ 对齐
- **设计动机**：点云适合在连续稠密潜在空间编码几何细节，Mesh需要离散序列化表示生成结构拓扑——不同阶段用不同潜在空间让各阶段可以专精化

#### 2. 分层潜在扩散

- **功能**：分两级恢复建筑的完整几何先验
- **核心思路**：
    - 粗级扩散模型 $p_{\theta_d}(\mathcal{G}_d | \mathcal{P}_{in})$：在稠密网格上去噪，恢复基础形状
    - 细级扩散模型 $p_{\theta_s}(\mathcal{G}_s | \mathcal{G}_d)$：条件于粗级输出在稀疏网格上去噪，精细化高分辨率几何
    - 训练目标：标准去噪损失 $\min_\theta \mathbb{E}[\|\epsilon - \epsilon_\theta(\mathbf{z}_t, t)\|_2^2]$
    - 条件化：点云编码器将 $\mathcal{P}_{in}$ 量化为体素网格，拼接到潜在特征上
- **设计动机**：分层方式先恢复粗略结构再精细化，比单级扩散更稳定

#### 3. 自回归Mesh生成

- **功能**：从恢复的点云条件化生成低面数、拓扑一致的建筑Mesh
- **核心思路**：
    - 基于MeshAnything V2的decoder-only Transformer
    - 输入序列 $\mathcal{T} = [\mathcal{T}_P; \mathcal{T}_M^{<t}]$，自回归预测下一个mesh token
    - 训练目标：最大化条件对数似然 $\max_\phi \sum_{t=1}^N \log P(t_m^N | \mathcal{T}_P, \mathcal{T}_M^{<t}; \phi)$
- **设计动机**：恢复的高质量点云+法向量模拟了自回归Mesh生成器所需的"艺术级"输入

### 损失函数

- 稀疏VAE：BCE（生成 vs 目标）+ KL散度 + 法向量学习
- 扩散模型：去噪MSE损失
- Transformer：交叉熵next-token prediction损失

## 实验关键数据

### 主实验：建筑结构化抽象

| 方法 | #V↓ | #F↓ | #P↓ | FR↓ | CD↓ |
|------|-----|-----|-----|-----|-----|
| City3D（优化方法） | 173 | 72 | 14 | 6% | 0.167 |
| Point2Building（学习方法） | 20 | 34 | 18 | 1% | 0.043 |
| **BuildAnyPoint** | **10** | **16** | **8** | **0%** | **0.036** |

- **顶点数仅10个**（vs P2B的20），面数仅16（vs 34），更紧凑的低多边形Mesh
- 失败率0%，CD最低

### 点云补全基准

| 方法 | F-score↑ | CD↓ | Uniformity↓ | EMD↓ |
|------|----------|-----|-------------|------|
| PoinTr | 0.85 | 0.41 | 0.25 | 0.12 |
| AnchorFormer | 0.82 | 0.39 | 1.27 | 0.13 |
| **BuildAnyPoint** | **0.91** | **0.35** | **0.04** | **0.10** |

- 均匀性分数0.04，比所有竞争方法**低近一个数量级**

### 消融实验

| 设置 | #V↓ | #F↓ | CD↓ |
|------|-----|-----|-----|
| 无3D生成先验（直接从 $\mathcal{P}_{in}$ 生成） | 78 | 127 | 0.107 |
| **完整模型**（从 $\mathcal{P}_{out}$ 生成） | **38** | **70** | **0.034** |

- 去掉粗级 $\mathcal{G}_d$：恢复的点云混乱
- 去掉细级 $\mathcal{G}_s$：出现"双表面"效应误导后续Mesh生成
- 用传统solver替代Transformer：即使有恢复的好点云也无法生成有效表面

### 关键发现

1. **3D生成先验是关键**：无先验时CD从0.034恶化到0.107，面数从70爆增到127
2. **分层扩散二者缺一不可**：粗级提供基础形状、细级提供表面精度
3. **跨三种点云分布均有效**：LiDAR、SfM、稀疏噪声——真正实现了"BuildAnyPoint"的目标

## 亮点与洞察

1. **解耦思想出色**：不试图一步到位从异构点云生成Mesh，而是拆分为"恢复分布"+"生成Mesh"两个子问题，各用最适合的生成范式（扩散 vs 自回归）——这种级联设计哲学在其他跨模态生成中同样适用
2. **密化瓶颈的巧妙设计**：在稀疏VAE瓶颈处将稀疏网格密化，让解码器能同时看到"有"和"无"的区域进行形状推理
3. **中间产物即为SOTA**：仅作为框架中间表示的恢复点云，在建筑点云补全基准上就已达到SOTA

## 局限与展望

1. **数据集几何多样性受限**：公开建筑数据集偏向简单几何，复杂结构（如哥特式建筑、不规则形态）的建模能力受限
2. **无高度/地理坐标等先验**：未利用建筑的物理约束（如重力、对称性）和地理信息
3. **推理速度未报告**：扩散采样+自回归解码的级联可能较慢
4. 仅在The Hague/Rotterdam数据集测试，地理泛化性未知

## 相关工作与启发

- **MeshAnything系列**：从点云自回归生成Mesh的范式在快速演进，本文证明了**输入点云质量**对Mesh质量的决定性影响
- **XCube**：分层稀疏VAE+扩散的3D生成框架，本文在此基础上增加条件化和密化操作
- **松耦合级联**：将生成过程分解为概率独立的子阶段，每个阶段用最适合的生成模型——这种"分而治之"策略值得在更广泛的3D生成任务中推广

## 评分

⭐⭐⭐⭐ — 框架设计优雅，三种点云分布的通用性令人印象深刻，中间产物本身就是SOTA，但应用场景相对垂直

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](../../CVPR2025/autonomous_driving/weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2026\] Points-to-3D: Structure-Aware 3D Generation with Point Cloud Priors](points-to-3d_structure-aware_3d_generation_with_point_cloud_priors.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](../../AAAI2026/autonomous_driving/understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](../../CVPR2025/autonomous_driving/reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](../../ICCV2025/autonomous_driving/robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)

<!-- RELATED:END -->
