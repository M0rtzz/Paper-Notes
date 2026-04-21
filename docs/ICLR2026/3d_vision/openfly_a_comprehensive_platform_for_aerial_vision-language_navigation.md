---
title: >-
  [论文解读] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation
description: >-
  [ICLR 2026][3D视觉][空中VLN] 构建OpenFly——航空视觉-语言导航(VLN)综合平台：集成4种渲染引擎(UE/GTA V/Google Earth/3DGS)+开发全自动数据生成工具链(点云获取→语义分割→轨迹生成→GPT-4o指令)+构建10万轨迹大规模数据集(18场景)+提出关键帧感知VLN模型OpenFly-Agent(关键帧选择+视觉token融合)，在已见/未见场景分别以14.0%/7.9%的成功率优势超越现有方法。
tags:
  - ICLR 2026
  - 3D视觉
  - 空中VLN
  - 无人机导航
  - 多渲染引擎
  - 自动数据生成
  - 关键帧感知
  - 3D高斯溅射
---

# OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation

**会议**: ICLR 2026  
**arXiv**: [2502.18041](https://arxiv.org/abs/2502.18041)  
**代码**: 有(开源)  
**领域**: 无人机导航/具身AI  
**关键词**: 空中VLN, 无人机导航, 多渲染引擎, 自动数据生成, 关键帧感知, 3D高斯溅射

## 一句话总结

构建OpenFly——航空视觉-语言导航(VLN)综合平台：集成4种渲染引擎(UE/GTA V/Google Earth/3DGS)+开发全自动数据生成工具链(点云获取→语义分割→轨迹生成→GPT-4o指令)+构建10万轨迹大规模数据集(18场景)+提出关键帧感知VLN模型OpenFly-Agent(关键帧选择+视觉token融合)，在已见/未见场景分别以14.0%/7.9%的成功率优势超越现有方法。

## 研究背景与动机

**VLN领域发展**：VLN是具身AI核心任务，让agent根据语言指令和视觉观察导航到目标。已有大量室内/地面数据集(R2R、RxR、TouchDown、VLN-CE等)推动了方法发展，但无人机(UAV)作为航拍/救援/货运的关键平台，其VLN研究仍然不足。

**先驱工作的局限**：AerialVLN和OpenUAV利用AirSim+UE模拟器初步建立了空中VLN数据集，但面临三大挑战：数据多样性受限、收集成本高、数据规模小。

**数据多样性瓶颈**：现有方法依赖AirSim和Unreal Engine控制无人机，仅能使用与这些平台兼容的数字资产，限制了环境的多样性和真实感，无法引入更多高保真数据源。

**人工标注成本高**：轨迹生成依赖飞行员在模拟器中操作无人机，再由标注员手动编写语言指令。整个流程劳动密集、耗时长、难以规模化。

**数据规模严重不足**：当前航空VLN数据集仅约1万条轨迹，远远落后于机器人操作领域——Open X-Embodiment和EO-1已收集超过100万操作episode，数据匮乏严重制约模型能力。

**核心思路**：(1)多渲染引擎集成→解决多样性; (2)全自动化工具链→解决成本; (3)10万规模数据集→解决规模; (4)关键帧感知模型→解决长序列视觉冗余。

## 方法详解

### 1. 多渲染引擎集成

OpenFly集成4种渲染引擎/技术，极大丰富了场景资源：

- **Unreal Engine (UE)**：提供8个城市场景，覆盖超过 $100 \text{km}^2$，包含建筑、车辆、行人等丰富资产
- **GTA V**：贡献高度真实的城市景观，以洛杉矶为原型建模
- **Google Earth**：提供4个城市区域(Berkeley/大阪/华盛顿D.C./圣路易斯)，覆盖 $53.60 \text{km}^2$
- **3D Gaussian Splatting (3DGS)**：利用层级3DGS从无人机采集的真实图像重建3D场景，覆盖超过 $7 \text{km}^2$ 的5个校园场景，实现real-to-sim渲染

### 2. 自动数据生成工具链

工具链包含4个自动化模块，设计了3个统一接口控制agent运动和获取传感器数据：

**点云获取**：
- 光栅化采样重建(UE/GTA V)：在适当分辨率的采样点获取局部点云并拼接
- 基于图像的稀疏重建(3DGS)：使用COLMAP从输入图像生成稀疏点云

**语义分割**（三种方法灵活选择）：
- 3D场景理解：捕获俯视图序列 → Octree-Graph提取语义3D实例
- 点云投影+轮廓提取：体素化点云投影到地面 → 分割轮廓 → GPT-4o标注语义
- 手动标注：点云质量低或需精细分割时的备选方案

**自动轨迹生成**：
- 从场景点云构建全局体素地图 $M_{global}$
- 随机选择地标作为目标，在一定距离选起点，靠近地标选终点
- 基于 $M_{global}$ 和自定义动作空间，使用A*算法搜索无碰撞轨迹
- 迭代选择终点为新起点，可生成复杂轨迹

**自动指令生成**：
- 关键策略：按动作转换点将完整轨迹分割为子轨迹，而非输入全部图像
- 提取每段子轨迹的关键动作和最后3帧图像 → GPT-4o生成子指令
- LLM整合所有子指令为完整导航指令
- 随机抽样3K样本人工检查 → 合格率达91%

### 3. OpenFly-Agent：关键帧感知VLN模型

基于OpenVLA构建，核心创新是**关键帧选择**和**视觉token融合**：

**关键帧选择**：
- 动机：均匀帧采样不适合空中VLN，可能错过包含关键地标的帧
- 启发式方法：识别无人机运动变化点 → 提取变化点及前后两帧 → 构成候选关键帧集
- 地标定位模块：3层交叉注意力，融合LLM隐状态中的文本和图像特征，预测指令相关地标的边界框 $\mathbf{b} \in \mathcal{R}^4$
- 筛选规则：候选帧中边界框面积大于阈值 $\theta$ 的保留为最终关键帧

**视觉Token融合 (VTM)**：
- 选择关键帧集中边界框最大的帧作为参考帧（包含最关键的地标观测）
- 密集计算参考帧与其他帧各视觉token对的余弦相似度
- 高相似度token通过平均融合，未融合的对比帧token被丢弃
- 迭代执行直到遍历整个关键帧集
- 维护容量为 $K$ 的记忆库（FIFO策略保留最新关键帧）
- 关键帧内部通过grid pooling进一步压缩，当前帧保持未压缩以获取最新视觉观测

**动作预测**：词表最后256个token作为动作特殊token，定义6种无人机动作 $\{$Forward, Turn Left, Turn Right, Move Up, Move Down, Stop$\}$。

## 实验结果

### 表1：VLN数据集对比

| 数据集 | 轨迹数 | 词表大小 | 路径长度(m) | 指令长度 | 动作空间 | 环境 |
|--------|--------|---------|------------|---------|---------|------|
| R2R | 7189 | 3.1K | 10.0 | 29 | graph | Matterport3D |
| RxR | 13992 | 7.0K | 14.9 | 129 | graph | Matterport3D |
| AerialVLN | 8446 | 4.5K | 661.8 | 83 | 4 DoF | AirSim+UE |
| CityNav | 32637 | 6.6K | 545 | 26 | 4 DoF | SensatUrban |
| OpenUAV | 12149 | 10.8K | 255 | 104 | 6 DoF | AirSim+UE |
| **OpenFly** | **100K** | **15.6K** | **99.1** | **59** | **4 DoF** | **多引擎** |

### 表2：测试集导航性能对比

| 方法 | NE↓(seen) | SR↑(seen) | OSR↑(seen) | SPL↑(seen) | NE↓(unseen) | SR↑(unseen) | OSR↑(unseen) | SPL↑(unseen) |
|------|----------|----------|-----------|-----------|------------|------------|-------------|-------------|
| Random | 242m | 0.7% | 0.8% | 0% | 301m | 0.1% | 0.1% | 0% |
| Seq2Seq | 205m | 2.9% | 24.3% | 2.6% | 229m | 2.1% | 20.6% | 1.1% |
| CMA | 161m | 5.4% | 28.1% | 4.8% | 217m | 4.6% | 24.4% | 2.1% |
| AerialVLN | 139m | 7.5% | 30.0% | 6.8% | 214m | 7.3% | 28.1% | 4.4% |
| Navid | 153m | 13.0% | 38.2% | 11.6% | 210m | 10.8% | 27.2% | 5.0% |
| NaVila | 132m | 20.3% | 53.5% | 17.8% | 202m | 14.7% | 42.1% | 9.6% |
| **OpenFly-Agent** | **93m** | **34.3%** | **64.3%** | **24.9%** | **154m** | **22.6%** | **56.2%** | **19.1%** |

### 表3：消融实验(test-seen)

| 方法 | NE↓ | SR↑ | OSR↑ | SPL↑ |
|------|-----|-----|------|------|
| OpenVLA (baseline) | 231m | 2.3% | 10.8% | 2.2% |
| History (均匀采样) | 223m | 6.9% | 23.3% | 5.6% |
| Random KS | 264m | 8.7% | 26.6% | 5.8% |
| KS (关键帧选择) | 275m | 9.2% | 28.1% | 6.1% |
| History + VTM | 215m | 16.6% | 40.5% | 9.1% |
| **KS + VTM** | **93m** | **34.3%** | **64.3%** | **24.9%** |

## 关键发现

1. **关键帧选择+视觉token融合的协同效应极为显著**：单独使用KS(SR 9.2%)或History+VTM(SR 16.6%)效果有限，组合使用(SR 34.3%)产生超线性提升。原因是VTM解决了文本-图像token数量不平衡问题，避免背景噪声稀释对关键线索的注意力。

2. **多引擎训练数据的泛化优势**：在真实世界23个场景实验中，用OpenFly数据训练的模型(SR 26.09%, OSR 34.78%)大幅优于AerialVLN数据训练的模型，证明多引擎数据有效弥合sim-to-real gap。

3. **VLM在空中VLN中的巨大潜力**：基于VLM的Navid/NaVila显著优于传统Seq2Seq/CMA方法，尤其在Oracle SR上差距明显(38-53% vs 24-28%)，说明VLM的推理能力对导航至关重要。

4. **短-中程指令更贴近实际使用**：OpenFly平均轨迹长度99.1m、指令长度59词，远低于AerialVLN(661.8m/83词)。作者论证这更符合自然人类使用习惯，且对推动空中VLN更有益。

5. **自动指令生成质量可靠**：基于GPT-4o的子轨迹分割策略+LLM整合，随机抽样3K样本人工检查合格率91%，且支持高并发快速生成大量指令。

## 亮点与洞察

- **系统级创新而非单点突破**：OpenFly的贡献不在于某个模型组件的创新性，而在于整合4引擎+自动工具链+10万数据集+关键帧模型的完整platform，形成闭环。
- **3DGS的real-to-sim应用**：无人机采集真实图像→3DGS重建→在重建场景中自动生成训练数据→部署到真实无人机，闭环验证了新范式。
- **工程价值极高**：用户可利用OpenFly工具链在自己的场景快速生成定制数据，具有基础设施级贡献。
- **数据规模的量变到质变**：10万轨迹（vs 现有~1万）首次让空中VLN数据规模与地面VLN可比，OpenVLA的迁移效果也依赖这一规模。

## 局限性

1. **绝对成功率仍然不高**：即使是最优的OpenFly-Agent，test-seen SR也仅34.3%、test-unseen仅22.6%，说明空中VLN仍极具挑战性，距离实用部署差距明显。
2. **泛化能力有限**：所有方法在unseen场景上性能显著下降（SR从34.3%→22.6%），跨场景泛化仍是核心瓶颈。
3. **依赖GPT-4o**：指令生成和语义标注依赖商业闭源VLM，成本和可复现性受限。
4. **动作空间简化**：使用固定步长(3/6/9m)的离散动作，与真实无人机的连续控制有差距。虽然提供了连续轨迹支持，但主要实验仍基于离散动作。
5. **Google Earth数据仅限高空视角**：为保证视觉质量，Google Earth仅采集高空数据(4.46%)，限制了低空真实场景覆盖。

## 相关工作对比

### vs AerialVLN (ICCV 2023)
AerialVLN是首个空中VLN数据集(8446轨迹)，但仅使用AirSim+UE单一引擎，轨迹和指令依赖人工飞行+标注。OpenFly在多引擎多样性(4种 vs 1种)、数据规模(100K vs 8.4K)、自动化程度(全自动 vs 人工)三方面全面超越。在NE指标上OpenFly-Agent(93m) vs AerialVLN方法(139m)提升33%。

### vs OpenUAV (2024)
OpenUAV同样使用AirSim+UE构建12149条轨迹的VLN数据集，引入人类反馈(RLHF)引导导航。但其仍依赖飞行员操作和人工标注，数据多样性受限。OpenFly的工具链实现了数据生成的全自动化，且通过3DGS引入了real-to-sim能力，在真实世界部署中展现更强的迁移性。

### vs CityNav (2024)
CityNav基于SensatUrban点云数据+CityRefer语言标注构建32637条轨迹，但依赖预有的2D地图辅助定位地标。OpenFly不需要外部地图，通过端到端的视觉-语言方法直接从一人称视角导航，更接近实际无人机应用场景。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多引擎集成+全自动pipeline+关键帧感知的系统级创新，单点技术创新有限
- **实验充分度**: ⭐⭐⭐⭐⭐ 多方法对比+消融+真实无人机部署+跨数据集对比+规模分析，非常全面
- **写作质量**: ⭐⭐⭐⭐ 系统描述清晰完整，图表丰富
- **价值与影响**: ⭐⭐⭐⭐⭐ 对空中VLN研究有基础设施级贡献，工具链+数据集+benchmark三位一体

<!-- RELATED:START -->

## 相关论文

- [3D Gaussian Map with Open-Set Semantic Grouping for Vision-Language Navigation](../../ICCV2025/3d_vision/3d_gaussian_map_with_openset_semantic_grouping_for_visionlan.md)
- [EgoNight: Towards Egocentric Vision Understanding at Night with a Challenging Benchmark](egonight_towards_egocentric_vision_understanding_at_night_with_a_challenging_ben.md)
- [Vision-Language Embodiment for Monocular Depth Estimation](../../CVPR2025/3d_vision/vision-language_embodiment_for_monocular_depth_estimation.md)
- [Generalizable Coarse-to-Fine Robot Manipulation via Language-Aligned 3D Keypoints](generalizable_coarse-to-fine_robot_manipulation_via_language-aligned_3d_keypoint.md)
- [VPN: Visual Prompt Navigation](../../AAAI2026/3d_vision/vpn_visual_prompt_navigation.md)

<!-- RELATED:END -->
