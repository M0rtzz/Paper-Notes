---
title: >-
  [论文解读] CityNav: A Large-Scale Dataset for Real-World Aerial Navigation
description: >-
  [ICCV 2025][遥感][视觉语言导航] 构建了首个面向真实城市环境的大规模空中视觉语言导航数据集 CityNav（32,637 条人类演示轨迹，覆盖 4.65 km²），并提出地理语义地图（GSM）辅助表示，显著提升基线模型的导航性能。
tags:
  - ICCV 2025
  - 遥感
  - 视觉语言导航
  - 无人机
  - 真实世界
  - 地理语义地图
  - 大规模数据集
---

# CityNav: A Large-Scale Dataset for Real-World Aerial Navigation

**会议**: ICCV 2025  
**arXiv**: [2406.14240](https://arxiv.org/abs/2406.14240)  
**代码**: [项目页面](https://water-cookie.github.io/city-nav-proj/)  
**领域**: Remote Sensing / Aerial Navigation  
**关键词**: 视觉语言导航, 无人机, 真实世界, 地理语义地图, 大规模数据集

## 一句话总结

构建了首个面向真实城市环境的大规模空中视觉语言导航数据集 CityNav（32,637 条人类演示轨迹，覆盖 4.65 km²），并提出地理语义地图（GSM）辅助表示，显著提升基线模型的导航性能。

## 研究背景与动机

视觉语言导航（VLN）旨在让智能体根据自然语言描述在真实环境中导航。虽然室内和地面户外 VLN 已有大量工作，但**空中导航（特别是真实城市环境下）仍严重欠缺**：

现有空中 VLN 数据集的局限：
- **LANI**（6K 轨迹）：虚拟小环境，2D 动作空间
- **AVDN**（3K 轨迹）：仅在 2D 卫星图像上操作
- **AerialVLN**（8.4K 轨迹）：3D 但是合成城市，缺乏真实世界复杂性

关键问题：**真实城市环境下**，智能体必须理解真实地标的空间关系，并整合视觉与地理信息进行导航，而现有合成数据集无法有效训练这种能力。

## 方法详解

### 整体框架

CityNav 的构建包含三大部分：
1. CityFlight 仿真环境（基于真实 3D 扫描数据）
2. 大规模人类演示轨迹采集
3. 地理语义地图（GSM）辅助表示

### 关键设计

1. **CityFlight 仿真环境**：

    - 基于 SensatUrban 数据集的 3D 点云数据构建，覆盖剑桥和伯明翰两座真实城市
    - 使用 Potree（WebGL 点云渲染器）实现浏览器中的 3D 场景可视化，支持众包数据采集
    - 与 OpenStreetMap 实时同步，提供 3D 坐标与 2D 地图坐标互转、地标名称检索等功能
    - **动作空间**：5D 位姿 $\bm{p} = (x, y, z, \theta, \psi)$，6 种动作（前进 5m、左转/右转 30°、上升/下降 2m、停止）
    - 最大飞行高度 200 米，起始点在目标 500 米半径内随机选择，高度 100-150 米

2. **人类演示轨迹采集**：

    - 通过 Amazon MTurk 众包，171 名标注员贡献 32,637 条高质量轨迹
    - 三阶段质量控制：资格测试（排除不合格标注员）→ 初始采集（排除 18.4% 不达标轨迹）→ 重新采集（7.2% 仍不达标被彻底移除）
    - 目标描述来自 CityRefer 数据集，包含至少一个地标并描述目标与周围物体的空间关系
    - 成功判定：智能体停止在目标 20 米球形半径内
    - 目标类型分布：48.3% 建筑、40.7% 汽车、7.4% 地面、3.6% 停车场

3. **地理语义地图（GSM）**：

    - 五个类别：当前视野、已探索区域、地标、潜在目标、周围物体
    - 当前视野和已探索区域从 GNSS 坐标获取
    - 地标从 OpenStreetMap 检索
    - 潜在目标和周围物体使用 Grounding DINO 检测
    - 地标和物体名称使用 GPT-3.5 提取
    - 编码方式：对齐 2D 地图的二进制掩码 → 5 层 CNN 编码器 $E$ → 特征 $\bm{z}^{(t)}_{map}$
    - 集成到 VLN 模型：将 $\bm{z}^{(t)}_{map}$ 追加到 $[\bm{z}^{(t)}_{RGB}, \bm{z}^{(t)}_{depth}]$ 序列输入 GRU

### 评估指标

- **NE (Navigation Error)**：停止点到目标的欧氏距离
- **SR (Success Rate)**：满足成功标准的比例
- **OSR (Oracle Success Rate)**：轨迹中至少一点满足成功标准的比例
- **SPL (Success weighted by Path Length)**：考虑路径效率的成功率

## 实验关键数据

### 主实验：三种基线模型在 CityNav 上的表现

| 方法 | GNSS | Val-seen NE↓ | Val-seen SR↑ | Test-unseen NE↓ | Test-unseen SR↑ | Test-unseen SPL↑ |
|------|------|-------------|-------------|----------------|----------------|-----------------|
| Seq2Seq | ✗ | 257.1 | 1.81% | 245.3 | 1.50% | 1.30% |
| Seq2Seq + GSM | ✓ | 58.5 | 8.43% | 98.1 | 3.81% | 2.79% |
| CMA | ✗ | 240.8 | 0.95% | 252.6 | 0.82% | 0.79% |
| CMA + GSM | ✓ | 68.0 | 6.25% | 94.6 | 4.68% | 4.05% |
| AerialVLN | ✗ | 185.2 | 1.73% | 187.7 | 1.79% | 0.62% |
| **AerialVLN + GSM** | ✓ | **56.6** | **10.16%** | **85.1** | **6.72%** | **5.16%** |
| Human | - | 9.1 | 89.31% | 9.8 | 87.86% | 57.04% |

GSM 使所有模型的导航误差下降超过 50%，成功率提升数倍。

### 消融实验

| 配置 | NE↓ | SR↑ | OSR↑ | SPL↑ | 说明 |
|------|-----|-----|------|------|------|
| AerialVLN + GSM (完整) | 85.1 | 6.72% | 18.21% | 5.16% | 基线 |
| w/o landmarks | 190.7 | 0.60% | 6.94% | 0.56% | 地标信息最关键 |
| w/o potential destination | 92.8 | 3.97% | 13.08% | 3.86% | 潜在目标重要 |
| w/o surrounding objects | 87.5 | 5.17% | 15.16% | 5.10% | 周围物体辅助 |

### 训练数据对比

| 训练数据 | 数据量 | NE↓ | SR↑ |
|---------|--------|-----|-----|
| 最短路径 | 22k | 95.1 | 4.96% |
| **人类演示** | **22k** | **85.1** | **6.72%** |
| 最短路径 + 噪声 | 22k | 123.1 | 2.37% |
| **人类演示 + 噪声** | **22k** | **95.0** | **4.92%** |

### 关键发现

- **人类与机器差距巨大**：人类成功率 87-90%，最好模型仅 6-10%，表明真实空中 VLN 仍是极具挑战的开放问题
- **地标信息是 GSM 最关键组件**：移除地标后成功率从 6.72% 暴跌至 0.60%
- **人类演示优于最短路径训练**：人类轨迹更频繁经过地标附近，提供了更丰富的视觉-地理关联信息
- **人类演示训练的模型更鲁棒**：在添加位置噪声后性能差距进一步拉大（NE 差距从 10.0 扩大到 28.1）
- 更长的描述和更多地标数量均能提升所有模型的成功率

## 亮点与洞察

- 首个基于真实世界 3D 扫描构建的空中 VLN 数据集，填补了重要的研究空白
- 32,637 条轨迹是迄今最大规模的空中 VLN 数据集，采集流程设计严谨（三阶段质量控制）
- GSM 的设计理念——将 OpenStreetMap 地理信息与视觉观测统一为可学习的辅助模态——简单但效果显著
- 人类演示 vs 最短路径的对比揭示了"认知导航"与"几何导航"的本质差异

## 局限与展望

- 仅覆盖两座城市（剑桥+伯明翰），全球泛化性有限
- 当前模型的绝对性能仍很低（最好 SR ~10%），需要更强的基线模型
- 未探索多智能体协作的大范围搜索场景
- 无人机的真实物理约束（电量、避障）未纳入考虑
- 3D 点云渲染的视觉真实感与实际无人机拍摄仍有差距

## 相关工作与启发

- **AerialVLN** 是最直接的前序工作，CityNav 将其从合成环境推进到真实世界
- **TouchDown** 和 **Talk2Nav** 在地面户外 VLN 中的经验（利用街景地标）与 CityNav 的空中地标利用形成互补
- GSM 的设计启发：其他多模态导航任务也可考虑将结构化地理知识作为辅助输入
- 巨大的人机差距暗示未来需要更强的空间推理、3D 理解和多步规划能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个真实世界空中 VLN 大规模数据集，数据贡献度高
- 实验充分度: ⭐⭐⭐⭐ 三种基线×有无 GSM + 训练数据消融 + 鲁棒性评测
- 写作质量: ⭐⭐⭐⭐⭐ 数据集论文应有的全面性：任务定义、采集流程、统计分析、基线对比
- 价值: ⭐⭐⭐⭐ 为空中 VLN 社区提供了关键基础设施，人机差距指明了明确的研究方向

<!-- RELATED:START -->

## 相关论文

- [Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](../../CVPR2026/remote_sensing/olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)
- [RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events](../../NeurIPS2025/remote_sensing/rscc_a_large-scale_remote_sensing_change_caption_dataset_for_disaster_events.md)
- [Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](../../CVPR2026/remote_sensing/cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)
- [OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [Pan-Crafter: Learning Modality-Consistent Alignment for Pan-Sharpening](pan-crafter_learning_modality-consistent_alignment_for_pan-sharpening.md)

<!-- RELATED:END -->
