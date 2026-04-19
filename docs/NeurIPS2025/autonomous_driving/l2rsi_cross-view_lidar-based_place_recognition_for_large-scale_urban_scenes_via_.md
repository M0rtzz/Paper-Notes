---
title: >-
  [论文解读] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery
description: >-
  [NeurIPS 2025][自动驾驶][位置识别] 提出 L2RSI，首个利用高分辨率遥感影像实现超大规模（100km²）城市场景 LiDAR 位置识别的框架，通过语义对比学习对齐 LiDAR BEV 与遥感语义空间，并引入时空粒子估计（STPE）聚合连续查询的时空信息，在 100km² 范围内 Top-1 精度达 83.27%。
tags:
  - NeurIPS 2025
  - 自动驾驶
  - 位置识别
  - LiDAR
  - 遥感影像
  - 跨视角检索
  - 粒子估计
---

# L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery

**会议**: NeurIPS 2025  
**arXiv**: [2503.11245](https://arxiv.org/abs/2503.11245)  
**代码**: [项目页面](https://shizw695.github.io/L2RSI/)  
**领域**: 自动驾驶  
**关键词**: 位置识别, LiDAR, 遥感影像, 跨视角检索, 粒子估计

## 一句话总结

提出 L2RSI，首个利用高分辨率遥感影像实现超大规模（100km²）城市场景 LiDAR 位置识别的框架，通过语义对比学习对齐 LiDAR BEV 与遥感语义空间，并引入时空粒子估计（STPE）聚合连续查询的时空信息，在 100km² 范围内 Top-1 精度达 83.27%。

## 研究背景与动机

位置识别是自动驾驶和机器人导航中 GPS 受限场景的关键任务。传统 LiDAR 位置识别依赖预建的三维地图，但其获取和维护成本极高。

现有方法的局限：

**单模态方法**（PointNetVLAD、MinkLoc3D等）需要可靠的先验3D地图

**跨模态方法**（LIP-Loc、VXP等）将LiDAR与其他模态对齐，但检索范围局限于已知路线或极小区域（<2km²）

**基于地图的方法**（OSM方案）受限于地图信息的稀疏性

**核心创新点**：利用遥感影像（卫星图）替代3D地图作为参考数据库——遥感影像具有全球覆盖、成本低廉、时效性好的优势，能以极低成本实现大规模定位。

**核心挑战**：
- **跨视角 + 跨模态**的双重鸿沟：LiDAR（地面视角、3D点云）vs 遥感（鸟瞰视角、2D图像）
- 单次查询在大范围检索中不稳定且存在歧义

## 方法详解

### 整体框架

L2RSI 包含三个模块：
1. **数据预处理**：将遥感影像和 LiDAR 点云分别转为语义图
2. **语义对比学习网络**：在共享语义空间对齐全局描述符
3. **时空粒子估计（STPE）**：聚合连续查询的时空信息精化检索结果

### 关键设计

#### 1. 数据预处理——语义域统一

**核心观察**：LiDAR 点云和遥感影像虽在内容/风格上差异巨大，但在**语义域**上高度相关（道路、人行道、植被、建筑在两个模态中都可识别）。

**遥感端**：
- 使用阿里 AIE-SEG 语义分割模型分割遥感影像为4类语义（道路、人行道、植被、建筑）
- 以 $d=60m$ 的滑动窗口等间距提取子图，形成语义数据库
- 过滤远离道路的子图

**LiDAR端**：
- 使用 FastGICP 配准短序列 LiDAR 扫描构建点云子图
- SphereFormer 对同类语义进行分割
- 利用磁力计方向将点云压缩为鸟瞰（BEV）语义图（北向朝上对齐）

#### 2. 语义对比学习网络

双分支网络，两个分支**完全共享权重**（因为都在语义域中操作）：
- 语义编码器：MAE 预训练的 ViT-B 初始化
- GeM 池化 + 全连接层生成全局描述符

使用对称 InfoNCE 损失进行对比学习：

$$\mathcal{L} = -\log\frac{\exp(f_i^Q \cdot f_i^P / \tau)}{\sum_{j \in N} \exp(f_i^Q \cdot f_j^P / \tau)} - \log\frac{\exp(f_i^P \cdot f_i^Q / \tau)}{\sum_{j \in N} \exp(f_i^P \cdot f_j^Q / \tau)}$$

温度系数 $\tau=0.1$。相比传统 triplet loss，增加批内负样本数量可大幅加速训练。

#### 3. 时空粒子估计（STPE）

单次查询在大范围检索中不稳定，STPE 聚合连续查询序列 $\{Q_j\}_{t-L+1}^{t}$ 的时空信息：

**粒子建模**：每次查询的 Top-K 检索结果视为粒子，用 DBSCAN 聚类（半径 $r=30m$）确定高斯分量数 $M$，将粒子分布建模为 GMM：

$$P(x,y) = \sum_{m=1}^{M} A_m \cdot \exp\left(-\frac{(x-\mu_{xm})^2}{2\sigma_{xm}^2} - \frac{(y-\mu_{ym})^2}{2\sigma_{ym}^2}\right)$$

**时序传播**：利用 FastGICP 估计帧间相对位移，将历史查询的粒子分布传播到当前时刻。

**概率聚合**：对传播后的分布取平均得到当前位置的概率密度函数 $P_t(x,y) = \frac{1}{L}\sum_j \tilde{P}_j(x,y)$，据此重排序检索结果。

**关键区别**：与传统粒子滤波不同，STPE 进行粒子**聚合**而非**过滤**——在跨视角跨模态的困难检索中，单个有毒查询可能导致滤波器丢弃全部可信粒子，聚合策略更鲁棒。

### 损失函数 / 训练策略

- 损失：对称 InfoNCE（上述），正样本对的中心距小于 $d/2=30m$
- 训练在 LiRSI-XA 数据集上进行（约 12194 个点云子图 + 47913 个遥感子图）
- 默认参数：$L=50$（序列长度），$K=30$（Top-K 粒子数），$\lambda=30\%$（采样率）

## 实验关键数据

### 主实验

在 LiRSI-XA 不同规模数据库上的 Recall@1 (<30m)：

| 方法 | 4km² | 9km² | 16km² | 100km² |
|------|------|------|-------|--------|
| LIP-Loc | 16.82 | 13.69 | 11.60 | 11.02 |
| Sample4Geo | 28.63 | 25.39 | 24.23 | 22.95 |
| L2RSI (w/o STPE) | 30.05 | 23.90 | 21.93 | 20.07 |
| L2RSI (w. PF) | 71.66 | 55.17 | 50.87 | 47.85 |
| **L2RSI (w. STPE)** | **88.93** | **87.95** | **85.49** | **83.27** |

跨场景泛化（LiRSI-Oxford，无微调）：

| 轨迹 | Recall@1 | Recall@10 |
|------|----------|-----------|
| 11-14-02-26 | 42.29 | 59.41 |
| 14-12-05-52 | 43.77 | 62.76 |

### 消融实验

| 组件 | Test-S R@1 | Oxford R@1 |
|------|-----------|------------|
| 完整 L2RSI | **88.93** | **42.29** |
| 去掉语义分割 | 50.43 | 8.13 |
| 去掉 STPE | 30.05 | 10.19 |
| 去掉方向信息 | 54.37 | 4.92 |

### 关键发现

1. **STPE 是性能飙升的核心**：从无 STPE 的 30.05% 到有 STPE 的 88.93%（+58.88%），证实时空聚合的巨大价值
2. **语义域统一是跨模态对齐的关键**：去掉语义分割导致 R@1 从 88.93% 降至 50.43%
3. **STPE 远优于传统粒子滤波**：88.93% vs 71.66%，聚合优于过滤
4. 检索范围从 4km² 扩大到 100km² 仅引起 5.66% 的 R@1 下降，鲁棒性出色
5. 采样率 30% 即可达到最优性能，STPE 开销仅 31.7ms
6. 运动模型可容忍 ±10° 的额外偏航噪声和 ±1m 的平移噪声

## 亮点与洞察

1. **首次实现 100km² 量级的跨视角 LiDAR 位置识别**，具有重大实际应用价值
2. **语义域作为桥梁**的思想简洁有效：将两个完全不同的模态映射到同一语义空间
3. **STPE 的聚合思想**：颠覆了粒子滤波的传统范式，更适合"单次观测高不靠谱"的困难检索场景
4. **跨场景零样本泛化**：在完全不同城市（厦门→牛津）无需微调即可工作

## 局限与展望

1. 依赖额外的磁力计提供方向信息，增加了硬件需求
2. 跨场景泛化精度（~42%）尚不足以实际部署，需进一步提升
3. 语义分割质量是瓶颈——天桥下方、密集树荫等遮挡区域会导致 LiDAR 与遥感语义不匹配
4. 遥感影像与 LiDAR 数据存在3年时间差，建筑/植被变化会引入噪声
5. 目前仅在城市道路场景验证，乡村或非结构化环境有待探索

## 相关工作与启发

- **单模态3D位置识别**：PointNetVLAD、MinkLoc3D 等需要先验3D地图，L2RSI 用遥感替代
- **跨视角2D位置识别**：GeoDTR、Sample4Geo 直接用于跨模态效果不佳，凸显语义域统一的重要性
- **粒子滤波定位**：L2RSI 的 STPE 将粒子思想从"滤波收敛"改为"概率聚合"
- **启发**：遥感影像作为廉价替代地图的思路可推广到城市无人机导航、灾后搜救等场景

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个百公里级跨模态LiDAR位置识别，STPE设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ — 自建大规模数据集 + 多尺度评估 + 跨场景泛化 + 充分消融
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，实验组织有条理
- 价值: ⭐⭐⭐⭐⭐ — 极强的实际应用导向，有望推动GPS受限场景的大规模定位

<!-- RELATED:START -->

## 相关论文

- [ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images](../../CVPR2025/autonomous_driving/forestlpr_lidar_place_recognition_in_forests_attentioning_multiple_bev_density_i.md)
- [CuMoLoS-MAE: A Masked Autoencoder for Remote Sensing Data Reconstruction](cumolos-mae_a_masked_autoencoder_for_remote_sensing_data_reconstruction.md)
- [X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)
- [Extrapolated Urban View Synthesis Benchmark](../../ICCV2025/autonomous_driving/extrapolated_urban_view_synthesis_benchmark.md)

<!-- RELATED:END -->
