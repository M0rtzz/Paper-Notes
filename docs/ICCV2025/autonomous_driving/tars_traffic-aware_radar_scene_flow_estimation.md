---
title: >-
  [论文解读] TARS: Traffic-Aware Radar Scene Flow Estimation
description: >-
  [ICCV2025][自动驾驶][雷达场景流] 提出 TARS，一种交通感知的雷达场景流估计方法，通过联合目标检测构建交通向量场（TVF），在交通层面而非实例层面捕获刚体运动，在 VOD 和专有数据集上分别超越 SOTA 15% 和 23%。
tags:
  - ICCV2025
  - 自动驾驶
  - 雷达场景流
  - 交通向量场
  - 点云运动估计
  - 多任务学习
  - 自动驾驶感知
---

# TARS: Traffic-Aware Radar Scene Flow Estimation

**会议**: ICCV2025  
**arXiv**: [2503.10210](https://arxiv.org/abs/2503.10210)  
**代码**: 待确认  
**领域**: autonomous_driving  
**关键词**: 雷达场景流, 交通向量场, 点云运动估计, 多任务学习, 自动驾驶感知

## 一句话总结

提出 TARS，一种交通感知的雷达场景流估计方法，通过联合目标检测构建交通向量场（TVF），在交通层面而非实例层面捕获刚体运动，在 VOD 和专有数据集上分别超越 SOTA 15% 和 23%。

## 研究背景与动机

- 场景流为自动驾驶提供关键运动信息，描述两帧点云间的点位移向量
- 现有 LiDAR 场景流方法利用**实例级刚体运动假设**：场景由多个刚体运动物体和静止部分组成
- 但实例级方法**不适合雷达点云**，原因有三：
  1. **极度稀疏**：雷达点云比 LiDAR 稀疏一个数量级（VOD 数据集每帧仅 ~256 点）
  2. **缺乏形状信息**：无法可靠地匹配实例对
  3. **帧间"变形"**：稀疏性导致同一物体在连续帧中的点分布差异显著
- 雷达的优势：对天气条件更鲁棒，成本低一个数量级
- 核心问题：如何在保留刚体运动假设的同时适应雷达稀疏性？
- 本文方案：**将刚体运动假设从实例级提升到交通级**

## 方法详解

### 整体架构

TARS 采用层次化架构（L 层），结合两个分支：
1. **场景流分支**：层次化粗到细的场景流估计
2. **目标检测（OD）分支**：提供交通上下文信息的特征图

两个分支**联合训练**，OD 分支的特征图为场景流提供交通级先验知识。

### 输入与编码

- 两帧点云 $P \in \mathbb{R}^{N \times 5}$ 和 $Q \in \mathbb{R}^{M \times 5}$
- 每个点 5 维特征：x, y, z 坐标 + 相对径向速度（RRV）+ 雷达截面积（RCS）
- 多尺度点编码器（PointNet）提取特征
- 最远点采样逐层下采样，产生多尺度点集对

### 双层运动理解

#### 1. 点级运动理解

- 使用**双注意力机制**从邻近点中提取运动线索（替代不稳定的 MLP）
- **交叉注意力**：点 $p_i$ 与其在 Q 中 K 个最近邻之间计算，获取匹配嵌入
- **自注意力**：结合匹配嵌入与 P 中邻域点的信息，获取点级流嵌入
- 先用上一层粗流进行 warp 对齐，减少搜索范围
- 与 HALFlow 不同，移除了方向向量以缓解点间距问题，使用异构 key/value

#### 2. 交通级场景理解

核心创新：通过 TVF（交通向量场）建模交通级运动一致性。

**TVF 定义**：离散网格图，包含道路参与者和环境的交通信息，每个单元包含运动向量。使用**粗网格**（如 2m×2m）来获得高层理解而非陷入点级细节。

### TVF 编码器

分两阶段构建 TVF：

**场景更新（Scene Update）**：
- 使用 GRU 跨层更新 TVF，输入为 OD 特征图（经 CNN 和池化适配到 TVF 形状）
- TVF 作为 GRU 的隐藏状态，OD 特征为输入
- 跨层次逐步精化场景表示

**流绘制（Flow Painting）**：
- 将上一层的流嵌入和点特征投影到粗网格上
- 由于每个网格单元可包含多个不同运动模式的点，使用**点到网格自注意力**自适应提取运动特征
- 通过空间注意力融合交通特征和运动特征
- 使用**轴向注意力**（ω 个 block）提供全局感受野，建模交通中的刚体运动依赖关系（如同车道车辆的运动模式）

### TVF 解码器

- 在空间上下文中感知隐藏的刚体运动
- 对每个点 $p_i$ 执行**网格到点交叉注意力**：查询周围 $\mathcal{N}_{TVF}$ 个 TVF 单元
- 注意力感受野限制在点的局部区域，聚焦相关的局部刚体运动
- query：上一层流嵌入 + 点特征；key/value：TVF 网格

### 场景流预测

- 拼接点级和交通级流嵌入：$\textbf{e}^l = \text{Concat}(\textbf{e}_\text{point}, \textbf{e}_\text{traffic}, \text{Interp}(\textbf{e}^{l-1}))$
- 再经自注意力后预测最终场景流 $F^l$

### 时序更新模块

- 使用 PointGRU 层利用多帧时序信息（区别于 TVF 编码器的跨层 GRU）
- 以 t-2 时刻的点特征初始化隐藏状态
- 训练时采样 T 帧 mini-clip 作为序列

### 训练策略

弱监督训练，不使用场景流 GT 标注，而是使用组合损失：
1. **Soft Chamfer 损失** $\mathcal{L}_{sc}$：对齐 warp 后的 P 与 Q
2. **空间平滑损失** $\mathcal{L}_{ss}$：相邻点应有相似流向量
3. **径向位移损失** $\mathcal{L}_{rd}$：利用雷达 RRV 测量约束径向流分量
4. **前景损失** $\mathcal{L}_{fg}$：使用 LiDAR 多目标跟踪模型的伪 GT
5. **背景损失** $\mathcal{L}_{bg}$：静态点使用自车运动变换作为伪 GT（λ=0.5）

### 自车运动处理

- **TARS-ego**：训练额外的自车运动头（与 CMFlow 公平对比）
- **TARS-superego**：自车运动作为已知输入进行补偿（模拟真实自动驾驶）

## 实验关键数据

### VOD 数据集

| 方法 | EPE↓(m) | AccS↑(%) | AccR↑(%) | RNE↓ | MRNE↓ | SRNE↓ |
|------|---------|----------|----------|------|-------|-------|
| RaFlow | 0.226 | 19.0 | 39.0 | 0.090 | 0.114 | 0.087 |
| CMFlow (SOTA) | 0.130 | 22.8 | 53.9 | 0.052 | 0.072 | 0.049 |
| **TARS-ego** | **0.092** | **39.0** | **69.1** | **0.037** | **0.061** | **0.034** |
| TARS-superego | 0.048 | 76.6 | 86.4 | 0.019 | 0.057 | 0.014 |

TARS-ego 将 EPE 从 0.130m 降至 0.092m（首次突破 AccR 阈值 0.1m），AccS/AccR 分别提升 16.2%/15.2%。

### 专有数据集（高分辨率雷达）

| 方法 | MEPE↓(m) | MagE↓ | DirE↓(rad) | AccS↑(%) | AccR↑(%) |
|------|----------|-------|------------|----------|----------|
| PointPWC-Net+GRU | 0.213 | 0.178 | 0.762 | 49.0 | 60.5 |
| HALFlow+GRU | 0.170 | 0.135 | 0.721 | 50.9 | 63.8 |
| **TARS** | **0.069** | **0.059** | **0.599** | **69.8** | **86.8** |

MEPE 从 0.170m 降至 0.069m（-59%），AccS/AccR 提升 18.9%/23.0%。

### 消融实验（专有数据集）

| 配置 | MEPE↓ | AccS↑ | AccR↑ |
|------|-------|-------|-------|
| 仅点级 | 0.178 | 47.9 | 61.6 |
| + 交通级（无OD特征图） | 0.144 | 45.0 | 63.3 |
| + OD特征图（细网格） | 0.104 | 51.4 | 69.9 |
| + 粗网格（无全局注意力） | 0.074 | 65.6 | 84.2 |
| + 全局注意力（完整 TARS） | **0.069** | **69.8** | **86.8** |

关键发现：
- 粗网格（2m×2m vs 1m×1m）是交通级理解的关键
- 全局注意力（vs 局部卷积）提升 AccS +4.2%
- TVF 解码器中 $\mathcal{N}_{TVF}=9$（周围邻域）效果最佳

### 损失函数消融（VOD 数据集）

- 背景损失 $\mathcal{L}_{bg}$ 对整体性能显著提升：AccR 从 62.4% 到 69.1%
- 权重 λ=0.5 在运动点精度和整体精度间取得平衡

## 优势与局限

**优势**：
- 首次将刚体运动假设从实例级提升到交通级，有效适应雷达稀疏性
- TVF 的粗网格设计避免了陷入点级细节的问题
- 通过 OD 分支的特征图（而非检测结果）获取交通上下文，降低了对检测精度的依赖
- 在两个数据集上大幅超越 SOTA（15% 和 23%）
- 有效缓解了雷达无法测量切向速度的固有问题

**局限**：
- 仍依赖 LiDAR 多目标跟踪模型生成前景伪 GT（非完全无 LiDAR）
- VOD 数据集点云极稀疏（256 点/帧），TVF 解码器的 $\mathcal{N}_{TVF}$ 影响不显著
- 未探索端到端训练（OD 和场景流联合优化）

## 个人思考

- "实例级 vs 交通级"的问题定位非常精准，直接击中了雷达场景流的核心痛点
- TVF 的设计思路有启发性：当数据太稀疏无法做精细匹配时，提升抽象层级是有效策略
- 粗网格 + 全局注意力的组合很有道理：粗网格保持高层语义，全局注意力建模车道级运动关联
- 联合目标检测的方式比直接使用检测结果更鲁棒，因为特征图包含的信息比框检测更丰富
- PointGRU 时序模块独立于 TVF 的层间 GRU，两者分工明确：一个负责时序积累，一个负责跨尺度场景更新

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

## 相关论文

- [GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [Embracing Large Language Models in Traffic Flow Forecasting](../../ACL2025/autonomous_driving/embracing_large_language_models_in_traffic_flow_forecasting.md)
- [Meta Dynamic Graph for Traffic Flow Prediction](../../AAAI2026/autonomous_driving/meta_dynamic_graph_for_traffic_flow_prediction.md)
- [V2X-Radar: A Multi-Modal Dataset with 4D Radar for Cooperative Perception](../../NeurIPS2025/autonomous_driving/v2x-radar_a_multi-modal_dataset_with_4d_radar_for_cooperative_perception.md)
- [FlowScene: Learning Temporal 3D Semantic Scene Completion via Optical Flow Guidance](../../NeurIPS2025/autonomous_driving/learning_temporal_3d_semantic_scene_completion_via_optical_flow_guidance.md)

<!-- RELATED:END -->
