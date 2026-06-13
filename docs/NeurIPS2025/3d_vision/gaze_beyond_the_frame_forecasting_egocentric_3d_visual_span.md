---
title: >-
  [论文解读] Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span
description: >-
  [NeurIPS 2025][3D视觉][第一人称视觉] 提出 EgoSpanLift 方法，将第一人称 2D 注视预测提升到 3D 空间，构建多层级体积视觉跨度表示，结合 3D U-Net 和单向 Transformer 实现对未来 3D 视觉关注区域的预测。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "第一人称视觉"
  - "注视预测"
  - "3D视觉跨度"
  - "SLAM"
  - "体素预测"
---

# Gaze Beyond the Frame: Forecasting Egocentric 3D Visual Span

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2511.18470](https://arxiv.org/abs/2511.18470)  
**代码**: [有](https://hs-yn.github.io/GazeBeyondFrame/)  
**领域**: 3D视觉  
**关键词**: 第一人称视觉, 注视预测, 3D视觉跨度, SLAM, 体素预测

## 一句话总结

提出 EgoSpanLift 方法，将第一人称 2D 注视预测提升到 3D 空间，构建多层级体积视觉跨度表示，结合 3D U-Net 和单向 Transformer 实现对未来 3D 视觉关注区域的预测。

## 研究背景与动机

人们通过持续感知和与环境交互来完成日常活动，而视觉感知是引导人类行为的基础——"先看后做"。现有第一人称理解研究主要关注动作预测和接触式交互，对**视觉感知本身的预测**探索不足。

现有 2D 注视预测的局限：

**动态场景中 2D 注视预测定义不明确**：需要同时建模用户的自身运动和注意力，但两者在 3D 空间中才有自然表达

**2D 投影丢失信息**：注视方向和自身运动本质上指向 3D 空间的特定位置，而非 2D 图像上的任意区域

**缺乏超出当前视野的预测能力**：用户可能会转头关注当前画面之外的区域

本文提出**第一人称 3D 视觉跨度预测**这一新问题：预测用户的视觉感知在 3D 环境中的未来关注位置。

## 方法详解

### 整体框架

系统由三部分组成：
1. **EgoSpanLift**：将 2D 注视从图像平面提升到 3D 场景中的体积区域
2. **预测网络**：3D U-Net（空间编码器）+ 单向 Transformer（时序建模）进行未来视觉跨度预测
3. **基准测试**：从原始第一人称多传感器数据策划的 364.6K 样本 benchmark

### 关键设计

#### 1. EgoSpanLift — 从 2D 到 3D 的注视提升

**输入**：SLAM 提供的半稠密 3D 关键点 $\mathcal{P}$（每点含位置 $\mathbf{p}_i \in \mathbb{R}^3$、置信度 $\sigma_i$、观测时间 $t_i$）和定位信息 $\mathcal{E}$（SE(3) 变换矩阵 $\mathbf{E}_t$）。

**基于观测的关键点选择**：

$$\mathcal{P}_t = \{p_i \in \mathcal{P} \mid t_i = t, \|\mathbf{p}_i - \mathbf{t}_t\|_1 < D/2, \mathcal{I}_f(p_i; \mathcal{P}_t) = 1\}$$

三重过滤：时间窗口过滤（仅保留最近几秒）、空间过滤（$D=3.2$m 立方体内）、统计异常点过滤（保留动态物体如手、移动中的人，同时去除无效点）。

**基于注视的关键点分类**：将关键点变换到局部坐标系后，用 3D 注视锥判断是否在视觉跨度内：

$$Q_t^{\theta, \mathbf{g}_t} = \left\{p_i \in \mathcal{P}_t \mid \frac{\langle \mathbf{E}_t^{-1}\mathbf{p}_i, \mathbf{g}_t \rangle}{\|\mathbf{E}_t^{-1}\mathbf{p}_i\| \|\mathbf{g}_t\|} > \cos\theta \right\}$$

其中 $\theta$ 为离心角阈值，$\mathbf{g}_t$ 为注视方向。

#### 2. 多层级体积区域定位

受视觉科学文献启发，定义四个层级视觉跨度：

| 层级 | 离心角 $\theta$ | 含义 |
|------|:--------------:|------|
| **中央凹 (Foveal)** | 2° | 传统 2D 注视定位区域 |
| **中心 (Central)** | 8° | 补偿半稠密关键点的覆盖不足 |
| **近外周 (Near Peripheral)** | 30° | 更广的外周感知范围 |
| **方向 (Orientation)** | 55° | 以头部朝向为中心的视野范围 |

将分类后的关键点体素化为 3D 栅格（分辨率 $R$，边长 $D$），计算二值占用：

$$V_{[t_b,t_e]}^{\theta,\mathbf{g}_t}(i,j,k) = \mathcal{I}(|\{p_i \in \cup_{t} Q_t^{\theta,\mathbf{g}_t} \mid \text{落在对应体素}\}| > 0)$$

#### 3. 预测网络

**自回归编码器**：
- 输入：$T_p \times (4+1) \times R \times R \times R$ 的体素网格（4 层级视觉跨度 + 完整场景）
- 3D U-Net 编码器压缩空间维度（缩减因子 $R$），得到 $T_p \times C$ 的时序特征
- 附加全局嵌入作为预测头，形成 $(T_p+1) \times C$ 的特征序列
- **单向 Transformer** 学习时序依赖，确保信息向最终全局嵌入汇聚

**解码器**：
- 通过 U-Net 解码器对输出嵌入上采样
- 利用编码器中间特征的残差连接
- Sigmoid 输出 $4 \times R \times R \times R$ 的 0-1 软占用图

### 损失函数 / 训练策略

由于视觉跨度仅占空间极小区域（中央凹 <1%），传统交叉熵难以学习有意义信号。采用 **Dice Loss**：

$$\mathcal{L} = 1 - \frac{2 \times \sum \tilde{Y}_{ijk} \odot Y_{ijk}}{\sum \tilde{Y}_{ijk} + \sum Y_{ijk} + 1}$$

多层级联合训练优于单任务训练，尤其对中央凹预测有显著提升。

## 实验关键数据

### 主实验

**FoVS-Aria** 测试集（日常活动，23.2K 样本）：

| 方法 | Orientation IoU | Peripheral IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-------------:|:-----------:|:----------:|
| CSTS + EgoSpanLift | - | 0.457 | 0.234 | 0.139 |
| EgoChoir | 0.496 | 0.430 | 0.261 | 0.199 |
| **Ours (full)** | **0.584** | **0.489** | **0.351** | **0.284** |

**FoVS-EgoExo** 测试集（技能活动，341.4K 样本）：

| 方法 | Orientation IoU | Peripheral IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-------------:|:-----------:|:----------:|
| CSTS + EgoSpanLift | - | 0.498 | 0.287 | 0.156 |
| EgoChoir | 0.329 | 0.285 | 0.198 | 0.127 |
| **Ours (full)** | **0.523** | **0.511** | **0.421** | **0.369** |

3D 中央凹定位误差（距离分布）：

| 方法 | 最小(cm) | 平均(cm) | 最大(cm) |
|------|:-------:|:-------:|:-------:|
| CSTS | 59.71 | 73.79 | 87.68 |
| **Ours** | **19.04** | **34.85** | **51.23** |

### 消融实验

| 配置 | Orientation IoU | Central IoU | Foveal IoU |
|------|:--------------:|:-----------:|:----------:|
| w/o 先前跨度 | 0.342 | 0.107 | 0.059 |
| BCE 损失 | 0.573 | 0.284 | 0.206 |
| 单任务训练 | 0.583 | 0.335 | 0.249 |
| w/o 全局嵌入 | 0.560 | 0.324 | 0.262 |
| **完整模型** | **0.584** | **0.351** | **0.284** |

### 关键发现

1. 在所有层级上大幅超越基线，中央凹跨度预测超出 50%+
2. 先前视觉跨度信息至关重要——移除后性能暴跌
3. 多层级联合训练显著优于单任务，利用了外周与注视之间的相互线索
4. 3D 预测回投到 2D 后（无需 2D 专门训练）与 2D 专用方法性能持平
5. 推理延迟仅 71.2ms，满足实时要求

## 亮点与洞察

- **定义新问题**：首次形式化第一人称 3D 视觉跨度预测任务
- **视觉科学驱动**：多层级视觉跨度定义来自严谨的视觉科学分类
- **实用性强**：基于 SLAM 关键点（非稠密重建），延迟低（71ms），可用于 AR/VR
- **双向验证**：2D→3D 和 3D→2D 都有效，证明 3D 建模的优越性
- **大规模 benchmark**：策划了 364.6K 样本的测试平台

## 局限与展望

1. 依赖 SLAM 提供的半稠密关键点质量，某些场景关键点不足
2. 中央凹跨度绝对 IoU 仍不算高（~0.28-0.37），精细预测仍有空间
3. 排除了足球、篮球等大范围动态场景，适用性有限
4. 体素化是主要瓶颈（45ms），需优化以用于更受限设备
5. 未纳入非视觉感知（听觉、本体感觉），可能丢失重要线索

## 相关工作与启发

- **CSTS**：当前最优 2D 注视预测方法，多模态对比时空融合
- **EgoChoir**：从合成几何和运动中预测 3D 交互热点，是最强基线
- **Ego-Exo4D / Aria Everyday Activities**：数据来源
- 启发：将 2D 任务提升到 3D 是一种通用且有效的策略，可推广到其他第一人称预测任务

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 定义了全新问题并提出了系统性解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集（364.6K 样本）、多基线、全面消融、2D 回投验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，视觉科学背景融入自然
- 价值: ⭐⭐⭐⭐ 对 AR/VR 和辅助技术有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](../../ICCV2025/3d_vision/uniegomotion_a_unified_model_for_egocentric_motion_reconstruction_forecasting_an.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[NeurIPS 2025\] IndEgo: A Dataset of Industrial Scenarios and Collaborative Work for Egocentric Assistants](indego_a_dataset_of_industrial_scenarios_and_collaborative_work_for_egocentric_a.md)
- [\[NeurIPS 2025\] Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting](mesh_interpolation_graph_network_for_dynamic_and_spatially_irregular_global_weat.md)
- [\[NeurIPS 2025\] 3D Visual Illusion Depth Estimation](3d_visual_illusion_depth_estimation.md)

</div>

<!-- RELATED:END -->
