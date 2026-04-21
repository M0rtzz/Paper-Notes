---
title: >-
  [论文解读] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets
description: >-
  [CVPR 2026][自动驾驶][毫米波雷达] 提出 EMDUL 管线，通过伪标签标注无标注毫米波数据（含新设计的无监督时序一致性损失 UTCL）和闭式 LiDAR→mmWave 点云转换器（含基于流的点过滤 FPF），大幅扩展毫米波 HPE 数据集的规模与多样性，域内误差降低 15.1%、跨域误差降低 18.9%。
tags:
  - CVPR 2026
  - 自动驾驶
  - 毫米波雷达
  - 人体姿态估计
  - 数据扩展
  - LiDAR点云转换
  - 半监督学习
---

# EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets

**会议**: CVPR 2026  
**arXiv**: [2603.14507](https://arxiv.org/abs/2603.14507)  
**代码**: [GitHub](https://github.com/Shimmer93/EMDUL)  
**领域**: 自动驾驶  
**关键词**: 毫米波雷达, 人体姿态估计, 数据扩展, LiDAR点云转换, 半监督学习

## 一句话总结

提出 EMDUL 管线，通过伪标签标注无标注毫米波数据（含新设计的无监督时序一致性损失 UTCL）和闭式 LiDAR→mmWave 点云转换器（含基于流的点过滤 FPF），大幅扩展毫米波 HPE 数据集的规模与多样性，域内误差降低 15.1%、跨域误差降低 18.9%。

## 研究背景与动机

毫米波（mmWave）雷达因其 3D 感知、隐私保护和光照鲁棒性优势，在人体姿态估计（HPE）领域受到持续关注。主流方法使用处理后的 3D 点云作为输入，但当前毫米波 HPE 数据集存在两大瓶颈：

**数据稀缺**：有标注的 mmWave HPE 数据集极少，采集和标注成本高

**多样性不足**：
   - **点云属性单一**：设备型号和环境种类有限，检测噪声、点密度、运动灵敏度分布狭窄
   - **姿态单调**：受试者多为面向雷达的站立姿势，缺乏蹲、游泳等复杂动作

与此同时，**无标注 mmWave 数据**易于采集，**LiDAR HPE 数据集**（如 LiDARHuman26M、HmPEAR）丰富且姿态多样。然而 LiDAR 与 mmWave 的点云属性因物理传感原理不同而存在根本差异（LiDAR 可感知静态物体，mmWave 依赖多普勒效应更擅长检测运动目标），直接混用效果不佳。此前的扩展方法要么仅扩展骨架而不改善点云属性分布，要么需要 mmWave-LiDAR 配对数据（采集困难）。

## 方法详解

### 整体框架

EMDUL 由两个独立模块组成：

1. **伪标签估计器**：利用已标注 mmWave 数据和 UTCL 损失训练，为无标注 mmWave 数据生成伪标签 $D^{\text{pl}}$
2. **闭式点云转换器**：将 LiDAR 数据集转换为 mmWave 点云 $D^{\text{conv}}$

工作流程为：先将 LiDAR 数据通过转换器生成 $D^{\text{conv}}$，与原始 $D^{\text{mm}}$ 合并为 $D^{\text{lab}}$；再用 $D^{\text{lab}}$ 训练伪标签估计器，标注无标注数据得到 $D^{\text{pl}}$；最终扩展数据集 $D^{\text{exp}} = D^{\text{lab}} \cup D^{\text{pl}}$。训练时每个 epoch 重新生成 $D^{\text{conv}}$（新随机种子）和更新 $D^{\text{pl}}$，实现迭代精炼。

### 关键设计

1. **无监督时序一致性损失（UTCL）**

    - 功能：在无标注 mmWave 数据上施加时序一致性约束，提升伪标签质量
    - 核心思路：利用 mmWave 雷达的运动检测物理先验——靠近检测点的关节可能在运动，远离检测点的关节可能静止。UTCL 包含两个互补分量：
        - **动态一致性损失（DCL）**：鼓励靠近点云的关节具有非零光流
       $$L^{\text{dyn}} = \frac{1}{|F_t^{\text{dyn}}|} \sum_{k} \max(0, \eta - \|F_t^{\text{dyn}}[k]\|_2)$$
        - **静态一致性损失（SCL）**：鼓励远离点云的关节光流趋近零
       $$L^{\text{sta}} = \frac{1}{|F_t^{\text{sta}}|} \sum_{k} \|F_t^{\text{sta}}[k]\|_2$$
    - 设计动机：单独使用 DCL 或 SCL 会导致预测偏向过动或过静，二者组合才能保持平衡，配合受监督的 $L^{\text{lab}}$ 使用

2. **基于流的点过滤（FPF）**

    - 功能：模拟 mmWave 雷达的运动检测机制，将 LiDAR 点云转换为具备 mmWave 特征的点云
    - 核心思路：基于骨架光流，通过反距离加权插值计算每个点的 3D 流向量，然后按流量大小概率性过滤——低流量（静态）点以更高概率被丢弃：
    $\mathcal{P}(P_t[i] \in P_t^{\text{conv}}) = \min\left(\frac{\|F_t^P[i]\|_2}{\upsilon_t}, 1\right)$
    - 设计动机：mmWave 雷达依赖多普勒效应，运动部位更容易被检测，FPF 直接在点云层面模拟这一物理机制

3. **完整点云转换管线（NPA→FPF→RS→NI）**

    - **NPA（噪声点添加）**：添加固定数量噪声点，模拟环境杂波
    - **FPF（基于流的点过滤）**：模拟运动检测机制（见上）
    - **RS（随机采样）**：降低点密度，模拟 mmWave 的稀疏性
    - **NI（噪声注入）**：在坐标上添加随机噪声，模拟较低的空间分辨率

### 损失函数 / 训练策略

伪标签估计器的总损失：

$$L = L^{\text{lab}} + \lambda^{\text{con}} L^{\text{con}}$$

- $L^{\text{lab}}$：标注数据上的 MSE 损失
- $L^{\text{con}} = L^{\text{dyn}} + L^{\text{sta}}$：UTCL
- $\lambda^{\text{con}} = 0.01$

训练 100 epoch，使用 AdamW 优化器，学习率 $10^{-4}$，带余弦退火和线性 warmup。推理模型 $\theta^{\text{infer}}$ 与估计器共同训练，每个 epoch 先更新 $\theta^{\text{pl}}$，再更新 $\theta^{\text{infer}}$。

## 实验关键数据

### 主实验

使用 10% 有标注 MM-Fi（F）或 mmBody（B）数据，在 full setting（+无标注+HmPEAR）下的 MPJPE（cm）：

| 设置 | 方法 | 骨干 | MPJPE | PA-MPJPE |
|------|------|------|-------|----------|
| F→F（域内） | P4T baseline | P4T | 12.23 | 7.95 |
| F→F | EMDUL | P4T | **10.06** | 7.01 |
| F→F | EMDUL | SPiKE | 10.40 | 7.23 |
| B→F（跨域） | P4T baseline | P4T | 33.62 | 15.85 |
| B→F | EMDUL | SPiKE | **22.80** | 14.09 |
| B→F | Mean Teacher | SPiKE | 32.71 | 15.74 |

- 域内（F→F）MPJPE 降低 **15.1%**（12.23→10.06，使用 SPiKE 可达 17.5% 优于 Mean Teacher）
- 跨域（B→F）MPJPE 降低 **18.9%**，使用 SPiKE 优于 Mean Teacher **30.3%**

### 消融实验

| 配置 | F→B MPJPE | PA-MPJPE | 说明 |
|------|-----------|----------|------|
| 无伪标签（仅 PCC） | 15.22 | 11.51 | 转换 LiDAR 已有显著提升 |
| +$L^{\text{lab}}$ 仅 | 15.12 | 11.44 | 受标注数据监督 |
| +$L^{\text{lab}}$+DCL+SCL（完整 UTCL） | **14.89** | **11.11** | 二者互补 |
| PCC w/o FPF | 15.85 | 12.23 | FPF 贡献最大 |
| PCC 完整 | 14.89 | 11.11 | 各组件互补 |

**LiDAR 点云转换真实性验证**：训练二分类器区分 mmWave 和 LiDAR 点云，使用 FPF 后 60.46% 的转换点云被判为 mmWave（不用 FPF 仅 43.06%）。

### 关键发现

- FPF 是点云转换管线中贡献最大的单一模块，直接模拟了 mmWave 的运动检测物理机制
- UTCL 的 DCL 和 SCL 单独使用不改善跨域性能，但组合后效果显著——单独时会导致预测偏向过动/过静
- 1% 有标注数据时 EMDUL 的优势最为显著（MPJPE 从 18.40 降至 14.77）
- 跨域性能的提升幅度大于域内，说明 EMDUL 真正改善了数据多样性而非过拟合

## 亮点与洞察

- **巧妙的物理先验利用**：UTCL 将 mmWave 的多普勒运动检测机制转化为无监督损失约束，是物理知识驱动的半监督设计典范
- **闭式转换无需配对数据**：LiDAR→mmWave 转换不依赖任何配对的跨模态数据，只需独立的 LiDAR 数据集即可
- **每 epoch 重新随机生成转换数据**：有效增加训练数据的多样性，类似在线数据增强

## 局限与展望

- 点云转换管线依赖经验参数（阈值 $\gamma$、$\delta$ 等），对不同场景可能非最优——未来可探索自适应或可学习转换
- UTCL 对复杂运动模式的建模还不够充分,可引入更精细的时序模型
- 目前仅支持单人 HPE，扩展到多人场景（遮挡、交互）是重要方向

## 相关工作与启发

- 与 Video2mmPoint 等视频转点云方法不同，EMDUL 直接在 3D 点云域操作且改善了点云属性分布
- FPF 的运动检测模拟思路可推广到其他基于多普勒效应的传感器（如 FMCW 雷达）
- UTCL 的时序一致性思想可启发其他自监督 3D 估计任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统研究从 LiDAR 到 mmWave 的跨模态点云转换用于 HPE 数据扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖两个 mmWave 和两个 LiDAR 数据集，消融全面，多种骨干验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，物理直觉表达到位
- 价值: ⭐⭐⭐⭐ 解决了 mmWave HPE 领域的数据瓶颈问题，方法通用性强

<!-- RELATED:START -->

## 相关论文

- [Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multimodal_learning_in_3d_human_p.md)
- [InCaRPose: In-Cabin Relative Camera Pose Estimation Model and Dataset](incarpose_in-cabin_relative_camera_pose_estimation_model_and_dataset.md)
- [PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](spectral-geometric_neural_fields_for_pose-free_lidar_view_synthesis.md)
- [SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)

<!-- RELATED:END -->
