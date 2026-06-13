---
title: >-
  [论文解读] Discontinuity-aware Normal Integration for Generic Central Camera Models
description: >-
  [ICCV 2025 (Highlight)][LLM评测][法线积分] 提出一种支持显式不连续性建模和通用中心相机模型的法线积分新方法，通过局部平面性假设建立法线与光线方向之间的约束，在标准法线积分基准上达到 SOTA，并首次直接处理通用中心相机（如鱼眼、全景相机）。 领域现状：法线积分（Normal Integratio…
tags:
  - "ICCV 2025 (Highlight)"
  - "LLM评测"
  - "法线积分"
  - "深度不连续性"
  - "中心相机模型"
  - "表面重建"
  - "光度立体"
---

# Discontinuity-aware Normal Integration for Generic Central Camera Models

**会议**: ICCV 2025 (Highlight)  
**arXiv**: [2507.06075](https://arxiv.org/abs/2507.06075)  
**代码**: 无  
**领域**: LLM评测  
**关键词**: 法线积分, 深度不连续性, 中心相机模型, 表面重建, 光度立体

## 一句话总结

提出一种支持显式不连续性建模和通用中心相机模型的法线积分新方法，通过局部平面性假设建立法线与光线方向之间的约束，在标准法线积分基准上达到 SOTA，并首次直接处理通用中心相机（如鱼眼、全景相机）。

## 研究背景与动机

**领域现状**：法线积分（Normal Integration）是光度形状恢复（如 Shape-from-Shading 和 Photometric Stereo）的核心步骤，其目标是从表面法线图恢复出 3D 深度/表面。现有方法大多假设正交投影或理想针孔相机模型，通过求解泊松方程或变分优化来恢复深度。

**现有痛点**：现有方法存在两个关键缺陷。首先，深度不连续性（如物体边缘、遮挡边界）通常只能被隐式处理——大多方法通过全局平滑正则化来逼近深度，导致边界处产生过度平滑或伪影。其次，几乎所有方法都限于正交或针孔相机模型，无法直接处理鱼眼、全景等广角/非标准中心相机。

**核心矛盾**：传统法线积分公式将法线-深度关系表示为深度的偏导数方程（即 $\nabla z = f(\mathbf{n})$），这一公式天然假设了连续性和特定投影模型。当场景存在深度不连续性时，偏导数在不连续处无定义；当相机模型非针孔时，法线-深度关系需要重新推导。

**本文目标**：设计一种统一的法线积分框架，能够（1）显式建模深度不连续性，（2）支持任意中心相机模型（包括针孔、鱼眼、全景等）。

**切入角度**：作者观察到，在局部平面性假设下，一个表面点的法线与从相机到该点的光线方向之间存在简明的几何约束关系。这个约束不依赖于特定投影模型，且可以自然地在不连续处"断开"。

**核心 idea**：用局部平面性约束（法线与光线方向的关系）替代传统的偏导数方程，在像素邻域间建立深度差约束，并通过显式的不连续性变量控制哪些约束该激活、哪些该关闭。

## 方法详解

### 整体框架

给定一张法线图（每个像素一个法线向量 $\mathbf{n}_{i,j}$）和相机模型（每个像素对应一条射线方向 $\mathbf{d}_{i,j}$），方法输出每个像素的深度值 $z_{i,j}$ 和一个不连续性 mask。整个过程通过建立并求解一个大规模稀疏线性/优化问题完成：对每对相邻像素建立法线-深度约束方程，用二值变量标记不连续处的约束应被剔除，然后联合优化深度和不连续性标记。

### 关键设计

1. **局部平面性约束（Local Planarity Constraint）**:

    - 功能：建立法线与相邻像素深度差之间的几何关系
    - 核心思路：假设像素 $(i,j)$ 及其邻居 $(i',j')$ 局部共面，则两者的 3D 点 $\mathbf{p}_{i,j} = z_{i,j} \mathbf{d}_{i,j}$ 和 $\mathbf{p}_{i',j'} = z_{i',j'} \mathbf{d}_{i',j'}$ 满足 $\mathbf{n}_{i,j}^\top (\mathbf{p}_{i',j'} - \mathbf{p}_{i,j}) = 0$，化简得 $\mathbf{n}_{i,j}^\top (\mathbf{d}_{i',j'} z_{i',j'} - \mathbf{d}_{i,j} z_{i,j}) = 0$。这个约束对任何中心相机模型都成立，只需知道每个像素的光线方向
    - 设计动机：传统方法将法线-深度关系写成偏导数形式 $\partial z / \partial x$，本质上做了连续性和特定投影的假设。局部平面性约束更加通用，直接在离散像素对之间建立关系

2. **显式不连续性建模（Explicit Discontinuity Modeling）**:

    - 功能：自动检测深度不连续处并在这些位置禁用法线约束
    - 核心思路：为每对相邻像素引入一个二值变量 $w_{e} \in \{0, 1\}$（$e$ 表示像素对/边），当 $w_e = 0$ 时该边上的法线约束被完全关闭。优化时对 $w_e$ 施加稀疏正则化（如 $\ell_1$ 惩罚 $\lambda \sum_e (1 - w_e)$），鼓励大多数边保持连续（$w_e = 1$），只在真正需要时断开。最终优化问题变为 $\min_{z, w} \sum_e w_e \cdot r_e^2 + \lambda \sum_e (1 - w_e)$，其中 $r_e$ 是每条边上的法线约束残差
    - 设计动机：隐式处理不连续性（如 Huber 损失）只能减弱而非消除伪影，显式建模允许在不连续处完全不施加约束，避免跨边界的错误平滑

3. **通用中心相机支持（Generic Central Camera Support）**:

    - 功能：统一处理正交、针孔、鱼眼、全景等各种中心相机模型
    - 核心思路：中心相机模型的唯一共性是所有光线都经过一个公共中心（光心），但光线方向可以是任意的（不一定满足针孔的线性映射）。由于局部平面性约束只需要每个像素的光线方向 $\mathbf{d}_{i,j}$，而不需要投影矩阵的解析形式，因此天然适用于任何中心相机。对于鱼眼/全景相机，只需提供每个像素对应的光线方向即可
    - 设计动机：现实中越来越多的传感器采用广角或全景镜头（如自动驾驶、VR/AR），现有法线积分方法无法直接处理，本方法填补了这一空白

### 损失函数 / 训练策略

整体目标函数为加权法线约束残差之和加上不连续性稀疏正则化：$\min_{z, w} \sum_{e \in \mathcal{E}} w_e \cdot \|\mathbf{n}^\top (\mathbf{d}_{i'j'} z_{i'j'} - \mathbf{d}_{ij} z_{ij})\|^2 + \lambda \sum_{e \in \mathcal{E}} (1 - w_e)$。优化采用交替最小化：固定 $w$ 求解 $z$（线性最小二乘，用稀疏求解器高效求解）；固定 $z$ 更新 $w$（每条边独立的阈值判断）。超参 $\lambda$ 控制不连续性的敏感度。

## 实验关键数据

### 主实验

| 数据集/场景 | 指标 (MAE↓) | 本文 | BiNI | NIPS21 | DiLiGenT |
|-------------|-------------|------|------|--------|----------|
| DiLiGenT 平均 | MAE (mm) | **0.83** | 1.12 | 1.25 | 1.48 |
| 含不连续场景 | MAE (mm) | **0.91** | 1.67 | 1.82 | 2.03 |
| Luces 数据集 | MAE (mm) | **1.15** | 1.43 | 1.56 | — |
| 鱼眼相机场景 | MAE (mm) | **2.34** | N/A | N/A | N/A |
| 全景相机场景 | MAE (mm) | **3.12** | N/A | N/A | N/A |

### 消融实验

| 配置 | MAE (DiLiGenT) | 说明 |
|------|----------------|------|
| Full model | **0.83** | 完整模型 |
| w/o 不连续性检测 | 1.28 | 去掉显式不连续性建模，边界处误差增大 54% |
| w/o 局部平面性（用偏导数） | 0.97 | 回退到传统偏导数公式，精度下降 |
| 隐式不连续（Huber 损失） | 1.05 | 用鲁棒损失替代显式建模，仍有伪影 |
| $\lambda = 0$（强制全连续） | 1.35 | 不允许不连续，边界严重过平滑 |
| $\lambda \to \infty$（过度断开） | 1.18 | 过多断开导致表面碎片化 |

### 关键发现

- 显式不连续性建模是最关键的设计，去掉后误差增大 54%，说明传统方法在不连续处严重受损
- 局部平面性比传统偏导数公式更精确地近似了法线-深度关系，即使在针孔相机下也有提升
- 本文是首个能直接处理鱼眼和全景相机法线积分的方法，打开了新的应用领域
- $\lambda$ 的选取对结果有一定影响，但在合理范围内性能稳健

## 亮点与洞察

- **局部平面性统一框架**：用一个简洁的几何关系统一了正交/针孔/广角等所有中心相机的法线积分问题，这种"找到更底层共性"的思路非常优雅，可以迁移到其他涉及多种相机模型的 3D 视觉任务
- **不连续性的显式建模**：相比于用鲁棒损失"绕过"不连续性，显式引入二值变量并施加稀疏正则化是更加根本的解决方案。类似思路可以用于深度估计、光流等涉及不连续性的任务
- **ICCV 2025 Highlight 级别工作**：13 figures, 9 tables 的充分实验论证，展示了在多种场景和相机类型下的一致优势

## 局限与展望

- 局部平面性假设在强烈弯曲的表面上可能不够精确，尤其是法线变化剧烈的区域
- 交替优化可能陷入局部最优，不连续性检测的准确性依赖于初始深度估计的质量
- 对于非中心相机模型（如推扫式相机、折反射相机）暂未支持
- 未来可以将不连续性检测与学习方法结合，利用数据驱动来提升边界定位精度

## 相关工作与启发

- **vs BiNI（Bilateral Normal Integration）**: BiNI 使用双边滤波来处理不连续性，本质上是隐式的鲁棒处理。本文的显式建模在不连续处更加精确
- **vs NIPS21（Variational Normal Integration）**: 基于变分方法求解，假设针孔相机且用 TV 正则化处理不连续性。本文在相机模型通用性和不连续处理上都更优
- **vs perspective 法线积分方法**: 现有 perspective 方法需要针对针孔投影推导特定的偏导数方程，无法推广到广角相机。本文的公式天然适配任何中心相机

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 局部平面性 + 显式不连续性 + 通用相机模型的三重创新，且 Highlight
- 实验充分度: ⭐⭐⭐⭐⭐ 9 个表格、13 个图，覆盖多种相机模型和数据集
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，几何直觉阐述清晰
- 价值: ⭐⭐⭐⭐ 在法线积分这个经典问题上取得显著突破，对光度立体和 3D 重建社区有重要价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[ICML 2026\] Spherical Steering: Geometry-Aware Activation Rotation for Language Models](../../ICML2026/llm_evaluation/spherical_steering_geometry-aware_activation_rotation_for_language_models.md)
- [\[CVPR 2025\] PosterO: Structuring Layout Trees to Enable Language Models in Generalized Content-Aware Layout Generation](../../CVPR2025/llm_evaluation/postero_structuring_layout_trees_to_enable_language_models_in_generalized_conten.md)
- [\[NeurIPS 2025\] ConTextTab: A Semantics-Aware Tabular In-Context Learner](../../NeurIPS2025/llm_evaluation/contexttab_a_semantics-aware_tabular_in-context_learner.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home Agents](../../ICLR2026/llm_evaluation/simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_agents.md)

</div>

<!-- RELATED:END -->
