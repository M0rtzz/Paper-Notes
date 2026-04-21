---
title: >-
  [论文解读] Generating Physically Stable and Buildable Brick Structures from Text
description: >-
  [ICCV2025][3D视觉][text-to-3D] BrickGPT 首次实现从文本提示生成物理稳定且可组装的互锁砖块结构，核心思想是将积木组装问题建模为自回归文本生成任务，并在推理时集成物理感知的有效性检查和回滚机制，确保生成结构的稳定性和可构建性。
tags:
  - ICCV2025
  - 3D视觉
  - text-to-3D
  - brick assembly
  - physical stability
  - autoregressive LLM
  - LEGO generation
---

# Generating Physically Stable and Buildable Brick Structures from Text

**会议**: ICCV2025  
**arXiv**: [2505.05469](https://arxiv.org/abs/2505.05469)  
**代码**: [项目主页](https://avalovelace1.github.io/BrickGPT/)  
**领域**: 3d_vision  
**关键词**: text-to-3D, brick assembly, physical stability, autoregressive LLM, LEGO generation  

## 一句话总结

BrickGPT 首次实现从文本提示生成物理稳定且可组装的互锁砖块结构，核心思想是将积木组装问题建模为自回归文本生成任务，并在推理时集成物理感知的有效性检查和回滚机制，确保生成结构的稳定性和可构建性。

## 研究背景与动机

- **问题定义**：从自由形式的文本提示直接生成由互锁积木组成的 3D 结构，要求满足两个条件：(1) 物理稳定——建在底板上不会坍塌、悬浮或断裂；(2) 可构建——兼容标准积木件，可逐块由人或机器人组装
- **现有方法局限**：
    - 标准 text-to-3D 方法（DreamFusion 等）生成的数字设计无法直接物理实现——难以用标准组件组装，且可能物理不稳定
    - 现有积木设计方法主要从给定 3D 物体转换（Luo et al.），或限于单一物体类别（Ge et al.）
    - 少数学习方法（Thompson et al.）使用图生成但仅限简单类别和单一砖块类型
    - 无法直接从文本生成，也未考虑物理约束
- **动机**：利用 LLM 的序列建模和文本理解能力，将"next-token prediction"重新定义为"next-brick prediction"，并通过训练数据和推理约束双管齐下确保物理稳定性

## 方法详解

### 整体框架

BrickGPT 包含三个阶段：
1. **数据集构建**（StableText2Brick）：47000+ 物理稳定的砖块结构 + 描述文本
2. **模型微调**：在 LLaMA-3.2-1B-Instruct 上进行指令微调
3. **物理感知推理**：逐块拒绝采样 + 物理感知回滚

### 数据集：StableText2Brick

**砖块表示**：每个结构 $B = [b_1, b_2, \ldots, b_N]$，每块 $b_i = [h_i, w_i, x_i, y_i, z_i]$（长宽+坐标），网格世界 $20 \times 20 \times 20$。

**构建流程**：
1. **Shape-to-Brick**：ShapeNetCore 的 3D 网格 → 体素化 → delete-and-rebuild 算法生成砖块布局
2. **结构增强**：对同一物体随机化砖块布局生成多种变体，增加多样性和获得稳定结构的概率
3. **稳定性过滤**：基于力学分析（非线性规划）计算每块砖的稳定性分数 $s_i \in [0,1]$，仅保留全部 $s_i > 0$ 的稳定结构
4. **描述生成**：24 视角渲染 → GPT-4o 生成 5 种详细程度的几何描述（不含颜色）

**规模**：28000+ 独特 3D 物体、21 个常见物体类别、47000+ 不同砖块结构。

### 模型微调

**自定义砖块文本格式**：每行一块 `"{h}×{w} ({x},{y},{z})"`，相比 LDraw 格式大幅减少 token 数量，同时包含尺寸信息便于 3D 推理。砖块按光栅扫描顺序从底到顶排列。

**自回归生成**：

$$p(b_1, b_2, \ldots, b_N | \theta) = \prod_{i=1}^{N} p(b_i | b_1, \ldots, b_{i-1}, \theta)$$

### 物理感知推理

**稳定性分析**基于力学建模，对结构中每块砖建立力模型（重力、垂直力、剪切力、knob 连接力），通过求解非线性规划达到静力平衡：

$$\arg\min_{\mathcal{F}} \sum_i^N \left\{ \left|\sum_j^{M_i} F_i^j\right| + \left|\sum_j^{M_i} \tau_i^j\right| + \alpha \mathcal{D}_i^{\max} + \beta \sum \mathcal{D}_i \right\}$$

使用 Gurobi 求解。约束包括：力非负、互斥力不能共存、牛顿第三定律。

**逐块拒绝采样**：对每个生成的砖块检查格式合法性、边界内、无碰撞——轻量约束不显著影响推理时间。

**物理感知回滚**：生成完成后计算稳定性分数。若不稳定，回滚到第一个不稳定砖块之前的状态 $B' = [b_1, \ldots, b_{\min \mathcal{I} - 1}]$，从 $B'$ 继续生成。迭代至多 100 次。中位回滚次数仅 2 次，中位生成时间 40.8 秒。

### 砖块上色与纹理

- **UV 纹理**：合并可见砖块为网格 → 立方体投影生成 UV 图 → FlashTex 生成纹理
- **均匀颜色分配**：体素化 → UV 展开 → FlashTex 生成纹理 → 逐体素/砖块取均值 → 匹配标准颜色库

## 实验关键数据

### 主实验结果

| 方法 | 有效率% | 稳定率% | 平均稳定性 | 最小稳定性 | CLIP ↑ | DINO ↑ |
|------|---------|---------|-----------|-----------|--------|--------|
| Pre-trained LLaMA (0-shot) | 0.0 | 0.0 | N/A | N/A | N/A | N/A |
| In-context learning (5-shot) | 2.4 | 1.2 | 0.675 | 0.479 | 0.284 | 0.814 |
| LLaMA-Mesh | 94.8 | 50.8 | 0.894 | 0.499 | 0.317 | 0.851 |
| Hunyuan3D-2 + stability | 100 | 88.4 | 0.976 | 0.813 | 0.324 | 0.868 |
| **BrickGPT** | **100** | **98.8** | **0.996** | **0.915** | **0.324** | **0.880** |

### 消融实验

| 变体 | 有效率% | 稳定率% | 平均稳定性 | 最小稳定性 | CLIP |
|------|---------|---------|-----------|-----------|------|
| 无拒绝采样和回滚 | 37.2 | 12.8 | 0.956 | 0.325 | 0.329 |
| 无回滚 | 100 | 24.0 | 0.947 | 0.228 | 0.322 |
| **完整 BrickGPT** | **100** | **98.8** | **0.996** | **0.915** | **0.324** |

### 关键发现

1. 预训练 LLaMA（0-shot）完全无法生成合格结构，5-shot 仅 2.4% 有效率——证明微调和约束的必要性
2. 拒绝采样将有效率从 37.2% 提升到 100%，回滚将稳定率从 24.0% 提升到 98.8%
3. 即使 Hunyuan3D-2 等先进 text-to-3D 方法转换为砖块后+稳定性分析，稳定率仍低于 BrickGPT（88.4% vs 98.8%）
4. 中位回滚仅 2 次说明模型已在训练中学到较好的稳定性先验
5. 生成结构可被人手动组装和机器人自动组装验证

## 亮点与洞察

1. **跨域创新**：将 LLM 的 next-token prediction 机制巧妙适配为 next-brick prediction，思路简洁优雅
2. **物理约束的优雅集成**：不在每步都做物理检查（过度约束），而是轻量逐块检查 + 完成后回滚，平衡了效率和稳定性
3. **端到端可实现**：生成结构可直接物理组装，包括双臂机器人自动装配，真正实现了从文本到真实物体的闭环
4. **数据集贡献**：StableText2Brick（47000+ 结构 + 描述）是大规模的物理稳定砖块结构数据集
5. **自定义文本格式设计**：相比 LDraw 减少冗余信息（方向、缩放），显著降低 token 数量

## 局限性

1. 结构限制在 $20 \times 20 \times 20$ 网格，分辨率有限，无法生成更精细的设计
2. 训练样本限制在 4096 token 以内，大型结构可能被截断
3. 仅使用 ShapeNetCore 的 21 个类别，自然场景物体覆盖有限
4. 稳定性分析依赖 Gurobi 商业求解器
5. 颜色/纹理生成与几何形状生成是分离的后处理步骤

## 相关工作与启发

- **LLM → 3D**：LLaMA-Mesh 证明 LLM 可微调输出 3D 形状（OBJ 格式），本文进一步将其适配到受物理约束的砖块结构
- **物理感知生成**：从简单碰撞避免 → 结构稳定性分析 → 力学建模+非线性规划，物理约束的精细程度在不断提升
- **text-to-3D 的可实现性**：本文开辟了从文本到可物理组装物体的新方向，对制造、教育、建筑等领域有实际意义
- **受约束推理**：拒绝采样+回滚的推理范式可推广到其他需要满足硬约束的生成任务

## 评分 ⭐⭐⭐⭐

选题新颖，首次实现文本到物理稳定砖块结构，方法设计巧妙，物理约束集成合理。数据集规模大且质量高。实验全面，消融清晰证明各组件贡献。可直接物理组装的验证增加说服力。局限在于分辨率和类别覆盖，但作为开创性工作已很出色。

<!-- RELATED:START -->

## 相关论文

- [Stable Score Distillation](stable_score_distillation.md)
- [Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence](../../CVPR2025/3d_vision/stable-score_a_stable_registration-based_framework_for_3d_shape_correspondence.md)
- [SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [Bolt3D: Generating 3D Scenes in Seconds](bolt3d_generating_3d_scenes_in_seconds.md)
- [GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)

<!-- RELATED:END -->
