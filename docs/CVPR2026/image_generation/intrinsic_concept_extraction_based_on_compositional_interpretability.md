---
title: >-
  [论文解读] Intrinsic Concept Extraction Based on Compositional Interpretability
description: >-
  [CVPR 2026][图像生成][概念提取] HyperExpress 提出组合可解释本征概念提取（CI-ICE）新任务，利用双曲空间的层次建模能力和等球面投影模块，从单张图像中提取可组合的物体级和属性级概念，实现可逆的复杂视觉概念分解。
tags:
  - CVPR 2026
  - 图像生成
  - 概念提取
  - 双曲空间
  - 组合可解释性
  - 扩散模型
  - 概念解耦
---

# Intrinsic Concept Extraction Based on Compositional Interpretability

**会议**: CVPR 2026  
**arXiv**: [2603.11795](https://arxiv.org/abs/2603.11795)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 概念提取, 双曲空间, 组合可解释性, 扩散模型, 概念解耦

## 一句话总结

HyperExpress 提出组合可解释本征概念提取（CI-ICE）新任务，利用双曲空间的层次建模能力和等球面投影模块，从单张图像中提取可组合的物体级和属性级概念，实现可逆的复杂视觉概念分解。

## 研究背景与动机

**领域现状**：无监督概念提取（UCE）旨在从单张图像中提取人类可理解的视觉概念。现有方法如 ConceptExpress、AutoConcept 只能提取物体级概念，ICE 虽能提取属性级概念但不考虑组合性。

**核心问题**：
   - 现有方法仅关注概念解耦（disentanglement），忽视了可组合性（composability），导致提取的概念无法可靠地重组回原始图像
   - CCE 方法虽考虑了组合性，但需要从包含相同概念的多张图像中学习
   - 欧几里得空间难以捕捉物体级和属性级概念之间的层次结构和关联关系

**本文方案**：提出 CI-ICE 任务和 HyperExpress 方法，通过双曲空间学习概念层次结构，通过等球面投影确保概念嵌入空间的组合性

## 方法详解

### 整体框架

HyperExpress 由两个核心部分组成：**概念学习**和**概念优化**。先使用 ICE 第一阶段定位主要物体并获取 mask 和文本描述，然后通过双曲对比学习和蕴含学习模块学习概念，最后通过等球面投影优化组合性。输入包含 N 个物体（每个有 M 个属性）的图像，目标是学习 (M+1)*N 个概念 token 及其嵌入向量。

### 关键设计

#### 1. 双曲文本编码器

将 CLIP 编码的文本嵌入通过指数映射投射到庞加莱球（Poincare ball），添加可学习权重 W 学习从标准编码器空间到切空间的映射。使得概念嵌入天然具有层次性：距球心越近代表越抽象的物体级概念，越靠近边界代表越具体的属性级概念。

#### 2. 双曲对比学习模块（HCL）

利用双曲空间的层次建模能力区分物体级和属性级概念：

- **物体-属性区分**：双曲三元组损失，使物体级概念锚点与对应物体嵌入的距离小于与属性嵌入的距离
- **不同属性区分**：属性级三元组损失，使同一属性类型内不同属性的距离保持合理
- 核心思路：在双曲空间中，不同层级的概念自然被放置在不同位置

#### 3. 双曲蕴含学习模块（HEL）

在洛伦兹模型中建立物体与属性间的蕴含（entailment）关系：

- 若概念 i 蕴含概念 j，则空间角 theta(v_i, v_j) 小于蕴含锥半径 omega(v_i)
- 蕴含损失确保属性概念落入对应物体概念的蕴含锥内
- 通过庞加莱球到洛伦兹模型的变换来计算锥半径和空间角

#### 4. 等球面投影模块（HP）

将概念嵌入空间映射到可组合子流形：

- 在锚点上训练，寻找 n 个测地方向使投影后方差最大化
- 利用等距性质确保不破坏概念间层次结构和关联关系
- 投影后子流形继承等球面零曲率特性，支持向量加法，实现概念组合
- 通过正交矩阵完成旋转操作

### 损失函数 / 训练策略

总损失由四部分组成：重建损失（扩散模型去噪）、双曲三元组损失（物体级+属性级）、Wasserstein 注意力对齐损失、双曲蕴含损失，各项通过 lambda 加权。

## 实验关键数据

### 主实验

**UCEBench 性能对比（表1）**：

| 方法 | SIM^I (%) | SIM^C (%) | ACC^1 (%) | ACC^3 (%) |
|------|-----------|-----------|-----------|-----------|
| Break-A-Scene | 0.627 | 0.773 | 0.174 | 0.282 |
| ConceptExpress | 0.689 | 0.784 | 0.263 | 0.385 |
| AutoConcept | 0.690 | 0.770 | 0.350 | 0.520 |
| ICE | 0.738 | 0.822 | 0.325 | 0.518 |
| **HyperExpress** | **0.699** | **0.786** | **0.504** | **0.736** |

**ICBench 性能对比（表2）**：

| 方法 | SIM^T-T_obj | SIM^T-T_mat | SIM^T-T_color | SIM^T-V_obj | SIM^T-V_mat | SIM^V-T_color |
|------|------------|------------|--------------|------------|------------|--------------|
| ICE | 0.249 | 0.101 | 0.093 | 0.264 | 0.208 | 0.215 |
| **HyperExpress** | **0.280** | **0.115** | **0.098** | **0.305** | **0.211** | **0.222** |

### 消融实验

| HCL | HEL | HP | SIM^I | SIM^C | ACC^1 | ACC^3 |
|-----|-----|-----|-------|-------|-------|-------|
| Y | N | N | 0.625 | 0.769 | 0.326 | 0.509 |
| Y | Y | N | 0.688 | 0.771 | 0.330 | 0.518 |
| Y | N | Y | 0.621 | 0.765 | 0.348 | 0.522 |
| Y | Y | Y | **0.699** | **0.786** | **0.504** | **0.736** |

### 关键发现

- HyperExpress 在 ACC^1 和 ACC^3 上大幅领先（0.504 vs 0.350），代价是 SIM^I 略低于 ICE
- 三模块协同效果显著：仅 HCL 时 ACC^3=0.509，三模块完整达 0.736（+44.6%）
- HP 对组合性贡献最大，HEL 对 SIM^I 提升最大

## 亮点与洞察

1. **创新任务定义**：CI-ICE 同时要求解耦和可组合性，填补研究空白
2. **双曲几何巧妙应用**：庞加莱球天然层次建模处理概念层级
3. **等距投影理论保证**：HP 的等距性质确保不破坏已学习的概念关系
4. **可解释组合路径**：如 "robot" + "metal" + "gold" -> "golden robot made of metal"

## 局限与展望

1. 在 SIM^I 上不如 ICE，组合性约束带来保真度损失
2. 依赖 ICE 第一阶段物体定位
3. 双曲空间计算增加复杂度
4. 仅在 D1 数据集评估
5. 属性类型固定为颜色和材质

## 相关工作与启发

- **ICE**：直接前驱，单图本征概念提取但忽视组合性
- **CCE**：组合性理论框架但需多图像
- **HoroPCA**：启发等球面投影模块
- **启发**：双曲空间在视觉概念学习中值得更广泛探索

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 新任务+双曲空间创新应用 |
| 技术深度 | 4 | 严谨数学框架和理论证明 |
| 实验完整性 | 3 | 数据集和基线较少 |
| 写作质量 | 4 | 逻辑清晰 |
| 实用价值 | 3 | 任务较学术化 |
| 总分 | 3.6 | |

<!-- RELATED:START -->

## 相关论文

- [coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](codrawagents_a_multi-agent_dialogue_framework_for_compositional_image_generation.md)
- [ICE: Intrinsic Concept Extraction from a Single Image via Diffusion Models](../../CVPR2025/image_generation/ice_intrinsic_concept_extraction_from_a_single_image_via_diffusion_models.md)
- [Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)

<!-- RELATED:END -->
