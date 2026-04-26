---
title: >-
  [论文解读] Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion
description: >-
  [CVPR 2026][3D视觉][3D城市生成] 提出 Yo'City 多智能体框架，通过"City–District–Grid"层次化规划 + produce–refine–evaluate 等距图像合成环 + 场景图引导扩展机制，实现用户个性化文本驱动的无界 3D 城市生成，在语义一致性和视觉质量上全面超过 SynCity 等现有方法。
tags:
  - CVPR 2026
  - 3D视觉
  - 3D城市生成
  - 多智能体框架
  - 层次化规划
  - 等距图像合成
  - 场景图扩展
  - LLM驱动
---

# Yo'City: Personalized and Boundless 3D Realistic City Scene Generation via Self-Critic Expansion

**会议**: CVPR 2026  
**arXiv**: [2511.18734](https://arxiv.org/abs/2511.18734)  
**代码**: 待确认  
**领域**: 3d_vision  
**关键词**: 3D城市生成, 多智能体框架, 层次化规划, 等距图像合成, 场景图扩展, LLM驱动

## 一句话总结

提出 Yo'City 多智能体框架，通过"City–District–Grid"层次化规划 + produce–refine–evaluate 等距图像合成环 + 场景图引导扩展机制，实现用户个性化文本驱动的无界 3D 城市生成，在语义一致性和视觉质量上全面超过 SynCity 等现有方法。

---

## 研究背景与动机

**3D 城市模型需求广泛**：虚拟现实、游戏、城市规划、数字孪生、机器人仿真等场景均依赖高质量 3D 城市模型，但人工建模极为耗时。

**传统方法局限性大**：过程化建模和基于图像的重建方法依赖手工规则或街景数据，扩展性差；基于 GAN/扩散模型的方法需要地图或卫星数据训练，难以处理用户文本输入。

**SynCity 的问题**：SynCity 采用自回归 tile-by-tile 流水线，缺少显式层次化规划，在大规模城市生成时出现空间不一致（部分 tile 密集、部分稀疏）、纹理模糊、几何简化问题。

**LLM/VLM 的机遇**：大语言模型的世界知识和推理能力为城市规划提供了新可能，但 agentic 3D 城市生成仍几乎未被探索。

**核心挑战**：城市是开放、大规模、高度结构化的空间，物体多样性和空间组织密度远超室内场景，需要层次化推理和扩展机制。

---

## 方法详解

### 整体框架

Yo'City 由四个核心模块组成，贯穿 **"规划-生成-扩展"** 三阶段：

> 用户文本 $p_0$ → **Global Planner** (城市级规划) → **Local Designer** (网格级描述) → **3D Generator** (等距图像→3D资产) → **Expansion Module** (场景图引导扩展)

关键特点：**并行生成**所有 tile（消除自回归的因果依赖），避免误差累积并加速生成。

### 问题形式化

将城市空间划分为 $H \times W$ 的网格 $\mathcal{T} = \{0, \ldots, H-1\} \times \{0, \ldots, W-1\}$，每个 tile $(x, y)$ 对应一个 3D 场景块（如住宅小区）。与 SynCity 的自回归方式不同，Yo'City 基于全局提示 $p_0$ **同时并行**生成所有 tile 的属性，打破 tile $(x,y)$ 必须依赖已生成 tile 集合 $\mathcal{T}(x,y)$ 的严格因果约束。

### 关键模块 1：Global Planner（全局规划器）

- **功能**：将用户的抽象文本描述 $p_0$ 转化为高层城市布局。
- **三步规划流程**：
  1. **Size Estimation**：LLM 估计城市尺寸为 $H \times W$ 矩形网格，每个格子作为基本空间单元。
  2. **District Planning**：规划 $N$ 个功能区，生成蓝图集 $\{B_i \mid i=1,2,\ldots,N\}$，每个 $B_i$ 描述该区的功能（如"商业中心"）和建筑类型（如"高层写字楼"）。
  3. **Layout Allocation**：考虑区域间空间关系和邻近约束，将各功能区合理分配到 $H \times W$ 网格上，可跨多个格子。
- **RAG 增强**：针对真实城市引用（如"纽约风格"），从 Wikipedia 语料库检索城市结构、分区特征等信息，经 GPT-4o-mini 提炼后注入规划过程，使生成结果对齐真实城市的空间逻辑。

### 关键模块 2：Local Designer（局部设计器）

- **功能**：将 Global Planner 输出的粗粒度蓝图 $\{B_i\}$ 细化为网格级文本描述。
- **核心思路**：LLM 以蓝图 $B_i$ 和全局提示 $p_0$ 为条件，为城市中每个网格生成详细设计 $\{d_i \mid i=1,2,\ldots,H \times W\}$，涵盖建筑风格、密度、地标、周边环境等。
- **连续性保证**：对同一功能区内的所有网格进行联合规划，确保空间和风格的区域内一致性。
- **设计动机**：相比一步到位生成全城布局，粗到细策略为 LLM 提供了隐式推理过程——先全局组织再局部细化，产出的布局更具真实性和合理性。

### 关键模块 3：3D Generator（3D 生成器）

分两阶段将网格描述 $\{d_i\}$ 转化为 3D 资产：

#### 阶段 A：Produce–Refine–Evaluate 等距图像合成环

朴素的 text-to-image 方法容易产生物体错位、建筑不完整等问题。为此设计迭代循环：

1. **Produce（生成）**：在预定义地面平台上为网格 $d_i$ 生成初始等距图像。平台作为公共锚点，确保所有资产共享一致的尺度和空间对齐。
2. **Refine（精修）**：图像编辑模型移除平台并修正几何伪影、增强视觉多样性。
3. **Evaluate（评估）**：专用评估器从文图对齐、真实感、布局合理性三方面打分。若未达标则反馈重新生成，直到满足全部质量标准。

#### 阶段 B：3D 模型转换与场景装配

- 使用预训练 Hunyuan3D 模型将高质量等距图像转为 3D 模型。
- 利用并行网格对齐管线，所有 3D 模型按 Global Planner 的预定义布局直接排列，**无需 3D 融合即可消除边界不一致**。
- 添加道路、地面等连接元素，并支持用户按主题自定义地面材质（如古风 / 现代）。

### 关键模块 4：Relationship-guided Expansion（关系引导扩展）

城市空间遵循功能区邻近原则（住宅靠近学校和商业区，工业区远离居民区）。扩展机制实现无界城市演化：

#### 扩展流程

1. **VLM 推理**：给定已渲染城市和区域概览，VLM 为新扩展网格生成文本描述 $d_{\text{new}}$ 并构建**场景图**——新网格为中心节点，与现有功能区的边编码定性距离关系（near / relatively near / far 等）。
2. **候选位置搜索**：通过广度优先搜索在城市布局中找到可行候选位置集 $\mathcal{X}$。
3. **距离-语义联合优化**：确定最优放置位置。

#### 优化目标

**距离驱动空间目标**（拉近/推离满足空间关系）：

$$L_{\text{dist}}(x) = \sum_{g \in \mathcal{G}} \gamma_{r(g)} \|x - g\|_2$$

其中 $\gamma_{r(g)}$ 为有符号权重：正值表示拉近（proximity），负值表示推离（separation）。

**语义正则化**（确保新网格与邻域语义兼容）：

$$L_{\text{sem}}(x) = -\sum_{y \in \mathcal{N}(x)} \text{Embedding\_Sim}(d_{\text{new}}, d_y)$$

基于 Sentence-BERT 嵌入相似度，鼓励选择语义融合度高的位置。

**综合目标**：

$$x^* = \arg\min_{x \in \mathcal{X}} \left[ L_{\text{dist}}(x) + \lambda \, L_{\text{sem}}(x) \right]$$

得到最优位置 $x^*$ 后调用 3D Generator 生成新网格模型，完成一轮扩展。用户可迭代交互反复扩展，实现真正的开放世界城市生长。

---

## 实验设置

- **数据集**：100 条城市文本描述（30% 人工编写 + 70% GPT-4o 生成），涵盖多种风格。
- **基线方法**：Trellis（text-to-3D）、Hunyuan3D API（text-to-3D）、CityCraft（布局+资产检索）、SynCity（自回归 tile-by-tile）。
- **实现细节**：GPT-4o 作为 LLM，GPT-Image-1 进行图像编辑，Hunyuan3D API 进行 image-to-3D。

---

## 实验关键数据

### 表 1：城市级定量对比（VQAScore + GPT-5/人类 win rate）

| 方法 | VQAScore | 几何保真度 (GPT-5 / 人类) | 纹理清晰度 (GPT-5 / 人类) | 布局连贯性 (GPT-5 / 人类) | 整体真实感 (GPT-5 / 人类) |
|------|----------|--------------------------|--------------------------|--------------------------|--------------------------|
| Trellis | 0.6189 | 6.5% / 7.0% | 4.5% / 6.0% | 6.5% / 3.5% | 9.0% / 5.0% |
| Hunyuan3D | 0.6198 | 12.0% / 7.0% | 12.5% / 9.5% | 7.0% / 5.5% | 12.0% / 6.5% |
| CityCraft | 0.5639 | 9.5% / 8.0% | 6.0% / 6.0% | 15.0% / 16.5% | 12.0% / 13.5% |
| SynCity | 0.6975 | 15.0% / 12.0% | 21.5% / 18.5% | 14.0% / 10.5% | 15.5% / 12.0% |
| **Yo'City** | **0.7151** | **85–93.5%** | **78.5–95.5%** | **85–93.5%** | **84.5–95%** |

Yo'City 在所有维度上全面压制基线，VQAScore 最高（0.7151），对最强基线 SynCity 的 win rate 也稳定在 78.5% 以上。

### 表 2：网格级对比（SynCity vs. Yo'City）

| 方法 | Alignment Score | Aesthetic Score |
|------|----------------|-----------------|
| SynCity | 0.6572 | 4.95 |
| **Yo'City** | **0.6927** (+0.0355) | **5.52** (+0.57) |

网格级评估进一步证实 Yo'City 不仅全局一致性更好，每个局部网格的语义对齐和美学质量也更优。

### 消融实验：粗到细规划

| 指标 | w/o reason | w/ reason |
|------|-----------|-----------|
| VQAScore | 0.7034 | **0.7151** |
| Layout Coherence (win rate) | 27.0% | **73.0%** |
| Overall Realism (win rate) | 24.5% | **75.5%** |

粗到细规划（Global Planner + Local Designer）在布局连贯性上带来 +46% 的 win rate 提升，验证了层次化推理的必要性。

### 扩展机制稳定性

对 5 个城市各执行 4 步扩展，VQAScore 的变异系数仅 3.34%，表明扩展过程中语义一致性保持稳定。

---

## 亮点与洞察

- **首个层次化 agentic 城市生成框架**："City–District–Grid"三级规划模拟真实城市的组织逻辑，比 SynCity 的平坦 tile-by-tile 方式更符合城市内在结构。
- **并行生成突破自回归瓶颈**：消除 tile 间因果依赖，避免误差累积，同时大幅加速生成。
- **RAG 增强真实性**：从 Wikipedia 检索真实城市结构知识注入规划过程，使"纽约风格"等指令有事实依据而非 LLM 幻觉。
- **produce–refine–evaluate 闭环**：引入评估器反馈机制确保等距图像质量，解决了朴素 text-to-image 的空间对齐问题。
- **关系引导扩展支持无界演化**：场景图 + 距离-语义联合优化，新增区域自动满足功能区邻近原则（如商超靠近住宅、工业远离居民区）。
- **个性化能力强**：能处理"哈利波特主题公园"、"丝绸之路"等细粒度个性化描述。

---

## 局限性

1. **依赖闭源大模型**：核心流程使用 GPT-4o + GPT-Image-1 + Hunyuan3D API，成本高且不可复现，开源替代可能导致质量下降。
2. **评估主观性强**：视觉质量主要依赖 GPT-5 和人类的 pairwise 比较，缺少客观几何指标（如 FID、点云精度）。
3. **缺少与真实城市数据的对比**：未在 Google Earth、OSM 等真实城市数据上评估重建精度。
4. **扩展语义漂移风险**：多轮扩展后全局风格一致性的长期保持未充分验证（仅测试了 4–8 步）。
5. **生成速度未报告**：未给出单个城市的端到端生成耗时，包含多轮 API 调用的延迟可能很高。
6. **网格边界硬切分**：3D 模型按网格拼接时依赖道路/地面填充，缺少跨网格建筑（如横跨两个 tile 的大型体育馆）的处理。

---

## 相关工作与启发

- **SynCity**：自回归 tile-by-tile 生成，无层次化规划，大规模城市出现空间不一致和纹理退化。Yo'City 的层次化规划 + 并行生成直接解决了这两个问题。
- **CityCraft**：先生成语义布局再检索预定义资产，泛化性受限于资产库，对非现代城市风格表现差。
- **室内场景 LLM 生成（LayoutGPT, I-Design）**：室内环境闭合可控，而城市开放且物体密度高、类别多，Yo'City 的层次化分解是关键差异。
- **Hunyuan3D / Trellis**：通用 text-to-3D 模型，缺少城市级布局规划和拼接能力，直接用于城市生成效果差。
- **启发**：该框架可推广到其他大规模场景生成（如高速公路网络、园区规划），RAG + 层次规划的范式也适用于非城市领域的结构化生成任务。

---

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐ |

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images](from_orbit_to_ground_generative_city_photogrammetry_from_extreme_off-nadir_satel.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](../../ICCV2025/3d_vision/sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] Benchmarking Egocentric Visual-Inertial SLAM at City Scale](../../ICCV2025/3d_vision/benchmarking_egocentric_visualinertial_slam_at_city_scale.md)
- [\[ICCV 2025\] GeoProg3D: Compositional Visual Reasoning for City-Scale 3D Language Fields](../../ICCV2025/3d_vision/geoprog3d_compositional_visual_reasoning_for_city-scale_3d_language_fields.md)
- [\[CVPR 2026\] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)

<!-- RELATED:END -->
