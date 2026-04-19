---
title: >-
  [论文解读] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models
description: >-
  [ICCV 2025][3D视觉][物体空间关系] 提出从预训练 2D 扩散模型合成图像中学习物体间 3D 空间关系（OOR），通过 3D 提升管线构建配对数据集，训练文本条件化的 score-based 扩散模型对物体对的相对位姿和尺度分布建模，并扩展至多物体场景布局和场景编辑。
tags:
  - ICCV 2025
  - 3D视觉
  - 物体空间关系
  - 扩散模型
  - 3D场景布局
  - Score-based模型
  - 多物体场景生成
---

# Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models

**会议**: ICCV 2025  
**arXiv**: [2503.19914](https://arxiv.org/abs/2503.19914)  
**领域**: 3D视觉  
**关键词**: 物体空间关系, 扩散模型, 3D场景布局, Score-based模型, 多物体场景生成  
**作者**: Sangwon Baik, Hyeonwoo Kim, Hanbyul Joo（首尔国立大学 & RLWRLD）

## 一句话总结

提出从预训练 2D 扩散模型合成图像中学习物体间 3D 空间关系（OOR），通过 3D 提升管线构建配对数据集，训练文本条件化的 score-based 扩散模型对物体对的相对位姿和尺度分布建模，并扩展至多物体场景布局和场景编辑。

## 研究背景与动机

真实场景中物体之间存在特定的空间和功能性摆放模式。椅子围绕桌子放置、杯子放在桌上而非椅子上、披萨刀以特定角度切割披萨——这些直觉而多样的关系被定义为**物体-物体空间关系（Object-Object Relationships, OOR）**，描述物体对之间的相对位姿和尺度。理解并生成这些自然布局对内容创建、VR/AR、机器人操控等应用至关重要。

**现有方法的局限**：

- **手动标注/受控采集**：OOR 多样性极高，类别组合爆炸，人工方式成本不可承受
- **室内 3D 数据集**（ScanNet、3D-FRONT、HyperSim 等）：仅覆盖有限预定义类别，无法推广到开放类别
- **互联网真实图像**：场景杂乱，难以从 2D 图像提取精确 3D 空间关系
- **LLM 方法**（SceneTeller、SMC）：无法直接访问真实 3D 数据，缺乏精细空间控制能力

**核心洞察**：2D 扩散模型生成的图像天然包含合理的物体空间关系线索——茶壶倒茶的倾斜角度、刀切苹果的姿态等皆蕴含丰富的 3D 先验。利用这一特性可高效构建多样化的 3D OOR 数据集。

## 方法详解

### 整体框架（三阶段）

1. **OOR 形式化定义**：定义物体对的相对位姿和尺度表示空间
2. **3D OOR 数据集生成**：从 2D 合成图像通过 3D 提升管线构建数据
3. **OOR 扩散模型**：训练 score-based 扩散模型学习 OOR 分布

### 3.1 OOR 形式化表示

将一对物体中的一个指定为基准物体（base），另一个为目标物体（target）。OOR 样本定义为：

$$\phi = (\mathbf{R}^{\mathcal{T}\to\mathcal{B}},\; \mathbf{t}^{\mathcal{T}\to\mathcal{B}},\; \mathbf{s}^{\mathcal{T}\to\mathcal{B}},\; \mathbf{s}^{\mathcal{B}})$$

- **R ∈ SO(3)**：目标相对于基准的旋转
- **t ∈ ℝ³**：相对平移
- **s_target ∈ ℝ³₊**：目标物体的非各向同性缩放（保留长宽比）
- **s_base ∈ ℝ³₊**：基准物体的缩放因子

每个物体实例定义在各自的 canonical 空间中（包围盒中心为原点，y 轴朝上，z 轴朝前），并引入**尺度归一化 canonical 空间**（3D 包围盒归一化为单位立方体）处理类内不同长宽比。

### 3.2 3D OOR 数据生成 Pipeline

这是方法中最工程化也最关键的部分，解决了 3D OOR 数据稀缺问题。

**Step 1: 高质量 2D OOR 图像合成**

- 使用 FLUX.1-dev 文本到图像模型生成包含 OOR 线索的图像
- 提示策略：追加"white background"（确保物体完整可见）、加入形状纹理描述（对齐模板 mesh）、调整视角（处理尺度差异大的类别对如桌子-茶杯）
- 进一步用 image-to-video 模型（SV3D）扩增多样性，每帧作独立 2D 样本

**Step 2: 伪多视角生成与 SfM**

- SV3D 生成环形多视角图像
- VGGSfM 重建 3D 点云，丢弃重建失败的样本
- 输出：3D 点云及其 2D 关键点对应

**Step 3: 网格配准提取位姿和尺度**

- 视频分割模型（SAM2/Grounding DINO）分离基准/目标物体点云
- 语义特征提取：从 2D 视角提取 768 维语义特征，PCA 降至 15 维后按 3D 点聚合平均
- 余弦相似度建立模板 mesh 与点云的对应关系
- **Procrustes 分析 + RANSAC** 估计刚体变换，**ICP** 精化
- 多候选模板 mesh 通过 DINO 特征选择最匹配者
- 自动过滤去除不可靠样本

### 3.3 OOR 扩散模型

基于 GenPose 的 score-based 模型框架。

**训练**：模型 Ψ_θ 学习 OOR 分布的噪声 score function。条件输入包括文本上下文 c、基准类别 B、目标类别 T，均由预训练 T5 编码器编码。使用 Denoising Score Matching（DSM）损失训练。

**推理**：从纯高斯噪声出发，通过 Probability Flow ODE 的逆向过程生成 OOR 样本。

**文本上下文增强**（LLM 驱动）：

- **措辞多样化**：变换动词和句式，保持语义不变（如"倒茶"→"将茶倒入"）
- **类别替换**：相似形状物体共享 OOR 分布（如"茶壶→水壶"、"茶杯→咖啡杯"）
- 最终覆盖 **475 种上下文、188 个物体类别、23750 条文本提示**

### 3.4 多物体 OOR 扩展

场景表示为带有单一起始节点的**连通 DAG（有向无环图）**。每个节点是物体，每条边是成对 OOR。

**两大挑战及解决方案**：

1. **碰撞问题**：非相邻节点物体可能重叠 → 碰撞损失 C(Φ) 惩罚 AABB 重叠
2. **不一致问题**：同一物体可由多条路径确定位姿（如键盘可从显示器或鼠标推导）→ 不一致损失 I(Φ) 最小化多路径 OOR 方差

修改后的逆向 ODE：

$$\frac{d\phi_t^{p_i}}{dt} = -\sigma(t)\dot{\sigma}(t)\nabla_{\phi_t^{p_i}}\log p_i(\phi_t^{p_i}) + \lambda_1 \nabla C(\Phi) + \lambda_2 \nabla I(\Phi)$$

权重设置：λ₁ = min(100/t, 10⁴)，λ₂ = min(100/t², 10⁵)，从 t=0.5 起施加约束。

## 实验结果

### 成对 OOR 生成（150 场景，30 对类别，92 受访者用户研究）

| 指标 | SMC | SceneTeller | **本文** |
|------|-----|-------------|----------|
| CLIP Score ↑ | 28.54 | 29.06 | **29.11** |
| VQA Score ↑ | 0.61 | 0.68 | **0.69** |
| VLM Score ↑ | 49.83 | 64.67 | **75.67** |
| 用户研究(%) ↑ | 22.21 | 23.77 | **54.02** |

- SMC 平移合理但经常完全误判旋转
- SceneTeller 受益于 LLM 上下文学习能力，可估计大致位置关系，但缺乏精细 3D 数据支撑
- 本文在功能性关系（如"倒茶""切割"）上尤为突出

### 多物体 OOR 生成（20 场景，3-5 物体，81 受访者）

| 指标 | GraphDreamer | **本文** |
|------|------------|----------|
| VLM Score ↑ | 2.50 | **97.50** |
| 用户研究(%) ↑ | 11.88 | **88.12** |

GraphDreamer 常失捕 OOR（如"刀切苹果"），甚至丢失物体（如鼠标、盐罐），本文通过组合成对 OOR 知识稳定生成多物体场景。

### 应用验证

**3D 场景编辑**：利用 score function 梯度驱动优化（50 步内完成，η=0.01, λ₁=0.01）：

- 噪声场景去噪→合理布局
- 切换场景语义（如茶壶从"放在茶杯旁"变为"向茶杯倒茶"）
- 向已有场景添加新物体并应用新关系

**人体运动合成**：结合 VPoser 体姿先验和接触约束，从初始人体-物体交互状态生成连贯运动序列（如人抓茶壶倒茶到茶杯中）。OOR 序列由优化过程产出，接触约束保持人体与物体间初始接触对在整个序列中的距离不变。

## 亮点与洞察

1. **全新任务定义**：首次形式化定义 OOR 概念和参数空间，填补 3D 关系建模的形式化空白
2. **巧妙利用 2D 扩散模型的隐含 3D 知识**：物体摆放先验蕴含在生成图像中，无需真实 3D 标注
3. **Score-based 模型捕捉多模态分布**：同一 OOR 上下文可有多种合理配置（茶壶可从不同方向倒茶），扩散模型自然建模这种多模态性
4. **DAG + 推理时损失的多物体扩展**：无需重新训练模型，仅通过推理时约束即可组合成对 OOR 为多物体场景
5. **Score function 的灵活应用**：直接用 score 梯度驱动场景编辑优化，展现 score-based 模型在下游任务的天然优势
6. **LLM 驱动数据增强**：语义和类别两个维度同时扩展，475 上下文覆盖广泛

## 局限性

- 3D 提升质量依赖 SV3D 伪多视角的几何一致性，不一致时配准失败率较高
- 仅建模静态空间关系，未考虑动态变化过程（如倒茶过程中水位上升）
- 碰撞检测使用 AABB 近似，非凸形物体可能误判
- Pipeline 冗长（text-to-image → video → SfM → 分割 → 特征 → 配准 → 过滤），计算开销显著
- 功能性关系的多样性受限于 2D 扩散模型的生成能力上限
- 评估指标（VLM Score 等）偏主观，缺乏标准量化基准

## 相关工作

- **物体空间关系学习**：机器人摆放任务、语言条件空间推理，多限于预定义类别
- **2D 扩散先验提取**：CHORUS（人-物交互）、ComA（人-物关系），但物体位姿估计比人体更困难
- **Score-based 扩散模型**：GenPose（6D 位姿）、DAViD（动态 HOI）是直接技术基础
- **3D 场景生成**：GraphDreamer（文本到 3D）、SceneTeller（LLM 布局），均未达到精细 3D 控制

## 评分

- **新颖性**: ★★★★★ — 全新任务定义、新颖的"2D 扩散→3D 关系"范式
- **技术深度**: ★★★★☆ — Pipeline 设计合理完整但模块组合性较强
- **实验充分度**: ★★★★☆ — 多指标+用户研究有力，但缺 ablation 对各模块贡献的分析
- **实用性**: ★★★★☆ — 场景编辑和运动合成展示应用潜力
- **总分**: 8.5/10

<!-- RELATED:START -->

## 相关论文

- [4D Visual Pre-training for Robot Learning](4d_visual_pretraining_for_robot_learning.md)
- [Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [Spatial-Temporal Aware Visuomotor Diffusion Policy Learning](spatial-temporal_aware_visuomotor_diffusion_policy_learning.md)
- [Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [Bridging Diffusion Models and 3D Representations: A 3D Consistent Super-Resolution Framework](bridging_diffusion_models_and_3d_representations_a_3d_consistent_super-resolutio.md)

<!-- RELATED:END -->
