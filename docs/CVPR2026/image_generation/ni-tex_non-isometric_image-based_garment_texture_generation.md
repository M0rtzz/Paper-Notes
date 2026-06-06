---
title: >-
  [论文解读] NI-Tex: Non-isometric Image-based Garment Texture Generation
description: >-
  [CVPR 2026][图像生成][服装纹理生成] 提出NI-Tex框架，通过构建3D Garment Videos数据集、基于图像编辑的跨拓扑增强以及不确定性引导的迭代烘焙算法，首次以前馈架构实现了非等距条件下从单图到3D服装PBR纹理的高质量生成。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "服装纹理生成"
  - "PBR材质"
  - "非等距变形"
  - "不确定性引导烘焙"
  - "跨拓扑增强"
---

# NI-Tex: Non-isometric Image-based Garment Texture Generation

**会议**: CVPR 2026  
**arXiv**: [2511.18765](https://arxiv.org/abs/2511.18765)  
**代码**: [有](https://github.com/SII-Hui/NI-Tex)  
**领域**: 3D视觉  
**关键词**: 服装纹理生成, PBR材质, 非等距变形, 不确定性引导烘焙, 跨拓扑增强  

## 一句话总结

提出NI-Tex框架，通过构建3D Garment Videos数据集、基于图像编辑的跨拓扑增强以及不确定性引导的迭代烘焙算法，首次以前馈架构实现了非等距条件下从单图到3D服装PBR纹理的高质量生成。

## 研究背景与动机

现有工业级3D服装网格已覆盖大多数真实世界的服装几何形状，但纹理多样性仍然有限。为获取更逼真的纹理，生成式方法常从大量真实图像中提取PBR（基于物理的渲染）纹理并投射回服装网格。然而，现有的图像条件纹理生成方法面临两个核心限制：

**拓扑一致性要求**：大多数方法要求输入图像与目标3D网格之间具有严格的拓扑一致性，例如Hunyuan3D和Meshy在图像-网格拓扑不匹配时生成质量严重下降

**网格变形依赖**：部分方法（如Pix2Surf、Cloth2Tex）依赖精确的网格变形来匹配图像姿态，但变形过程引入累积误差，灵活性受限

实际应用中，用户提供的图像与目标网格之间经常存在显著的拓扑差异（如从裙子图像生成长裤网格纹理）和几何差异（不同姿态、不同体型），这使得现有方法在非等距场景下难以胜任。本文的核心切入点是：将非等距问题转化为数据增强问题，利用图像编辑模型制造跨拓扑训练对，再用物理仿真数据覆盖跨姿态场景。

## 方法详解

### 整体框架

NI-Tex采用三阶段设计：

- **数据空间**：从BEDLAM构建3D Garment Videos提供跨姿态训练对；用Nano Banana图像编辑制造跨拓扑训练对
- **生成网络**：双分支前馈架构——引导分支提取参考图像特征，生成分支根据法线/位置图输出多视角PBR纹理
- **迭代烘焙**：训练UQ（不确定性量化）模型，迭代选择新视角、加权融合为mesh纹理

输入为一张RGB图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和一个目标服装网格，输出为UV空间下的albedo ($C=3$)、roughness ($C=1$) 和metallic ($C=1$) 纹理贴图。

### 关键设计

#### 1. **3D Garment Videos跨姿态增强**

核心思路：利用BEDLAM物理仿真数据构建跨姿态训练对，消除姿态差异对纹理生成的影响。

- 从BEDLAM中提取每个运动序列的逐帧服装几何，形成序列 $V = \{M_1, M_2, \ldots, M_n\}$，所有帧共享同一albedo纹理贴图
- 为每帧补充PBR材质属性：$\text{roughness} \sim \mathcal{U}(0,1)$，$\text{metallic} = 0$
- 训练时随机采样两帧：一帧作为**条件帧**（选取一个光照视角作为输入图像prompt，使用10个视角的法线和位置图作为几何约束），另一帧作为**监督帧**（使用10个视角的PBR纹理属性作为监督信号）
- 这种跨帧监督使数据集从数十万帧组合式扩展到数百亿训练样本

设计动机：同一序列的不同帧保持纹理一致但姿态不同，天然构成了跨姿态的训练对，无需额外的配对标注。同时随机应用不同类型光源（点光源、面光源、环境光）增强光照多样性。

#### 2. **基于Nano Banana的跨拓扑增强**

核心思路：利用图像编辑模型修改条件图像的服装拓扑，同时保留原始纹理信息，构建跨拓扑训练对。

- 从3D Garment Videos中随机采样渲染视图，使用Nano Banana编辑服装拓扑（如将长裤编辑为短裤、裙子→长裤等）
- 使用光照渲染图（而非albedo图）作为编辑输入，减小推理时的域差距
- 编辑后的图像替换原始条件图像，而监督仍来自原始监督帧，本质上是从Nano Banana蒸馏纹理身份一致性

为避免错误蒸馏，遵循三条语义完整性约束：

- **类别一致性**：编辑全身服装时，上下衣纹理不能漂移或交换
- **内外层一致性**：分层穿搭中，外衣纹理必须与内衣保持区分
- **允许辅助人体部分**：允许偶尔生成额外人体区域，促进模型聚焦于服装材质本身

总计生成约50K编辑图像用于跨拓扑训练。

#### 3. **双分支生成网络与可切换多通道U-Net**

核心思路：采用前馈双分支架构，并引入可切换机制处理MR通道不一致的问题。

**双分支架构**：引导分支从输入图像提取层次化特征，生成分支接收多视角法线和位置图并生成纹理。通过**多通道对齐注意力（MCAA）**连接两个分支：

$$\text{Attn}_{albedo} = \text{Softmax}\left(\frac{Q_{albedo} K_{ref}^T}{\sqrt{d}}\right) \cdot V_{ref}$$

将albedo注意力注入MR潜在表示以实现空间和几何对齐：

$$z_{MR}^{new} = z_{MR} + \text{Attn}_{albedo}$$

**可切换机制**：由于Nano Banana生成的图像无法保持一致的MR属性，设计可切换U-Net——处理编辑图像时关闭MR通道使用单通道注意力，正常帧则使用多通道对齐注意力。这避免了不一致MR监督对训练的污染。

#### 4. **不确定性引导的迭代烘焙**

核心思路：训练UQ模型预测纹理贴图的逐像素不确定性，通过迭代选择高不确定性视角并重新生成来修复烘焙伪影。

**UQ模型训练**：采用ResNet-50架构。通过误差模拟流程收集训练数据——对GT网格渲染10个视角，用Nano Banana编辑随机视角，然后用纹理生成模型重建，优化latent code使前/后视角纹理与GT匹配：

$$\min_{\boldsymbol{z}} \| \Gamma^{\text{front}}(\boldsymbol{z}) - T_{\text{gt}}^{\text{front}} \|^2 + \| \Gamma^{\text{back}}(\boldsymbol{z}) - T_{\text{gt}}^{\text{back}} \|^2$$

收集所有视角的预测-GT对，用SSIM计算逐像素不确定性标签，监督损失为：

$$\sum_{p_i} \| \text{UQ}(p_i) - y^{\text{SSIM, GT}}(p_i) \|_2^2$$

**视角选择**：从候选视角中选取平均不确定性最高的视角，针对该视角重新推理。迭代直至达到最大视角数 $N_{view}$ 或新视角不确定性低于阈值 $\epsilon$。

**多视角重加权融合**：最终纹理通过不确定性加权与视角权重联合计算：

$$t_i^{\star} = \frac{\sum_j (1 - \text{UQ}(p_{ij})) c_j p_{ij}}{\sum_j (1 - \text{UQ}(p_{ij})) c_j + \epsilon_1}$$

其中 $c_j$ 为视角权重，前后视角设为1，其余按距离衰减至0.5、0.25、0.125、0.1。

### 损失函数/训练策略

多通道优化阶段（albedo + MR联合监督）：

$$\mathcal{L}_1 = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1), t} \left[ \| \epsilon - \epsilon_t^{MR} \|_2^2 + \| \epsilon - \epsilon_t^{Albedo} \|_2^2 \right]$$

单通道优化阶段（仅albedo监督，用于Nano Banana编辑样本）：

$$\mathcal{L}_2 = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1), t} \left[ \alpha \cdot \| \epsilon - \epsilon_t^{Albedo} \|_2^2 \right]$$

两种损失交替优化，平衡因子 $\alpha = 2$ 用于平滑训练损失曲线。**MR校正（MR Rectification）**：由于跨帧MR值不一致，从条件帧MR图中采样代表性前景像素，替换监督帧MR图中所有前景像素值，实现一致的跨帧MR监督。

训练基于Stable Diffusion 2.1，8×H200 GPU训练约10天，batch size=2，分辨率512×512。数据规模：100K Objaverse + 90K TexVerse（通用3D数据）+ 150K BEDLAM（服装仿真）+ 50K编辑图像（跨拓扑）。为防MR过拟合于均匀值，额外引入Objaverse/TexVerse数据做交叉混合训练。

## 实验关键数据

### 主实验

| 方法 | KID ↓ | FID ↓ |
|------|-------|-------|
| Paint3D | 0.0695 | 293.45 |
| Hyper3D OmniCraft | 0.0471 | 285.45 |
| Hunyuan3D | 0.0528 | 272.34 |
| Meshy 6 Preview | 0.0383 | 246.39 |
| **NI-Tex (Ours)** | **0.0364** | **0.0364** |

实验设置：10个工业/生成网格 × 10个图像prompt × 多视角渲染 × 42随机种子。NI-Tex在KID和FID上均为最优，KID比次优Meshy低5.0%，FID降低3.6%。

### 烘焙策略对比

| 烘焙策略 | 网格覆盖 | 伪影处理 | PSNR |
|----------|----------|----------|------|
| 6正交视图 | 大量自遮挡缺失 | 无 | 基线 |
| Coverage-based视角选择 | 改善但仍有小区域缺失 | 无 | 中等 |
| **UQ迭代烘焙 (Ours)** | **完全覆盖** | **主动修复模糊/空洞** | **最高** |

### 关键发现

1. **跨拓扑鲁棒性**：NI-Tex在图像-网格拓扑差异显著时（如裙子→长裤）仍能生成高质量纹理，而Hunyuan3D和Meshy出现严重的纹理扭曲甚至生成失败
2. **野外图像适应性**：在DeepFashion2真实图像（经SAM2掩码）上，NI-Tex能有效捕获正确的纹理信息，包括logo和精细图案
3. **跨姿态一致性**：在4D-Dress数据集上验证，同一人不同姿态下纹理生成一致性良好
4. **UQ烘焙优于覆盖率烘焙**：不确定性引导的视角选择能捕获传统覆盖率方法遗漏的中间烘焙伪影（模糊、接缝、空洞等），在最差视角上PSNR显著更高
5. **工业+生成mesh通用**：在Hunyuan3D生成的含更多褶皱的mesh上也能稳定工作，保留logo、花纹等细节

## 亮点与洞察

- **图像编辑工具作为数据增强引擎**：将非等距问题转化为图像编辑问题，用Nano Banana从已有3D资产低成本制造跨拓扑训练对，这一策略可迁移到任何需要几何多样性的3D生成任务
- **组合式数据扩展**：3D Garment Videos通过帧对组合将数据量从数十万级扩展到数百亿级，是一种极高效的数据增强范式
- **可切换架构设计务实**：针对编辑图像MR不一致的问题，设计可切换U-Net而非强行统一监督，体现工程上的务实思路
- **不确定性闭环**：UQ模型不仅用于评估质量，还直接驱动视角选择和融合权重，形成完整的质量检测-修复闭环
- **从蒸馏视角理解跨拓扑增强**：本质上是将Nano Banana的纹理身份一致性能力蒸馏到纹理生成模型中

## 局限性

- 对**复杂刚性变形**的泛化能力有限，因为缺乏一般物体的物理仿真数据，目前主要适用于服装类柔性物体
- 依赖Nano Banana等外部图像编辑模型的质量，编辑失败会引入训练噪声
- 训练成本较高（8×H200 GPU训练约10天），推理需迭代烘焙多轮
- 定量评估主要依赖KID/FID，缺乏针对纹理一致性和PBR材质准确性的专用评估指标
- MR Rectification假设每件服装MR属性全局均匀，对复杂多材质服装可能不适用

## 评分

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次以前馈架构解决非等距纹理生成，图像编辑驱动的跨拓扑增强思路新颖；但骨干网络借用Hunyuan3D，UQ部分借鉴AVS |
| 实验 | ⭐⭐⭐⭐ | 对比多个商业模型（Hyper3D、Meshy、Hunyuan3D），覆盖工业/生成mesh两类场景，烘焙策略有定量消融；但定量指标仅KID/FID |
| 写作 | ⭐⭐⭐⭐ | 框架图清晰完善，问题定义明确，跨拓扑/跨姿态的区分和处理逻辑连贯；附录补充详实 |
| 价值 | ⭐⭐⭐⭐⭐ | 直接面向工业级3D服装设计需求，生成PBR材质可用于实际渲染管线，代码将开源，实用价值高 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GarmentAligner: Text-to-Garment Generation via Retrieval-augmented Multi-level Corrections](../../ECCV2024/image_generation/garmentaligner_text-to-garment_generation_via_retrieval-augmented_multi-level_co.md)
- [\[CVPR 2026\] Agentic Retoucher for Text-To-Image Generation](agentic_retoucher_for_texttoimage_generation.md)
- [\[NeurIPS 2025\] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling](../../NeurIPS2025/image_generation/track_inpaint_resplat_subject-driven_3d_and_4d_generation_with_progressive_textu.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Image Generation as a Visual Planner for Robotic Manipulation](image_generation_as_a_visual_planner_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
