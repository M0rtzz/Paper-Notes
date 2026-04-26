---
title: >-
  [论文解读] RelationField: Relate Anything in Radiance Fields
description: >-
  [CVPR 2025][3D视觉][神经辐射场] RelationField 首次将物体间关系建模引入神经辐射场，通过从多模态大语言模型（如 GPT-4o）蒸馏关系知识到 NeRF 中的隐式关系特征头，实现了开放词汇的3D场景关系查询与场景图生成，在 3DSSG 基准上显著超越现有方法。
tags:
  - CVPR 2025
  - 3D视觉
  - 神经辐射场
  - 关系理解
  - 3D场景图
  - 开放词汇
  - 知识蒸馏
---

# RelationField: Relate Anything in Radiance Fields

**会议**: CVPR 2025  
**arXiv**: [2412.13652](https://arxiv.org/abs/2412.13652)  
**代码**: https://relationfield.github.io/  
**领域**: 3D视觉  
**关键词**: 神经辐射场, 关系理解, 3D场景图, 开放词汇, 知识蒸馏

## 一句话总结

RelationField 首次将物体间关系建模引入神经辐射场，通过从多模态大语言模型（如 GPT-4o）蒸馏关系知识到 NeRF 中的隐式关系特征头，实现了开放词汇的3D场景关系查询与场景图生成，在 3DSSG 基准上显著超越现有方法。

## 研究背景与动机

**领域现状**：近年来，神经辐射场（NeRF）已经从单纯的新视角合成扩展到了语义场景理解，如 LERF、OpenNeRF 等工作通过从 CLIP/DINO/SAM 等视觉-语言基础模型蒸馏特征到 3D 中，实现了开放词汇的物体分割和检测。与此同时，3D 场景图作为一种紧凑的场景表示，能够同时捕获场景中的物体和物体间关系。

**现有痛点**：现有将语义特征蒸馏到辐射场的方法（如 LERF、LangSplat）主要关注以物体为中心的语义特征，只能做物体级别的分割和检测，无法理解物体间的复杂关系（如"灯开关控制了灯"、"枕头躺在沙发上"）。而现有的 3D 场景图方法（如 Open3DSG）依赖显式的 3D 表示（点云或网格）和深度传感器数据，应用受限。

**核心矛盾**：辐射场只为每个 3D 点预测颜色和密度等属性，天然不支持两个物体之间的关系建模——因为关系需要同时考虑两个空间位置。而直接用 2D 多模态 LLM 逐帧推理关系，会受遮挡和视角变化影响，缺乏 3D 一致性。

**本文目标**：在神经辐射场中建模开放词汇的物体间关系，使其能够回答诸如"什么东西站在架子上"、"哪个物体和另一个物体相似"等关系查询，并能直接从辐射场中提取 3D 场景图。

**切入角度**：作者观察到 CLIP 虽然擅长物体级语义但关系理解能力有限，而多模态 LLM（如 GPT-4o）具有强大的关系推理能力但只能在 2D 层面工作。如果能将 LLM 的关系知识蒸馏到 3D 表示中，就能兼得两者优势。

**核心 idea**：通过引入额外的"查询位置" $\mathbf{z}$ 扩展 NeRF 的输入，使辐射场能够预测任意两个位置之间的关系特征，再通过 Set-of-Mark 提示从 GPT-4o 提取关系标注来监督训练。

## 方法详解

### 整体框架

RelationField 基于 Nerfacto 模型构建，输入是一组带位姿的 RGB 图像，输出是一个支持物体和关系查询的丰富 3D 表示。整体 pipeline 包含三个阶段：（1）从训练视图中通过 SoM 提示 GPT-4o 提取 2D 物体关系标注；（2）将关系知识蒸馏到辐射场的隐式关系特征头中；（3）通过指定查询位置和文本关系描述的方式交互式查询关系、或自动构建 3D 场景图。

模型在标准 NeRF 基础上增加了三个额外的输出头：768 维 CLIP/OpenSeg 嵌入的开放词汇语义头、256 维的实例分组头、以及 512 维 jina-embeddings-v3 嵌入空间的关系特征头。

### 关键设计

1. **隐式关系特征预测头（Relationship Field）**:

    - 功能：在辐射场中为任意两个 3D 位置之间的关系编码特征表示
    - 核心思路：标准 NeRF 的输入是 $(\mathbf{x}, \mathbf{d})$（位置和方向），RelationField 额外引入一个查询位置 $\mathbf{z} \in \mathbb{R}^3$，使得模型函数变为 $g_\theta(\mathbf{x}, \mathbf{d}, \mathbf{z}) \mapsto (\mathbf{c}, \sigma, \mathbf{o}, \mathbf{r})$，其中 $\mathbf{r}$ 是关系特征。实现上，将射线采样点和查询位置拼接后送入一个 MLP 头，输出 512 维的关系特征向量。这个特征处于语言嵌入空间中，可以通过余弦相似度与任意文本关系描述进行匹配。
    - 设计动机：关系本质上是两个实体之间的属性，需要两个空间位置作为输入才能定义。通过将"查询位置"作为额外输入，巧妙地将成对关系编码为辐射场的条件生成问题，保持了辐射场的连续体积表示优势。

2. **基于 SoM 的关系知识提取**:

    - 功能：从多模态 LLM 中提取像素对齐的稠密关系特征监督信号
    - 核心思路：首先用 SAM 对每张训练图像提取物体掩码，然后在图像上叠加 Set-of-Mark（SoM）标注（字母/数字标记），将标注后的图像输入 GPT-4o，提示其识别临近物体对之间的关系（例如 "A stands on B"）。LLM 输出的文本描述 $t_{ij}$ 使用 jina-embeddings-v3 编码为高维特征 $\phi_{t_{ij}}$，再通过 SAM 掩码投影到图像平面，得到像素级的关系特征标注。
    - 设计动机：CLIP 等模型擅长物体理解但关系理解弱，而 GPT-4o 具有强大的空间推理和关系推理能力。SoM 提示技术能显著提升 LLM 的视觉定位能力，使其能够精确地关联标记物体并描述关系。

3. **成对像素采样与渲染训练**:

    - 功能：高效训练关系特征场
    - 核心思路：训练时使用"成对像素采样器"从训练视图中均匀随机采样射线和查询位置的配对。利用辐射场的密度预测估计查询位置沿射线的深度，将射线和查询样本拼接送入关系 MLP 头。渲染得到的关系特征与 ground-truth 关系特征之间最大化余弦相似度，即最小化损失 $\mathcal{L} = 1 - \frac{\mathbf{r}}{||\mathbf{r}||_2} \cdot \frac{\hat{\mathbf{r}}}{||\hat{\mathbf{r}}||_2}$。
    - 设计动机：成对采样策略既保证了训练效率，又能充分覆盖场景中不同物体对之间的关系。使用辐射场的渲染权重来聚合 3D 关系特征到 2D，确保了多视角一致性。

### 损失函数 / 训练策略

总损失由三部分组成：（1）标准 NeRF 的颜色重建损失；（2）CLIP 语义特征的余弦相似度损失，用于物体级语义学习；（3）关系特征的余弦相似度损失。实例分组头通过对比学习方式训练，使同一实例的射线在嵌入空间中聚集。查询时采用 pairwise softmax 与规范短语（"and"、"next to"、"none"）比较，给出关系置信度。

## 实验关键数据

### 主实验

在 3DSSG 数据集（RIO10 子集）上的 3D 场景图预测：

| 方法 | Object R@5 | Object R@10 | Predicate R@3 | Predicate R@5 | Relation R@50 | Relation R@100 |
|------|-----------|-------------|--------------|--------------|--------------|----------------|
| GPT-4 (2D+depth) | 0.34 | 0.42 | 0.55 | 0.58 | 0.52 | 0.54 |
| Open3DSG | 0.56 | 0.61 | 0.58 | 0.65 | 0.55 | 0.56 |
| ConceptGraphs | 0.37 | 0.46 | 0.74 | 0.79 | 0.69 | 0.71 |
| **RelationField** | **0.69** | **0.80** | **0.76** | **0.82** | **0.73** | **0.74** |

### 消融实验

关系引导的 3D 实例分割（ScanNet++）：

| 方法 | IoU | Accuracy |
|------|-----|----------|
| LERF | 0.25 | 0.50 |
| OpenNeRF | 0.45 | 0.83 |
| LangSplat | 0.49 | 0.87 |
| **RelationField** | **0.53** | **0.96** |

3D 一致性消融：将 GPT-4 直接在 2D 上推理关系 vs. RelationField 的 3D 蒸馏，后者在 Object R@5 上从 0.34 提升到 0.69，Predicate R@3 从 0.55 提升到 0.76。使用 Llama 3.2 替代 GPT-4o 作为关系提取器，关系预测 recall 仅有轻微下降，证明方法对 LLM 后端不敏感。

### 关键发现

- 3D 蒸馏显著优于 2D 逐帧推理，因为 2D 方法受遮挡和视角影响严重（部分可见的物体会被 GPT-4 遗漏），而 3D 表示通过多视角聚合信息实现了一致性理解
- RelationField 在关系引导分割任务中准确率达到 96%，远超仅用 CLIP 特征的方法（50-87%），说明 bag-of-words 式的 CLIP 嵌入无法区分通过关系区分的重复物体
- 更换开源 Llama 3.2 替代闭源 GPT-4o 后性能损失很小，说明方法的通用性好

## 亮点与洞察

- **将关系建模为辐射场的条件查询**：通过引入额外的查询位置 $\mathbf{z}$ 扩展 NeRF 输入，将物体对间关系优雅地编码为体积表示的一部分，这个思路可以推广到任何需要建模成对/多元关系的隐式表示中
- **SoM + LLM 的关系知识提取 pipeline**：将 LLM 的文本推理能力转化为像素级的稠密特征监督，是一种通用的从大模型中提取结构化知识的范式
- **首次从辐射场生成 3D 场景图**：证明了仅用 RGB 图像（不需要深度传感器或显式 3D 表示）就能构建高质量的开放词汇 3D 场景图

## 局限与展望

- 关系知识完全依赖 LLM 的提示输出质量，如果 LLM 对特定关系理解有误或提示设计不当，错误会传播到 3D 关系场中
- 需要已知的相机内参和高质量多视角采集，这在很多实际应用场景中不易满足
- 训练成本较高：需要对每张训练图像都调用 GPT-4o 提取关系，API 成本和时间成本可观
- 关系场的质量上限受底层辐射场重建质量制约，重建较差的场景关系理解也会受影响
- 未来可探索将关系场扩展到 3D Gaussian Splatting 以获得实时性能

## 相关工作与启发

- **vs Open3DSG**: Open3DSG 使用 CLIP+InstructBLIP 蒸馏到 3D 图神经网络，但依赖预先给定的类别无关实例分割和显式 3D 网格；RelationField 不需要显式 3D 表示，直接从 RGB 图像训练
- **vs ConceptGraphs**: ConceptGraphs 也使用 GPT-4 但结合 SLAM 管线，先重建再用 GPT-4 标注场景级标题；RelationField 将关系知识直接蒸馏到连续体积表示中，实现了端到端的关系学习
- **vs LERF/LangSplat**: 这些方法专注于物体级 CLIP 特征蒸馏，无法处理复杂关系查询；RelationField 通过额外的关系头扩展了特征场的能力

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次在辐射场中建模开放词汇关系，查询位置扩展输入的设计简洁优雅
- 实验充分度: ⭐⭐⭐⭐ 有场景图预测和关系引导分割两大定量任务，消融覆盖 3D 一致性和 LLM 选择，但新提出的关系引导分割 benchmark 较小
- 写作质量: ⭐⭐⭐⭐ 思路清晰，图示直观，从动机到方法到实验逻辑连贯
- 价值: ⭐⭐⭐⭐ 为 3D 场景理解开辟了关系建模的新方向，对机器人交互和 AR/VR 有潜在应用价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2025\] Exploiting Deblurring Networks for Radiance Fields](exploiting_deblurring_networks_for_radiance_fields.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)
- [\[NeurIPS 2025\] HyRF: Hybrid Radiance Fields for Memory-efficient and High-quality Novel View Synthesis](../../NeurIPS2025/3d_vision/hyrf_hybrid_radiance_fields_for_memory-efficient_and_high-quality_novel_view_syn.md)
- [\[CVPR 2025\] MVPaint: Synchronized Multi-View Diffusion for Painting Anything 3D](mvpaint_synchronized_multi-view_diffusion_for_painting_anything_3d.md)
- [\[CVPR 2025\] Material Anything: Generating Materials for Any 3D Object via Diffusion](material_anything_generating_materials_for_any_3d_object_via_diffusion.md)

<!-- RELATED:END -->
