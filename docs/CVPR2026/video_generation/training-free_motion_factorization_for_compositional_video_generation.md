---
title: >-
  [论文解读] Training-free Motion Factorization for Compositional Video Generation
description: >-
  [CVPR 2026][视频生成][组合式视频生成] 提出一个运动分解框架，将场景中多实例的运动分解为静止、刚体运动和非刚体运动三类，通过结构化运动图推理（SMR）解决 prompt 的语义歧义，通过解耦运动引导（DMG）在扩散过程中针对性地调控三类运动的生成…
tags:
  - "CVPR 2026"
  - "视频生成"
  - "组合式视频生成"
  - "运动分解"
  - "结构化推理"
  - "解耦引导"
  - "免训练"
---

# Training-free Motion Factorization for Compositional Video Generation

**会议**: CVPR 2026  
**arXiv**: [2603.09104](https://arxiv.org/abs/2603.09104)  
**代码**: 待发布  
**领域**: 扩散模型 / 视频生成 / 运动控制  
**关键词**: 组合式视频生成, 运动分解, 结构化推理, 解耦引导, 免训练

## 一句话总结

提出一个运动分解框架，将场景中多实例的运动分解为静止、刚体运动和非刚体运动三类，通过结构化运动图推理（SMR）解决 prompt 的语义歧义，通过解耦运动引导（DMG）在扩散过程中针对性地调控三类运动的生成，无需额外训练即可在 VideoCrafter-v2.0 和 CogVideoX-2B 上显著提升运动多样性和保真度。

## 研究背景与动机

1. **领域现状**：组合式视频生成（CVG）旨在从复杂 prompt 生成多实例、多运动的场景。现有方法（LVD、VideoDirectorGPT 等）通常用 LLM 生成 bounding box 序列来引导实例运动。
2. **现有痛点**：(1) 运动语义歧义——从文本直接生成 box 序列会导致断裂的运动路径和异常尺寸变化；(2) 运动引导粗糙——统一的扩散引导无法区分不同运动类别，导致运动趋同、不自然。
3. **核心矛盾**：现有方法对所有实例的运动一视同仁，缺乏对运动类别多样性的建模。静止物体、直线运动的车辆、跳舞的人需要完全不同的生成策略。
4. **本文目标** 如何在不训练的前提下，让视频生成模型为每个实例生成符合其运动类别的多样化运动？
5. **切入角度**：将运动分解为三个基本类别——静止、刚体运动、非刚体运动，分别设计针对性的推理和引导策略。
6. **核心 idea**：运动分解 + 先规划后生成——用结构化运动图推理出每个实例的运动表示，再用解耦引导分支针对性地合成三类运动。

## 方法详解

### 整体框架

框架遵循"先规划后生成"范式。(1) 规划阶段（SMR）：将用户 prompt 转换为运动图，对每个实例推理出逐帧 bounding box 序列作为运动表示。(2) 生成阶段（DMG）：根据运动类别，通过三个专用引导分支（外观一致性 / 几何不变性 / 空间变形）分别调控注意力图来合成运动。框架是模型无关的，适配 3D U-Net（VideoCrafter）和 DiT（CogVideoX）两种架构。

### 关键设计

1. **结构化运动推理（SMR）模块**:

    - 功能：将语义模糊的文本 prompt 转换为结构化运动表示，为每个实例生成运动类别标签和逐帧 bounding box 序列
    - 核心思路：首先构建运动图 $\mathcal{R} = (\mathcal{V}, \mathcal{E})$，每个实例为节点（标注运动属性和类别标签），有向边表示实例间空间关系和动态交互。然后基于运动类别推理 box 序列——静止实例 $\mathcal{B}_f(v_n) = \mathcal{B}_1(v_n)$ 保持不变；刚体运动实例根据估计速度 $\vec{u}$ 和加速度 $\vec{a}$ 更新 $\mathcal{B}_f = \mathcal{B}_{f-1} + \vec{u} + \frac{1}{2}\vec{a}$；非刚体运动用边界位移向量 $\Delta_f(v_n)$ 建模非对称变形。
    - 设计动机：直接从 prompt 生成 box 序列会因语义歧义产生错误运动。运动图作为中间结构化表示，使 LLM 可以分步推理——先理解实例关系和运动类别，再推导具体运动参数，显著减少歧义。

2. **参考条件引导（静止实例）**:

    - 功能：抑制静止区域的帧间伪变化，保持外观一致性
    - 核心思路：选择帧间特征差异最小的帧作为参考帧 $f^* = \arg\min_f \sum_{f'} D(\varphi(\mathbf{z}_f^t), \varphi(\mathbf{z}_{f'}^t))$。通过掩码 $\mathcal{G}_m$ 强制所有帧只与参考帧交互，实现像素级外观对齐。掩码定义为 $\mathcal{G}_m[x,y,f,f'](v_n) = \mathbb{1}(f'=f^* \& (x,y) \in \mathcal{B}(v_n))$。
    - 设计动机：视频扩散模型常在静态区域引入伪闪烁。锚定到稳定参考帧可在注意力层面消除不必要的跨帧变化。

3. **几何不变性引导（刚体运动实例）**:

    - 功能：在刚体运动过程中保持实例的几何形状不变
    - 核心思路：先用 k-means 聚类从 box 中分离前景，再通过像素投票聚合多帧粗掩码生成形状模板，将模板反投影到每帧得对齐掩码 $\mathcal{M}_f$。同时用位移惩罚因子 $\Gamma[f,f'] = \exp(-\alpha \cdot \|\mathbf{C}_f - \mathbf{C}_{f'}\|_2) + 1$ 调控帧间特征交互强度——距离近的帧交互更强。最终引导掩码 $\mathcal{G}_r = \mathcal{M} \cdot \mathcal{M}^\top \odot \Gamma$。
    - 设计动机：无几何约束时，视频模型在刚体运动中常产生形变。帧无关的形状模板确保几何一致性，位移惩罚实现平滑运动过渡。

4. **空间变形引导（非刚体运动实例）**:

    - 功能：建模非刚体运动中的复杂像素级变形
    - 核心思路：用最近邻搜索从扩散特征中提取感知变形场 $\mathcal{D}_{\text{perc}}$，用 box 角点位移的双线性插值得到 box 变形场 $\mathcal{D}_{\text{box}}$。通过变形惩罚因子 $\Lambda[i,j] = \exp(-\alpha \cdot (\mathcal{D}_{\text{perc}}[i,j] - \mathcal{D}_{\text{box}}[i,j])) + 1$ 最小化两者差异，使实际变形跟随预期。最终掩码 $\mathcal{G}_{\text{nr}} = (\mathcal{M} \cdot \mathcal{M}^\top) \odot \Lambda$。
    - 设计动机：非刚体运动中每个像素速度方向不同（如人体关节运动），需要像素级变形场而非全局平移来建模。

### 损失函数 / 训练策略

无需额外训练。对 3D U-Net 架构，通过梯度更新噪声嵌入 $\mathbf{z}^{t-1} \leftarrow \mathbf{z}^t - \nabla\mathcal{L}$，其中 $\mathcal{L} = 1 - \frac{\beta}{P}\sum(\mathbf{A} \odot (\mathcal{G}_m + \mathcal{G}_r + \mathcal{G}_{nr}))$。对 DiT 架构，直接修改注意力分数 $\mathbf{A} = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top (1 + \beta \odot (\mathcal{G}_m + \mathcal{G}_r + \mathcal{G}_{nr}))}{\sqrt{d}})$。VideoCrafter-v2.0 用 $\beta=10$, 引导步 1-25；CogVideoX-2B 用 $\beta=0.15$, 引导步 1-10。

## 实验关键数据

### 主实验

在自建 CVGBench-m（1665 样本自 MSR-VTT）和 CVGBench-p（994 样本自 Panda-70M）上评估：

| 模型设置 | Subject Consis. | Background Consis. | Temporal Flicker. | Motion Smooth. | Dynamic Degree |
|---------|----------------|-------------------|-------------------|---------------|----------------|
| VideoCrafter-v2.0 (基线) | 97.68% | 97.28% | 96.28% | 98.16% | 33.11% |
| + A&R | 97.48% | 97.05% | 96.43% | 98.27% | 38.40% |
| + **Ours** | **98.40%** | **98.11%** | **97.39%** | **98.63%** | **82.21%** |
| CogVideoX-2B (基线) | 91.33% | 92.78% | 95.01% | 96.88% | 87.80% |
| + R&P | 91.00% | 90.85% | 95.07% | 96.96% | 91.02% |
| + **Ours** | **98.27%** | **97.73%** | **98.25%** | **98.74%** | **96.00%** |

### 消融实验

引导分支消融（VideoCrafter-v2.0 基线）：

| RCG | GIG | SDG | Subject Consis. | Dynamic Degree | 说明 |
|-----|-----|-----|----------------|---------------|------|
| ✗ | ✗ | ✗ | 97.48% | 38.40% | 仅语义引导 |
| ✓ | ✗ | ✗ | 98.11% | 51.60% | 静止引导 |
| ✗ | ✓ | ✗ | 98.07% | 53.60% | 刚体引导 |
| ✗ | ✗ | ✓ | 97.71% | 74.85% | 非刚体引导 |
| ✓ | ✓ | ✓ | **98.40%** | **82.21%** | 完整模型 |

运动推理模块消融（CogVideoX-2B 基线）：

| 配置 | Subject Consis. | Dynamic Degree | 说明 |
|------|----------------|---------------|------|
| w/o SMR | 93.16% | 88.21% | 直接文本到运动 |
| w/ SMR (Ours) | **98.27%** | **96.00%** | 运动图推理 |

### 关键发现

- **Dynamic Degree 提升最为显著**：在 VideoCrafter-v2.0 上从 33.11% 提升到 82.21%（+49.1 pp），说明基线模型生成的运动过于保守，本框架有效激发了大幅运动。
- **非刚体引导对动态度贡献最大**：单独使用 SDG 即可将 Dynamic Degree 从 38.40% 提至 74.85%（+36.45 pp）。
- **SMR 模块是关键**：去掉 SMR 后 Subject Consistency 掉 5.11%、Dynamic Degree 掉 7.79%，证明结构化推理对消解语义歧义至关重要。
- **模型规模影响推理质量**：LLaMA-70B 比 8B 在 Dynamic Degree 上提升 6.87%（VideoCrafter）和 1.23%（CogVideoX）。
- **跨架构泛化**：框架在 3D U-Net 和 DiT 上均有效，验证了架构无关性。

## 亮点与洞察

- **运动三分类的简洁抽象**：将复杂运动分解为静止/刚体/非刚体三个基本类别，每类有明确的数学建模方式（常数/运动方程/位移场），简洁而有效。这种分类思路可迁移到运动估计、视频编辑等任务。
- **运动图作为中间表示**：将文本→运动的歧义问题转化为文本→结构化图→运动的两步推理，利用图结构编码实例间关系，是应对 LLM 运动推理不可靠的聪明策略。
- **注意力级调控的架构无关性**：通过直接操作注意力图/分数实现运动引导，无需修改模型权重或架构，天然适配不同骨干。

## 局限与展望

- 无法处理罕见语义概念（如 "Dendroid"），受限于基线模型的生成能力
- 对情感线索（如"悲伤"表情）生成效果差，因为视频模型倾向忽略形容词/副词
- 仅支持平面运动（bounding box），不处理深度方向的运动和 3D 旋转
- 未探索相机运动的建模，所有运动都在固定视角下进行
- 改进方向：引入参考图像提供罕见概念先验；建模相机位姿变化；扩展到 3D bounding box

## 相关工作与启发

- **vs VideoTetris/Vico**: 关注语义绑定和 token 重要性，但忽略运动类别多样性；本文互补地解决了运动问题
- **vs LVD/VideoDirectorGPT**: 用 LLM 生成 box 序列但统一引导，运动趋同；本文通过运动图和解耦引导显著提升多样性
- **vs FreeTraj/TrailBlazer**: 用稀疏运动场做引导，但不区分运动类别；本文的分类引导更精细
- **vs MotionPrompting**: 用鼠标拖拽提供运动信号，需要用户交互；本文完全自动化

## 评分

- 新颖性: ⭐⭐⭐⭐ 运动三分类和运动图推理的抽象有创意，但各组件（attention guidance/LLM reasoning）相对成熟
- 实验充分度: ⭐⭐⭐⭐ 自建 benchmark 覆盖多种语言模式，消融详尽，但缺少与 SOTA 商业模型的对比
- 写作质量: ⭐⭐⭐⭐ 框架清晰，公式完整，但符号定义较多
- 价值: ⭐⭐⭐⭐ 免训练+架构无关的特性使其实用性强，运动分解思路有普适性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowMotion: Training-Free Flow Guidance for Video Motion Transfer](flowmotion_training-free_flow_guidance_for_video_motion_transfer.md)
- [\[CVPR 2026\] SWIFT: Sliding Window Reconstruction for Few-Shot Training-Free Generated Video Attribution](swift_sliding_window_reconstruction_for_few-shot_training-free_generated_video_a.md)
- [\[CVPR 2026\] SwitchCraft: Training-Free Multi-Event Video Generation with Attention Controls](switchcraft_training-free_multi-event_video_generation_with_attention_controls.md)
- [\[AAAI 2026\] DreamRunner: Fine-Grained Compositional Story-to-Video Generation with Retrieval-Augmented Motion Adaptation](../../AAAI2026/video_generation/dreamrunner_fine-grained_compositional_story-to-video_genera.md)
- [\[CVPR 2026\] When to Lock Attention: Training-Free KV Control in Video Diffusion](when_to_lock_attention_training-free_kv_control_in_video_diffusion.md)

</div>

<!-- RELATED:END -->
