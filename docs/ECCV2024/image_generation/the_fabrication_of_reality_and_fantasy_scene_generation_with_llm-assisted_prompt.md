---
title: >-
  [论文解读] The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation
description: >-
  [ECCV 2024][图像生成][文本到图像生成] 提出 Realistic-Fantasy Benchmark (RFBench) 评估扩散模型在创意/知识密集型 prompt 上的表现，并设计 training-free 的 RFNet 框架，通过 LLM 辅助 prompt 解读和语义对齐评估模块来增强扩散模型对抽象和想象性概念的生成能力。
tags:
  - ECCV 2024
  - 图像生成
  - 文本到图像生成
  - 大语言模型
  - benchmark
  - 扩散模型
  - 创意场景生成
---

# The Fabrication of Reality and Fantasy: Scene Generation with LLM-Assisted Prompt Interpretation

**会议**: ECCV 2024  
**arXiv**: [2407.12579](https://arxiv.org/abs/2407.12579)  
**代码**: [有](https://leo81005.github.io/Reality-and-Fantasy/)  
**领域**: 图像生成  
**关键词**: 文本到图像生成, 大语言模型, benchmark, 扩散模型, 创意场景生成

## 一句话总结

提出 Realistic-Fantasy Benchmark (RFBench) 评估扩散模型在创意/知识密集型 prompt 上的表现，并设计 training-free 的 RFNet 框架，通过 LLM 辅助 prompt 解读和语义对齐评估模块来增强扩散模型对抽象和想象性概念的生成能力。

## 研究背景与动机

现有文本到图像模型在处理需要创造力或专业知识的复杂 prompt 时表现不佳，原因在于训练数据中缺少违反常规现实的场景（如"老鼠猎杀狮子"）。传统解决方案（数据收集+重训练/微调）代价高昂且可能导致灾难性遗忘。

本文认为核心问题是：**扩散模型如何更好地捕捉想象性和抽象概念？** 同时指出缺少针对此类任务的评估基准。

关键观察：
- 训练数据偏差导致模型无法理解角色冲突（猫被老鼠追）、科学推理（国际空间站水滴）等场景
- 现有 benchmark 不覆盖创意和幻想维度的评估
- LLM 拥有逻辑推理和知识推测能力，可弥补训练数据偏差

## 方法详解

### 整体框架

RFNet（Realistic-Fantasy Network）是一个两阶段的 training-free 方法：

- **阶段一（LLM 驱动细节生成）**：用 LLM 解析 prompt，生成布局（bounding boxes）、详细描述、背景场景和 negative prompts
- **阶段二（综合图像合成）**：通过两步生成过程——先独立生成前景对象，再无缝融合到背景中

### 关键设计

#### 1. RFBench 基准构建

包含 229 个组合文本 prompt，分为两大类共九个子类：

**Realistic & Analytical（现实与分析）**：
- 科学和经验推理（如"国际空间站上的一滴水"）
- 文化和时间意识（如"10月31日挨家挨户敲门的化装儿童"）
- 事实或字面描述（如"在沙滩上放了50年的坦克"）
- 概念和隐喻思维（如"一个人像狮子一样勇敢"）

**Creativity & Imagination（创造力与想象力）**：
- 常见物体在异常情境中（如"橡皮鸭在熔岩田上航行"）
- 想象性场景（如"章鱼和海马下象棋"）
- 反事实场景（如"鱼在云中游泳"）
- 角色反转或冲突（如"猫被老鼠追"）
- 拟人化场景（如"雪人在暴风雪中堆朋友"）

收集流程采用混合方法：交替使用 ChatGPT 和 Bard 的 in-context learning + 预定义规则，确保多样性。

#### 2. LLM 驱动细节生成（LLM-Driven Detail Synthesis）

给定 prompt 后，通过指定任务要求和 in-context learning 引导 LLM 生成增强响应，包括：
- 主要对象的 bounding box 布局
- 每个对象的详细描述
- 背景场景描述
- Negative prompts

核心目标是通过 LLM 的逻辑推理能力弥补扩散模型训练数据中的偏差。

#### 3. 语义对齐评估（Semantic Alignment Assessment, SAA）

解决 LLM 为不同对象生成的描述可能冲突的问题。例如对"狮子"的描述可能是"毫不知情地沉睡"或"惊恐逃亡"，两者虽各自合理但语义冲突。

SAA 模块通过计算不同对象描述向量之间的余弦相似度，选择最兼容的描述组合，确保文本精确性和一致性，为后续扩散模型提供清晰指令。

#### 4. 综合图像合成（Comprehensive Image Synthesis）

**步骤一——深度对象生成（In-Depth Object Generation）**：
- 为每个前景对象独立生成，输入格式为 `[background prompt] with [target object], [descriptions]`
- 使用 cross-attention 约束函数将对象限制在 bounding box 内：

$$\mathcal{L}_{obj}(\mathbf{A}, i, v) = [1 - \text{Topk}_u(\mathbf{A}_{uv} \cdot \mathbf{m}_i)] + [\text{Topk}_u(\mathbf{A}_{uv} \cdot (1 - \mathbf{m}_i))]$$

- 在去噪前每步用梯度更新隐变量：$z'_t \leftarrow z_t - \alpha \cdot \nabla_{z_t} \sum_{v \in V} \mathcal{L}$
- 提取 cross-attention map 并转换为 saliency mask 用于下一步

**步骤二——无缝背景融合（Seamless Background Integration）**：
- 用步骤一的 masked latent 替换当前 latent 对应区域
- 引入两个约束函数：
    - **引导约束（Guidance Constraint）**：减小当前 cross-attention 与步骤一生成对象 attention 的差异，保留精细细节
    - **抑制约束（Suppression Constraint）**：最小化 bounding box 外的 cross-attention，减少多对象干扰

$$\mathcal{L}_{bg} = \beta \cdot \underbrace{\sum_u |(\mathbf{A}'_{uv} - \mathbf{A}^{(i)}_{uv}) \cdot \mathbf{m}_i|}_{\text{guidance}} + \gamma \cdot \underbrace{\text{Topk}_u(\mathbf{A}'_{uv} \cdot (1 - \mathbf{m}_i))}_{\text{suppression}}$$

### 损失函数 / 训练策略

RFNet 是 training-free 方法，不需要训练。核心利用：
- 预训练 Stable Diffusion（v1.4 / v2.1）作为基础生成模型
- 预训练 LLM（ChatGPT/GPT-4）进行 prompt 解析
- 去噪步数 50 步，guidance scale 7.5，分辨率 512×512
- 超参数 α 控制梯度更新幅度，β/γ 控制引导/抑制约束强度
- 替换操作限制在前 rT 个时间步内

## 实验关键数据

### 主实验

**RFBench 上 GPT4-CLIP 和 GPT4Score 对比：**

| 方法 | GPT4-CLIP R&A | GPT4-CLIP C&I | GPT4-CLIP Avg | GPT4Score R&A | GPT4Score C&I | GPT4Score Avg |
|------|-------------|-------------|-------------|--------------|--------------|--------------|
| Stable Diffusion | 0.573 | 0.552 | 0.561 | 0.667 | 0.440 | 0.541 |
| MultiDiffusion | 0.510 | 0.510 | 0.510 | 0.517 | 0.493 | 0.504 |
| Attend and Excite | 0.523 | 0.560 | 0.546 | 0.633 | 0.520 | 0.570 |
| LMD | 0.457 | 0.536 | 0.501 | 0.550 | 0.600 | 0.578 |
| BoxDiff | 0.532 | 0.553 | 0.543 | 0.583 | 0.520 | 0.548 |
| SDXL | 0.536 | 0.619 | 0.582 | 0.567 | 0.587 | 0.578 |
| **RFNet (Ours)** | **0.587** | **0.623** | **0.607** | **0.833** | **0.627** | **0.719** |

RFNet 在 GPT4Score 上比 Stable Diffusion 提升 33%，在 Creativity & Imagination 任务上提升 43%，在 Realistic & Analytical 的 GPT4Score 上比 MultiDiffusion 提升 61%。

**DrawBench 子集对比 Imagen：**

| Prompt | Imagen | RFNet |
|--------|--------|-------|
| A shark in the desert | 0.194 | **0.713** |
| An elephant under the sea | 0.300 | **0.900** |
| A panda making latte art | 0.050 | **0.250** |
| A pizza cooking an oven | 0.700 | **0.831** |
| Rainbow coloured penguin | 0.394 | **0.519** |

在大部分需要创意想象的 prompt 上显著优于 Imagen。

### 消融实验

**各组件对 RFBench GPT4Score 的影响：**

| SAA | Guidance | Suppression | GPT4Score |
|-----|----------|-------------|-----------|
| ✗ | ✗ | ✗ | 0.295 |
| ✓ | ✗ | ✗ | 0.407 |
| ✗ | ✓ | ✗ | 0.554 |
| ✗ | ✓ | ✓ | 0.572 |
| **✓** | **✓** | **✓** | **0.719** |

- SAA 模块贡献最大，从 0.572 提升至 0.719（+25.7%）
- 引导约束和抑制约束互补，共同将 baseline 从 0.295 提升至 0.572
- 完整模型比 baseline 提升 143.7%

### 关键发现

- SAA 至关重要：没有语义对齐评估，LLM 生成的冲突描述会导致图像质量显著下降
- 高相似度描述产生高质量图像，低相似度描述导致视觉不一致
- 120 人用户研究中，RFNet 在图像质量和文本 prompt 忠实度两个维度上均获最高评分
- 传统 CLIPScore 在评估创意场景时存在局限性，GPT4-CLIP 和 GPT4Score 更适用

## 亮点与洞察

1. **首个 Realistic-Fantasy 基准**：系统化定义了 9 个子类覆盖现实推理和创意想象，填补了评估空白
2. **Training-free 架构**：兼容独立训练的 LLM 和扩散模型，无需参数调整，部署灵活
3. **双约束机制区别于传统 layout loss**：引导约束保真度 + 抑制约束减干扰，两者功能互补
4. **LLM 弥补数据偏差**：利用 LLM 的知识和推理能力弥补扩散模型训练数据中对非常规场景的缺失

## 局限与展望

- 依赖 LLM 推理质量，LLM 理解错误会传播到生成结果
- SAA 基于文本余弦相似度选择描述，可能不适用于高度抽象的概念
- 两步生成流程增加推理时间
- 基准规模有限（229 个 prompt），可扩展更多子类和更大规模
- 可开发更多评估指标以补充 GPT4-CLIP 和 GPT4Score

## 相关工作与启发

- **LMD (LLM-grounded Diffusion)**：用 LLM 生成前景对象布局再引导扩散模型，RFNet 在此基础上增加了 SAA 和双约束机制
- **Attend and Excite**：通过注意力机制增强语义理解，但缺乏处理复杂创意 prompt 的能力
- **RPG**：以闭环方式集成 LLM，通过 Chain-of-Thought 改善生成质量
- **SDXL**：高分辨率合成能力强但有时无法捕捉 prompt 中的创意意图

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次系统性定义现实-幻想生成任务并构建基准
- **有效性**: ⭐⭐⭐⭐ — 在自动和人工评估中均显著优于 SOTA 方法
- **工程价值**: ⭐⭐⭐ — Training-free 但流程较复杂，依赖多个预训练模型
- **推荐度**: ⭐⭐⭐⭐ — 基准数据集和评估方法对社区有重要参考价值

<!-- RELATED:START -->

## 相关论文

- [EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion](echoscene_indoor_scene_generation_via_information_echo_over_scene_graph_diffusio.md)
- [HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](hybridbooth_hybrid_prompt_inversion_for_efficient_subje.md)
- [Soft Prompt Generation for Domain Generalization](soft_prompt_generation_for_domain_generalization.md)
- [Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [Source Prompt Disentangled Inversion for Boosting Image Editability with Diffusion Models](source_prompt_disentangled_inversion_for_boosting_image_editability_with_diffusi.md)

<!-- RELATED:END -->
