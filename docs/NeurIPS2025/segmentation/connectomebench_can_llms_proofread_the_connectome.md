---
title: >-
  [论文解读] ConnectomeBench: Can LLMs Proofread the Connectome?
description: >-
  [NeurIPS 2025][图像分割][connectomics proofreading] 提出 ConnectomeBench，首个评估多模态 LLM 在连接组校对（片段识别、分裂错误修正、合并错误检测）三项关键任务上能力的标准化基准；o4-mini 在分裂修正多选任务达 85%，但合并错误检测仍显著落后于人类专家。
tags:
  - NeurIPS 2025
  - 图像分割
  - connectomics proofreading
  - 多模态
  - 3D neuron mesh
  - benchmark
---

# ConnectomeBench: Can LLMs Proofread the Connectome?

**会议**: NeurIPS 2025  
**arXiv**: [2511.05542](https://arxiv.org/abs/2511.05542)  
**代码**: [项目主页](https://connectomebench.github.io/)  
**领域**: segmentation / 连接组计算神经科学  
**关键词**: connectomics proofreading, multimodal LLM, 3D neuron mesh, segmentation error detection, benchmark

## 一句话总结

提出 ConnectomeBench，首个评估多模态 LLM 在连接组校对（片段识别、分裂错误修正、合并错误检测）三项关键任务上能力的标准化基准；o4-mini 在分裂修正多选任务达 85%，但合并错误检测仍显著落后于人类专家。

## 研究背景与动机

**领域瓶颈**：连接组学通过高分辨率电子显微镜成像并自动分割来重建大脑神经连接，但分割算法不可避免地产生分裂（split）和合并（merge）两类错误，需要大量人工校对——果蝇全脑连接组消耗约 33 人年的校对工作量。

**扩展困境**：随着科学界推进更大规模的脑连接组重建（如小鼠视觉皮层 1mm³ 体积），人工校对成本将呈指数增长，亟需自动化方案。

**LLM 视觉推理进展**：多模态 LLM（如 o3 在 CharXiv 上接近人类水平）展现出越来越强的视觉推理能力，为科学任务自动化提供了新可能。

**现有方法局限**：已有工作包括基于启发式的图方法（NEURD）、基于 CNN 的引导校对和 RoboEM 等，但它们要么脆弱不泛化，要么需要针对特定任务训练——缺乏通用评估框架。

**缺少标准评测**：目前没有标准化基准来系统衡量 AI 系统在连接组校对任务上的表现，无法跨模型、跨时间进行一致比较。

**核心想法**：将 3D 神经元网格渲染为多视角 2D 图像（正交三视图），转化为 LLM 可处理的视觉问答任务，构建涵盖三种关键校对能力的标准化 benchmark。

## 方法详解

### 整体框架

ConnectomeBench 基于两个大规模开源连接组数据集构建：
- **MICrONS**：小鼠视觉皮层 1mm³ 体积，约 20 万已校对神经元
- **FlyWire**：果蝇全脑，约 14 万已校对神经元

利用 CAVEClient 访问分割编辑历史，获取校对前后的分割状态作为 ground truth。每个 3D mesh 从顶视、侧视、正视三个方向渲染为 1024×1024 图像，直接作为 LLM 的视觉输入。评估三项任务：片段类型识别、分裂错误修正、合并错误检测。

### 关键设计一：片段类型识别（Segment Identification）

- **功能**：将分割片段归为 5 类——单体神经元、多体合并神经元、无胞体突起、细胞核、非神经元细胞
- **核心思路**：渲染完整 3D mesh 的三个视角，提示 LLM 进行多选分类；探索 "Description"（提供类别形态描述）和 "Null"（无额外上下文）两种提示策略
- **设计动机**：分类是校对的基础步骤——识别多体合并本身就等同于发现合并错误；类别先验可辅助后续校对决策

### 关键设计二：分裂错误修正（Split Error Correction）

- **功能**：判断两个分离片段是否应被合并（二元分类），或在多个候选中选择正确合并对象（多选分类）
- **核心思路**：正例来自编辑历史中人类实际执行的合并操作；负例通过在界面点（interface point）附近 128nm 横向 / 120-880nm 纵向范围内采样邻近片段生成；使用 4096nm³ 边界框裁剪渲染区域
- **设计动机**：纵向采样范围 120-880nm 是为了处理缺失成像切片导致的间隙——这是真实数据中分裂错误的主要来源

### 关键设计三：合并错误检测（Merge Error Identification）

- **功能**：判断某个片段中是否存在合并错误（二元分类），或从两个候选中识别哪个有合并错误（多选分类）
- **核心思路**：正例来自编辑历史中人类标注的合并修正；裁剪边界框为 max(4096nm, 2×较小片段尺寸)，动态适应不同大小的错误区域；负例使用最终校对完成的 mesh
- **设计动机**：合并错误通常在突起的异常分支或不自然连接处可见，变尺寸裁剪确保足够的空间上下文

### 关键设计四：启发式推理增强（Heuristic-Guided Reasoning）

- **功能**：分析 o4-mini 的推理错误模式，提取 7 条启发式规则嵌入提示中
- **核心思路**：发现模型存在偏见，如"正确合并对应的片段应是小的延伸"、"大间隙意味着不应合并"等——但实际中分裂片段可能与原片段同样大，缺失数据可造成大间隙。将纠正这些偏见的启发式写入提示
- **设计动机**：利用 LLM 的自然语言推理能力理解其失败原因并有针对性地改进，避免训练/微调成本

### 评估策略

- 每个提示重复 5-10 次，取多数投票作为最终答案
- 每项任务随机抽取 100 个样本进行分析
- 专家基线由训练过的研究生/本科生标注约 50 个样本
- 额外训练 ResNet-50 分类器作为传统深度学习基线

## 实验关键数据

### 表1：片段识别 Balanced Accuracy

| 模型 | FlyWire | MICrONS |
|------|---------|---------|
| Claude 3.7+Desc | 0.459 | **0.822** |
| o4-mini+Desc | 0.511 | 0.728 |
| GPT-4.1+Desc | **0.529** | 0.655 |
| GPT-4o+Desc | 0.396 | 0.588 |
| InternVL-3 78B+Desc | 0.320 | 0.493 |
| InternVL-3 8B+Desc | 0.303 | 0.417 |
| NVLM+Desc | 0.234 | 0.258 |
| ResNet-50 | 0.552 | 0.587 |
| 随机基线 | 0.200 | 0.250 |

### 表2：分裂错误修正（二元 & 多选）最佳配置

| 模型+提示 | 二元 FlyWire | 二元 MICrONS | 多选 FlyWire | 多选 MICrONS |
|-----------|-------------|-------------|-------------|-------------|
| o4-mini+Heuristics | **0.754** | **0.786** | 0.788 | **0.850** |
| o4-mini+Desc | 0.631 | 0.679 | **0.828** | 0.790 |
| Claude 4+Heuristics | 0.551 | 0.587 | 0.677 | 0.770 |
| GPT-4o+Heuristics | 0.556 | 0.536 | 0.667 | 0.720 |
| ResNet-50 | 0.720 | 0.667 | 0.721 | 0.693 |
| 人类专家 | 0.840 | 0.902 | 0.900 | 0.920 |

### 表3：合并错误检测（二元 & 多选）

| 模型+提示 | 二元 FlyWire | 二元 MICrONS | 多选 FlyWire | 多选 MICrONS |
|-----------|-------------|-------------|-------------|-------------|
| o4-mini+Desc | **0.628** | **0.615** | 0.670 | **0.703** |
| o4-mini+Null | 0.553 | 0.591 | **0.740** | 0.689 |
| Claude 4+Desc | 0.487 | 0.480 | 0.560 | 0.530 |
| GPT-4o+Desc | 0.538 | 0.517 | 0.345 | 0.361 |
| ResNet-50 | 0.769 | 0.798 | 0.569 | 0.541 |
| 人类专家 | 0.740 | 0.800 | 0.840 | 0.796 |

### 关键发现

- **片段识别表现惊人**：所有专有模型远超随机基线（20-25%），Claude 3.7 在 MICrONS 上达到 82.2%，超过 ResNet-50（58.7%）
- **分裂修正中多选优于二元**：相对比较比绝对判断更容易，o4-mini 多选最高达 85.0%
- **启发式推理有效提升**：o4-mini 二元分裂修正从 67.9% 提升至 78.6%（+10.7pp），多选从 79.0% 到 85.0%（+6.0pp）
- **合并错误检测最难**：最佳模型 o4-mini（62.8%）仍显著落后于人类（74-80%）和 ResNet-50（76.9-79.8%）
- **开源模型差距明显**：InternVL-3 78B/8B、NVLM 在各任务上均明显弱于专有模型
- **Description 提示效果因模型而异**：对 Claude 3.7 片段识别几乎无帮助（其内部先验已足够），但在分裂修正多选中普遍有效

## 亮点与洞察

- **首个系统性连接组 AI 校对基准**：为 LLM 在神经科学关键任务上的能力评估提供标准化框架，可跨模型、跨时间追踪进展
- **3D→多视图 2D 的零样本方案**：无需任何训练即可让 LLM 处理 3D 数据，展示了多模态模型的泛化潜力
- **推理分析→启发式反馈闭环**：通过分析 LLM 推理中的系统性偏见，产出可直接嵌入提示的纠正规则——这种"分析失败→改进提示"的方法论具有广泛适用性
- **o4-mini 一致领先**：在三项任务中 o4-mini 几乎全面优于其他模型，暗示推理能力（而非纯视觉能力）是此类任务的关键

## 局限性

- **合并错误检测不足**：当前最佳 LLM 性能（~63%）远低于人类（~80%）和 ResNet-50（~80%），且甚至低于部分简单基线——这是走向自动校对的主要障碍
- **3D 信息损失**：三视图渲染不可避免地丢失 3D 拓扑信息，尤其对复杂分支结构的合并错误判断不利
- **评估规模有限**：每项任务仅分析 100 个样本，专家基线约 50 个——统计功效较低，置信区间较宽
- **未覆盖完整校对流程**：缺少突触识别、合并错误修正（仅检测不修正）等关键环节
- **计算成本高**：每个样本需多次 API 调用+多数投票，大规模部署面临成本约束
- **数据集偏差**：仅涵盖 MICrONS（小鼠）和 FlyWire（果蝇），可能不代表其他物种或成像条件

## 相关工作与启发

- **NEURD (Celii et al., 2025)**：将 3D 神经元 mesh 转为标注图表示，用启发式图规则校正合并错误——可解释但脆弱
- **RoboEM (Schmidt et al., 2024)**：CNN 将轴突追踪建模为飞行控制问题——巧妙但任务特定
- **RLCorrector (Nguyen et al., 2021)**：强化学习 agent 模拟人类校对工作流——预示了 AI agent 在此领域的可能性
- **启发**：未来可结合 LLM 的推理能力与专用 3D 模型（如点云 Transformer），在 agent 框架下构建端到端校对系统

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个评估 LLM 连接组校对能力的系统化 benchmark，3D→多视图渲染的零样本评估方案新颖
- **实验充分度**: ⭐⭐⭐⭐ 三任务×两数据集×八模型×多提示策略，有人类专家和 ResNet 基线，但样本量偏小
- **写作质量**: ⭐⭐⭐⭐ 领域背景清晰，任务动机充分，数据构建过程详尽
- **价值**: ⭐⭐⭐⭐ 为 LLM 在科学任务中的应用提供了重要的能力边界分析，启发式推理增强方法具有通用参考价值

<!-- RELATED:START -->

## 相关论文

- [\[ICCV 2025\] ReferEverything: Towards Segmenting Everything We Can Speak of in Videos](../../ICCV2025/segmentation/refereverything_towards_segmenting_everything_we_can_speak_of_in_videos.md)
- [\[ACL 2026\] TemporalVLM: Video LLMs for Temporal Reasoning in Long Videos](../../ACL2026/segmentation/temporalvlm_video_llms_for_temporal_reasoning_in_long_videos.md)
- [\[ICCV 2025\] Can Generative Geospatial Diffusion Models Excel as Discriminative Geospatial Foundation Models?](../../ICCV2025/segmentation/can_generative_geospatial_diffusion_models_excel_as_discriminative_geospatial_fo.md)
- [\[ICML 2025\] Using Multiple Input Modalities Can Improve Data-Efficiency and O.O.D. Generalization for ML with Satellite Imagery](../../ICML2025/segmentation/using_multiple_input_modalities_can_improve_data-efficiency_and_ood_generalizati.md)
- [\[NeurIPS 2025\] MultiHuman-Testbench: Benchmarking Image Generation for Multiple Humans](multihuman-testbench_benchmarking_image_generation_for_multiple_humans.md)

<!-- RELATED:END -->
