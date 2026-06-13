---
title: >-
  [论文解读] Olympus: A Universal Task Router for Computer Vision Tasks
description: >-
  [CVPR 2025][3D视觉][任务路由] Olympus 将多模态大模型（MLLM）作为统一的任务路由器，通过设计任务特定路由 token 和构建大规模指令数据集，将超过 20 种计算机视觉任务（涵盖图像/视频/3D）分派到专用模型，实现了 94.75% 的单任务路由准确率和 91.82% 的链式动作精度。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "任务路由"
  - "多模态大模型"
  - "链式动作"
  - "统一框架"
  - "视觉任务"
---

# Olympus: A Universal Task Router for Computer Vision Tasks

**会议**: CVPR 2025  
**arXiv**: [2412.09612](https://arxiv.org/abs/2412.09612)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 任务路由, 多模态大模型, 链式动作, 统一框架, 视觉任务

## 一句话总结

Olympus 将多模态大模型（MLLM）作为统一的任务路由器，通过设计任务特定路由 token 和构建大规模指令数据集，将超过 20 种计算机视觉任务（涵盖图像/视频/3D）分派到专用模型，实现了 94.75% 的单任务路由准确率和 91.82% 的链式动作精度。

## 研究背景与动机

**领域现状**：多模态大模型（如 LLaVA、GPT-4V）在视觉问答等理解任务上表现优异，而 Emu3、Omni-Gen 等统一模型试图在单一网络中同时完成理解和生成任务。

**现有痛点**：All-in-one 模型面临三大困境：(1) 不同任务目标之间存在冲突（如文本生成与图像生成），导致单个任务性能下降；(2) 输入输出格式多样化使得扩展到更多任务十分困难；(3) 训练这类综合模型需要巨大的计算资源（如 Omni-Gen 需要 104×A800 GPU 和五阶段训练）。

**核心矛盾**：在单一模型中同时优化数十种不同的视觉任务，任务间的优化冲突和架构限制使得性能和可扩展性难以两全。

**本文目标**：不追求 all-in-one，而是让 MLLM 扮演"调度员"角色——自身处理视觉语言理解任务，同时将生成和经典视觉任务分配给外部专家模型。

**切入角度**：受 HuggingGPT 启发，但不同于其纯 prompt engineering 方式，Olympus 通过训练 MLLM 学习任务路由能力，结合 GPT-4o 生成的大规模指令数据集，实现从用户指令到专家模型的精确映射。

**核心 idea**：设计任务特定的路由 token，让 MLLM 在生成响应时自动输出对应的路由 token 和精炼后的提示词，从而零冲突地调度 20+ 视觉任务。

## 方法详解

### 整体框架

Olympus 以一个可训练的 MLLM（基于 Mipha 架构，含 SigLIP 视觉编码器 + Phi-2 语言模型）为核心控制器。对于 VQA 等理解类任务，MLLM 直接用自身能力回答；对于图像/视频/3D 生成、图像编辑、深度估计等任务，MLLM 生成包含路由 token 和精炼提示词的响应，分派给对应的专家模型（如 Stable Diffusion、ControlNet、InstantMesh 等）执行。框架支持在单条指令中执行多达 5 个链式任务。

### 关键设计

1. **OlympusInstruct 指令数据集构建**:

    - 功能：为 20 种视觉任务提供高质量的用户指令-响应对训练数据
    - 核心思路：为每种任务设计专用的 GPT-4o 提示模板，引入多样化的前缀/短语和三级复杂度分层（短/中/长），让 GPT-4o 生成风格、语气和结构各异的指令。同时构建 64.8K 链式动作指令对，训练模型在单条指令中调度多个任务。总计收集 446.3K 训练样本和 49.6K 评估样本
    - 设计动机：用户指令在实际场景中高度多样，需要覆盖不同表达风格；链式动作数据使模型能处理如"先生成图片再编辑"的复合请求

2. **任务特定路由 Token（Task-Specific Routing Tokens）**:

    - 功能：为每个视觉任务定义唯一的标记对（如 `<image_gen>...</image_gen>`、`<depth_est>...</depth_est>`），MLLM 通过预测这些 token 来指定应调度的专家模型
    - 核心思路：给定用户指令如"请生成一张吉娃娃的图片"，模型输出 `<image_gen>a chihuahua dog...</image_gen>`，系统解析路由 token 后将内容发送给对应的图像生成模型。在链式场景中，模型可顺序输出多个路由 token，如 `<pose_to_image>...</pose_to_image><image_edit>...</image_edit>`，实现任务流水线
    - 设计动机：路由 token 提供了明确的任务边界和模型选择信号，避免了自然语言解析的模糊性，同时支持灵活的任务组合

3. **链式动作（Chain-of-Action）能力**:

    - 功能：在单条用户指令中分解并顺序执行多个视觉任务
    - 核心思路：MLLM 被训练理解复合指令的意图，将其拆解为有序的子任务序列，每个子任务对应一个路由 token。前一个任务的输出作为后续任务的输入，形成任务流水线。训练数据中包含最多 5 个任务的链式指令
    - 设计动机：真实用户需求常常是复合的（如"根据这个姿态生成城堡图片并添加绿树"），单任务路由无法满足这种需求

### 损失函数 / 训练策略

训练采用标准的自回归语言模型损失（交叉熵），模型需要在理解任务上给出正确的文本回答，在路由任务上生成正确的路由 token + 精炼提示词。采用 Mipha 架构（SigLIP-SO + Phi-2, 2.7B 参数），训练分为两阶段：先用 558K 图文对预训练视觉-语言对齐，再用 OlympusInstruct 进行指令微调。

## 实验关键数据

### 主实验 — 多模态理解基准

| 方法 | LM 参数 | VQAv2 | GQA | SQAI | MME-P | MMB | MM-Vet | POPE |
|------|---------|-------|-----|------|-------|-----|--------|------|
| LLaVA-1.5 | 7B | 78.5 | 62.0 | 66.8 | 1510.7 | 64.3 | 30.5 | 85.9 |
| Mipha-3B | 2.7B | 81.3 | 63.9 | 70.9 | 1488.9 | 69.7 | 35.2 | 86.7 |
| Olympus | 2.7B | 80.8 | 63.6 | 72.5 | 1501.2 | 69.2 | 34.8 | 87.0 |

### 路由准确率

| 任务类型 | 准确率 |
|----------|--------|
| 单任务路由（20 类平均） | 94.75% |
| 链式动作（2-5 任务） | 91.82% 精度 |
| 2 任务链 | 94.82% |
| 5 任务链 | 87.81% |

### 消融实验

| 配置 | 单任务准确率 | 链式精度 |
|------|-------------|---------|
| 无路由 token（纯文本描述） | 71.32% | 62.15% |
| 路由 token + 无链式数据 | 94.75% | 78.43% |
| 完整 Olympus | 94.75% | 91.82% |

### 关键发现

- Olympus 在多模态理解基准上与同参数量的 Mipha-3B 基线性能相当（部分指标甚至略优），说明引入路由能力**不会损害原始理解能力**
- 路由准确率随链长增加而下降（2 任务 94.82% → 5 任务 87.81%），但即使 5 任务链仍保持接近 88% 的精度
- 任务特定路由 token 相比纯文本描述路由，准确率提升超过 23 个百分点

## 亮点与洞察

- **模块化 vs. All-in-one 的权衡**：Olympus 证明了在当前阶段，让 MLLM 做"调度器"比做"全能选手"更实际——既保留了理解能力，又通过外部专家模型获得了广泛的任务覆盖
- **路由 token 设计简洁有效**：相比 HuggingGPT 的纯提示工程方案，训练路由 token 提供了更强的任务识别鲁棒性
- **链式动作能力**是亮点，使得用户可以用自然语言描述复杂的多步骤工作流

## 局限与展望

- 路由质量依赖于 OlympusInstruct 的覆盖范围，当用户指令超出训练分布时可能失效
- 当前框架中专家模型的输出无法被 MLLM 感知和校验，存在错误传播风险
- 链式动作目前仅支持顺序执行，不支持条件分支或并行任务
- 扩展新任务需要重新构建指令数据并重新训练路由 token

## 相关工作与启发

- **vs HuggingGPT**：HuggingGPT 依赖 prompt engineering 调用 ChatGPT 作为任务分配器，不进行训练；Olympus 通过端到端训练 MLLM 实现更精准的路由
- **vs Emu3/Omni-Gen**：这些 all-in-one 模型试图在单一网络中完成一切，但计算成本极高且任务冲突明显；Olympus 采用分治策略更具可扩展性
- **vs Visual ChatGPT**：Visual ChatGPT 将视觉查询转为 Python 代码调用工具，灵活但不可控；Olympus 的路由 token 提供了更结构化的调度方式

## 评分

- 新颖性: ⭐⭐⭐ 思路直观，核心贡献更多在工程和数据层面，技术方法本身（路由 token + 指令微调）比较标准
- 实验充分度: ⭐⭐⭐⭐ 20 种任务的路由评估全面，链式动作实验有说服力，但缺少端到端任务质量的定量比较
- 写作质量: ⭐⭐⭐⭐ 架构图清晰，任务覆盖展示充分
- 价值: ⭐⭐⭐⭐ 提供了一种实用的统一视觉任务框架思路，OlympusInstruct 数据集有潜在复用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Scaling Properties of Diffusion Models for Perceptual Tasks](scaling_properties_of_diffusion_models_for_perceptual_tasks.md)
- [\[ICCV 2025\] A Simple yet Mighty Hartley Diffusion Versatilist for Generalizable Dense Vision Tasks](../../ICCV2025/3d_vision/a_simple_yet_mighty_hartley_diffusion_versatilist_for_genera.md)
- [\[CVPR 2025\] ASHiTA: Automatic Scene-grounded Hierarchical Task Analysis](ashita_automatic_scene-grounded_hierarchical_task_analysis.md)
- [\[CVPR 2026\] GeoCodeBench: Benchmarking PhD-Level Coding in 3D Geometric Computer Vision](../../CVPR2026/3d_vision/benchmarking_phd-level_coding_in_3d_geometric_computer_vision.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)

</div>

<!-- RELATED:END -->
