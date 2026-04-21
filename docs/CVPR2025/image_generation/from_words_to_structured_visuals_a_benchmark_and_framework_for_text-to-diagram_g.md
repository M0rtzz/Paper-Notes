---
title: >-
  [论文解读] From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing
description: >-
  [CVPR 2025][图像生成][文本到图表生成] 本文定义了文本到图表生成任务，构建了 DiagramGenBenchmark（涵盖 8 类图表），并提出多智能体框架 DiagramAgent（Plan + Code + Check + Diagram-to-Code），在图表生成、编码和编辑任务上显著超越现有文本到图像/代码方法。
tags:
  - CVPR 2025
  - 图像生成
  - 文本到图表生成
  - 结构化可视化
  - 多智能体框架
  - 代码生成
  - 基准测试
---

# From Words to Structured Visuals: A Benchmark and Framework for Text-to-Diagram Generation and Editing

**会议**: CVPR 2025  
**arXiv**: [2411.11916](https://arxiv.org/abs/2411.11916)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 文本到图表生成, 结构化可视化, 多智能体框架, 代码生成, 基准测试

## 一句话总结
本文定义了文本到图表生成任务，构建了 DiagramGenBenchmark（涵盖 8 类图表），并提出多智能体框架 DiagramAgent（Plan + Code + Check + Diagram-to-Code），在图表生成、编码和编辑任务上显著超越现有文本到图像/代码方法。

## 研究背景与动机

**领域现状**：将文本描述转换为结构化图表（如流程图、架构图、思维导图）在教育、科研和工业中有广泛需求。现有方法主要分为两条路线：文本到图像生成（GAN/扩散模型）和文本到代码生成（LLM 生成可执行代码）。

**现有痛点**：文本到图像方法（如 Imagen、DALL-E）虽然能生成逼真图像，但缺乏图表所需的逻辑组织和层次结构，生成的输出结构不严谨。文本到代码方法虽然能生成基础图表（柱状图、折线图），但对复杂图表（层次关系、颜色编码、分层结构）的表达能力不足，且生成结果难以后续修改。

**核心矛盾**：结构化图表同时需要逻辑精确性和可编辑性，但现有方法只能满足其一——图像方法视觉好但逻辑乱，代码方法有结构但表达力差。此外，缺乏专门的评测基准。

**本文目标** (1) 缺少文本到图表生成的标准化基准；(2) 缺少能同时处理生成、编码和编辑的统一框架。

**切入角度**：将图表生成分解为多个子任务（理解、编码、验证、逆向解析），由不同的专业 Agent 协同完成，形成一个可闭环纠错的工作流。

**核心 idea**：用多智能体协作框架（Plan + Code + Check + Diagram-to-Code）将文本到图表任务拆解为可验证的子步骤，辅以专门构建的 8 类图表基准来评估。

## 方法详解

### 整体框架
DiagramAgent 由四个核心 Agent 组成，支持三种任务流程：图表生成（文本→代码→图表）、图表编码（图表→代码）、图表编辑（图表→代码→修改→新图表）。输入为用户的文本指令或图表图像，输出为可编译的图表代码和对应的可视化结果。

### 关键设计

1. **Plan Agent（任务规划与查询扩展）**:

    - 功能：接收用户指令，判断任务类型（生成/编码/编辑），对不完整指令进行查询扩展
    - 核心思路：使用 Qwen-72B 分析用户指令完整性。若指令缺少节点、标签等信息，通过 LLM 补全为完整查询 $x_{comp} = f_{expand}(x_{ins})$，然后将完整查询路由到对应的下游 Agent
    - 设计动机：用户的自然语言描述通常不够精确，直接生成容易丢失关键元素。查询扩展确保了生成代码包含所有必要的结构和样式信息

2. **Code Agent（代码生成与微调）**:

    - 功能：将处理后的指令转化为可执行的图表代码（LaTeX/DOT 语言）
    - 核心思路：基于 Qwen2.5-Coder-7B 微调，在 DiagramGenBenchmark 训练集上进行 4 轮训练（max length 8192）。优化目标是最小化生成代码与参考代码的差异 $\mathcal{L}_{code}(f_{code}(x), c_{ref})$。对于编辑任务，同时接收原始代码和修改指令
    - 设计动机：通用代码模型对图表领域的语法和布局规则理解不够，微调能显著提升生成准确率和结构一致性

3. **Check Agent（调试与验证闭环）**:

    - 功能：对生成的代码进行编译调试和语义完整性验证
    - 核心思路：分两阶段——先编译代码，发现语法错误则返回 Code Agent 修正；编译通过后，再用 GPT-4o 验证代码的语义完整性（是否包含所有必要元素）。形成 $f_{check} = f_{debug} + f_{verify}$ 的双重保障
    - 设计动机：LLM 生成的代码经常有语法错误或遗漏元素，人工编译+AI验证的组合能有效提高最终输出质量

### 损失函数 / 训练策略
Code Agent 和 Diagram-to-Code Agent 的训练均采用标准的代码生成损失，最小化生成代码与参考代码之间的编辑距离和语义差异。微调使用 8×80G A100 GPU，Code Agent 基于 7B 模型、Diagram-to-Code Agent 基于 Qwen2-VL-7B。

## 实验关键数据

### 主实验

| 模型 | Pass@1↑ | ROUGE-L↑ | CodeBLEU↑ | CLIP-FID↓ | PSNR↑ | MS-SSIM↑ |
|------|---------|----------|-----------|-----------|-------|----------|
| DiagramAgent (7B) | **58.15** | **51.97** | **86.83** | **11.16** | **6.38** | **24.78** |
| DeepSeek-Coder (33B) | 55.56 | 44.26 | 83.29 | 15.49 | 6.02 | 19.80 |
| GPT-4o | 49.81 | 44.59 | 82.83 | 13.26 | 5.56 | 18.21 |
| DeepSeek V2.5 | 54.44 | 43.00 | 82.83 | 13.32 | 5.56 | 16.98 |
| Code-Llama (34B) | 8.89 | 22.92 | 76.78 | 30.12 | 0.89 | 2.32 |

7B 的 DiagramAgent 在 Pass@1 上超过 33B 的 DeepSeek-Coder 约 3 个点，超过 GPT-4o 约 8 个点。

### 消融实验

| 配置 | Pass@1↑ | chrF↑ | LPIPS↓ | MS-SSIM↑ |
|------|---------|-------|--------|----------|
| Full model | 58.15 | 53.49 | 45.95 | 24.78 |
| w/o GPT-4o 验证 | 57.78 (-0.37) | 52.81 (-0.68) | 46.66 (+0.71) | 20.80 (-3.98) |
| w/o 编译调试 | 57.41 (-0.74) | 51.74 (-1.75) | 48.13 (+2.18) | 24.10 (-0.68) |
| w/o 两者 | 57.41 (-0.74) | 51.69 (-1.80) | 48.17 (+2.22) | 20.37 (-4.41) |

### 关键发现
- 编译调试（Compiler）对代码质量的影响比 GPT-4o 验证更大，去掉后 chrF 降 1.75
- GPT-4o 验证对图像保真度（MS-SSIM）影响最大，去掉后降 3.98，说明验证模块主要保障视觉完整性
- 两个组件互补：同时去掉导致 MS-SSIM 降幅最大（-4.41），说明调试保证语法正确、验证保证语义完整

## 亮点与洞察
- **多智能体任务分解**：将文本到图表拆成规划→编码→验证的闭环流水线，每个 Agent 专注一个子任务，这种模式可迁移到任何"生成+验证"的任务
- **DiagramGenBenchmark 的实用性**：覆盖 8 种图表类型（流程图、架构图、思维导图等），是该领域首个全面基准，填补了重要空白
- **小模型 beat 大模型**：7B 的微调模型超过了 GPT-4o 和 33B 模型，说明领域微调在结构化生成任务上的优势

## 局限与展望
- 图表类型仅限于可用代码（LaTeX/DOT）编译的类型，无法处理手绘风格或更自由的可视化
- 查询扩展依赖 72B 的大模型（Qwen-72B），推理成本较高，可考虑用更小模型或 prompt engineering 替代
- Check Agent 依赖 GPT-4o（闭源），部署受限且成本高，可探索开源替代方案
- 数据集规模（训练集 ~7K）相对有限，扩展到更多图表类型和更大规模数据可能进一步提升性能

## 相关工作与启发
- **vs 文本到图像方法（Imagen/DALL-E）**: 它们生成自然图像优秀但无法保证结构逻辑，DiagramAgent 通过代码中间表示解决了结构化和可编辑性问题
- **vs 文本到代码方法（Qwen2.5-Coder）**: 通用代码模型缺乏图表领域知识，DiagramAgent 通过领域微调+多agent验证显著提升了图表特有的准确性
- 多 Agent 协作模式（Plan→Execute→Verify）是一个通用设计范式，可迁移到其他需要"生成→验证→修正"的任务中

## 评分
- 新颖性: ⭐⭐⭐⭐ 文本到图表生成任务定义新颖，但方法上是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ 16个模型对比+消融+人工评估，覆盖面广
- 写作质量: ⭐⭐⭐⭐ 框架和数学定义清晰，但部分公式过于形式化
- 价值: ⭐⭐⭐⭐ 基准和框架都有实用价值，但图表生成场景仍属小众

<!-- RELATED:START -->

## 相关论文

- [AutoPresent: Designing Structured Visuals from Scratch](autopresent_designing_structured_visuals_from_scratch.md)
- [Towards Scalable Human-Aligned Benchmark for Text-Guided Image Editing](towards_scalable_human-aligned_benchmark_for_text-guided_image_editing.md)
- [Factuality Matters: When Image Generation and Editing Meet Structured Visuals](../../ICLR2026/image_generation/factuality_matters_when_image_generation_and_editing_meet_structured_visuals.md)
- [PQPP: A Joint Benchmark for Text-to-Image Prompt and Query Performance Prediction](pqpp_a_joint_benchmark_for_text-to-image_prompt_and_query_performance_prediction.md)
- [SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing](salad_skeleton-aware_latent_diffusion_for_text-driven_motion_generation_and_edit.md)

<!-- RELATED:END -->
