---
title: >-
  [论文解读] ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting
description: >-
  [CVPR 2025][图像生成][自动T2I] 提出 ChatGen，将文本到图像生成中的 prompt 编写、模型选择和参数配置三个繁琐步骤自动化，通过多阶段进化训练策略（ChatGen-Evo）让用户以自由聊天方式描述需求即可获得高质量生成图像。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "自动T2I"
  - "自由聊天"
  - "多阶段进化"
  - "模型选择"
  - "提示学习"
---

# ChatGen: Automatic Text-to-Image Generation From FreeStyle Chatting

**会议**: CVPR 2025  
**arXiv**: [2411.17176](https://arxiv.org/abs/2411.17176)  
**代码**: [https://chengyou-jia.github.io/ChatGen-Home](https://chengyou-jia.github.io/ChatGen-Home)  
**领域**: 扩散模型 / LLM Agent  
**关键词**: 自动T2I, 自由聊天, 多阶段进化, 模型选择, Prompt改写

## 一句话总结
提出 ChatGen，将文本到图像生成中的 prompt 编写、模型选择和参数配置三个繁琐步骤自动化，通过多阶段进化训练策略（ChatGen-Evo）让用户以自由聊天方式描述需求即可获得高质量生成图像。

## 研究背景与动机

**领域现状**：T2I 生成模型日趋丰富，Civitai 等平台上有数千个定制化模型。但用户要获得满意的图像，需要手动完成三个繁琐步骤：编写合适的 prompt（需了解模型特定语法）、选择合适的模型（需了解各模型特长）、配置参数（如采样步数、CFG scale）。

**现有痛点**：（1）BeautifulPrompt 只优化 prompt 不选模型；（2）DiffAgent 只做模型选择不改 prompt；（3）没有方法同时处理全部三个步骤。更关键的是，现有方法不支持自由式聊天输入（如"帮我画一个在咖啡馆的女孩"），要求用户提供格式化输入，远离真实场景。

**核心矛盾**：自动化 T2I 是一个多步推理任务——prompt 质量影响模型选择，模型选择决定参数配置。直接用 SFT 端到端训练 MLLM 会导致模型学习简单文本映射而非推理能力，泛化差。

**本文目标** 如何从用户的自由式聊天输入自动完成 prompt 改写、模型选择和参数配置，实现端到端的图像生成。

**切入角度**：受 OpenAI o1 的多步推理思路启发，将自动 T2I 分解为三个可学习的子技能，逐阶段训练 MLLM，每阶段提供针对性反馈。

**核心 idea**：用多阶段进化策略依次训练 MLLM 的 prompt 改写、模型选择（通过 ModelToken 词汇扩展）和参数配置能力，实现从自由聊天到图像的全自动化。

## 方法详解

### 整体框架
构建 ChatGenBench 基准（256K 训练 + 14K 测试，来自 6807 个 Civitai 定制模型），支持单文本/多模态/多轮历史三种输入格式。ChatGen-Evo 基于 MLLM 进行三阶段训练：Stage 1 用 SFT 学 prompt 改写，Stage 2 用 ModelToken 学模型选择，Stage 3 用上下文学习配置参数。推理时依次执行三步输出 prompt + 模型 + 参数。

### 关键设计

1. **ChatGenBench 基准构建**

    - 功能：提供首个支持全流程步骤级评估的自动 T2I 基准
    - 核心思路：从 Civitai 收集 44,881 条高质量人类示范（经下载量/点赞过滤），包含 prompt、模型、参数的完整链路。用 LLM 角色扮演（100+ 职业角色）将示范反向合成为自由式聊天输入。用多个 MLLM（GPT-4o、Claude 等）+ 温度 0.9 增加多样性。支持单文本、多模态（文本+图片）和多轮历史三种输入格式
    - 设计动机：此前无基准同时覆盖 prompt 质量、模型选择准确性和参数配置的评估，无法定位自动化瓶颈

2. **多阶段进化策略（Multi-Stage Evolution）**

    - 功能：将复杂的多步推理任务分解为三个可独立学习的子技能
    - 核心思路：**Stage 1（Prompt 改写）**：用 SFT 训练 MLLM 从自由聊天输入生成高质量 prompt，只优化 prompt 预测目标。**Stage 2（模型选择）**：为每个 T2I 模型创建可学习的 ModelToken 嵌入并扩展词汇表，训练时仅更新 ModelToken 嵌入（冻结其他参数），模型选择变为在扩展词汇表上的分类任务。**Stage 3（参数配置）**：基于前两步的输出，用上下文学习（ICL）方式让 MLLM 根据模型文档和示例配置参数，无需额外训练
    - 设计动机：消融实验表明直接端到端 SFT（ChatGen-Base 8B）在所有指标上都不如分阶段的 ChatGen-Evo 2B，证明了分解多步推理的必要性

3. **ModelToken 模型选择**

    - 功能：让 MLLM 高效选择数千个定制化 T2I 模型中的最佳模型
    - 核心思路：为每个 T2I 模型添加一个新的 token 嵌入到 MLLM 词汇表。训练时输入 prompt 和模型 token，仅更新 ModelToken 嵌入（不影响 MLLM 的其他能力）。模型选择本质上变成了"给定 prompt，预测最合适的 ModelToken"
    - 设计动机：直接用模型名字或描述进行选择，MLLM 无法区分语义相近的模型。ModelToken 为每个模型提供了可学习的独特表示

### 损失函数 / 训练策略
三阶段分别用：Stage 1 用标准 SFT 交叉熵损失（只监督 prompt）；Stage 2 用交叉熵损失（冻结 MLLM 只训 ModelToken 嵌入）；Stage 3 无训练（ICL）。ChatGen-Evo 2B 训练约 100 小时，推理 1.9s/样本。

## 实验关键数据

### 主实验

| 方法 | 参数 | Prompt BERTScore↑ | 模型选择Acc↑ | 参数Acc↑ | FID↓ | CLIP↑ | ImageReward↑ | 综合↑ |
|------|------|-------------------|------------|---------|------|-------|-------------|------|
| ChatGen-Base 8B | 8B | 0.208 | 0.264 | 0.509 | 20.8 | 70.7 | 4.0 | 60.7 |
| **ChatGen-Evo 2B** | **2B** | **0.247** | **0.328** | **0.537** | **19.1** | **72.9** | **8.9** | **65.9** |

### 消融实验

| 配置 | Prompt BERTScore↑ | Selection Acc↑ | 综合↑ |
|------|-------------------|---------------|------|
| ChatGen-Base (端到端SFT) | 0.208 | 0.264 | 60.7 |
| Stage 1 only | 0.247 | — | — |
| Stage 1+2 | 0.247 | 0.328 | — |
| **Stage 1+2+3 (Full)** | **0.247** | **0.328** | **65.9** |

### 关键发现
- 分阶段进化 2B 模型全面优于端到端 8B 模型，证明多步推理分解的有效性
- ModelToken 设计对模型选择准确率提升关键（0.264→0.328），且不影响 prompt 改写质量
- Few-shot 设置下性能下降明显（模型选择 Acc 从 0.328 降到 0.231），新/少见模型的泛化是开放问题
- 多阶段推理虽慢（1.9s vs 1.1s），但质量显著更高

## 亮点与洞察
- **问题定义有价值**：首次将 T2I 全流程自动化作为统一问题研究，ChatGenBench 提供了标准化评估框架
- **ModelToken 设计优雅**：将模型选择转化为词汇扩展 + 嵌入学习，比用文本描述选择更精确且不干扰其他能力
- **分阶段训练符合认知**：先学表达（prompt）、再学选择（模型）、最后配置（参数），每步用前一步结果作条件，类似人类工作流

## 局限与展望
- 模型选择在 few-shot 场景下泛化差，新模型上线后需要更新 ModelToken
- 依赖 Civitai 社区数据，可能存在分布偏差（特定风格/主题过度代表）
- 参数配置使用 ICL 无训练，准确率（0.537）有较大提升空间
- 评估中 prompt 质量用 BERTScore 衡量，可能不完全反映生成质量

## 相关工作与启发
- **vs BeautifulPrompt**: 只做 prompt 改写不选模型。ChatGen 覆盖全流程
- **vs DiffAgent**: 只做模型选择。ChatGen 同时改写 prompt + 选模型 + 配参数
- **vs DiffusionGPT**: 用 LLM 做模型路由，但只支持约 20 个模型且无步骤级评估。ChatGen 支持 6807 个模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题定义和 ModelToken 设计有创新，但各组件技术较标准
- 实验充分度: ⭐⭐⭐⭐⭐ 256K 数据+14K 测试+步骤级评估+多种基线+消融，非常扎实
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法描述清晰，基准设计细致
- 价值: ⭐⭐⭐⭐ 直接解决 T2I 使用痛点，对非专业用户有明确价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Minority-Focused Text-to-Image Generation via Prompt Optimization](minority-focused_text-to-image_generation_via_prompt_optimization.md)
- [\[CVPR 2025\] EasyCraft: A Robust and Efficient Framework for Automatic Avatar Crafting](easycraft_avatar_crafting.md)
- [\[CVPR 2025\] Towards Understanding and Quantifying Uncertainty for Text-to-Image Generation](towards_understanding_and_quantifying_uncertainty_for_text-to-image_generation.md)
- [\[CVPR 2025\] From Elements to Design: A Layered Approach for Automatic Graphic Design Composition](from_elements_to_design_a_layered_approach_for_automatic_graphic_design_composit.md)
- [\[CVPR 2025\] PreciseCam: Precise Camera Control for Text-to-Image Generation](precisecam_precise_camera_control_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
