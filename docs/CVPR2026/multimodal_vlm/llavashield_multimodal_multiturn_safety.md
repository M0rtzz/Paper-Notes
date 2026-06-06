---
title: >-
  [论文解读] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models
description: >-
  [CVPR 2026][多模态VLM][多模态多轮对话安全] 针对VLM多模态多轮对话中的恶意意图隐蔽性、上下文风险累积和跨模态联合风险三大挑战，构建4,484个标注对话的MMDS数据集和基于MCTS的MMRT红队框架，提出LLaVAShield审计模型…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "多模态多轮对话安全"
  - "内容审核"
  - "红队测试"
  - "MCTS"
  - "风险分类"
---

# LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2509.25896](https://arxiv.org/abs/2509.25896)  
**代码**: [项目主页](https://leost123456.github.io/LLaVAShield/)  
**领域**: 多模态VLM / AI安全  
**关键词**: 多模态多轮对话安全, 内容审核, 红队测试, MCTS, 风险分类

## 一句话总结
针对VLM多模态多轮对话中的恶意意图隐蔽性、上下文风险累积和跨模态联合风险三大挑战，构建4,484个标注对话的MMDS数据集和基于MCTS的MMRT红队框架，提出LLaVAShield审计模型，在用户/助手两侧分别达到F1 95.71%/92.24%，大幅超越GPT-5-mini等基线。

## 研究背景与动机

**领域现状**：VLM正大规模部署在智能助手、教育等交互场景中，安全问题日益突出。已有内容审核工具（如BingoGuard、WildGuard、LlamaGuard）取得了初步进展，但主要面向单轮或单模态设置。

**现有痛点**：多模态多轮对话具有三个独特的风险特征，使得现有审核方法失效。(1) **恶意意图隐蔽性**——攻击者以无害开场逐步升级，在多模态中将目标拆散为分散的文本和视觉线索，跨轮次关联后大幅放大危害；(2) **上下文风险累积**——攻击者将最终目标分解到多个轮次中，利用模型的"局部顺从"逐步拓宽攻击面，风险随对话推进而叠加；(3) **跨模态联合风险**——即使正常的图文配对也可能触发不安全生成，跨模态关联可被利用来诱导有害输出。

**核心矛盾**：现有审核方法要么只看单轮、要么只处理单模态，无法捕捉多轮上下文累积的风险和图文联合攻击。同时，多模态多轮对话安全数据集严重缺乏，限制了该方向的研究。

**本文目标**：需要(1)覆盖多维度风险的标注数据集；(2)能自动高效生成对抗对话的红队框架；(3)能理解完整上下文和跨模态信号的安全审计模型。

**切入角度**：从数据→攻击→防御三位一体出发，构建MMDS数据集、MMRT红队框架和LLaVAShield审计模型。

**核心 idea**：通过MCTS高效探索攻击路径生成安全数据集，训练一个能同时审计用户输入和助手响应的多模态多轮对话安全模型。

## 方法详解

### 整体框架
系统分为三个阶段：(1) MMDS数据集构建——通过恶意意图生成、图像挖掘、MMRT红队攻击和多层标注，构建包含4,484个对话的安全数据集；(2) MMRT红队框架——基于MCTS的自动化攻击路径探索，生成跨轮次跨模态的不安全对话；(3) LLaVAShield模型——在MMDS上微调VLM，输入指令+安全策略+对话历史，输出双侧安全评分、违反维度和证据链。

### 关键设计
1. **MMRT（基于MCTS的多模态多轮红队框架）**:

    - 功能：自动化生成不安全的多模态多轮对话，为MMDS提供高质量攻击样本
    - 核心思路：将红队攻击建模为攻击者$\mathcal{A}$、目标VLM $\mathcal{T}$、评估器$\mathcal{E}$的迭代交互。在每轮$t$，攻击者根据恶意意图$g$、对话上下文$c_{t-1}$和策略集$\Sigma$生成攻击计划$(q_t, \mathcal{I}_t)$，目标模型回复$r_t = \mathcal{T}(q_t, \mathcal{I}_t, c_{t-1})$，评估器打分$s_t \in \{1,...,5\}$。关键是使用MCTS（选择-扩展-模拟-反向传播）突破线性攻击链的搜索空间限制，用PUCT公式选择节点，通过$k$步rollout估计下游风险，奖励$z = (s_{t+k}-1)/4$
    - 设计动机：线性的$\mathcal{A} \to \mathcal{T} \to \mathcal{E}$循环搜索空间有限，MCTS可以高效探索分支攻击路径，找到更隐蔽有效的攻击序列。攻击策略包括渐进引导、目的反转、查询分解和角色扮演

2. **MMDS数据集构建与标注流程**:

    - 功能：构建首个面向多模态多轮对话安全的标注数据集，包含4,484个对话（2,756原始+1,728增强），覆盖8大维度60子维度的风险分类
    - 核心思路：数据源有两类——MMRT生成的756个不安全对话和从MMDU-45k采样的2,000个安全对话。每个对话对用户和助手双侧独立标注安全评级、违反的策略维度。再通过四种数据增强策略提升泛化性：(a)随机移除未违反的策略维度；(b)用GPT-5-mini将不安全回复改写为合规文本，减少误报；(c)单侧遮蔽训练；(d)调整策略配置防止过度审核
    - 设计动机：仅靠标签监督信息有限，因此引入"角色解耦双通道理由机制"——为用户和助手独立生成解释性推理链，要求每条推理必须提供关键证据，确保分类的可溯源性和可验证性

3. **LLaVAShield审计模型**:

    - 功能：在指定策略维度下同时审计多模态多轮对话中用户输入和助手响应的安全性
    - 核心思路：将安全审计建模为统一的seq2seq任务。输入由三部分拼接：指令$\mathcal{G}$、策略集合$\mathcal{P}$和$T$轮对话历史$\mathcal{C} = \{(V_t^u, x_t^u, x_t^a)\}_{t=1}^T$。模型输出包含六个组件的结构化JSON：双侧安全评级$S_u, S_a$、违反策略$D_u, D_a$和证据推理$R_u, R_a$，全部在`<OUTPUT>...</OUTPUT>`标签内生成
    - 设计动机：统一格式使输出可机器解析，支持下游自动化处理。策略维度作为输入参数，使模型能灵活适配不同部署场景的安全规范

### 损失函数 / 训练策略
基于LLaVA-OV-7B初始化，在MMDS训练集上微调。优化目标为最大化条件对数似然$\max_\theta \sum \log p(\mathcal{Y} | \mathcal{G}, \mathcal{P}, \mathcal{C}; \theta)$。学习率$2 \times 10^{-5}$，cosine调度+0.03%总步数warmup，batch size=1 + 4步梯度累积，训练3个epoch。8×NVIDIA A6000 (48GB)，约3小时完成训练。

## 实验关键数据

### 主实验

| 模型 | 用户侧F1(%) | 助手侧F1(%) | 备注 |
|------|-----------|-----------|------|
| LLaVAShield-7B | **95.71** | **92.24** | 开源，7B参数 |
| GPT-5-mini | 75.46 | 77.93 | 闭源，最强基线 |
| Gemini-2.5-Pro | 64.00 | 65.62 | 闭源 |
| GPT-4o | 61.54 | 57.92 | 闭源 |
| InternVL3.5-38B | 29.15 | 36.71 | 开源 |
| Llama Guard-4-12B | 14.21 | 28.21 | 专用审核工具 |
| Qwen2.5-VL-7B | 1.17 | 1.54 | 同规模开源 |

LLaVAShield在用户侧精确率100%、召回率91.76%；相比GPT-5-mini分别提升+20.25和+14.31个F1点。

### 外部基准泛化

| 基准 | 指标 | LLaVAShield | GPT-5-mini | Llama Guard-4 |
|------|------|------------|------------|--------------|
| MM-SafetyBench | Avg Recall(%) | **97.62** | 48.44 | 44.49 |
| VLGuard-Test | F1(%) | **90.55** | 86.39 | 64.87 |

### 消融实验

| 配置 | 用户侧F1(%) | 助手侧F1(%) | 说明 |
|------|-----------|-----------|------|
| Vanilla (含理由) | 95.71 | 92.24 | 完整模型 |
| w/o rationale | 95.12 | 93.93 | 去除推理链 |
| 策略适配FPR | 0% / 0% | - | GPT-5-mini为30%/34% |

### 关键发现
- 所有开源VLM在多模态多轮场景下recall极低（接近0%），说明内置安全对齐在此场景下形同虚设
- 图像的加入使评估器平均风险分提升0.375（ASG），视觉线索将泛化指导转为高风险操作性内容
- 对话轮次增加使VLM更倾向产生有害内容，但6轮后效果趋于饱和
- 主流VLM的攻击成功率极高：Qwen2.5-VL-72B达100%，GPT-4o达98.21%，即使Claude-3.7-Sonnet也有73.77%

## 亮点与洞察
- **问题定义精准**：首次系统刻画多模态多轮对话的三大风险特征（隐蔽性、累积性、跨模态性），问题建模清晰
- **MCTS红队框架**：借鉴博弈搜索思想探索攻击路径，比线性尝试高效得多，可生成多样化攻击轨迹
- **全面的数据增强**：四种增强策略分别应对误报、单侧信息不全、策略过度审核等部署中的实际问题
- **7B模型碾压闭源大模型**：LLaVAShield-7B在专项任务上大幅超越GPT-5-mini和Gemini-2.5-Pro，说明专用微调在安全审计领域远优于通用大模型
- **策略适配能力突出**：FPR为0%，模型真正按照输入策略维度做判断，不会泛化到不相关的风险类别

## 局限与展望
- MMDS仅4,484个对话，且主要由两个目标VLM（GPT-4o和Qwen2.5-VL-72B）生成，多样性可能不足
- 风险分类体系由人工设计，难以覆盖新兴风险类型（如deepfake、AI生成的社会工程攻击）
- 安全审计模型引入额外推理延迟，尚未报告推理速度和部署开销
- MMRT的攻击效率取决于MCTS超参数和rollout步数，未做搜索效率分析
- 训练仅用LLaVA-OV-7B，未验证在更强backbone上的效果

## 相关工作与启发
- **vs Llama Guard系列**: Llama Guard-4-12B在MMDS上F1仅14.21%/28.21%，说明单轮审核工具无法处理多轮上下文
- **vs ShieldVLM**: 同期工作聚焦多模态隐式毒性，但仍限于单轮设置，缺乏多轮推理能力
- **vs Red Queen**: 多轮红队攻击工作，但仅限纯文本LLM，本文拓展到多模态并使用MCTS提升搜索效率
- **vs IDEATOR**: 本文的攻击策略组合（渐进引导+目的反转+查询分解+角色扮演）受其启发，但扩展到图文跨模态攻击

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统解决多模态多轮对话安全，MCTS红队框架和角色解耦推理链设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 主实验+8维度细粒度分析+2个外部基准+策略适配+VLM脆弱性分析+组件贡献分析，非常全面
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、三大风险特征论述有说服力、实验设计严谨
- 价值: ⭐⭐⭐⭐⭐ 数据集+红队框架+审计模型三位一体，对VLM安全部署有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs](dictionary_aligned_concept_control_for_safeguarding_multimodal_llms.md)
- [\[ICCV 2025\] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models](../../ICCV2025/multimodal_vlm/multiverse_a_multi-turn_conversation_benchmark_for_evaluating_large_vision_and_l.md)
- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[ICML 2026\] MM-Snowball: Evaluating and Mitigating Hallucination Snowballing in Multimodal Multi-Turn Dialogue](../../ICML2026/multimodal_vlm/mm-snowball_evaluating_and_mitigating_hallucination_snowballing_in_multimodal_mu.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)

</div>

<!-- RELATED:END -->
