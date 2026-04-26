---
title: >-
  [论文解读] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][图像生成][T2I后处理] Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。
tags:
  - CVPR 2026
  - 图像生成
  - T2I后处理
  - 感知-推理-行动循环
  - 扭曲检测
  - 局部修复
  - GenBlemish-27K
---

# Agentic Retoucher for Text-To-Image Generation

**会议**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**代码**: 无  
**领域**: 图像生成 / Agent / 图像质量评估  
**关键词**: T2I后处理, 感知-推理-行动循环, 扭曲检测, 局部修复, GenBlemish-27K

## 一句话总结

Agentic Retoucher 将 T2I 生成图像的局部缺陷修复重构为感知→推理→行动的多 agent 闭环决策流程，通过上下文感知的显著性检测、人类偏好对齐的诊断推理和自适应工具选择实现自主修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 修复结果被人类评为优于原图。

## 研究背景与动机

**领域现状**：T2I 扩散模型（SDXL、FLUX、Qwen-Image 等）已能生成高度逼真的图像，广泛应用于设计、影视和娱乐产业。然而即使是最先进的模型，仍然频繁产生局部小尺度扭曲——手指畸形、面部不对称、文字不可读、肢体错位、物体交互不合理等。这些缺陷通常出现在整体质量较高的图像内部，检测困难且修复代价高昂。

**现有痛点**：当前改善生成质量的路线主要有三条：prompt 增强、基于强化学习的优化、以及噪声空间对齐。它们能提升整体真实感，但缺乏显式的空间推理能力，无法解释或修复局部失败。后处理编辑方案（如 Imagic、Step1x-Edit）虽然支持局部修复，却依赖手工 mask 或启发式文本指令，无法自主识别需要修复的区域。

**核心矛盾**：VLM 看似可以作为自动化评判器，但实验表明即使是 GPT-5 也无法可靠定位 AI 生成图像中的扭曲——六指人像被判为正常，明显的面部畸变被忽略。根本原因在于 VLM 针对高层语义对齐而非像素级验证进行优化，其强大的知识先验会覆盖视觉证据，产生幻觉式判断。

**本文目标** 如何让 T2I 系统具备自主感知、诊断和修复局部生成缺陷的能力，同时避免 VLM 在细粒度空间定位上的不可靠性。

**切入角度**：将后处理修复建模为人类修图师的感知-推理-行动决策循环，而非一次性前馈编辑。三个专门化 agent 各司其职，形成可迭代收敛的闭环系统。

**核心 idea**：用层级化的多 agent 决策框架将 T2I 后处理重构为感知扭曲→推理诊断→精准修复的自校正循环。

## 方法详解

### 整体框架

Agentic Retoucher 输入一张 T2I 生成图像 $I_t$ 及其对应 prompt $P$，输出修复后的无扭曲图像。整体流程为三阶段迭代闭环：(1) Perception Agent 生成扭曲显著性图 $S_t$ 定位异常区域；(2) 若 $S_t$ 超过阈值 $\tau_s$，Reasoning Agent 对检测到的区域进行分类诊断并生成文字描述 $\{D_i\}$ 和 mask $\{M_i\}$；(3) Action Agent 根据推理结果从模块化工具库中选择合适的修复方式执行局部 inpainting，得到更新图像 $I_{t+1}$。修复后的图像重新送入 Perception Agent 验证，迭代 2-3 轮直至所有显著扭曲消除。

### 关键设计

1. **Context-Aware Perception Agent（上下文感知扭曲检测器）**:

    - 功能：从生成图像中检测上下文依赖的局部扭曲，生成扭曲显著性图 $S \in [0,1]^{H \times W}$
    - 核心思路：采用双编码器架构（ViT 编码图像 + T5 编码 prompt），通过自注意力机制融合视觉和文本表征，捕捉视觉结构与文本语义之间的内在对应关系；轻量级注意力细化模块进一步聚合多尺度上下文线索。显著性图经二值化和形态学膨胀生成 mask 候选 $\{M_i\}$
    - 设计动机：T2I 扭曲往往是上下文依赖的（如手指数量需要结合全局人体结构判断），传统像素级检测不可靠。双编码器融合 prompt 语义信息可以利用文本-图像一致性线索辅助定位，而 KLD 损失项与人类注视分布对齐，避免显著性图过度平滑

2. **Human-Aligned Reasoning Agent（人类对齐推理 Agent）**:

    - 功能：对检测到的扭曲区域进行诊断推理，输出扭曲类型分类和自然语言描述 $\{D_i\}$
    - 核心思路：基于 VLM（如 Qwen2.5-VL-7B）+ LoRA 微调，采用两阶段渐进式偏好对齐训练——(a) SFT 阶段用交叉熵损失建立结构化输出格式和扭曲分类体系（LoRA rank=64, $\alpha$=32）；(b) GRPO 阶段通过偏好对齐强化学习减少幻觉，奖励基于分类准确率和文本描述与人类标注的对齐度
    - 设计动机：简单的分类或 captioning 不足以描述扭曲的类型、局部特征和上下文关系。直接用 GRPO（无 SFT 初始化）会导致输出格式不稳定和事实漂移，因此需要渐进式训练。实验证实 GPT-5/Gemini 2.5 Pro 零样本准确率仅约 61%，说明通用 VLM 无法胜任此任务

3. **Adaptive Action Agent（自适应修复 Agent）**:

    - 功能：将推理结果 $\{M_i, D_i\}$ 转化为可控的局部编辑操作，执行精准修复
    - 核心思路：从模块化工具库中根据计算约束和用户偏好动态选择修复方式——VLM-based（Qwen-Edit、Gemini 2.5 Flash Image）或 Mask-based（Flux-Fill、SD-inpainting）。为每个区域确定空间范围、工具选择和 inpainting 指令
    - 设计动机：不同类型的扭曲适合不同的修复工具（如文字渲染适合 VLM-based，几何扭曲适合 Mask-based）。工具解耦设计使框架可即插即用新工具，不受具体编辑模型限制

### 损失函数 / 训练策略

- **Perception Agent 损失**: 混合损失 $\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE}(S, \hat{S}) + (1-\alpha) \mathcal{L}_{KLD}(S, \hat{S})$，MSE 保证像素级精度，KLD 与人类注视分布对齐保持歧义区域的判别力，学习率 $2 \times 10^{-5}$
- **Reasoning Agent 训练**: 第一阶段 SFT（交叉熵损失 + LoRA），第二阶段 GRPO 偏好优化（$\mathcal{L}_{GRPO}$ 含归一化优势函数 $\hat{A}_t$ 和 KL 正则化项 $\beta D_{KL}[\pi_\theta || \pi_{ref}]$）
- **Action Agent**: 无需训练，通过 API 调用预训练编辑模型

### 数据集：GenBlemish-27K

为支持细粒度监督和定量评估，作者构建了 GenBlemish-27K 数据集：6,025 张 T2I 图像（来自 20+ 模型）中标注了 27,507 个像素级扭曲区域，覆盖 6 大维度 12 个细粒度类别（肢体畸形、面部扭曲、文字异常等）。标注经四阶段人机协作流程完成，标注一致率超过 95%。手部扭曲占 46.8%，面部缺陷占 15.7%，每张图平均 4.6 个标注区域。

## 实验关键数据

### 主实验

在 GenBlemish-27K 和 SynArtifacts-1K 上与多种 inpainting baseline 对比（四项感知指标：plausibility/aesthetics/alignment/overall）：

| 数据集 | 方法 | Plausibility↑ | Aesthetics↑ | Alignment↑ | Overall↑ |
|--------|------|:---:|:---:|:---:|:---:|
| GenBlemish-27K | Original | 44.21 | 53.69 | 57.89 | 47.15 |
| GenBlemish-27K | Qwen-Edit (直接) | 44.44 | 53.71 | 57.69 | 47.15 |
| GenBlemish-27K | **Ours w/ Qwen-Edit** | **47.10** | **55.75** | **59.54** | **49.27** |
| GenBlemish-27K | Gemini Flash (直接) | 44.41 | 53.80 | 57.93 | 47.27 |
| GenBlemish-27K | **Ours w/ Gemini Flash** | 46.81 | 55.47 | 59.22 | 48.97 |
| SynArtifacts-1K | Original | 61.53 | 61.63 | 60.65 | 55.35 |
| SynArtifacts-1K | **Ours w/ Gemini Flash** | **65.96** | **65.27** | **62.94** | **58.43** |
| SynArtifacts-1K | **Ours w/ SD-inpainting** | **66.66** | 64.67 | 62.33 | 58.27 |

人类评估（5 名评估者盲评）：

| 方法 | 明显更好(≫) | 略好(>) | 持平(≈) | 略差(<) | 明显更差(≪) |
|------|:---:|:---:|:---:|:---:|:---:|
| Baseline | 4.2% | 22.8% | 60.8% | 9.2% | 3.0% |
| **Ours** | **48.8%** | **34.4%** | 10.2% | 5.8% | 0.8% |

### 消融实验

**Perception Agent 消融**（注意力机制与 KLD 损失的影响）：

| 配置 | AUC-Judd↑ | NSS↑ | CC↑ | SIM↑ | KLD↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| w/o attn & KLD | 0.9335 | 1.1957 | 0.5518 | 0.3766 | 1.4436 |
| w/o attn | 0.9335 | 1.2153 | 0.5544 | 0.3731 | 1.4412 |
| w/o KLD | 0.9313 | 1.1892 | 0.5546 | 0.3525 | 1.5008 |
| **Full model** | **0.9336** | **1.2087** | **0.5568** | **0.3822** | **1.4313** |

**Reasoning Agent 消融**（Qwen2.5-VL-7B 为例）：

| 训练策略 | Accuracy↑ | SimCSE↑ | Word2Vec↑ | METEOR↑ | ROUGE↑ |
|----------|:---:|:---:|:---:|:---:|:---:|
| Zero-Shot | 57.76% | 0.6658 | 0.6110 | 0.1678 | 0.0733 |
| +GRPO only | 58.97% | 0.7020 | 0.6592 | 0.1741 | 0.1003 |
| +SFT only | 78.34% | 0.8405 | 0.7768 | 0.4011 | 0.3515 |
| **+SFT+GRPO** | **80.10%** | **0.8426** | **0.7785** | **0.4037** | **0.3530** |

### 关键发现

- 所有修复工具（VLM-based 和 Mask-based）接入 Agentic Retoucher 后均获得一致提升，证明框架与具体工具无关
- GPT-5 和 Gemini 2.5 Pro 零样本扭曲分类准确率仅 61.31%/60.28%，远低于本文方法的 80.10%，说明通用 VLM 不适合细粒度扭曲检测
- 仅用 GRPO 而跳过 SFT 阶段效果极差（58.97%），渐进式训练是必要的
- Perception Agent 的 AUC-Judd 达 0.9336，显著超过 SALICON（0.9230）、RichHF（0.9211）和所有通用 VLM

## 亮点与洞察

- **首个 T2I 后处理 agent 系统**：将修复从一次性前馈操作升级为可迭代的感知-推理-行动闭环，使 T2I 系统首次具备自主诊断和修复能力。这种范式天然支持多轮迭代收敛，比单次修复更稳健
- **揭示 VLM 的细粒度定位盲区**：实验定量证明了包括 GPT-5 在内的最强 VLM 在 AI 生成图像扭曲检测上的严重不足，这一发现对 VLM 评估 AIGC 质量的可信度提出了警示
- **工具解耦的架构设计**：Action Agent 以模块化工具库形式组织修复方法，使框架可以即插即用任何新的编辑模型，具有良好的可扩展性

## 局限与展望

- 迭代修复引入额外计算开销（每张图 2-3 轮推理），实时应用困难
- 修复工具库是预定义的静态集合，无法从修复经验中学习新策略或自我进化
- 主要针对局部几何扭曲（手指、面部、文字），对风格不一致或全局语义错误覆盖较弱
- GenBlemish-27K 数据分布偏斜（手部扭曲占 46.8%），可能导致对其他类型扭曲的感知不够均衡

## 相关工作与启发

- **vs RichHF**: RichHF 仅做评估不做修复，且过度关注面部和肢体区域。本文将评估和修复统一到闭环流程中
- **vs AgenticIR / JarvisArt**: 这些是通用图像修复/修图 agent 系统，而本文专门针对 AI 生成图像的特有扭曲类型设计感知和推理模块
- **vs Imagic / Step1x-Edit**: 这些后处理方法需要手动提供 mask 或编辑指令，本文实现了从检测到修复的全自动化
- 感知-推理-行动闭环范式可迁移至视频生成、3D 生成等其他需要自动质量控制的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 agent 决策系统引入 T2I 后处理是新视角，但各组件（显著性检测、VLM 推理、inpainting）本身并非全新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 + 四种修复工具 + 多模型消融 + 人类盲评，体系完整；缺少与端到端修复方法的直接对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰、图表精美，闭环范式的形式化描述简洁
- 价值: ⭐⭐⭐⭐ 填补了 T2I 自动质量修复的系统性空白，GenBlemish-27K 数据集有独立学术价值
---
title: >-
  [论文解读] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][图像生成][T2I后处理] Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。
tags:
  - CVPR 2026
  - 图像生成
  - T2I后处理
  - 感知-推理-行动循环
  - 扭曲检测
  - 局部修复
  - GenBlemish-27K
---

# Agentic Retoucher for Text-To-Image Generation

**会议**: CVPR 2026  
**arXiv**: [2601.02046](https://arxiv.org/abs/2601.02046)  
**代码**: 无  
**领域**: 图像生成 / Agent / 图像质量评估  
**关键词**: T2I后处理, 感知-推理-行动循环, 扭曲检测, 局部修复, GenBlemish-27K  

## 一句话总结
Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。

## 背景与动机
T2I 扩散模型（SDXL、FLUX 等）虽然能生成高质量图像，但仍然经常出现局部扭曲——手指畸形、面部不对称、文字不可读、肢体错位等。现有修复方案要么需要昂贵的全图重生成，要么依赖 VLM 做自动评估但 VLM 的空间定位能力很弱（六指图片被 VLM 判为正常）。缺乏一个能**自主发现 → 诊断 → 修复**局部缺陷的自动化系统。

## 核心问题
如何让 T2I 模型具备自主感知和修复生成缺陷的能力？如何解决 VLM 在细粒度缺陷检测上的不可靠性（幻觉导致的误判）？

## 方法详解

### 整体框架
Agentic Retoucher 由三个协作 agent 组成闭环：(1) Perception Agent 生成扭曲显著性图定位问题区域；(2) Reasoning Agent 对定位区域做诊断推理（分类 + 文字描述）；(3) Action Agent 根据推理结果选择工具做局部修复。修复后的图片再送回 Perception Agent 检查，迭代 2-3 轮直到无显著扭曲。

### 关键设计
1. **Context-Aware Perception Agent（上下文感知扭曲检测器）**: 采用双编码器架构（ViT 编码图像 + T5 编码 prompt），通过自注意力融合视觉和文本信息，生成扭曲显著性图 $S \in [0,1]^{H \times W}$。用混合损失训练：$\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE} + (1-\alpha) \mathcal{L}_{KLD}$，其中 KLD 项与人类注视分布对齐。比传统显著性模型和通用 VLM 在 AUC-Judd 上高出 10+ 个百分点。

2. **Human-Aligned Reasoning Agent（人类对齐推理 agent）**: 基于 Qwen2.5-VL-7B + LoRA 微调。两阶段训练：(a) SFT 阶段建立结构化输出格式和扭曲分类能力（LoRA rank=64）；(b) GRPO 阶段用偏好对齐减少幻觉。最终在扭曲类型分类准确率达到 80.10%（vs GPT-5 Zero-Shot 61.31%），语义描述 SimCSE 达 0.8517。

3. **Adaptive Action Agent（自适应修复 agent）**: 从模块化工具库中选择修复方式——VLM-基（Qwen-Edit、Gemini 2.5 Flash Image）或 Mask-基（Flux-Fill、SD-inpainting）。根据推理结果确定修复的空间范围、工具选择和指令，闭环后再验证。

### 损失函数 / 训练策略
- Perception Agent: MSE + KLD 混合损失
- Reasoning Agent: SFT（交叉熵）+ GRPO（偏好优化，奖励基于分类准确率和文本对齐度）

## 实验关键数据

| 数据集 | 条件 | Plausibility | Aesthetics | Alignment | Overall |
|--------|------|------|----------|------|------|
| GenBlemish-27K | Original | 44.21 | 53.69 | 57.89 | 47.15 |
| GenBlemish-27K | Ours w/ Qwen-Edit | **47.10** | **55.75** | **59.54** | **49.27** |
| SynArtifacts-1K | Ours w/ Gemini Flash | 65.96 | 65.27 | 62.94 | **58.43** |

人类评估：83.2% 的修复结果被判为优于原图（48.8% 明显更好 + 34.4% 略好）。

### 消融实验要点
- **Perception Agent**: 去掉注意力机制降低 SIM 和 CC；去掉 KLD 损失降低 NSS 和 AUC-Judd
- **Reasoning Agent**: 仅 GRPO（无 SFT）效果很差（准确率 58.97%）；SFT+GRPO 最优（80.10%）
- **工具选择**: 所有工具（Qwen-Edit、Gemini、Flux-Fill、SD-inpainting）配合 Agentic Retoucher 均有提升，说明框架与工具无关
- **GPT-5 和 Gemini 2.5 Pro Zero-Shot** 在扭曲推理上仅 61.31%/60.28%，说明通用 VLM 不擅长此任务

## 亮点
- 首次将 T2I 后处理修复建模为"感知-推理-行动"闭环 agent 系统，而非简单的一次性修复
- GenBlemish-27K 数据集提供了 27K 个像素级标注的扭曲区域，覆盖 12 类缺陷，是首个大规模 T2I 缺陷标注数据集
- 实验证明 VLM（包括 GPT-5）在零样本设置下无法可靠检测 AI 生成图像的扭曲——这是一个重要发现
- 框架与具体修复工具解耦，可以即插即用不同的编辑模型

## 局限与展望
- 迭代修复引入额外计算开销（2-3 轮推理）
- 当前修复工具是预定义的，无法学习新的修复策略
- 主要针对局部几何扭曲（手指、面部），对风格不一致或全局语义错误的覆盖较弱
- GenBlemish-27K 中手部扭曲占 46.8%，数据分布偏斜

## 与相关工作的对比
- **vs RichHF**: RichHF 做评估但不做修复，且过度关注面部/肢体区域。Agentic Retoucher 不仅评估还能闭环修复
- **vs AgenticIR/JarvisArt**: 这些是通用图像修复/修图 agent。Agentic Retoucher 专门针对 AI 生成图像的特有缺陷类型设计
- **vs Imagic/Step1x-Edit**: 这些需要手动提供 mask 或编辑指令。Agentic Retoucher 全自动定位和修复

## 启发与关联
- "感知-推理-行动"闭环范式对其他需要自动质量控制的生成任务有启发（视频生成、3D 生成）
- GenBlemish-27K 的扭曲分类体系（6 维度 12 类）可以作为评估 T2I 模型质量的标准化工具
- VLM 在细粒度空间定位上的失败案例值得关注——可能需要专门的空间理解训练

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 agent 系统应用于 T2I 后处理修复是新视角，但各组件（显著性检测、VLM 推理、inpainting）本身不新
- 实验充分度: ⭐⭐⭐⭐ 两个数据集 + 多种修复工具 + 消融 + 人类评估，但缺少与端到端修复方法的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表精美
- 价值: ⭐⭐⭐⭐ 填补了 T2I 自动质量修复的空白，GenBlemish-27K 数据集有独立价值

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Improving Text-to-Image Generation with Intrinsic Self-Confidence Rewards](improving_text-to-image_generation_with_intrinsic_self-confidence_rewards.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)

<!-- RELATED:END -->
