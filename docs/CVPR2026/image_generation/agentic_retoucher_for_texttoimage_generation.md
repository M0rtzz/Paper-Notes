---
title: >-
  [论文解读] Agentic Retoucher for Text-To-Image Generation
description: >-
  [CVPR 2026][图像生成][T2I后处理] Agentic Retoucher 将 T2I 生成后的缺陷修复重构为"感知→推理→行动"的人类式闭环决策过程，用三个协作 agent 分别做上下文感知的扭曲检测、人类对齐的诊断推理和自适应局部修复，在 GenBlemish-27K 上 plausibility 提升 2.89 分，83.2% 的结果被人类评为优于原图。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "T2I后处理"
  - "感知-推理-行动循环"
  - "扭曲检测"
  - "局部修复"
  - "GenBlemish-27K"
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

Agentic Retoucher 要解决的是 T2I 生成图的局部缺陷——手指畸形、面部不对称、文字不可读、肢体错位——而现有方案要么重生成整张图太贵，要么靠 VLM 自动评估但 VLM 空间定位太弱（六指的手会被判成正常）。它把修复重构成人类式的「感知→推理→行动」闭环：Perception Agent 先生成扭曲显著性图定位问题区域，Reasoning Agent 对定位区域做诊断推理（分类 + 文字描述），Action Agent 据此从工具库里选合适的编辑工具做局部修复；修完再送回 Perception Agent 复检，迭代 2-3 轮直到没有显著扭曲。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["T2I 生成图<br/>含局部缺陷"] --> B["Perception Agent<br/>双编码器输出扭曲显著性图定位"]
    B --> C["Reasoning Agent<br/>SFT+GRPO 诊断：分类 + 文字描述"]
    C --> D["Action Agent<br/>从工具库选编辑器做局部修复"]
    D --> E["送回复检"]
    E -->|"仍有显著扭曲：迭代 2-3 轮"| B
    E -->|"无显著扭曲"| F["输出修复图"]
```

### 关键设计

**1. 上下文感知的 Perception Agent：把缺陷定位做成显著性图**

通用 VLM 连「哪儿坏了」都定位不准，所以第一步得有个专门的检测器。Perception Agent 用双编码器（ViT 编码图像 + T5 编码 prompt），通过自注意力融合视觉和文本，输出扭曲显著性图 $S \in [0,1]^{H \times W}$。训练用混合损失 $\mathcal{L}_{sal} = \alpha \mathcal{L}_{MSE} + (1-\alpha) \mathcal{L}_{KLD}$，其中 KLD 项把预测分布对齐到人类注视分布——这让它在 AUC-Judd 上比传统显著性模型和通用 VLM 高出 10+ 个百分点，定位才靠谱。

**2. 人类对齐的 Reasoning Agent：两阶段训练压住幻觉**

定位之后要判断「这是什么缺陷」，VLM 在这一步极易幻觉。Reasoning Agent 基于 Qwen2.5-VL-7B + LoRA 微调，分两阶段：先做 SFT（LoRA rank=64）建立结构化输出格式和扭曲分类能力，再用 GRPO 做偏好对齐减少幻觉。两阶段缺一不可——消融里只用 GRPO 不做 SFT 时准确率只有 58.97%，而 SFT+GRPO 把扭曲类型分类准确率做到 80.10%（GPT-5 零样本只有 61.31%），语义描述 SimCSE 达 0.8517。

**3. 自适应的 Action Agent：与具体修复工具解耦**

诊断完由 Action Agent 决定怎么修，它从模块化工具库里按推理结果选——VLM 基的 Qwen-Edit、Gemini 2.5 Flash Image，或 Mask 基的 Flux-Fill、SD-inpainting——并确定修复的空间范围和指令，修完闭环复检。工具是即插即用的，消融显示换任何一个工具配合该框架都有提升，说明效果来自框架本身而非某个特定编辑器。

### 一个完整示例：修一张六指的手

给一张 SDXL 生成、手部多出一根手指的图：Perception Agent 先在显著性图上把手部区域点亮、定位到畸形所在；Reasoning Agent 读这块区域，分类为「手指畸形」并给出文字描述；Action Agent 据此挑 Mask 基的 Flux-Fill 对手部局部 inpaint，按描述把多出的手指修掉；修复图再送回 Perception Agent 复检，若仍有残留就进入下一轮，整个过程迭代 2-3 轮收敛到无显著扭曲。这正是「感知→推理→行动→再感知」区别于一次性修复的地方——它能发现自己没修干净并继续修。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Vinedresser3D: Agentic Text-guided 3D Editing](vinedresser3d_agentic_text-guided_3d_editing.md)
- [\[ICML 2026\] AtelierEval: Agentic Evaluation of Humans & LLMs as Text-to-Image Prompters](../../ICML2026/image_generation/ateliereval_agentic_evaluation_of_humans_llms_as_text-to-image_prompters.md)
- [\[CVPR 2026\] Resolving the Identity Crisis in Text-to-Image Generation](resolving_the_identity_crisis_in_text-to-image_generation.md)
- [\[ACL 2025\] A Unified Agentic Framework for Evaluating Conditional Image Generation](../../ACL2025/image_generation/a_unified_agentic_framework_for_evaluating_conditional_image_generation.md)
- [\[ICML 2026\] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes](../../ICML2026/image_generation/scenesmith_agentic_generation_of_simulation-ready_indoor_scenes.md)

</div>

<!-- RELATED:END -->
