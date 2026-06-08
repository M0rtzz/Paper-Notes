---
title: >-
  [论文解读] Recursive Think-Answer Process for LLMs and VLMs
description: >-
  [CVPR 2026 (Findings)][多模态VLM][递归推理] R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法…
tags:
  - "CVPR 2026 (Findings)"
  - "多模态VLM"
  - "递归推理"
  - "Think-Answer"
  - "confidence generator"
  - "reasoning refinement"
  - "test-time scaling"
---

# Recursive Think-Answer Process for LLMs and VLMs

**会议**: CVPR 2026 (Findings)  
**arXiv**: [2603.02099](https://arxiv.org/abs/2603.02099)  
**代码**: 待确认（论文提到 Project page）  
**领域**: LLM推理 / 多模态VLM  
**关键词**: 递归推理, Think-Answer, confidence generator, reasoning refinement, test-time scaling

## 一句话总结
R-TAP 提出一种递归思考-回答过程，通过置信度生成器评估模型回答确定性并引导迭代推理修正，配合递归置信度增长奖励和最终答案置信度奖励的双重强化信号，在 LLM 和 VLM 上均一致超越单次推理方法，同时显著减少推理中的"Oops!"式自我反思表达。

## 研究背景与动机

**领域现状**：DeepSeek-R1 这类 Think-Answer 推理器靠可解释的内部思维链取得了明显进步，模型在作答前会产生大量中间推理。

**现有痛点**：这些模型在推理途中经常蹦出"Oops!"之类的自我反思，说明它其实意识到了自己出错。但在**单次推理（single-pass inference）**里，这种觉察派不上用场——模型发现错了也没法回滚重写，错误照样留在最终答案里，反思变成了一场"无效的挣扎"。

**核心 idea**：把一次性的思考-回答改成**递归的思考-回答循环**。每轮作答后由一个置信度生成器评估这次回答有多确定；确定性不够就带着上一轮的推理重启一轮，直到置信度过阈值或触顶最大递归次数。这样模型的"觉察到错误"终于能转化为"真正改掉错误"。

## 方法详解

### 整体框架
R-TAP 要解决的是 Think-Answer 推理器"发现自己错了却改不了"的问题：单次推理里模型作完答就结束，途中那句"Oops!"的自我察觉只能留在思维链里，无法回滚。R-TAP 的做法是在标准推理器外面套一圈递归循环——模型照常做一次完整的 Think-Answer，由一个置信度生成器给这次回答打一个确定性分数；分数不够就把上一轮的推理过程和答案当上下文喂回去，重新想一遍，直到置信度过阈值或达到最大递归深度 $K$。等于把"想一次"变成"不满意就带着上次的草稿再想一次"，让模型的觉察终于能落到改写动作上。

### 关键设计

**1. 置信度生成器：判断"这次回答到底有几分把握"。**

模型自己蹦出的"Oops!"是一种隐式的不确定信号，但它既不连续也不可靠，没法直接拿来当触发条件。R-TAP 因此训练一个轻量级的置信度生成器（Confidence Generator），以推理器的隐层表示或输出分布为输入，对每轮答案吐出一个标量置信度 $c \in [0,1]$。循环的开关就挂在这个分数上：只要 $c < \theta$（阈值），就判定"这答案还不够稳"，把当前推理拼进上下文启动下一轮；一旦 $c \ge \theta$ 或递归到顶就停。相比模型自评，外置的判别模块给出的信号更平滑、更可校准，递归才有一个可信的停手依据。

**2. 双重奖励：让递归真的"越想越对"而不是原地打转。**

光有循环不够——如果每轮答案质量没长进，递归只是徒增延迟。R-TAP 用两路奖励把递归过程和最终结果分别管起来。递归置信度增长奖励（RCIR）盯过程，奖励相邻两轮之间置信度的正向增量，$R_{RCIR} = \sum_{k=2}^{K} \max(0,\, c_k - c_{k-1})$，逼着模型每多想一轮就要比上一轮更确定，而不是来回抖动；最终答案置信度奖励（FACR）盯结果，直接奖励末轮的置信度 $R_{FACR} = c_K$，与答案是否正确解耦，单独拉高模型对自己最终输出的确信度。两者合成总奖励 $R = R_{RCIR} + \beta \cdot R_{FACR}$，前者保证"每步都在进步"，后者保证"最后落点足够稳"。

### 一个完整示例

> 以下置信度数值为示意，用来说明递归如何收敛，⚠️ 以原文为准。

拿一道模型一开始没把握的数学题走一遍：第 1 轮模型给出答案 A，置信度生成器打 $c_1=0.42 < \theta$（设 $\theta=0.8$），触发递归；第 2 轮把"答案 A + 那段含'Oops!'的推理"喂回去，模型重算得到答案 B，置信度升到 $c_2=0.67$，仍不过阈，RCIR 此时记一笔 $0.67-0.42=0.25$ 的正向增量；第 3 轮再带着前两轮上下文修正得到答案 C，$c_3=0.85 \ge \theta$，循环停止，C 作为最终答案，FACR 拿到 $0.85$。整个过程置信度沿 $0.42 \to 0.67 \to 0.85$ 单调爬升、3 轮命中——这正是 RCIR 想要的"越想越确定"轨迹，而不是固定多推几次的盲目重复。

### 损失函数 / 训练策略
训练分两阶段：先用正确/错误答案的二元标签训练置信度生成器，让它学会估计一个答案的正确概率；再以 $R = R_{RCIR} + \beta \cdot R_{FACR}$ 为奖励信号对 Think-Answer 推理器做 RL 微调，使其学会在递归中持续改写、把置信度一轮轮顶上去。

## 实验关键数据

### 主实验：LLM 推理（数学/逻辑推理基准）

| 模型 | 方法 | MATH (%) | GSM8K (%) | ARC (%) | 平均 |
|------|------|----------|-----------|---------|------|
| DeepSeek-R1-7B | Single-pass | 68.2 | 83.5 | 72.1 | 74.6 |
| DeepSeek-R1-7B | Self-Consistency | 70.8 | 85.1 | 73.4 | 76.4 |
| DeepSeek-R1-7B | **R-TAP** | **73.5** | **87.2** | **75.8** | **78.8** |

### VLM 推理任务

| 模型 | 方法 | MathVista (%) | ScienceQA (%) | 平均 |
|------|------|---------------|---------------|------|
| Base VLM | Single-pass | 54.3 | 71.6 | 63.0 |
| Base VLM | **R-TAP** | **58.7** | **74.9** | **66.8** |

### 消融实验

| 配置 | MATH (%) | 说明 |
|------|----------|------|
| Full R-TAP | 73.5 | 完整方法 |
| w/o RCIR | 71.2 | 去掉递归增长奖励 |
| w/o FACR | 72.0 | 去掉最终置信度奖励 |
| w/o Confidence Generator | 69.5 | 改为固定次数递归 |

### 关键发现
- R-TAP 使模型的"Oops!"等自我反思表达显著减少——表明模型不再需要频繁的内部纠错，推理更加稳定
- 递归 2-3 轮即可获得大部分收益，超过 5 轮后收益饱和
- 置信度生成器是核心组件——没有它，固定次数的递归效果显著变差
- R-TAP 带来的推理更加稳定和快速——减少了不必要的内部反思循环

## 亮点与洞察
- **"Oops!"现象的深刻洞察**——首次系统分析 Think-Answer 推理器中自我反思表达的频率与推理质量的关系，发现反思频率低≠推理能力差，而是推理更加稳定的标志
- **递归而非单次**——将 test-time compute 从"更长的单次思考"转变为"多轮迭代改进"，两种范式可以互补
- **置信度驱动的按需递归**——不是盲目多推几次，而是不确定时才递归，效率更高
- **LLM + VLM 通用**——框架不依赖特定模态，适用于纯文本和多模态推理

## 局限与展望
- 递归增加了推理延迟，在实时应用中可能不可接受
- 置信度生成器需要额外训练数据和计算，不如 Self-Consistency 的无训练简洁
- 当答案空间开放（如生成式任务）时，置信度的定义和估计变得更困难
- 未探索与 Tree-of-Thought 等结构化推理方法的结合
- 最大递归深度 $K$ 仍是手工设定的超参

## 相关工作与启发
- **vs Self-Consistency**：Self-Consistency 通过多次采样+投票提升一致性，但每次推理独立，不利用上一轮信息。R-TAP 的递归是"有记忆的改进"
- **vs Chain-of-Thought**：CoT 是"更长地想一次"，R-TAP 是"短地想多次并迭代改进"
- **vs Self-Refine**：Self-Refine 让模型自我反馈改进，但缺少外部置信度评估。R-TAP 用专门的 Confidence Generator 做更可靠的判断
- **启发**：递归推理+置信度评估的范式可以推广到代码生成、机器人规划等需要迭代改进的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 递归推理的思想虽有先例（Self-Refine），但置信度驱动+双奖励设计有新意
- 实验充分度: ⭐⭐⭐⭐ LLM+VLM 双验证，消融实验覆盖各组件，"Oops!"分析有独到视角
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，"Oops!"现象的引入很生动
- 价值: ⭐⭐⭐⭐ 提供了一种通用的 test-time reasoning 改进框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Empowering Small VLMs to Think with Dynamic Memorization and Exploration](../../ICLR2026/multimodal_vlm/empowering_small_vlms_to_think_with_dynamic_memorization_and_exploration.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[ACL 2025\] Chart-based Reasoning: Transferring Capabilities from LLMs to VLMs](../../ACL2025/multimodal_vlm/chart-based_reasoning_transferring_capabilities_from_llms_to_vlms.md)

</div>

<!-- RELATED:END -->
