---
title: >-
  [论文解读] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation
description: >-
  [CVPR 2026][图像生成][compositional T2I generation] 提出coDrawAgents交互式多智能体对话框架，Interpreter、Planner、Checker、Painter四个专业智能体闭环协作，以分治策略按语义优先级逐组增量规划布局…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "compositional T2I generation"
  - "multi-agent dialogue"
  - "layout planning"
  - "visual context grounding"
  - "error correction"
---

# coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation

**会议**: CVPR 2026  
**arXiv**: [2603.12829](https://arxiv.org/abs/2603.12829)  
**代码**: 待发布  
**领域**: 图像生成 / 多智能体系统  
**关键词**: compositional T2I generation, multi-agent dialogue, layout planning, visual context grounding, error correction

## 一句话总结

提出coDrawAgents交互式多智能体对话框架，Interpreter、Planner、Checker、Painter四个专业智能体闭环协作，以分治策略按语义优先级逐组增量规划布局，基于画布视觉上下文接地推理并显式纠错，在GenEval上以0.94 Overall Score大幅领先GPT Image 1（0.84），在DPG-Bench上达85.17 SOTA。

## 研究背景与动机

**领域现状**：文本到图像（T2I）生成在多对象复杂场景中面临组合保真度问题。现有探索包括：LLM辅助布局生成（LayoutLLM-T2I、LMD）、扩散注意力引导（Attend-and-Excite）、生成思维链（GoT）、和早期多智能体框架（MCCD、T2I-Copilot）。

**现有痛点**：

1. 单智能体方法将解析/规划/验证交给一个模型，早期空间错误难以检测修复
2. 现有多智能体框架本质是固定流水线，缺乏协商和视觉反馈，错误仍然传播
3. 全局布局规划面临对象间关系的二次复杂度 $O(N^2)$，N个对象同时规划极困难
4. 绝大多数方法在无视觉上下文下预测布局，只能"想象"场景

**核心矛盾**：复杂场景需要的布局推理能力随对象数量呈二次增长，但单次规划和固定流水线均无法有效处理这种复杂度爆炸。

**本文目标** 在复杂多对象场景中实现忠实的组合T2I生成，同时解决布局复杂度、缺乏视觉感知、早期错误无法纠正三大挑战。

**切入角度**：四智能体闭环对话式协作——分治降低复杂度 + 画布视觉接地 + 显式检查纠错。

**核心 idea**：让Planner看着正在生成的画面来规划下一步、让Checker回溯所有历史布局来纠错、按语义优先级分组来降低单轮复杂度。

## 方法详解

### 整体框架

coDrawAgents要解决的是「一句话里塞了七八个对象、还带空间关系和属性约束」时，T2I模型画不忠实的问题。它不再让一个模型一口气把所有对象的位置都想好，而是把四个专业智能体围成一个闭环：Interpreter读题、Planner排布局、Checker挑错、Painter上色，画一笔、看一眼、再画下一笔。

具体怎么转：Interpreter先判断这句话复杂不复杂。简单场景（比如"一只猫"）直接走layout-free模式，调一次T2I就完事；复杂场景则切到layout-aware模式——Interpreter把文本拆成一组带属性的对象描述，并按语义重要性把它们分成几个优先级组。接下来按优先级一轮一轮迭代：每轮Planner只盯着当前这一组对象，看着已经画好的画布去规划它们该放哪儿，Checker两阶段查错并修正，Painter把这一组渲染上画布，这张更新后的画布又成为下一轮Planner的视觉输入。所有优先级组都画完，输出最终图像。

### 关键设计

**1. Interpreter + 分治策略：把 $O(N^2)$ 的全局布局摊成几轮 $O(k^2)$ 的局部布局**

复杂场景最难啃的地方在于，对象之间的空间关系数量随对象数呈平方增长——N个对象同时规划，要同时摆平 $O(N^2)$ 对关系，一次性想清楚几乎不可能。Interpreter的做法是先用LLM配CoT提示走三步：先把文本切成一个个语义单元，再按语义显著性给这些对象排序、把同等重要的归成一组，最后给每个对象做CoT引导的属性补全和背景描述。分组之后，每一轮Planner只处理同一优先级的 $k$ 个对象（$k \ll N$），单轮复杂度从 $O(N^2)$ 降到 $O(k^2)$，整张图的规划被摊到几轮里完成。代价很小：DPG-Bench场景平均2.79个对象，平均只需1.52轮Planner调用就收敛。

**2. Planner + Visualization Chain-of-Thought（VCoT）：让布局规划从"闭眼想象"变成"看着画布画"**

绝大多数布局方法的硬伤是在没有任何视觉上下文的情况下预测坐标，模型只能凭文本"想象"场景长什么样，想出来的布局和实际画面常常对不上。Planner用GPT-5作为MLLM，把规划拆成三步VCoT：先做Canvas State Analysis，接收上一轮的画布图像 $I_{i-1}$ 和已有布局，读出现有对象的空间状态；再做Context-Aware Planning，结合世界知识推理新对象该怎么和已有场景合理交互；最后做Physics Constraint Enforcement，保证物理上说得通（不漂浮、接触面合理）。其中关键的一步是对象接地（grounding），把文本里的实体明确对应到画布上的具体区域——LLM对纯坐标数字本来不敏感，接地这一步等于给它一个"看图说话"的锚点。因为规划是基于真实画布而非想象，布局和视觉天然一致，这也是Position指标能从0.75一路冲到0.95的根源。

**3. Checker 两阶段检查-修正：在布局阶段就把错误纠掉，别等扩散把它"烤死"**

扩散模型有个特性：早期去噪步一旦确定了粗结构，后面就很难再改，错误会被"bake in"进最终图像。所以Checker选择在布局阶段、图还没生成时就显式纠错，分两阶段做。第一阶段只看当前这一轮的布局 $L_i$，做对象级检查（尺寸、比例、覆盖是否合理）加全局级检查（相对位置、对象关系是否符合描述），就地修正。第二阶段把目光放到历史上所有布局 $\{L_1, ..., L_i\}$ 上，专门找跨对象、跨轮次的冲突——比如这一轮新放的对象和前几轮的某个对象重叠了、遮挡了、或者尺度突然漂了，逐个修复并把修正传播下去。这种跨迭代回溯是分步增量生成才有的能力：固定流水线一次成型，根本没有"回头看前几步"的机会。

**4. Painter 即插即用渲染：把"画"的能力和"想"的逻辑解耦**

Painter负责每一轮把规划好的对象增量渲染到画布上，渲染结果立刻成为下一轮Planner的视觉上下文。它本身不掺和规划和验证，只做执行：layout-free模式用Flux这类T2I模型，layout-aware模式用3DIS这类L2I模型，都是直接拿预训练模型来用、不需要额外训练。这种解耦的好处是底层绘制模型升级时，整个框架不用动就能自然受益——规划和验证的智能逻辑沉淀在Planner和Checker里，Painter只是一支可以随时换的"笔"。

### 一个完整示例

以一句"桌上有三个红苹果和一个蓝碗"为例走一遍。Interpreter先判定这是复杂场景（多对象+计数+空间关系），拆出对象 {蓝碗, 苹果×3, 桌子(背景)}，按语义显著性分组：桌子和碗作为承载体优先级高、归第一组，三个苹果归第二组。

第一轮，Planner拿到还是空白的画布，做VCoT：分析画布为空 → 规划桌子铺满下半部、蓝碗居中放在桌面上 → 检查物理（碗稳稳坐在桌面、不漂浮）。Checker确认这一轮布局没问题。Painter用3DIS把桌子和蓝碗画上画布。

第二轮，Planner接收的已经是画了桌子和碗的画布图像，VCoT分析出"碗在画面中央、内部有空间" → 规划三个苹果，其中一两个放进碗里、其余摆在碗边的桌面上 → 物理检查（苹果之间不互相穿模、放碗里的没溢出碗沿）。Checker第二阶段回溯第一轮的碗布局，发现某个苹果的bbox和碗沿轻微重叠超界，把它往内收一点。Painter渲染三个苹果，输出最终图像。

整个过程Interpreter调用1次、Planner 2轮、对象数4个但轮次只用了2——分组让"四个对象"的规划压成了"两组、两轮"，而每一轮Planner都是看着上一轮的真实画面在排，计数（恰好3个苹果）和空间关系（在桌上、在碗里）都因此得到保证。

### 损失函数 / 训练策略

无需额外训练，整个框架是training-free且plug-and-play的。全程复用预训练LLM（GPT-5）做解析/规划/检查，复用现成的T2I（Flux）和L2I（3DIS）模型做渲染，不引入任何可训练参数。

## 实验关键数据

### 主实验

**GenEval基准对比**

| 模型 | Single | Two Obj. | Counting | Colors | Position | Color Attr. | Overall↑ |
|------|--------|----------|----------|--------|----------|-------------|----------|
| DALL-E 3 | 0.96 | 0.87 | 0.47 | 0.83 | 0.43 | 0.45 | 0.67 |
| FLUX.1-dev | 0.99 | 0.81 | 0.79 | 0.74 | 0.20 | 0.47 | 0.67 |
| GoT | 0.99 | 0.69 | 0.67 | 0.85 | 0.34 | 0.27 | 0.64 |
| UniWorld-V1 | 0.99 | 0.93 | 0.79 | 0.89 | 0.49 | 0.70 | 0.80 |
| GPT Image 1 [High] | 0.99 | 0.92 | 0.85 | 0.92 | 0.75 | 0.61 | 0.84 |
| **coDrawAgents** | **1.00** | **0.96** | **0.94** | **0.97** | **0.95** | **0.81** | **0.94** |

**DPG-Bench对比**

| 模型 | Global | Entity | Relation | Overall↑ |
|------|--------|--------|----------|----------|
| DALL-E 3 | 90.97 | 89.61 | 90.58 | 83.50 |
| SD3-Medium | 87.90 | 91.01 | 80.70 | 84.08 |
| OmniGen2 | 88.81 | 88.83 | 89.37 | 83.57 |
| **coDrawAgents** | 84.78 | **90.15** | **92.92** | **85.17** |

### 消融实验

| 配置 | DPG Overall↑ | 说明 |
|------|-------------|------|
| Layout-free baseline | 77.60 | 仅直接T2I |
| + Layout-aware | 82.61 (+5.01) | 分治策略降低复杂度 |
| + Visual context | 84.51 (+1.90) | 画布接地增强空间一致性 |
| + Checker (完整) | **85.17** (+0.66) | 显式纠错提升忠实度 |

**效率统计（DPG-Bench 1074图）**

| 智能体 | 平均调用次数/图 |
|--------|--------------|
| Interpreter | 1.00 |
| Planner | 1.52 |
| Checker | 1.62 |
| Painter | 1.95 |
| 场景平均对象数 | 2.79 |

### 关键发现

- GenEval Overall从GPT Image 1的0.84跃升到0.94（+11.9%），全子指标均为最高
- Position指标从0.75暴涨到0.95，说明画布视觉接地+分治策略极大增强了空间推理能力
- Counting从0.85→0.94，分组生成有效解决了计数问题
- 智能体平均调用次数远少于场景对象数（1.52 vs 2.79），因分组策略减少迭代轮次

## 亮点与洞察

- 分治策略将N对象全局布局分解为按语义优先级逐组规划，优雅降低复杂度
- 画布视觉上下文作为Planner输入是核心创新——让布局推理从"想象"变为"看着画"
- Checker的跨迭代回溯修正可处理早期错误的级联效应，这在固定流水线中不可能实现
- VCoT三步推理（状态分析→上下文规划→物理约束）结构清晰，可推广到其他需要空间推理的生成任务

## 局限与展望

- 多智能体调用引入计算开销（多次LLM推理 + 多次图像生成），推理时间比单次方法长
- Painter性能依赖底层T2I/L2I模型能力，属性渲染不完美（如"黑皮萝卜"）会传播
- Planner和Checker依赖GPT-5 MLLM，存在幻觉和过度自信风险
- 仅支持2D合成，未扩展到3D场景生成
- DPG-Bench Global指标（84.78）低于部分单模型（如DALL-E 3的90.97），分步生成可能损失全局一致性

## 相关工作与启发

- **vs GoT**：GoT一次性推理所有bbox且无视觉反馈（Overall 0.64 vs 0.94），验证了闭环交互式协作的根本优势
- **vs T2I-Copilot**：固定流水线无对话协商和视觉接地（Overall 74.34 vs 85.17）
- **vs MCCD**：仅做文本分解无画布感知，本质是并行生成再融合
- **启发**：闭环多智能体协作范式可推广到视频生成（逐帧规划+一致性检查）、3D场景构建（逐物体放置+碰撞检测）等需要增量式组合的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 闭环多智能体对话框架和VCoT视觉接地规划有创新，但核心技术是LLM/MLLM的prompt工程
- 实验充分度: ⭐⭐⭐⭐ GenEval和DPG-Bench全面对比+消融+效率分析，定性比较清晰
- 写作质量: ⭐⭐⭐⭐ 四智能体定位和分工描述清晰，框架图直观
- 价值: ⭐⭐⭐ 组合生成效果惊艳但工程性强，依赖GPT-5的成本和可复现性是主要顾虑

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)
- [\[CVPR 2026\] Intrinsic Concept Extraction Based on Compositional Interpretability](intrinsic_concept_extraction_based_on_compositional_interpretability.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[CVPR 2026\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)

</div>

<!-- RELATED:END -->
