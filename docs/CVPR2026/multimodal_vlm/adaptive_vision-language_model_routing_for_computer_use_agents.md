---
title: >-
  [论文解读] AVR: Adaptive VLM Routing for Computer Use Agents
description: >-
  [CVPR 2026][多模态VLM][Computer Use Agent] 提出 AVR 自适应路由框架，通过轻量多模态嵌入模型评估动作难度 + 小模型 logprob 置信度探测 + warm agent 记忆注入，实现三层路由（简单→小模型，困难→大模型，高风险→大模型+guardrail）…
tags:
  - "CVPR 2026"
  - "多模态VLM"
  - "Computer Use Agent"
  - "VLM路由"
  - "自适应推理"
  - "成本优化"
  - "GUI grounding"
---

# AVR: Adaptive VLM Routing for Computer Use Agents

**会议**: CVPR 2026  
**arXiv**: [2603.12823](https://arxiv.org/abs/2603.12823)  
**代码**: [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router)  
**领域**: 多模态VLM  
**关键词**: Computer Use Agent, VLM路由, 自适应推理, 成本优化, GUI grounding

## 一句话总结

提出 AVR 自适应路由框架，通过轻量多模态嵌入模型评估动作难度 + 小模型 logprob 置信度探测 + warm agent 记忆注入，实现三层路由（简单→小模型，困难→大模型，高风险→大模型+guardrail），在推理成本降低 78% 的同时仅损失 2pp 准确率。

## 研究背景与动机

**CUA 的兴起与瓶颈**：Computer Use Agents（CUA）通过 VLM 直接操作 GUI 执行复杂任务（点击、输入、滑动等），是当前 Agent 领域最热门的方向之一。但现有 CUA 系统（如 OpenAI CUA、Claude Computer Use）采用单一大型 VLM 处理所有操作，每次动作都需要调用 GPT-4o 或 Claude 级别的模型，API 成本极高——一个 50 步的任务可能花费数美元。

**动作难度的巨大差异**：CUA 任务中的动作复杂度高度不均匀。简单动作（如点击显眼的按钮、在文本框中输入已知内容）占据了大量操作，完全不需要顶级 VLM 的推理能力；而复杂动作（如在密集 UI 中定位小图标、理解上下文做多步推理）则确实需要强模型。这种"杀鸡用牛刀"的现象造成了严重的资源浪费。

**现有路由方案的局限**：文本领域的 LLM 路由（如 RouterBench、RouteLLM）已有一些探索，但直接迁移到 CUA 面临挑战：(a) GUI 操作涉及视觉复杂度（屏幕布局、元素密度）和语义复杂度（指令歧义度），纯文本特征不够；(b) CUA 的动作是序列化的，上下文累积效应（之前做了什么）显著影响当前动作难度；(c) 错误的路由决策在 CUA 中代价更高——一次误点可能导致整个任务失败。

**模型大小 ≠ grounding 准确率**：作者的关键发现是，在 GUI grounding 任务上，模型大小与准确率并不完全正相关。在 ScreenSpot-Pro 上，GPT-4o 仅获得 0.8% 的准确率（可能因为其视觉编码器对 GUI 元素的理解不佳），而开源 7B 的 OS-Atlas 达到 18.9%，Qwen2.5-VL-72B 达到 43.6%。这意味着路由不能简单地按模型大小分层，需要考虑模型在 GUI 领域的实际能力。

**记忆的不对称效果**：作者观察到，将之前成功动作的经验（warm agent memory）注入模型时，对小模型的提升远大于大模型。小模型从 0.83 提升到 0.96 的任务成功率，而大模型只有边际改善。这为路由策略提供了关键洞察：通过记忆注入，小模型的"能力天花板"可以被显著抬高。

## 方法详解

### 整体框架

AVR 要解决的是 CUA「每个动作都调用顶级 VLM」带来的成本浪费。它的做法是把一次动作的处理拆成一条**级联路由管线**：先让一个轻量难度评估器快速看一眼当前截图和指令，把明显简单的动作直接交给小 VLM；剩下的动作让小 VLM 先试着生成一次，根据它输出的置信度决定是否放行；若信心不足，再注入历史成功经验重试一次；只有到这一步仍不确信，才升级到大 VLM（高风险操作再加一道 guardrail 验证）。整条管线越往后越贵，绝大多数动作在前两层就被消化掉。

三层的分工大致是：Layer 1 的难度评估器过滤掉约 40-50% 的明显简单动作；Layer 2 让小 VLM 实际尝试并用 logprob 判断置信度；Layer 3 在记忆注入后再探一次，仍无信心才升级到大模型。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["截图 + 指令"] --> B["多模态难度评估器<br/>120M 模型估难度 d"]
    B -->|明显简单| S["小 VLM 直接执行"]
    B -->|其余动作| C["Logprob 置信度探测<br/>小 VLM 生成动作，算置信度 C"]
    C -->|"C > θ_high"| S
    C -->|"C ≤ θ_low"| L["升级大 VLM"]
    C -->|中等置信| M["Warm Agent 记忆注入<br/>检索 top-k 经验重生成，再探 C"]
    M -->|够信| S
    M -->|仍不足| L
    L -->|高风险动作| G["Guardrail 验证"]
    L -->|普通动作| O["执行动作"]
    S --> O
    G --> O
```

### 关键设计

**1. 多模态难度评估器：用 120M 小模型先把简单动作筛走**

CUA 里大量动作（点显眼按钮、往文本框填已知内容）根本不需要顶级 VLM，但单模型架构对它们也照样调用大模型。难度评估器就是为了在最便宜的环节把这些动作拦下来：它是一个 120M 参数的多模态嵌入模型（类似 SigLIP-small），吃当前截图 + 指令文本，吐出一个标量难度分 $d \in [0,1]$。这个分数同时综合视觉复杂度（UI 元素密度、目标显著性、布局复杂度）和语义复杂度（指令歧义度、推理步数、是否需要上下文）。

训练数据来自在 ScreenSpot-Pro 和 OpenClaw 上收集不同规模模型（7B/14B/72B）的成功/失败记录——小模型能搞定的动作标「简单」，只有大模型能完成的标「困难」。它的推理开销极低（~2ms/帧），远小于任何 VLM 一次推理，所以放在管线最前端不会成为瓶颈，相当于用近乎免费的一步换掉了大量不必要的大模型调用。

**2. Logprob 置信度探测：让小模型自己「报」有没有把握**

被难度评估器放过、没被直接判简单的动作，未必真需要大模型——很多时候小模型自己就能做对，关键是怎么知道它做对了。AVR 的办法是让小 VLM（如 Qwen2.5-VL-7B）实际推理一次，生成动作序列 $a = (a_1, a_2, ..., a_n)$ 并收集每个 token 的 log 概率，用几何平均概率当置信度：

$$C(a) = \exp\left(\frac{1}{n}\sum_{i=1}^{n} \log p(a_i | a_{<i}, s)\right)$$

其中 $s$ 是当前状态（截图+历史）。然后按两个阈值分流：$C(a) > \theta_{\text{high}}$ 高置信，直接执行小模型动作；$\theta_{\text{low}} < C(a) \leq \theta_{\text{high}}$ 中等置信，进入记忆注入；$C(a) \leq \theta_{\text{low}}$ 低置信，直接升级大模型。阈值 $\theta_{\text{high}} = 0.85$、$\theta_{\text{low}} = 0.60$ 由验证集上的 F1 网格搜索确定。用模型自己的输出概率做路由信号，既省去再训一个单独的难度判别器，也让「该不该升级」有了可量化的依据。

**3. Warm Agent 记忆注入：用历史经验抬高小模型的能力天花板**

中等置信的动作如果直接升级大模型就太浪费了——作者发现小模型的瓶颈往往不是能力不够，而是缺 GUI 操作的先验经验。记忆模块为此维护一个动态经验库 $\mathcal{M} = \{(s_j, a_j, r_j)\}$，记录过去成功的（状态, 动作, 奖励）三元组。注入分三步：用当前截图和指令的嵌入从 $\mathcal{M}$ 检索最相似的 top-$k$（$k=3$）成功经验；把它们以 few-shot 示例形式塞进小 VLM 的 prompt；让小 VLM 在记忆增强后重新生成动作，再查一遍置信度。

这步的效果高度不对称：小模型（7B）任务成功率从 0.83 → 0.96（+13pp），大模型（72B）只有 0.94 → 0.95（+1pp）。也就是说同一份记忆喂给小模型收益巨大、喂给大模型几乎无感，这正是 AVR 经济性的来源——靠少量示例就能把大量本要升级的动作留在小模型这一层，既提成功率又省钱。

### 损失函数 / 训练策略

- 难度评估器：在人工标注的 5K 难度标签上微调 120M 嵌入模型，使用 BCE loss
- 路由策略本身无需训练，阈值通过验证集搜索确定
- 记忆库在线更新，采用 FIFO 策略维持固定大小（1000 条）

## 实验关键数据

### 主实验：ScreenSpot-Pro GUI Grounding 准确率

| 模型/方法 | 准确率 (%) | 推理成本 (相对) | 延迟 (ms/动作) |
|---|---|---|---|
| GPT-4o | 0.8 | 1.00× | ~3000 |
| OS-Atlas-7B | 18.9 | 0.05× | ~200 |
| Qwen2.5-VL-14B | 28.3 | 0.12× | ~400 |
| Qwen2.5-VL-72B | 43.6 | 0.80× | ~2000 |
| **AVR (7B+72B)** | **42.7** | **0.22×** | **~450** |

AVR 以仅 22% 的成本达到接近 72B 单模型的准确率（42.7% vs 43.6%），且平均延迟大幅降低。

### OpenClaw 任务成功率

| 方法 | 成功率 (%) | 平均步数 | 平均成本 ($) |
|---|---|---|---|
| Qwen2.5-VL-7B | 68.2 | 32.1 | 0.12 |
| Qwen2.5-VL-72B | 87.5 | 28.4 | 2.85 |
| 固定路由 (50/50) | 79.1 | 30.2 | 1.48 |
| RouteLLM (文本) | 76.8 | 31.0 | 0.89 |
| **AVR** | **85.7** | **29.1** | **0.63** |

AVR 在 OpenClaw 上成功率仅低于纯 72B 模型 1.8pp，但成本降低了 78%（$0.63 vs $2.85）。

### 消融实验

| 难度评估器 | 置信度探测 | 记忆注入 | 成功率 (%) | 成本 ($) |
|---|---|---|---|---|
| ✗ | ✗ | ✗ | 68.2 | 0.12 |
| ✓ | ✗ | ✗ | 78.3 | 0.95 |
| ✓ | ✓ | ✗ | 82.1 | 0.71 |
| ✓ | ✓ | ✓ | **85.7** | **0.63** |
| ✗ | ✓ | ✓ | 83.4 | 0.82 |

三个组件均有显著贡献。记忆注入不仅提升成功率（+3.6pp），还进一步降低成本（从 $0.71 到 $0.63），因为它减少了升级到大模型的次数。

### 关键发现

- **GPT-4o 在 GUI grounding 上表现极差**（0.8%），说明闭源模型在特定领域并不一定优于开源模型，模型选择需要基于任务评测而非品牌
- **记忆注入的不对称效应**是核心发现：小模型受益巨大（+13pp），大模型几乎无感（+1pp），这为路由策略的经济性提供了理论支撑
- **动作难度分布呈长尾**：约 45% 的动作属于"简单"类别，可直接由小模型处理；30% 为中等难度，记忆注入后小模型可搞定；仅 25% 的动作真正需要大模型
- **Guardrail 对高风险动作的必要性**：在涉及不可逆操作（删除文件、发送消息、支付）时，即使大模型也有约 3% 的错误率，额外的 guardrail 验证将其降至 0.5%

## 亮点与洞察

- **切中 CUA 落地痛点**：成本是 CUA 规模化部署的最大障碍，AVR 将推理成本降低近 5 倍，使 CUA 的商业化更可行
- **记忆注入的巧妙复用**：记忆既服务于路由决策（降低升级率），又提升小模型本身的能力，一举两得
- **三层级联设计优雅**：每层都有明确的分工和退出条件，避免了复杂的联合优化，工程上易于实现和调优
- **挑战"大模型万能"迷思**：GPT-4o 在 GUI grounding 上的惨淡表现是对行业盲目追求大模型的有力反驳

## 局限与展望

- 难度评估器需要针对特定 GUI 域（桌面、移动、Web）分别训练，跨域泛化能力未验证
- 置信度阈值为静态设定，理想情况下应该根据任务类型和当前进度动态调整
- 记忆库采用简单 FIFO，未考虑经验的多样性和代表性，可能导致某些类型的操作经验被冲刷
- 仅在 Qwen2.5-VL 系列上验证路由效果，对其他 VLM 系列（如 InternVL、LLaVA-OneVision）的适用性未知
- 三层路由引入额外的工程复杂度和故障点，实际部署中的鲁棒性需要更多验证
- 缺乏对多轮任务中错误传播和恢复机制的讨论

## 相关工作与启发

- **LLM 路由**：RouteLLM、RouterBench 等在文本任务上探索了模型路由，但均为纯文本场景，AVR 将其扩展到需要视觉理解的 GUI 操作领域
- **CUA 系统**：OpenAI CUA、Claude Computer Use、OS-Atlas 等定义了 CUA 的基本范式，AVR 从系统效率角度提出优化
- **GUI 理解**：ScreenSpot、OmniACT 等 benchmark 揭示了 VLM 在 GUI 理解上的差异，为 AVR 的路由策略提供了实证基础
- **MoE 与混合推理**：AVR 的思路与 MoE 的专家路由有相似之处，但在模型级别而非层级别做路由，粒度更粗但更实用
- **启发**：这种"先评估难度再分配资源"的范式可以推广到所有 Agent 场景——不仅是 CUA，代码 Agent、数据分析 Agent 等都存在类似的"简单任务占多数"的分布特征

## 评分

- 新颖性: ⭐⭐⭐⭐ — 多层路由思路不算全新，但在 CUA 场景的应用和记忆注入的设计有创新
- 实验充分度: ⭐⭐⭐⭐ — 两个主流 benchmark 验证，消融完整，但缺少更多 VLM 组合的对比
- 写作质量: ⭐⭐⭐⭐ — 动机阐述清晰，系统设计讲解直观，但评估器的训练细节偏少
- 价值: ⭐⭐⭐⭐⭐ — 直击 CUA 成本痛点，78% 的成本降低非常实用，对 Agent 部署有重要参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] "Are We Done Yet?": A Vision-Based Judge for Autonomous Task Completion of Computer Use Agents](../../AAAI2026/multimodal_vlm/are_we_done_yet_a_vision-based_judge_for_autonomous_task_completion_of_computer_.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[ICML 2026\] AgentHijack: Benchmarking Computer Use Agent Robustness to Common Environment Corruptions](../../ICML2026/multimodal_vlm/agenthijack_benchmarking_computer_use_agent_robustness_to_common_environment_cor.md)
- [\[ACL 2026\] AdaTooler-V: Adaptive Tool-Use for Images and Videos](../../ACL2026/multimodal_vlm/adatooler-v_adaptive_tool-use_for_images_and_videos.md)
- [\[CVPR 2026\] VL-RouterBench: A Benchmark for Vision-Language Model Routing](vl-routerbench_a_benchmark_for_vision-language_model_routing.md)

</div>

<!-- RELATED:END -->
