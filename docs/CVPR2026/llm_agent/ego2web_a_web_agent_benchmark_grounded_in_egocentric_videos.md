---
title: >-
  [论文解读] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos
description: >-
  [CVPR 2026][LLM Agent][Web Agent] 提出 Ego2Web，首个将第一人称视频感知与 Web 代理执行相结合的基准测试，配套半自动数据构建流程和 Ego2WebJudge 自动评测框架，实验揭示当前最强 Agent 在真实视觉感知到在线行动的跨模态迁移上仍有巨大差距…
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "Web Agent"
  - "第一人称视频"
  - "多模态基准"
  - "跨模态迁移"
  - "自动评测"
---

# Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos

**会议**: CVPR 2026  
**arXiv**: [2603.22529](https://arxiv.org/abs/2603.22529)  
**代码**: [https://github.com/Yui010206/Ego2Web](https://github.com/Yui010206/Ego2Web)  
**领域**: Agent  
**关键词**: Web Agent, 第一人称视频, 多模态基准, 跨模态迁移, 自动评测

## 一句话总结

提出 Ego2Web，首个将第一人称视频感知与 Web 代理执行相结合的基准测试，配套半自动数据构建流程和 Ego2WebJudge 自动评测框架，实验揭示当前最强 Agent 在真实视觉感知到在线行动的跨模态迁移上仍有巨大差距，最高仅 48.2% 成功率。

## 研究背景与动机

**领域现状**：多模态 AI Agent 正在快速发展，从简单的对话式助手向能够在真实网页环境中执行操作（如购物、搜索、查询地图）的方向演进。当前已有多个 Web Agent 基准（如 WebArena、MiniWoB++、Mind2Web 等）用于评测 Agent 在在线环境中的任务完成能力。

**现有痛点**：现有 Web Agent 基准存在一个根本性限制——它们完全聚焦于 Web 端的交互和感知，缺乏与用户真实物理环境的关联。这意味着一个关键场景无法评测：当 Agent 需要先通过第一人称视觉（如 AR 眼镜）识别用户周围环境中的物体，然后在线完成相关任务（如看到一个零食后在 Amazon 上搜索购买）。这种从"看见"到"在线执行"的桥接能力是未来 AI 助手的核心需求。

**核心矛盾**：当前 Web Agent 的评测只考虑了数字世界内的能力，完全忽略了 Agent 从物理世界获取视觉线索并将其转化为数字世界行动的能力。这导致我们无法了解当前模型在"见到→理解→行动"完整链路上的真实水平。

**本文目标**：构建一个将第一人称视频感知与 Web 行动执行相结合的基准，系统评测 Agent 的视觉理解、任务规划和在线交互能力。

**切入角度**：利用已有的大规模第一人称视频数据集（如 Ego4D），结合 VLM+LLM 的自动数据生成流水线和人工验证，构建高质量的视频-Web 任务对。

**核心 idea**：将第一人称视频中的视觉证据（如品牌、物体、动作）作为 grounding 信息，要求 Agent 在真实 Web 环境中完成相关任务，从而评测跨物理-数字世界的 Agent 能力。

## 方法详解

### 整体框架

Ego2Web 的核心是一个包含三步的系统：(1) **半自动数据构建流水线**：从第一人称视频生成视觉元数据，再由 LLM 生成 Web 任务指令，最后经人工验证和精炼；(2) **基准数据集**：包含多种 Web 任务类型（电商、媒体检索、知识查询、本地/地图服务等）的视频-任务对；(3) **Ego2WebJudge 评测框架**：基于 LLM-as-a-Judge 的自动评测方法，用于可扩展的在线评测。

### 关键设计

1. **半自动数据生成流水线（Semi-Automatic Data Pipeline）**:

    - 功能：将原始第一人称视频转化为高质量的"视频+Web任务"配对
    - 核心思路：首先用 VLM（如 Gemini）对第一人称视频进行结构化解析，生成片段级字幕和视觉元数据（识别物体、品牌、动作等）；然后用 LLM 基于这些视觉元数据，在真实活跃网站（如 Amazon、YouTube、Wikipedia）上生成 Web 任务指令；最后由人工标注员验证每个样本的视觉 grounding 准确性、Web 可行性和指令质量
    - 设计动机：纯手工标注成本过高，纯自动生成质量不可控。半自动流水线在效率和质量之间取得平衡，确保每个样本都有真实的视觉依据和可执行的 Web 任务

2. **多类型 Web 任务设计**:

    - 功能：覆盖日常 AI 助手需要处理的多种 Web 交互场景
    - 核心思路：任务被分为四大类——电商（如看到零食后搜索购买）、媒体检索（如看到健身动作后搜索教程视频）、知识查询（如看到大学名后查询录取信息）、本地/地图服务（如看到商店后搜索导航路线）。每种任务都要求 Agent 先从视频中提取关键视觉证据，再在 Web 上完成相应操作
    - 设计动机：不同任务类型对 Agent 的能力要求不同——电商需要精细的物体识别，媒体检索需要动作理解，知识查询需要文本识别，地图服务需要空间定位。多类型评测可以全面暴露 Agent 的能力短板

3. **Ego2WebJudge 自动评测方法**:

    - 功能：在实时 Web 环境中实现可扩展的自动化评测
    - 核心思路：给定任务指令、Agent 的操作轨迹、Web 截图和视频中标注的视觉证据，Ego2WebJudge 首先提取关键成功标准（success criteria），然后从 Agent 的 Web 操作轨迹中选取最相关的截图，最后判定 Agent 是否正确且一致地完成了任务。与简单的 URL/文本匹配不同，Ego2WebJudge 考虑了视觉证据与 Web 内容的一致性
    - 设计动机：在实时 Web 环境中评测 Agent，传统的精确匹配方法（如 URL 匹配）过于脆弱；而人工评测不可扩展。Ego2WebJudge 达到约 84% 的人类判断一致率，远高于现有评测方法

### 损失函数 / 训练策略

Ego2Web 是一个评测基准而非训练方法，因此不涉及训练损失。评测使用**成功率（Success Rate, SR）**作为核心指标，由 Ego2WebJudge 自动判定。

## 实验关键数据

### 主实验

| Agent (模型) | 电商 SR | 媒体检索 SR | 知识查询 SR | 地图 SR | 总体 SR |
|--------|------|------|----------|------|------|
| Qwen3-VL-Flash | 21.7 | 30.1 | 50.0 | 23.1 | 29.0 |
| GPT-4o | 26.9 | 30.3 | 63.0 | 22.5 | 34.6 |
| Gemini-2.5 Pro | 38.2 | 50.7 | 75.0 | 48.3 | 48.2 |
| 人类评测 | - | - | - | - | 58.6 |

### 消融实验（视觉感知影响）

| 视频输入 | 详细描述 | 电商 | 知识查询 | 总体 SR |
|------|---------|------|------|------|
| ✗ | ✗ | 2.6 | 5.4 | 4.4 |
| ✗ | ✓ | 13.0 | 39.1 | 23.6 |
| ✓ | ✗ | 38.2 | 75.0 | 48.2 |

### 关键发现

- **当前最强 Agent 也远未完美**：Gemini-2.5 Pro 作为最佳模型仅达到 48.2% SR，人类评测也仅 58.6%（因为部分任务本身有难度），展示了巨大的提升空间
- **原始视频远优于文本描述**：直接输入视频比用 VLM 生成文本描述后输入效果好一倍以上（48.2% vs 23.6%），说明视觉 grounding 必须来自原始视觉信号
- **错误分析**：36% 的失败来自物体误识别，18% 来自时序/动作误解，16% 来自跨模态检索失败，证明视觉感知是当前 Agent 的首要瓶颈
- **电商和地图任务最难**：需要精细的视觉识别和空间理解，当前 Agent 在这两类任务上表现最差

## 亮点与洞察

- **首创性的物理-数字世界桥接基准**：Ego2Web 填补了现有 Web Agent 评测中"视觉感知→在线行动"链路的空白。这种设计反映了未来 AR/智能助手的真实使用场景，具有前瞻性
- **Ego2WebJudge 评测方案**：84% 的人类一致率使其成为可靠的自动评测工具，避免了在实时 Web 环境中人工评测的巨大成本。这一框架可以迁移到其他需要在线评测的 Agent 任务
- **视觉感知瓶颈的定量揭示**：实验清楚地展示了"看→理解→行动"链路中每一步的瓶颈，为未来研究提供了明确的改进方向

## 局限与展望

- 数据规模相对有限，尚未覆盖所有日常 Web 任务场景（如社交媒体操作、日程管理等）
- 基准依赖于实时 Web 环境，网站变更可能影响评测的可复现性
- 当前仅评测了已有的通用 Agent，尚未设计针对 Ego2Web 任务特点的专用 Agent 架构
- 未来可扩展到多轮对话场景（如用户在观看视频后提出后续需求）和多模态 Web 交互（如语音指令+视觉感知）

## 相关工作与启发

- **vs WebArena**：WebArena 聚焦纯 Web 端任务执行，没有真实世界的视觉输入；Ego2Web 引入了从第一人称视频到 Web 行动的完整链路
- **vs Ego4D**：Ego4D 是纯视频理解基准，不涉及在线行动执行；Ego2Web 利用 Ego4D 的视频数据但要求 Agent 在真实 Web 上完成任务
- **vs Mind2Web**：Mind2Web 使用静态 Web 页面截图，Ego2Web 则在实时 Web 环境中评测，更接近真实应用场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个桥接第一人称视频与 Web Agent 执行的基准，填补重要空白
- 实验充分度: ⭐⭐⭐⭐ 多模型评测+输入消融+错误分析，但缺少更多 Agent 架构的对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、问题定义精准、数据流程描述详细
- 价值: ⭐⭐⭐⭐⭐ 对 AI Agent 和具身智能社区有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ST-WebAgentBench: A Benchmark for Evaluating Safety and Trustworthiness in Web Agents](../../ICLR2026/llm_agent/st-webagentbench_a_benchmark_for_evaluating_safety_and_trustworthiness_in_web_ag.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[CVPR 2026\] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents](gui-ceval_a_hierarchical_and_comprehensive_chinese_benchmark_for_mobile_gui_agen.md)
- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[ICLR 2026\] VideoMind: A Chain-of-LoRA Agent for Temporal-Grounded Video Reasoning](../../ICLR2026/llm_agent/videomind_a_chain-of-lora_agent_for_temporal-grounded_video_reasoning.md)

</div>

<!-- RELATED:END -->
