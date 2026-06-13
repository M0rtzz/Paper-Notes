---
title: >-
  [论文解读] GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents
description: >-
  [CVPR 2026][LLM Agent][GUI Agent] 提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。
tags:
  - "CVPR 2026"
  - "LLM Agent"
  - "GUI Agent"
  - "中文移动端基准"
  - "多模态评估"
  - "分层诊断"
  - "手机交互"
---

# GUI-CEval: A Hierarchical and Comprehensive Chinese Benchmark for Mobile GUI Agents

**会议**: CVPR 2026  
**arXiv**: [2603.15039](https://arxiv.org/abs/2603.15039)  
**代码**: 待公开  
**领域**: LLM Agent  
**关键词**: GUI Agent, 中文移动端基准, 多模态评估, 分层诊断, 手机交互

## 一句话总结
提出 GUI-CEval，首个面向中文移动端 GUI Agent 的综合评测基准，覆盖 201 个主流中文 App、4 种设备类型，采用"基础能力+应用能力"两层结构从感知、规划、反思、执行、评估五个维度进行细粒度诊断，在 20 个代表性模型上的实验揭示当前模型在反思和自我评估方面仍有明显短板。

## 研究背景与动机
**领域现状**：MLLM 的快速发展催生了具有视觉感知、跨模态推理和交互控制能力的移动端 GUI Agent，已有多个基准如 ScreenSpot、AndroidControl、AndroidWorld 等推动了该领域发展。

**现有痛点**：(i) 语言偏向——绝大多数基准以英文为主，无法反映中文移动生态的语言和交互特点；(ii) 场景不统一——数据来自不同平台，缺乏专门针对移动端的聚焦评测；(iii) 任务片面——现有基准要么只测 UI 元素定位，要么只测离线 Agent 成功率，缺乏统一的全链路能力评估；(iv) 数据真实性不足——自动化采集和验证忽略了真实用户意图。

**核心矛盾**：缺乏一个统一、细粒度、可诊断的中文移动端 GUI Agent 评测框架，无法系统性地定位模型从感知到执行全链路中的薄弱环节。

**本文目标** 构建第一个中文移动端 GUI Agent 综合基准，同时评估原子能力和端到端应用能力。

**切入角度**：采用分层设计（基础+应用），沿着 Agent 完整工作流定义五大核心维度，所有数据在真实设备上人工采集和验证。

**核心 idea**：通过两层结构 + 五维度设计，在真实中文移动环境中实现从原子技能到端到端执行的全面、可诊断评测。

## 方法详解

### 整体框架
GUI-CEval 想回答一个被现有英文基准回避的问题：一个 GUI Agent 从"看懂中文界面"到"真机里把任务做完"，到底在哪一环掉链子？为此它沿着 Agent 的完整工作流定义五个能力维度——感知、规划、反思、执行、评估，并把它们分摊到上下两层，让原子技能和端到端执行共享同一批真实截图与 App，从而把失败精确归因到某个能力维度。基础任务层（Foundation）用 4,194 个多模态 QA 单独诊断其中四类**原子技能**（感知、规划、反思、评估）；应用任务层（Application）用 4,028 个 Agent 任务统一 GUI Grounding、Offline Agent、Online Agent 三种场景，考察从看到做的**全链路执行**——第五个维度"执行"正是在这一层端到端考察的。整套数据覆盖 201 个主流中文 App、4 种真实设备类型（手机、平板、折叠屏等），全部在真机上人工采集。

### 关键设计

**1. 基础能力层：把四类原子技能拆成可单独诊断的多模态 QA**

现有基准要么只测元素定位、要么只测离线成功率，一个综合得分掩盖了模型究竟弱在哪。GUI-CEval 沿 Agent 完整工作流定义五个维度（感知、规划、反思、执行、评估），其中四类原子技能——感知、规划、反思、评估——在基础层用多模态 QA 单独打分（第五个维度"执行"留到下面的应用层端到端考察），于是一张能力雷达图就能指出短板所在。**感知（Perception）**考察模型对当前 App、页面、可操作控件和屏幕文本的识别，并把元素定位进一步拆成外观、空间、功能三类线索；为了避免模型对坐标数字格式过度敏感，定位题改用 Set-of-Marks (SoM) 候选框作答，模型只需选编号而非吐坐标。**规划（Planning）**给定高级指令和当前截图要模型预测下一步动作，又细分为任务规划（理解并分解全局目标）和动作决策（从可行动作空间里选一个）。**反思（Reflection）**最能区分 Agent 的自主性，分短程反思（判断刚做的单步动作对不对）和长程反思（在整条轨迹里揪出错误或冗余步骤）。**评估（Evaluation）**则反过来考察模型当裁判的能力：判断任务是否完成、反推执行意图、甚至从被打乱的截图序列里重排出正确步骤顺序。

**2. 三级应用任务：用噪声递增的三种场景隔离"能不能做对"和"真机里能不能做成"**

光有原子技能不等于能端到端跑完任务，而真机里的弹窗广告又会把规划能力和环境鲁棒性搅在一起。GUI-CEval 按环境噪声从低到高摆出三档应用场景，逐层逼近真实使用。**GUI Grounding** 最纯粹，给一张截图和一句自然语言指令，让模型选出正确交互位置，只测点击精度。**Offline Agent** 在预先录好的静态快照里回放交互，模型迭代预测下一步动作及参数——因为快照固定、消除了系统噪声，这一档能把规划和决策能力单独隔离出来评测。**Online Agent** 直接把 Agent 放到真实设备上跑，弹窗、广告、权限提示、网络波动全都暴露出来，考的就是模型在真实中文移动环境里的综合存活能力。三档共享同一批 App 与页面，于是"Grounding 能定位 ≠ Online 能做成"这类落差可以被直接观察到。

**3. 真机人工采集 + 三阶段质控：用可复现的真实数据堵住自动化采集的模板偏差与泄露**

自动化脚本批量抓数据省事，但会引入模板化偏差、还可能与训练语料泄露重叠，让基准失真。GUI-CEval 索性所有数据都在真实移动设备上由人工采集，包含单图采集（截图配对其 XML 布局）和轨迹采集（每条 10–20 步的真实指令执行记录）。采完再过三道质控关卡：先人工交叉检查标注，再上自动化检验——用一强一弱两个模型双重验证（强模型答得对、弱模型答错的题才被认为难度合适、答案可靠），最后对全集 20% 抽样做人工复核。这套"真机采集 + 强弱模型互验 + 抽样复核"保证了数据的真实性和可复现性，也让基准更难被现成模型靠记忆刷分。

## 实验关键数据

### 主实验
在 20 个代表性模型（通用 MLLM + GUI 专用模型 + 多 Agent 系统）上进行评测：

| 模型 | 感知 | 规划 | 反思 | 评估 | Grounding | Offline | Online | 平均 |
|------|------|------|------|------|-----------|---------|--------|------|
| Qwen2.5-VL-72B | 82.28 | 66.68 | 21.01 | 40.09 | 88.10 | 70.30 | 26.94 | 61.41 |
| UI-TARS-72B-SFT | 70.28 | 45.49 | 10.97 | 41.08 | 90.10 | 79.40 | 33.33 | 56.22 |
| Qwen2.5-VL-32B | 75.25 | 63.57 | 19.24 | 49.24 | 88.70 | 70.00 | 31.87 | 55.46 |
| MIMO-VL-RL | 73.56 | 51.67 | 15.55 | 46.78 | 90.00 | 60.80 | 17.22 | 49.67 |
| GPT-4o | 37.55 | 26.06 | 13.60 | 35.72 | 35.10 | 25.50 | 0.83 | 27.69 |

### 关键发现

| 维度 | 最佳模型 | 最佳得分 | 说明 |
|------|---------|---------|------|
| 反思 | Qwen2.5-VL-72B | 21.01 | 所有模型反思能力都很弱，最好也只有 21% |
| Online Agent | UI-TARS-72B-SFT | 33.33 | 在线执行能力整体偏低 |
| Grounding | UI-TARS-72B-SFT | 90.10 | GUI 定位能力相对较好 |

### 关键发现
- **反思是最大短板**：所有模型在 Reflection 维度的得分都明显偏低（最高仅 21%），说明当前模型在自检和纠错方面严重不足
- **GUI 专用模型 vs 通用模型交叉优势**：UI-TARS 在 Grounding 和 Offline/Online Agent 上更强，但通用模型（Qwen2.5-VL）在感知、规划、反思方面更优
- **在线 Agent 挑战巨大**：即使最好的模型在 Online Agent 上也只有约 33%，说明真实环境中的弹窗、广告等干扰仍是严峻挑战
- **中文 GUI 与英文差距**：GPT-4o 在本基准上表现远低于其在英文基准上的水平，凸显中文移动端的独特挑战

## 亮点与洞察
- **两层五维度设计**非常系统化——将 Agent 能力从原子级到端到端进行了清晰的层次化分解，既能评估整体也能定位短板，比现有基准更具诊断价值
- **基础+应用任务共享同一 App 和页面**的设计巧妙——可以直接关联定位能力和执行结果，精确归因失败原因
- **反思维度的引入**很有前瞻性：反思是 Agent 自主性的关键能力，但几乎所有模型都很弱，这为未来研究指明了方向

## 局限与展望
- 仅覆盖中文移动端，缺少跨语言对比（同一 App 的中英文版对比会更有说服力）
- 201 个 App 的覆盖面虽广，但可能仍遗漏某些垂直领域（如金融、医疗 App 的复杂交互）
- Online Agent 评测受限于"不需要登录"的常用功能，真实场景中很多关键功能需要登录后才能评估
- 数据集规模（8222 条）相对于英文大规模基准仍偏小

## 相关工作与启发
- **vs ScreenSpot/ScreenSpot Pro**：仅评估 GUI Grounding，而 GUI-CEval 涵盖完整的感知→执行链路
- **vs AndroidWorld**：仅 116 个在线任务且英文，GUI-CEval 有 4028 个应用任务且专注中文
- **vs MMBench-GUI**：跨平台但无在线Agent评测，GUI-CEval 在移动端更深入

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个中文移动端GUI Agent综合基准，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 20个模型、47种配置的全面评测
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务定义明确
- 价值: ⭐⭐⭐⭐ 对中文移动端Agent研究有很好的推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] EchoTrail-GUI: Building Actionable Memory for GUI Agents via Critic-Guided Self-Exploration](echotrail-gui_building_actionable_memory_for_gui_agents_via_critic-guided_self-e.md)
- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[CVPR 2026\] HATS: Hardness-Aware Trajectory Synthesis for GUI Agents](hats_hardness-aware_trajectory_synthesis_for_gui_agents.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ICLR 2026\] M²-Miner: Multi-Agent Enhanced MCTS for Mobile GUI Agent Data Mining](../../ICLR2026/llm_agent/m2-miner_multi-agent_enhanced_mcts_for_mobile_gui_agent_data_mining.md)

</div>

<!-- RELATED:END -->
