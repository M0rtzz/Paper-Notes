---
title: >-
  [论文解读] Bias in ??#的
description: >-
  [NeurIPS 2025][AI安全][VLM偏见] 这篇论文不再用合成图或封闭式选择题测偏见，而是用真实新闻图片中的社会线索来问开放式问题，再让 GPT-4o 作为评判员衡量回答的准确性、偏见和忠实度，最终证明很多 VLM 即使“看图很准”，依然会在性别、职业和种族线索上偷偷补进刻板印象。
tags:
  - NeurIPS 2025
  - AI安全
  - VLM偏见
  - ---
---

# Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment

**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](https://arxiv.org/abs/2509.19659)  
**代码**: 有  
**领域**: AI安全 / 多模态公平性 / VLM 评测  
**关键词**: VLM 偏见, 社会线索, 新闻图片, LLM-as-judge, 忠实度, 刻板印象

## 一句话总结

这篇论文不再用合成图或封闭式选择题测偏见，而是用真实新闻图片中的社会线索来问开放式问题，再让 GPT-4o 作为评判员衡量回答的准确性、偏见和忠实度，最终证明很多 VLM 即使“看图很准”，依然会在性别、职业和种族线索上偷偷补进刻板印象。

## 研究背景与动机

多模态大模型已经被广泛用于图像问答、视觉助手、内容审核、搜索与辅助决策。

一旦模型开始同时理解图像和文字，偏见问题就不再只是文本提示里的偏见，而会变成“图像中的社会线索如何触发模型隐含刻板印象”的问题。

现实图片里最容易触发这类偏见的线索包括年龄、性别、种族、职业、服饰、运动场景和社会角色。

过去很多偏见评测其实并没有真正碰到这个难点。

有的工作只测文本 LLM。

有的工作用人工合成 caption 或模板化提示。

有的工作把任务做成分类题或 multiple-choice，这固然便于打分，但会把模型真实的开放式生成偏差压扁。

还有一类工作虽然关注视觉偏见，但通常把“偏见”和“是否忠实于图片内容”分开讨论。

这会带来一个方法论缺陷。

如果模型回答错了，它是因为视觉 grounding 差，还是因为社会刻板印象太强，常常分不清。

作者认为，真正值得测的是三者联动。

第一，模型有没有回答对。

第二，模型有没有把答案建立在图片能支持的证据上。

第三，模型会不会在证据不足时自行补入社会属性假设。

要做到这点，数据必须尽可能接近真实世界。

因此本文放弃合成设置，转而从 Google News RSS 抓取真实新闻图片。

新闻图像之所以重要，是因为它天然带有高密度社会语境。

同一张图里可能同时出现服饰、身份、职业背景、年龄线索、体育场景、地理或政治语境。

这类复杂输入更像未来 VLM 的实际部署环境，而不是实验室里的干净模板。

此外，作者还强调一个经常被忽略的问题。

偏见不是简单地和能力负相关。

一个模型越会看图，未必越公平。

它也可能正因为“会从视觉里抽取细微信号”，所以更容易把这些信号连接到有害先验。

论文最终验证的就是这一点：高 faithfulness 不等于低 bias。

## 方法详解

### 整体框架

这篇论文的核心不是提出新 VLM，而是提出新的 benchmark 和评估流水线。

整个流程分两层。

第一层是数据构建。

作者从 2024 年 7 月的 Google News RSS 中收集新闻图片，主题覆盖医疗、气候、教育、外交、社会正义、枪支、贫富差距、民主、科技和环境等多个议题。

为了减少来源噪声，他们对新闻源做了 whitelist 过滤，只保留 CNN、BBC、The New York Times、The Guardian、CBS News、ABC News、Fox News、Al Jazeera、Reuters、Associated Press、Bloomberg、USA Today 等主流媒体。

然后去重、筛样，并为每张图设计开放式问题。

最终得到 1343 个 image-question pair。

第二层是评估。

对每个样本，作者让 VLM 输出结构化 JSON，包括 answer 和 2 到 3 句 rationale。

这样做的好处是控制输出长度和风格，减少不同模型啰嗦程度不一样带来的评估噪声。

随后再把图片、问题和模型输出交给 GPT-4o judge，根据一套 rubric 评估准确性、偏见和忠实度，并辅以人工验证。

### 关键设计

1. **真实新闻图像而非合成或 caption 数据**

    - 功能：让 benchmark 更接近真实部署场景。
    - 核心思路：直接从新闻图片中挖掘社会线索，用开放式问答去触发模型的解释和归因习惯。
    - 设计动机：如果只用模板化或合成数据，模型容易被规则化题目“驯化”，不一定暴露真实偏见。

2. **多维社会属性标注**

    - 功能：让偏见分析不只停留在 gender 一维。
    - 核心思路：每张图都标注 Age、Gender、Race/Ethnicity、Occupation、Sport 等属性。论文正文写的是 5 名训练过的 annotator 复核并多数票裁决，附录还提到有 10 名跨学科专家对社会标签进行更广义校验。
    - 设计动机：多属性标签可以看出模型在哪一类社会线索上最不稳，而不是只汇总一个平均 bias 分数。

3. **开放式输出 + JSON 约束**

    - 功能：保留开放式生成的真实风险，同时控制评估格式。
    - 核心思路：模型不是选 ABCD，而是自由回答；但必须按 JSON 返回 answer 和 rationale。
    - 设计动机：开放式设置能更真实地暴露 stereotype completion，而 JSON 可以减小 judge 输入的格式方差。

4. **LLM-as-judge 与 human verification 结合**

    - 功能：把大规模评测变得可执行，同时尽量降低 judge 自身偏差。
    - 核心思路：采用 GPT-4o 作为 judge，输入图片、问题和模型回答，并给出严格规则，要求惩罚所有没有视觉证据支撑的人口统计学推断，同时鼓励显式表达不确定性。
    - 设计动机：人工逐条打分成本高，纯自动打分又容易漂。LLM judge 加部分人工校验，是当前这个问题上比较现实的折中。

5. **把 bias 与 faithfulness 联合看待**

    - 功能：区分“看不懂图”与“看懂了还在乱推断”。
    - 核心思路：论文同时报告 Accuracy、Bias、Faithfulness 三个维度。
    - 设计动机：如果只看 bias，模型只要回答得非常保守、非常空泛，就可能显得偏见低；但那不等于它真的更可信。

### 损失函数 / 训练策略

本文是 benchmark 论文，没有训练新模型。

但评估协议本身相当重要，几乎等同于“实验训练设置”。

作者默认采用 zero-shot prompt，temperature 设为 0 保证确定性，同时也在 0.2 下做鲁棒性 sweep。

top-p 设为 1，最大 token 数为 128。

few-shot 版本则覆盖不同属性类型，并要求 rationale 只能引用可见证据。

这套协议的思想很明确。

如果任务是审计偏见，就不能允许模型靠超长解释和冗余文本模糊责任。

同时 judge rubric 也特别强调，对没有图像证据的人口统计推断要明确扣分。

## 实验关键数据

### 主实验

作者评测了 15 个开源和商业 VLM，包括 Gemini 2.0、Janus-Pro-7B、InternVL2.5-8B、GLM-4V-9B、Qwen2.5-VL-7B、LLaMA 3.2 11B Vision、Phi-3.5 Vision、CogVLM2-19B、Molmo-7B、PaliGemma、LLaVA v1.6 7B/13B、MAGMA、Phi-4、Aya Vision 8B。

总体结果里最值得关注的不是谁单项最高，而是三项指标之间的张力。

| 模型 | Accuracy ↑ | Bias ↓ | Faithfulness ↑ | 结论 |
|------|------------|--------|----------------|------|
| Gemini 2.0 | 85.97 | 15.19 | 78.96 | 综合能力强，但偏见并不低 |
| Janus-Pro 7B | 82.02 | 16.79 | 78.68 | 生成扎实，但 bias 偏高 |
| Qwen2.5-VL 7B | 71.18 | **9.46** | 68.98 | 偏见最低之一，但答案更保守 |
| Phi-4 | 80.00 | 17.10 | **81.67** | 忠实度最高，却仍有明显偏见 |
| Aya Vision 8B | 83.76 | 9.84 | 56.78 | 低偏见但忠实度明显不足 |

这张表直接支持作者最核心的论点。

Phi-4 的 faithfulness 最高，却没有最低 bias。

Qwen2.5-VL 的 bias 更低，但准确率和忠实度都弱一些。

也就是说，避免 stereotype 并不只是“把模型做大、把 grounding 做强”就自然会发生的事情。

### 消融实验

这篇论文的“消融”主要体现在属性级分析，而不是网络模块拆除。

作者按照年龄、性别、种族、运动、职业五类社会属性分别报告指标，从而看出偏见最容易在哪些信号上被触发。

| 属性 | 代表模型与结果 | 论文观察 | 含义 |
|------|---------------|---------|------|
| Age | Janus-Pro 准确率 88.8，Phi-4 准确率 75.5 | 年龄相对更容易识别 | 视觉线索较直接 |
| Gender | Gemini Bias 19.2，LLaMA 3.2 Bias 21.8 | 性别偏见显著 | 说明模型会主动补全社会角色假设 |
| Race | 多数模型准确率低于 70 到 75 左右 | 最难答准 | 证据本就复杂且敏感 |
| Sports | Gemini 准确率 86.9，Phi-4 准确率 78.4 | 体育场景较容易 grounding | 视觉上下文更明确 |
| Occupation | Phi-4 准确率 92.0，但职业相关 bias 仍高 | 职业最容易又准又偏 | “高能力但高偏见”最典型领域 |

作者特别指出，gender 和 occupation 是最敏感的两类属性。

这很符合直觉。

因为职业判断往往会被服饰、工具、站位、社会角色暗示触发，而性别判断又常与职业角色联动，模型很容易把“可见线索”扩展成“社会先验”。

| 评估组件 | 具体做法 | 它解决的问题 |
|---------|---------|---------------|
| 新闻源白名单 | 主流媒体来源 + 去重 | 降低垃圾数据和图文不一致噪声 |
| 开放式问答 | 不做多选题 | 真实暴露 stereotype completion |
| JSON 输出 | answer + rationale | 控制格式，便于 judge 比较 |
| GPT-4o Judge | 评分 bias / relevance / faithfulness | 扩大规模且保持一致性 |
| 人工校验 | 训练过的 annotator 和专家复核 | 降低单一 judge 的系统偏差 |

### 关键发现

- 视觉上下文会系统性改变 VLM 输出，而不是只起到辅助作用。
- 高忠实度不等于低偏见，Phi-4 和 Janus-Pro 等模型就是典型例子。
- 低偏见有时来自“少说”，而不是“更公平地说”，这在 Qwen2.5-VL 和 Aya Vision 的表现里很明显。
- 种族属性最难答准，职业与性别最容易触发 stereotype priors。
- 真实新闻场景比合成模板更容易暴露多重社会线索叠加带来的偏见问题。

## 亮点与洞察

- **用真实新闻图像替代合成 benchmark**。这让评测第一次真正接近部署环境，而不是只停留在受控实验室问题。
- **把 faithfulness 和 bias 捆在一起看**。这点非常关键，因为很多所谓“公平回答”只是更保守、更空洞，不一定更可靠。
- **LLM-as-judge 的使用场景是合理的**。因为这里的偏见不是简单字符串匹配，而是要判断“是否超出了视觉证据边界”，这恰恰是大型 judge 比较擅长的地方。
- **属性级分析比总分更有用**。职业、性别、种族在风险模式上完全不同，平均分会掩盖很多细节。
- **这篇论文提醒我们，多模态安全不是文本安全的简单延伸**。图像中的社会线索会引入新的触发机制。

## 局限与展望

第一，数据规模仍然不大。

1343 个样本对于 benchmark 论文已经不算小，但若要覆盖更广泛的视觉社会情境、跨文化语义和更细粒度身份类别，仍明显不足。

第二，数据域集中在新闻图片。

新闻图像本身带有媒体选择偏差、事件偏差和地区偏差，因此 benchmark 的社会分布并不等于世界本身。

第三，人口统计学标签采用了离散类别。

这对评测很方便，但现实身份和职业语义远比这些标签更连续、更模糊、更受上下文影响。

第四，judge 仍然是 GPT-4o。

即便有人工验证，judge 自身的文化偏好和训练偏差也可能高估或低估某些细微伤害。

第五，论文主要评估 zero-shot 和少量 few-shot 场景。

模型经过 instruction tuning、RLHF 或特定安全对齐后，偏见模式可能会显著变化。

未来可沿几个方向扩展。

一个是加入非西方媒体和多语言新闻图像。

一个是引入视频、连续图文报道和时间演化场景，测试偏见是否会随上下文累积。

一个是把 judge 从单一 LLM 扩展为 human-in-the-loop 或 adversarial auditing 体系。

还有一个方向是把 benchmark 和 mitigation 结合起来，不只测谁更偏，还研究怎样让模型在保持信息量的同时减少无根据推断。

## 相关工作与启发

- **vs 文本公平 benchmark**：文本任务主要测 prompt 中显式词汇触发的偏见，而本文测的是视觉社会线索触发的隐式联想，问题本质更复杂。
- **vs VL-Stereoset、VisoGender 等工作**：这些工作更偏模板化、单属性或封闭任务；本文则转向开放式、多属性、真实新闻语境。
- **vs 单独报告准确率或 grounding 的 VLM benchmark**：本文证明只看能力指标会漏掉大量现实风险，因为“更会看图”并不自动推出“更不偏见”。
- **对我自己的启发**：多模态安全评测必须同时问三个问题，能不能答对、有没有根据、有没有多想。少一个都不够。
- **实务价值**：如果 VLM 将被用于教育、媒体分析、辅助问答或公共服务，这类 benchmark 应该成为上线前审计的一部分。

## 评分

- 新颖性: ⭐⭐⭐⭐☆ 真实新闻图像 + LLM judge + 联合评估 bias 与 faithfulness 的组合很有针对性。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 15 个模型、5 类属性、真实数据来源，已经很扎实，但跨语言和跨文化覆盖还不够。
- 写作质量: ⭐⭐⭐⭐☆ 论文结构清楚，问题定义明确，结果解读也比较直接。
- 价值: ⭐⭐⭐⭐⭐ 对多模态系统安全评测非常重要，尤其是它证明了“忠实度高不代表更公平”。
---
title: >-
  [论文解读] Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment
description: >-
  [NeurIPS 2025][多模态VLM][公平性] 构建 1,343 个真实新闻图片-问答对偏见评估基准，标注年龄/性别/种族/职业/运动等社会属性，用 GPT-4o 作为评判员评估 15 个 VLM 的偏见表现，揭示高忠实度不等于低偏见且性别与职业偏见最严重。
tags:
  - NeurIPS 2025
  - 多模态VLM
  - 公平性
  - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
---

# Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment

**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](https://arxiv.org/abs/2509.19659)  
**代码**: 有  
**领域**: 多模态VLM / AI公平性  
**关键词**: VLM偏见,---  
title: >-
  [论文解读] Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment
description: >-
  [NeurIPS 2025][多模态VLM][公平性] ?i??/性别/?escription: >-
  [NeurIPS 2025][多模态VLM][公平性] 构建 1,343 个真实新闻图片-问答对偏见诞?  [NeurIPS 202??ags:
  - NeurIPS 2025
  - 多模态VLM
  - 公平性
  - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
---

# Bias in the Picture: Benchmarking VLMs with Social-Cue News Images and LLM-as-Judge Assessment

**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](https://arxiv.org/abs/?? - ?? - 多模态VL??  - 公平性  
 ? - 偏见诚? - 新闻图牿?  - 社会线??  - LLM-as-judg??--

# Bias in ??

#??**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](https://arxiv.org/abs/2509.19659)  
**代?M **arXiv**: [2509.19659](h??**代码**: 有  
**领域**: 多模态VL??合成/配文设定，不使用真?*关键词**: VLM偏见,---  
title: >-
?itle: >-
  [论文解读] ?? [论??description: >-
  [NeurIPS 2025][多模态VLM][公平性] ?i??/性别/?escription: >-
  [NeurIPS 2025][多??  [NeurIPS 202?? [NeurIPS 2025][多模态VLM][公平性] 构建 1,343 个真实新闻囼?  - NeurIPS 2025
  - 多模态VLM
  - 公平性
  - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
?何系统性地改  - 公平性
 ??  - 偏见??  - 新闻图牟? - 社会线紱?  - LLM-as-judg?--

# Bias in ??#的

**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](https:/?是什么关系？

**本文要解决?*arX？** （1）系统性 ? - 偏见诚? - 新闻图牿?  - 社会线??  - LLM-as-judg??2）构建包含?# Bias in ??
#??**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](????**会耧**arXiv**: [2509.19659](https:?*代?M **arXiv**: [2509.19659](h??**代码**: 有  
*???*领域**: 多模态VL??合成/配文设定??择题）title: >-
?itle: >-
  [论文解读] ?? [论??description: >-
  [NeurIPS 2025][多?
?itle:?i ea一句话  [NeurIPS 2025][多模态VLM][公平性] ??  [NeurIPS 2025][多??  [NeurIPS 202?? [NeurIPS 2025][多模态VLM][公? - 多模态VLM
  - 公平性
  - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
?何系统性地改  - 公幖?闻图片，白名单过滤主流媒体，去重  - 社会线??  - LLM-as-judg????系统性?性 ??  - 偏见??  - 新闻图瞋
# Bias in ??#的

**会议**: NeurIPS 2025  
**arXiv**: [2509.为评判?*会议**: Neur??*arXiv**: [2509.19659](h??**本文要解决?*arX？** （1）系统性 ?###??**会议**: NeurIPS 2025  
**arXiv**: [2509.19659](????**会耧**arXiv**: [2509.19659](https:?*代?M **arXiv**: [2509.19659](h??*??*arXiv**: [2509.19659](?????*???*领域**: 多模态VL??合成/配文设定??择题）title: >-  
?itle: >-
  [论文解读] ?? [论斘。白名单过??title: >-
  [论文解读] ?? [论??description: >-
  [NeurIPS ?? [论文?  问题（探测场景理解和社会线索??title:?i ea一句??  - 公平性
  - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
?何系统性地改  - 公幖?闻图片，白名单过滤主流媒?LLM 初  - 偏见?   - 新闻图??  - 社会线細? - LLM-as-judg???何系统性在? Bias in ??#的
**会议**: NeurIPS 2025  
**arXiv**: [2509.为评判?*会议**: Neur??*arXiv**: [2509.19659](h??**本文要解决?*arX？** （1）系统模**会议**: Neur??的条件下评估  
    - ?*arXiv**: [2509.19659](????**会耧**arXiv**: [2509.19659](https:?*代?M **arXiv**: [2509.19659](h??*??*arXiv**: [2509.19659](?????*???*领? ?itle: >-
  [论文解读] ?? [论斘。白名单过??title: >-
  [论文解读] ?? [论??description: >-
  [NeurIPS ??规模和多种训练范式
    - 设计动机：标准化输出格式稳定下游 LLM 评判，JSO  [论文解读] ?? [论??description: >-
  [NeurIPS框架**

    - 功能：自动化、可规模? - 偏见评估
  - 新闻图片
  - 社会线索
  - LLM-as-judge
?何系统性地改  - 公幖?闻囤? - 新闻图牃设计的评判准  - LLM-as-judg樂何系统性圆?*会议**: NeurIPS 2025  
**arXiv**: [2509.为评判?*会议**: Neur??*arXiv**: [2509.19659](h??**本文要解决?*arX？** （1）系统模**会议**: Neur??的条件丈?**arXiv**: [2509.为?   -    - ?*arXiv**: [2509.19659](????**会耧**arXiv**: [2509.19659](https:?*代?M **arXiv**: [2509.19659](h??*??*arXiv**: [2509.19659](?????*? [论文解读] ?? [论斘。白名单过??title: >-  
  [论文解读] ?? [论??description: >-
  [NeurIPS ??规模和多种训练范式
    - 设计动机：标准化輈  [论文解?：越高越好，衡量回答是否基于? [NeurIPS ??规模和多种训练范式
  ?   - 设计动机：标准化输出格??  [NeurIPS框架**

    - 功能：自动化、可规模? - 偏见评估
  - 新闻图片
  - 社会线索
  - LL--
    - 功能：?**  - 新闻图片
  - 社会线索
  - 7B | 82.02 | 16.  - 社会线?y  - LLM-as-judg3.?何系统性?5**arXiv**: [2509.为评判?*会议**: Neur??*arXiv**: [2509.19659](h??**本文要解?en2.5-VL | 71.18 | 9.46 | 68.98 |

| LLaVA v1  [论文解读] ?? [论??description: >-
  [NeurIPS ??规模和多种训练范式
    - 设计动机：标准化輈  [论文解?：越高越好，衡量回答是否基于? [NeurIPS ??规模和多种训练范式
  ?   - 设计动机：标准化输出格??  [NeurIPS框架**

    - 功能：自动化、可规模? - 偏见评估
  - 新闻图片
  - 社会线索
  - LL--
    - ?8  [NeurIPS ??规模和多种训练范式
   1    - 设计动机：标准化輈  [论 |  ?   - 设计动机：标准化输出格??  [NeurIPS框架**

    - 功能：自动化、可规模? - 偏见评估
  - 新闻?8
    - 功能：自动化、可规模? - 偏见评估
  - 漈9.4  - 新闻图片
  - 社会线索
  - LL--
    - 功??  - 社会线??  - LL--
    - ?   - ??  - 社会线索
  - 7B | 82.02 | ??  - 7B | 82.02 ? LLaVA v1  [论文解读] ?? [论??description: >-
  [NeurIPS ??规模和多种训练范式
    - 设计动机：标准化輈  [论文解?：越高越好，衡量回答是否基于? [Neu??  [NeurIPS ??规模和多种训练范式
    - 设计??   - 设计动机：标准化輈  ??注? ?   - 设计动机：标准化输出格??  [NeurIPS框架**

    - 功???根据的回答，但仍会在种族和性别方面?    - 功能：自动化、可规模? - 偏见评估
  - 新?? - 新闻图片
  - 社会线索
  - LL--
    - ?8??  - 社会线??  - LL--
    - ??   - ?   1    - 设计动机：标准化輈  [论 |  廍
    - 功能：自动化、可规模? - 偏见评估
  - 新闻?8
    - 功能：自动化、?真实新闻  - 新闻?8
    - 功能：自动化、可规模? ??   -，新? -?片更接近实际部署场景，发现的偏见? - 社会线索
  - LL--? - LL--
    - dg    - ???展性**：用 GPT-4o 评判员替? - 7B | 82.02类标注，使偏? [NeurIPS ??规模和多种训练范式
    - 设计动机：标??设计（惩罚无视觉    - 设计动机：标准化輈  [论?   - 设计??   - 设计动机：标准化輈  ??注? ?   - 设计动机：标准化输出格??  [NeurIPS框架**

    - 功???根惨
    - 功???根据的回答，但仍会在种族和性别方面?    - 功能：自动化、可规模? - 偏见读? - 新?? - 新闻图片
  - 社会线索
  - LL--
    - ?8??  - 社会线??  - LL--
    - ??   - ???种族分为  - 社会线索
  - LL--? - LL--
    - ??   - ?    - ??   - ?   1    - 设计??    - 功能：自动化、可规模???偏见可能导致高估或? - 新闻?8
    - 功能：自动化、?真实新?*    - 功胴?    - 功能：自动化、可规模? ??   -，新??  - LL--? - LL--
    - dg    - ???展性**：用 GPT-4o 评判员替? - 7B | 82.02类标注，使偏? [NeurIPS ??规??   - dg    - ?

    - 设计动机：标??设计（惩罚无视觉    - 设计动机：标准化輈  [论?   - 设计??   - 设计动机：??
    - 功???根惨
    - 功???根据的回答，但仍会在种族和性别方面?    - 功能：自动化、可规模? - 偏见读? - 新?? - 新闻图片
  - 社会线索
 ?，生态效度更高
- *    - 功???根?X  - 社会线索
    - LL--
    - ?8??  - 社会线??  - LL--
    - ??   - ???种族分为  - 社会线索
    - LL--? - LL--
    - ??   - ?? - LL--
    - ??  ?定
-     - ??   - ???种族分为  -5)*  - LL--? - LL--
    - ??   - ?    - ?? ??    - ??   - ???   - 功能：自动化、?真实新?*    - 功胴?    - 功能：自动化、可规模? ??   -，新??  - LL--? - LL--
    -??    - dg    - ???展性**：用 GPT-4o 评判员替? - 7B | 82.02类标注，使偏? [NeurIPS ??规??   - dg    - ?

 ??
    - 设计动机：标??设计（惩罚无视觉    - 设计动机：标准化輈  [论?   - 设计??   - 设计动?     - 功???根惨
  ??署提供重要的偏见审计基准，"忠实度≠低偏见"的发现有实际指导意义

<!-- RELATED:START -->

## 相关论文

- [HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)
- [Towards Fairness Assessment of Dutch Hate Speech Detection](../../ACL2025/ai_safety/towards_fairness_assessment_of_dutch_hate_speech_detection.md)
- [Rethinking VLMs for Image Forgery Detection and Localization](../../CVPR2025/ai_safety/rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)

<!-- RELATED:END -->
