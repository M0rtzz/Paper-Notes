---
title: >-
  [论文解读] LongReward: Improving Long-context Large Language Models with AI Feedback
description: >-
  [ACL2025][LLM效率][长上下文LLM] 提出 LongReward，利用现成 LLM 从帮助性、逻辑性、忠实性和完整性四个维度为长上下文模型回复自动打分，结合 DPO 离线强化学习显著提升长上下文 SFT 模型的多维能力。
tags:
  - "ACL2025"
  - "LLM效率"
  - "长上下文LLM"
  - "强化学习"
  - "DPO"
  - "AI反馈"
  - "奖励模型"
---

# LongReward: Improving Long-context Large Language Models with AI Feedback

**会议**: ACL2025  
**arXiv**: [2410.21252](https://arxiv.org/abs/2410.21252)  
**代码**: [THUDM/LongReward](https://github.com/THUDM/LongReward)  
**领域**: LLM效率  
**关键词**: 长上下文对齐, AI Feedback, DPO, 幻觉抑制, 长文档问答

## 研究背景与动机

长上下文模型的工程能力这两年提升很快，128k 甚至更长窗口已经不再稀缺，但“能读很长”和“读懂很长”是两回事。
作者指出，很多长上下文模型的后训练阶段依赖自动合成的 QA 数据，这一步虽然便宜，却把质量问题一并放进了模型里。
这些缺陷在短上下文里还能被模板化回复掩盖，在长文档问答和摘要里则会直接暴露为遗漏、幻觉、逻辑断裂和答非所问。
已有 RLHF 或 RLAIF 证明奖励学习对短上下文有效，但长上下文场景长期缺一个可用奖励信号。
人工标长文偏好太贵，审阅一次长文回答本身就接近一次完整阅读任务。
传统短上下文奖励模型窗口太短，根本看不完原始文档，也无法判断“回答虽然顺畅，但是否真实来自上下文”。
作者因此把问题拆成两个层次。
第一层是奖励设计：怎样让一个现成大模型在长上下文场景下给出可信评分。
第二层是训练落地：怎样把这些评分转成可训练的偏好对，真正推动长上下文模型变强。
本文的核心观察是，长上下文回答质量不是单一标量，而是至少由四个相对独立的维度共同决定。
帮助性关注有没有回答到点子上。
逻辑性关注推理链有没有自相矛盾。
忠实性关注回答中的事实是否能被上下文支撑。
完整性关注重要信息是否被遗漏，尤其是长文中间和尾部信息是否被忽略。
这四个维度合在一起，才更接近真实用户对长文回答的主观评价。
从动机上看，本文并不是要训练一个新的长上下文 reward model，而是先用现成强模型搭一个可靠评分流水线，再把它嵌入 DPO 这类离线 RL 中。
这个路线的价值在于门槛更低、实现更快，也更适合当前长上下文数据还在快速变化的阶段。

## 方法详解

LongReward 的整体流程可以概括为“分维度评分，再用分数构偏好对，再做长上下文 DPO”。
输入是一个长上下文 prompt，通常由长文档上下文和用户 query 组成，以及候选回答。
输出是四个维度的 0 到 10 分评分，以及它们的平均值作为最终 reward。
作者没有强行用一套 prompt 解决所有问题，而是针对不同维度分别设计判分管线，这是本文最关键的工程设计。

第一部分是帮助性评分。
帮助性主要依赖 query 和 response，本质上问的是回答是否相关、是否满足需求、是否信息充分。
因为作者把真实性单独划给忠实性，所以帮助性评估不必强依赖长上下文原文。
具体做法是给 judge model 提供评分原则、few-shot 示例和 CoT 要求，让它先分析再打分。
这样做的好处是把“长上下文内容检索”从帮助性里拿掉，避免评估过重。

第二部分是逻辑性评分。
作者认为长上下文模型常见的一类错误不是事实错，而是前后推理不一致。
例如前文给出一个判断，后文又在另一个段落否定它，或者中间计算错但最后结论似乎正确。
逻辑性也主要依赖 query 与 response，因此仍然用 few-shot 加 CoT 的直接打分策略。
不同于帮助性，逻辑性 prompt 会显式要求 judge 先定位逻辑错误，再输出分数。
这等于把“找错”变成评分前置步骤，减少模型只凭语言流畅度给高分的风险。

第三部分是忠实性评分。
这是 LongReward 中最有技术含量的一环，因为它必须把“回答来自上下文”这件事显式验证出来。
作者借鉴 FactScore 的 break-and-check 思路，但针对长文场景做了三点改造。
第一，回答不是拆成原子事实，而是拆成句子级 factual statements。
这样检索召回更稳，也减少了超长回答下的核验次数。
第二，功能性句子会被过滤，例如“下面是几点原因”这种不承载事实的句子不会进入核验列表。
第三，每条陈述不是二元判断，而是 full support、partial support、no support 三档，对应 1、0.5、0。
最后忠实性得分按 $10 \cdot \sum_i a_i / n$ 聚合，其中 $a_i$ 是每个陈述的支持度。
为了给每条陈述找到证据，系统会以该陈述为查询，从原始上下文中检索 top-5 个 128-token 片段，再让 judge 判断支持程度。
这一步把“阅读整篇长文再核实一句话”变成“局部证据核验”，成本明显可控。

第四部分是完整性评分。
完整性不是看回答有没有胡说，而是看有没有漏掉关键点。
作者专门强调，长上下文模型往往会遗忘中间位置的信息，因此如果直接把整篇长文和回答一起喂给 judge，judge 自己也可能漏看。
LongReward 采用分治策略。
它先把原始上下文切成 4096-token 的粗粒度块。
然后让 judge 从每一块里提取与 query 相关的信息。
接着把所有块中抽到的信息汇总，再让 judge 评估回答是否覆盖这些关键点。
这一设计的本质是先做“信息摘要”，再做“覆盖率判定”，避免评分模型在超长输入上同样出现位置偏置。

四维评分之后，最终 reward 是四个分数的平均值。
作者没有手工调不同维度权重，而是默认等权。
这其实反映出论文的立场：帮助、逻辑、真实、完整缺一不可，先用均衡目标建立可用管线比再做精细加权更重要。

拿到 reward 之后，作者用它来构造长上下文偏好数据。
对每个长上下文 prompt，先从 SFT 模型用 temperature 为 1.0 采样 10 个候选回答。
然后对 10 个候选逐个跑 LongReward。
分数最高的回答作为 winning response，最低的作为 losing response。
这样就把“分数监督”转换成了 DPO 需要的 pairwise preference。

训练阶段使用 DPO，但作者没有只用标准 DPO 损失。
为了稳定训练，又加入了 winning response 上的交叉熵正则项。
总损失是 $\mathcal{L}_{merge} = \mathcal{L}_{DPO} + \lambda \mathcal{L}_{CE}$。
这里的直觉很清楚：DPO 能推开好坏回答的相对概率，但过强时可能损伤语言建模稳定性；CE 正则帮助模型别偏离原本的可生成分布太远。

从方法论上看，LongReward 的关键不在“又造了一个 judge prompt”，而在于它把长上下文评分拆成可操作、可局部验证的多个子问题。
这让一个现成大模型在长文任务中不只是凭感觉打总分，而是沿着结构化流程给出较稳的奖励。

| 维度 | 依赖输入 | 核心机制 | 解决的主要问题 |
|------|----------|----------|----------------|
| 帮助性 | query + response | few-shot + CoT 直接评分 | 是否答到用户需求 |
| 逻辑性 | query + response | 先找逻辑错误再评分 | 前后矛盾、推理断裂 |
| 忠实性 | response + 检索片段 | 拆句、检索、三级支持判定 | 幻觉与事实失真 |
| 完整性 | query + 分块上下文 + response | 分块抽取关键信息再评估覆盖率 | 长文关键信息遗漏 |

| DPO 数据构建步骤 | 具体做法 | 作用 |
|------------------|----------|------|
| 候选采样 | 每个 prompt 采样 10 个回答 | 保证同一问题下有足够候选差异 |
| LongReward 评分 | 对每个回答算四维平均分 | 生成稳定 reward |
| 偏好对抽取 | 最高分 vs 最低分 | 转换为 DPO 可训练样本 |
| DPO + CE 训练 | $\beta=0.15$，$\lambda=0.1$ | 同时提升偏好学习与稳定性 |

## 实验关键数据

实验基座包含 Llama-3.1-8B 和 GLM-4-9B，两者都支持 128k 上下文，并先做了长上下文 SFT 再做 DPO。
长上下文 SFT 数据由 1 万条长文 QA 和 7.6 万条 ShareGPT 通用数据混合而成，长文长度覆盖 8k 到 64k token，涉及 9 个领域。
LongReward 评分用 GLM-4 执行，忠实性检索器使用 Zhipu-Embedding-2，每条 factual statement 检索 top-5 片段。
训练上，SFT 进行 1800 步，DPO 再进行约 400 到 800 步，说明本文不是靠极长训练把效果“磨”出来，而是靠更好的偏好构建策略。

| 实验设置 | 配置 |
|----------|------|
| 基座模型 | Llama-3.1-8B, GLM-4-9B |
| 上下文长度 | 支持 128k，训练最长 64k |
| 长文 SFT 数据 | 10k 长文 QA + 76k ShareGPT |
| 候选回答数 | 每个 prompt 采样 10 个 |
| 忠实性检索 | Zhipu-Embedding-2, top-5 片段 |
| DPO 超参 | $\beta=0.15$, $\lambda=0.1$, lr=$1e-6$ |
| 训练资源 | 4 节点，每节点 8 张 H800 |

主结果最值得注意的是，LongReward 在两个基座上都显著优于 SFT、短上下文 reward model 和“拿大模型回答当正样本”的 Contrast 基线。
对 Llama-3.1-8B，LongReward DPO 的平均分达到 59.9，相比 SFT 的 55.0 提升 4.9 个点。
对 GLM-4-9B，平均分从 56.6 提升到 62.1，提升 5.5 个点。
尤其在 Multi-Doc QA 上，Llama 版本从 44.5 直接涨到 55.8，说明该方法特别擅长缓解“多文档信息聚合时的遗漏和错配”。

| 模型 | 方法 | LongBench-Chat | S-Doc QA | M-Doc QA | Summ | Avg |
|------|------|----------------|----------|----------|------|-----|
| Llama-3.1-8B | SFT | 69.8 | 66.1 | 44.5 | 39.6 | 55.0 |
| Llama-3.1-8B | DPO w/ SRM | 67.4 | 65.0 | 49.6 | 42.7 | 56.2 |
| Llama-3.1-8B | DPO w/ Contrast | 70.6 | 67.8 | 46.2 | 40.3 | 56.2 |
| Llama-3.1-8B | **DPO w/ LongReward** | **72.6** | **67.8** | **55.8** | **43.2** | **59.9** |
| GLM-4-9B | SFT | 64.8 | 68.4 | 50.9 | 42.1 | 56.6 |
| GLM-4-9B | DPO w/ SRM | 66.6 | 67.5 | 57.4 | 48.2 | 59.9 |
| GLM-4-9B | DPO w/ Contrast | 68.2 | 67.8 | 58.0 | 47.8 | 60.5 |
| GLM-4-9B | **DPO w/ LongReward** | **69.2** | **71.9** | **58.8** | **48.5** | **62.1** |

论文还做了两个特别关键的补充实验。
一是 FactScore 分析。
LongReward DPO 不只是让回答更“像好答案”，而是让回答中被上下文支撑的事实比例更高，同时包含更多原子事实。
Llama 版本的 FactScore 从 91.94 升到 92.85，平均 atomic facts 数从 21.76 升到 32.86。
这说明模型不是靠缩短回答来规避错误，而是在更详细的同时保持更高真实性。

二是人类评估。
在 LongBench-Chat 上，LongReward DPO 对 SFT 的总体 win-rate 是 54%，loss 只有 8%。
四个维度里提升最大的是真实性和完整性，这与方法设计完全一致，因为这两个维度正是 LongReward 专门下重功夫建模的部分。

| 分析项 | SFT | DPO w/ LongReward | 结论 |
|--------|-----|-------------------|------|
| Llama #Facts | 21.76 | 32.86 | 回答更详细 |
| Llama FactScore | 91.94 | 92.85 | 细节更多但没牺牲真实性 |
| GLM #Facts | 18.41 | 28.05 | 覆盖信息更多 |
| GLM FactScore | 91.43 | 93.62 | 幻觉进一步下降 |
| 人类 Overall Win/Loss | - | 54% / 8% | 人类主观偏好明显更高 |

另外一个容易被忽视但很重要的发现是，LongReward 带来的长上下文 DPO 还能提升短指令跟随能力。
例如 AlpacaEval2 上，Llama 从 12.4 升到 14.2，GLM 从 12.5 升到 15.4。
这说明本文学习到的偏好不是“只适用于长文问答的局部技巧”，而是更普遍的输出价值观，比如尽量真实、完整、逻辑一致。

作者还比较了不同奖励策略与人工偏好的对齐度。
LongReward 的预测准确率达到 66.2%，高于短上下文奖励模型的 58.3%，也高于直接成对比较的 57.1%。
更关键的是，删掉任何一个维度都会降分，其中去掉忠实性或完整性会跌到 57.8%。
这从实验上证明四维设计不是装饰，而是真正必要的组成部分。

## 亮点与洞察

最强的亮点是把“长上下文奖励难以获取”这个模糊问题拆成四个可执行子任务，每个子任务都有独立判分机制，而不是试图用一个总分 prompt 一把梭。
忠实性的 break-and-check 加检索验证非常务实，本质上把开放式长文核验转换成局部证据支持判断，工程上可复用性很强。
完整性的分块抽取再覆盖评估也很有启发，这其实是一种“先把长文压缩成 query-aware memory，再做 judge”的设计范式。
本文还说明，长上下文对齐不必和短上下文对齐冲突，二者甚至可以用混合偏好数据共同训练。
从研究角度看，LongReward 代表一种比训练新 reward model 更轻的路线：先用强 judge 搭可靠标注管线，再在此基础上蒸馏或训练更便宜模型。
从工程角度看，这种方案适合快速迭代，因为四维 prompt、检索器、chunk 粒度都可以独立替换。
我认为本文最重要的洞察不是“DPO 有用”，而是“长上下文回答质量必须结构化定义，否则 reward 本身会被长文噪声淹没”。

## 局限与展望

第一，成本仍然很高。
每个 QA 样本要做多次候选采样、多个维度评分、忠实性检索核验和完整性分块分析，整体 API 调用量不低。
第二，实验规模仍停留在 10B 左右模型和最长 64k 训练长度，尚未证明在更大模型和更长序列下仍保持同样收益。
第三，任务主要集中在长文 QA 与摘要，对多轮长历史对话、长期 agent 记忆、代码仓库级推理等场景尚未覆盖。
第四，四维等权虽然简单，但不同任务真实偏好权重可能不同，例如法律检索和创意摘要对完整性与帮助性的侧重点并不一样。
未来可行的方向有三条。
一条是用 LongReward 产生的大量偏好对训练专门的长上下文 reward model，降低推理成本。
一条是把完整性与忠实性进一步细分到引用级监督，例如要求回答附带证据跨度或段落引用。
一条是把该框架移植到 agent 轨迹评估里，把“计划完整性”和“执行忠实性”纳入奖励设计。

## 相关工作与启发

和传统 RLHF 相比，本文最大的不同是把 AI judge 深度结构化，而不是直接做 pairwise preference judgement。
和短上下文 reward model 相比，LongReward 明确承认短 RM 在长上下文下看不到证据，因此不能直接复用。
和 FactScore 一类事实核验工作相比，本文不是单独做评估，而是把事实核验嵌入训练数据构造流程，变成真正能改进模型的训练信号。
和“拿大模型回答直接当优质回答”的 Contrast 路线相比，本文不是把大模型当唯一老师，而是把老师的判断拆成多维奖励，这样更可解释，也更不依赖某个单一输出模板。
对我自己的启发有两点。
第一，凡是涉及长上下文、长轨迹、长文档的对齐问题，都应该优先把 reward 分解成几种局部可验证属性，而不是追求单一总分。
第二，完整性往往比帮助性更难，因为它要求知道“本该说什么但没说”，这类缺失型错误很适合采用分块摘要再覆盖评估的思路。
如果以后做长上下文 agent 或长代码审查，完全可以把 LongReward 的思想迁移成“正确性、证据性、完整性、可执行性”四维打分框架。

## 评分

- 新颖性: ⭐⭐⭐⭐☆  把长上下文奖励设计系统化分解为四维，并给出各自评估管线，方法层面很扎实。
- 实验充分度: ⭐⭐⭐⭐⭐  双基座、自动评测、人评、FactScore、短上下文泛化、混合 DPO 兼容性都做到了。
- 写作质量: ⭐⭐⭐⭐☆  论文的核心设计和实验逻辑都比较清楚，尤其是四维评分与 DPO 的衔接很顺。
- 价值: ⭐⭐⭐⭐⭐  对长上下文对齐是非常实用的基建型工作，后续可直接服务 reward model、RLAIF 和 agent 评估。
- 综合评价: 9.0/10。它不是靠花哨架构取胜，而是把一个长期缺位的训练信号问题做成了可复用、可扩展、能稳定带来收益的方案。---
title: >-
  [论文解读] LongReward: Improving Long-context Large Language Models with AI Feedback
description: >-
  [ACL2025][LLM效率][长上下文LLM] 提出 LongReward，利用现成 LLM 从帮助性、逻辑性、忠实性和完整性四个维度为长上下文模型回复自动打分，结合 DPO 离线强化学习显著提升长上下文 SFT 模型的多维能力。
tags:
  - ACL2025
  - LLM效率
  - 长上下文LLM
  - 强化学习
  - DPO
  - AI反馈
  - 奖励模型
---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] Dynamic Chunking and Selection for Reading Comprehension of Ultra-Long Context in Large Language Models](dynamic_chunking_and_selection_for_reading_comprehension_of_ultra-long_context_i.md)
- [\[ACL 2025\] Ref-Long: Benchmarking the Long-Context Referencing Capability of Long-Context Language Models](ref-long_benchmarking_the_long-context_referencing_capability_of_long-context_la.md)
- [\[ACL 2025\] Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models](sliding_windows_full_ranking.md)

</div>

<!-- RELATED:END -->
