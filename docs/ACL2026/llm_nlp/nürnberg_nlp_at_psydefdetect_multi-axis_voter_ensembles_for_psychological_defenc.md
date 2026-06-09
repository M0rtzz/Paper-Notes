---
title: >-
  [论文解读] Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification
description: >-
  [ACL2026][LLM/NLP][防御机制识别] 这篇 BioNLP 2026 PsyDefDetect 共享任务系统论文把心理防御机制分类看成一个边界模糊、标注一致性有限的问题，用 9 个跨粒度、跨训练方式、跨基座模型的投票器做集成，在隐藏测试集上取得 F1=.420，并在 21 支注册队伍中排名第一。
tags:
  - "ACL2026"
  - "LLM/NLP"
  - "防御机制识别"
  - "心理健康对话"
  - "模型集成"
  - "类别不均衡"
  - "共享任务系统"
---

# Nürnberg NLP at PsyDefDetect: Multi-Axis Voter Ensembles for Psychological Defence Mechanism Classification

**会议**: ACL2026  
**arXiv**: [2605.07606](https://arxiv.org/abs/2605.07606)  
**代码**: https://github.com/th-nuernberg/nuernberg-nlp-psydefdetect  
**领域**: NLP理解 / 心理健康 NLP  
**关键词**: 防御机制识别, 心理健康对话, 模型集成, 类别不均衡, 共享任务系统

## 一句话总结
这篇 BioNLP 2026 PsyDefDetect 共享任务系统论文把心理防御机制分类看成一个边界模糊、标注一致性有限的问题，用 9 个跨粒度、跨训练方式、跨基座模型的投票器做集成，在隐藏测试集上取得 F1=.420，并在 21 支注册队伍中排名第一。

## 研究背景与动机
**领域现状**：心理健康 NLP 正在从情绪识别、风险评估、咨询对话辅助，走向更细粒度的临床概念识别。PsyDefDetect 任务要求模型在情绪支持对话中，对求助者的目标话语判断其心理防御机制层级，标签来自 Defense Mechanism Rating Scale，包含 8 个正向防御层级以及 No Defence 类别。

**现有痛点**：这个任务的难点不是单纯语义分类。许多防御机制在表面语言上都像是理性化、解释、反思或自我安慰，真正的差别常常藏在语用功能和上下文意图里。论文指出，训练有素的标注者也只有中等一致性，Cohen's $\kappa=.639$，说明标签边界本身就不稳定。

**核心矛盾**：单个大模型很容易在这些模糊边界上形成固定偏差。即便换更强的模型，也不一定能解决 C5、C6、C7 等中高层级防御之间的混淆，因为它们共享大量“成熟、克制、反思”的表达形式。真正有价值的信号不是某个模型绝对更强，而是不同模型在不同样本上犯错不一样。

**本文目标**：作者希望构建一个共享任务系统，既能识别容易分开的 No Defence，又能在 8 个防御类别之间尽量减少多数类吸附；同时还要处理训练集类别极不均衡、隐藏测试集不可见、验证信号有限等现实约束。

**切入角度**：论文从“错误独立性”出发，把系统设计成多轴投票器，而不是单一路径微调。作者先用表示空间分析确认 C0 No Defence 最可分，而 8 个防御类别高度重叠；再把 C0 判断交给 9 类 gatekeeper，把防御内部判断交给 8 类 specialist。

**核心 idea**：用跨类别粒度、跨训练目标、跨基座模型的 voter ensemble 代替单模型分类器，把模糊心理防御边界上的独立错误转化为投票收益。

## 方法详解
这篇论文的方法更像一套面向共享任务的系统工程方案，而不是新网络结构：先用合成数据缓解少数类稀缺，再训练几组"会在不同样本上犯不同错误"的互补模型，最后用一条带 C0 override 的投票规则把它们融合起来。它真正的设计重点，是围绕"哪些模型组合能产生有用的分歧"来选轴。

### 整体框架

系统的输入是一段情绪支持对话和其中一个求助者目标话语，输出是一个 DMRS 防御机制标签——C0 表示 No Defence，C1 到 C8 是 8 个防御层级。核心是 9 个 voter 组成的三轴集成：第一轴是 3 个 Min-SFT 9c gatekeeper，由 Ministral-8B 生成式 SFT 微调、保留全部 9 类；第二轴是 3 个 Min-LR 8c specialist，在 Ministral-8B 的适配表示上训练只管 C1–C8 的 8 类逻辑回归；第三轴是 3 个 Phi4-LR 8c specialist，换 Phi-4-14B 的表示训练同样的 8 类 LR，专门提供跨基座的错误差异。每个分支都先做 5 折交叉验证，再按内部 CV 表现选 top-3 fold 进最终系统。

推理时走的是一条"先把关、再细分"的两段式规则：gatekeeper 先集体表态这条样本是不是 C0，只要它们里多数判 C0，系统就直接输出 C0；否则 9 个 voter 一起对 C1–C8 做多数投票，平票时偏向训练集最大类 C7（High-Adaptive）。这样一来，No Defence 的高可分性、8 类防御内部的细粒度混淆、以及不同模型的互补性，就被统一放进同一个投票框架里处理。

### 关键设计
**1. C0 gatekeeper 与 8 类 specialist 的粒度拆分：把"有没有防御"和"是哪种防御"拆成两个难度不同的子问题。**

作者对 9 类 SFT 模型的隐藏状态做 t-SNE，发现 C0 No Defence 是唯一相对清晰的簇，而 C1–C8 在表示空间里大量重叠。如果让所有模型一起做平坦的 9 分类，C0 那条清晰边界就会和防御类内部那些模糊边界被塞进同一个决策空间里互相干扰。于是本文保留 9 类模型只当 gatekeeper、专门触发 C0 override，再单独训练只看 C1–C8 的 specialist，让它们把全部容量集中在防御类别之间的细微差异上——既吃到了 C0 的高可分性，又不让 specialist 在 No Defence 上浪费决策力。

**2. 生成式 SFT 与判别式 LR 的训练方式互补：用训练目标的差异，人为制造有用的"错误独立性"。**

集成要有收益，前提是成员在不同样本上犯不同的错。本文让两条分支走完全不同的训练范式：SFT 分支用 QLoRA 做生成式监督微调，让 LLM 直接按提示生成标签数字；LR 分支则复用 ClsHead 适配后的表示、丢掉原分类头，在冻结的最后 token hidden state 上训练带 L2 正则的多项逻辑回归，几乎不增加算力却能快速扫一遍"模型×类别粒度"的组合。关键的是，作者没有简单挑 CV 分数最高的 9 类 LR 当 gatekeeper，而是特意保留生成式 SFT——因为它和 8 类 LR specialist 搭配时两种范式会在不同样本上失误；实验也印证了这点：SFT gatekeeper 配 SFT specialist 毫无提升，换成配 LR specialist 后隐藏测试 F1 从 .373 升到 .391。

**3. 跨基座模型的第三轴投票：再加一个不同血统的模型，专治 Ministral 内部意见分裂的样本。**

前两轴都长在 Ministral 上，错误模式难免相关。作者于是测了 Phi-4-14B、Llama-3.1-8B、PsychoCounsel-Llama3-8B 等多个 8 类 LR specialist，用它们和 Min-LR 8c 在 5 折上的 F1 profile 相关性来量互补性，挑出 profile 最反向的 Phi4-LR 8c 当第三分支。它的角色不是靠人多压过 6 个 Ministral voter 的强多数，而是只在 Ministral 自己分裂、投不出多数时才起作用，更像一个仲裁者而非简单叠模型——flip 分析显示它的翻转几乎都集中在 C6/C7 这条最大混淆边界上，正好踩在关键错误区。

### 损失函数 / 训练策略
所有微调都用 4-bit NF4 QLoRA、作用于全部线性投影层，dropout 0.05、cosine schedule、10% warm-up、训练 10 epoch、有效 batch size 8、最大序列长度 4096。其中 SFT 用 LoRA rank 32、$\alpha=64$，以标签数字的生成式交叉熵为目标；ClsHead 用 LoRA rank 16、$\alpha=32$、focal loss，类别权重取 $w_c=N/(K n_c)$；LR 则在冻结 hidden state 上训练多项逻辑回归，用 L2 正则和 balanced class weight，并在每折内扫正则强度 $C$。

数据增强方面，作者在一个 dialog-stratified 80/20 split 上用 GPT-5.2 为少数类生成合成对话，预算是每类最多补到 200 条、且合成数不超过原类样本数的 75%；C0 和 C7 因样本本就充足不做增强，最终共生成 738 条合成对话，且只放进训练折，验证集和隐藏测试集全程保持原始人工标注。最终投票规则可写成：若 gatekeeper 中预测 C0 的数量达到多数则 $\hat{y}=0$，否则在全部 voter 上取 $\arg\max_c \sum_j \mathbf{1}[v_j=c]$。

> ⚠️ 原文出现的 GPT-5.2 等模型名以原文为准。

## 实验关键数据

### 主实验
主实验关注隐藏测试集 macro-F1，指标只在 C1 到 C8 上计算。

| 系统 | 激活的多样性轴 | 隐藏测试 F1 | 相对 baseline |
|------|----------------|-------------|---------------|
| 组织方 baseline：Min-SFT 9c，无增强 | 单模型 | .315 | - |
| Min-SFT 9c full-train，增强，单模型 | 数据增强但无投票 | .307 | -0.008 |
| 5V Min-SFT 9c | 5 折投票 | .373 | +0.058 |
| 6V Min-SFT 9c + Min-LR 8c | 类别粒度 + 训练方式 | .391 | +0.076 |
| 9V Min-SFT 9c + Min-LR 8c + PCounsel-LR 8c | 再加模型轴 | .414 | +0.099 |
| 9V Min-SFT 9c + Min-LR 8c + Llama-LR 8c | 再加模型轴 | .417 | +0.102 |
| 9V Min-SFT 9c + Min-LR 8c + Phi4-LR 8c | 再加模型轴 | **.420** | **+0.105** |

可以看到，最大跳变来自投票本身：5 折 Min-SFT 9c 从 .315/.307 附近提升到 .373。

加入 LR specialist 后进一步提升到 .391，说明训练范式差异确实带来互补。

再加入第三个基座模型后，三个候选分支都能继续提升，Phi4-LR 8c 最终达到 .420。

这比 baseline 相对提升约 33.4%，也是论文报告的共享任务第一名成绩。

### 消融实验
论文最有启发的消融是合成数据增强。

| 系统 | 有 GPT-5.2 增强 | 无增强 | 差值 | 说明 |
|------|----------------|--------|------|------|
| Min-SFT 9c 单模型 | .307 | .315 | -0.008 | 合成数据单独使用会引入噪声 |
| 5V Min-SFT 9c | .373 | .319 | +0.054 | 投票能平均掉一部分合成噪声 |
| 6V + Min-LR 8c | .391 | .369 | +0.022 | 增强收益继续存在 |
| 9V + Phi4-LR 8c | .420 | .378 | +0.042 | 最终系统中增强贡献明显 |

这个表说明增强不是“越多越好”的独立模块。

单模型被合成样本拉低，但投票系统能把合成数据带来的召回收益和噪声抵消结合起来。

因此作者认为增强和 voter diversity 是交织发挥作用的，而不是简单相加。

### 多模型与训练方式对比
Appendix 的 CV5 表可以解释为什么 LR specialist 被选中。

| 模型 | SFT 8c | SFT 9c | ClsHead 8c | ClsHead 9c | LR 8c | LR 9c |
|------|--------|--------|------------|------------|-------|-------|
| Ministral-8B | .321 | .306 | .333 | .311 | **.342** | .315 |
| Phi-4-14B | - | .293 | .337 | - | **.337** | - |
| Llama-3.1-8B | .251 | .279 | .246 | .284 | **.312** | .284 |
| Qwen2.5-7B | .266 | .256 | .302 | .268 | **.307** | .283 |
| PsychoCounsel-8B | - | - | **.316** | - | .301 | - |
| PsyLLM-8B | - | - | **.295** | - | .289 | - |
| GPT-OSS-20B | .212 | .183 | .278 | - | **.292** | - |

LR 在多数模型上优于或接近 ClsHead，并且计算成本低。

Ministral-8B 和 Phi-4-14B 的 8 类 LR CV 分数最高，因此最终系统选择它们作为两个 specialist 分支。

### 类别级表现

| 类别 | 防御层级 | F1 | Precision | Recall | 测试样本数 |
|------|----------|----|-----------|--------|------------|
| C0 | No Defence | .899 | .855 | .947 | 75 |
| C1 | Action | .583 | .700 | .500 | 28 |
| C2 | Major Image-Dist. | .333 | .357 | .312 | 16 |
| C3 | Disavowal | .291 | .267 | .320 | 25 |
| C4 | Minor Image-Dist. | .350 | .368 | .333 | 21 |
| C5 | Neurotic | .200 | .286 | .154 | 13 |
| C6 | Obsessional | .436 | .386 | .500 | 44 |
| C7 | High-Adaptive | .833 | .844 | .823 | 243 |
| C8 | Needs More Info | .333 | .400 | .286 | 7 |

系统在 C0 和 C7 上明显最好。

C0 的 F1 达到 .899，符合表示空间中 No Defence 最容易分开的观察。

C7 的 F1 达到 .833，部分来自样本数最多和语言模式较稳定。

真正困难的是 C3、C5、C6 等中间层级，尤其 C5 Neurotic 只有 .200。

论文报告的主要混淆包括 28 个 C6/C7 互换，以及 13 个 C5 中有 7 个被预测成 C7。

这说明模型倾向于把临床上更值得关注的 neurotic defence 读成更成熟的 high-adaptive coping，存在明显的 under-flag 风险。

### 关键发现
- 投票比单模型更关键。5 折同构投票已经把隐藏测试 F1 从 baseline .315 提高到 .373，说明这个任务中方差和边界不确定性是核心问题。
- 第三模型轴的收益来自关键边界仲裁。Phi4-LR 只能翻转 Ministral 内部不够一致的样本，而 39 个实际 flip 中有 33 个触及 C6/C7 边界。
- 合成数据需要和投票绑定。单模型增强变差，但 9V 系统增强后比无增强高 .042，说明 ensemble 对合成噪声有一定容错。
- 类别不均衡仍然主导错误形态。C7 占训练集 52%，也吸收了许多中间层级误分，模型表现并不等价于临床可用性。

## 亮点与洞察
- 这篇论文最好的地方是把共享任务系统写成了一个“错误互补性”故事，而不是堆模型排行榜。它用 t-SNE、CV profile correlation、Krippendorff's $\alpha$、flip analysis 等证据去解释为什么投票有效。
- C0 gatekeeper 的设计很实用。很多实际分类任务也存在一个“容易分开的外部类”和一组“内部细分难类”，把两者拆开常常比强行做平坦多分类更稳。
- LR specialist 是一个性价比很高的 trick。冻结 LLM 表示后训练线性头，既能快速探索多个基座模型，也能产生和生成式 SFT 不同的错误模式。
- 增强消融很诚实。论文没有把 GPT 生成数据包装成万能解法，而是指出它单独会伤害模型，只有在投票平均噪声时才变成收益。
- 逐类分析具有临床警觉性。作者没有只报第一名成绩，而是指出把 C5 错判为 C7 会低估干预需求，这比单纯追 F1 更负责任。

## 局限与展望
- 统计支撑仍然有限。PSYDEFCONV 训练集只有 1,864 条原始样本，最终 9V 相比 6V 的 +.029 增益来自一次隐藏测试观察，不能证明 Phi4 一定是普适最优选择。
- top-3 fold 选择、C0 override 阈值、用 5 个 fold 的 Pearson 相关性选 specialist，这些决策都偏启发式，缺少更严格的验证。
- 标注上限很硬。标注者一致性只有 $\kappa=.639$，C2、C5、C8 等小类本来就处在较强主观判断区间，macro-F1 很可能受到标签噪声上限限制。
- 适用范围很窄。实验只在英文 ESConv/PSYDEFCONV 上完成，且对话是模拟支持场景，不等同于真实临床会谈。
- 合成数据可能带来生成器痕迹。738 条 GPT-5.2 合成对话只进入训练，但仍可能让模型学到不属于真实求助者话语的风格偏差。
- 推理成本偏高。9 个 voter 不适合实时部署，作者也指出需要蒸馏；实际应用中也许 5V 或 6V 是更合理的成本-收益折中。
- 临床伦理风险很大。F1=.420 意味着大多数防御类话语仍会被错分，系统只能作为有监督工作流里的辅助信号，不能独立用于心理诊断或干预决策。

## 相关工作与启发
- **vs PSYDEFCONV / PsyDefDetect baseline**: baseline 用 Ministral-8B 做 9 类生成式 SFT，本文沿用其任务设定和提示形式，但通过数据增强、CV voter pool、粒度拆分、训练方式拆分和跨模型 specialist 把 F1 从 .315 提升到 .420。
- **vs 传统 ensemble 方法**: Dietterich 的集成学习思想强调准确且多样的分类器组合，本文把这个经典原则落到 LLM 心理健康分类任务上，并用 flip analysis 证明多样性主要作用在 C6/C7 等高混淆边界。
- **vs 单模型 LLM 微调**: 单模型微调追求一个最优表示，但本文显示，在标注含糊和类别重叠明显的任务中，多个不完全一致的弱决策边界可能比一个强模型更可靠。
- **vs 类别不均衡处理方法**: focal loss、class weight 和合成数据都被尝试，但真正稳定的收益来自与投票结合。这启发后续少数类 NLP 任务：增强数据应该和不确定性建模、集成或校准一起评估。
- **对其他任务的启发**: 医疗分诊、风险等级评估、立场强度识别等任务也常有一个清晰的负类和多个模糊正类。可以借鉴本文的 gatekeeper-specialist 设计，把“是否触发风险”和“风险类型/等级”分离建模。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 核心技术不是新模型结构，但把三轴 voter diversity 系统化地用于心理防御机制检测，设计清楚且有针对性。
- 实验充分度: ⭐⭐⭐⭐☆ 主结果、增强消融、模型筛选、逐类错误和 flip analysis 都比较完整，但最终选择仍受小数据和一次隐藏测试限制。
- 写作质量: ⭐⭐⭐⭐⭐ 系统论文写得很清楚，动机、设计选择和错误分析形成闭环，伦理风险也没有回避。
- 价值: ⭐⭐⭐⭐☆ 对共享任务和小样本临床 NLP 很有参考价值，尤其适合借鉴其集成诊断思路；但当前 F1 仍不足以支撑高风险真实应用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Evaluating Customized vs. Generalist Transformer-based Models for Legal Contract Classification](evaluating_customized_vs_generalist_transformer-based_models_for_legal_contract_.md)
- [\[ICML 2025\] Theoretical Limitations of Ensembles in the Age of Overparameterization](../../ICML2025/llm_nlp/theoretical_limitations_of_ensembles_in_the_age_of_overparameterization.md)
- [\[ACL 2025\] The Nature of NLP: Analyzing Contributions in NLP Papers](../../ACL2025/llm_nlp/the_nature_of_nlp_analyzing_contributions_in_nlp_papers.md)
- [\[ACL 2025\] Unintended Harms of Value-Aligned LLMs: Psychological and Empirical Insights](../../ACL2025/llm_nlp/unintended_harms_of_value-aligned_llms_psychological_and_empirical_insights.md)
- [\[ACL 2025\] Computation Mechanism Behind LLM Position Generalization](../../ACL2025/llm_nlp/computation_mechanism_behind_llm_position_generalization.md)

</div>

<!-- RELATED:END -->
