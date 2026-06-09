---
title: >-
  [论文解读] On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models
description: >-
  [ACL2026][可解释性][Theory of Mind] 这篇论文用大规模行为评测和受认知神经科学启发的功能定位/消融实验，给出语言模型的 Theory of Mind 与语用推理可能共享内部计算机制的证据，因此把“社会世界模型”从单纯能力得分推进到可检验的功能整合假说。
tags:
  - "ACL2026"
  - "可解释性"
  - "Theory of Mind"
  - "语用推理"
  - "功能定位"
  - "因果消融"
  - "社会世界模型"
---

# On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models

**会议**: ACL2026  
**arXiv**: [2602.10298](https://arxiv.org/abs/2602.10298)  
**代码**: https://github.com/polina-tsvilodub/lm-emergent-social-world-models  
**领域**: 可解释性 / 社会认知评测  
**关键词**: Theory of Mind, 语用推理, 功能定位, 因果消融, 社会世界模型

## 一句话总结
这篇论文用大规模行为评测和受认知神经科学启发的功能定位/消融实验，给出语言模型的 Theory of Mind 与语用推理可能共享内部计算机制的证据，因此把“社会世界模型”从单纯能力得分推进到可检验的功能整合假说。

## 研究背景与动机
**领域现状**：关于大语言模型是否拥有“世界模型”的讨论，过去常落在棋类、导航或文本游戏这类边界清晰的环境中；而自然语言理解中的功能性能力更复杂，因为模型不仅要处理句法和词义，还要推断说话者的意图、知识、信念和社会承诺。Theory of Mind 与语用推理正好位于这个交界处：前者关心如何表征他人的心理状态，后者关心理解话语在具体交际情境中的真实意图。

**现有痛点**：已有工作常把 ToM 和语用能力分开测，只报告模型在某个 benchmark 上是否答对。这种行为层面的得分很难说明模型是不是用了同一套内部机制：两个任务表现相关，可能只是因为大模型更大、指令微调更强、世界知识更多；一个任务得分高，也可能来自数据污染、启发式匹配或答案格式偏好。

**核心矛盾**：如果 ToM 与语用推理只是两个孤立技能，那么模型可以分别学到两套专门机制；但如果语言中的社会意义必须反复调用对他人信念、意图和知识的表征，那么压缩和复用的压力可能促使模型形成可跨任务调用的“社会世界模型”。关键问题不是模型会不会做某个题，而是这些能力在内部是否功能性整合。

**本文目标**：作者把这个大问题拆成两个可测层次：第一，在不同模型之间，ToM 与语用任务的行为表现是否系统相关，并且这种相关性是否超过一般语言能力的解释；第二，模型内部能否定位出支持 ToM 的子网络，并且消融这些子网络是否也会损害语用推理。

**切入角度**：论文借鉴人类认知神经科学中的 functional localizer 思路，用目标条件与控制条件的激活差异来找出和 ToM 相关的单元。这个角度的好处是，它不只问“模型答得对不对”，还问“哪些内部单元在什么条件下被选择性调用，以及这些单元对下游任务是否有因果作用”。

**核心 idea**：用行为相关性 + 功能定位 + 因果消融三段证据，检验 ToM 与语用推理是否共享可复用的内部机制，而不是停留在单个 benchmark 的表面得分。

## 方法详解
这篇论文的方法可以理解为两条线并行推进：一条是行为层面的跨模型统计分析，另一条是机制层面的功能定位与消融。前者回答“两个能力是否一起涨落”，后者回答“同一批内部单元是否真的支撑这两个能力”。

### 整体框架
输入端是多组多选形式的 ToM、语用和一般语言能力数据集，以及一批开源/开放权重语言模型。作者先用条件 log-probability 给每个模型在各数据集上打分：对同一个问题的所有候选答案计算归一化后的条件对数概率，选择概率最高的选项作为模型预测，再计算 accuracy。

行为实验覆盖 48 个模型，来自 Llama、Qwen、Falcon、Mistral、OLMo、Pythia、Gemma 七个家族，参数规模从 0.5B 到 72B，并区分 base 与 fine-tuned 模型。任务侧包含 16 个语用数据集、22 个 ToM 数据集，以及 BLiMP 和 SNLI 两个一般语言能力控制数据集。

机制实验则选取其中 20 个模型，主要来自 Qwen-2.5、Llama、Falcon-3 和 Gemma-2。作者构造四套 ToM localizer suite：LatentBeliefs、CommunicativeIntent、GameBeliefs、MoralIntent。每套 localizer 都有需要心理状态推理的 target 条件，以及尽量匹配表面形式但不需要 ToM 的 control 条件。模型处理这些刺激时，作者记录 transformer block 在最后一个待作答 token 前的单元激活，用 target-control 激活差异定位功能子网络，然后通过置零消融验证其因果作用。

### 关键设计

**1. 行为层面的功能整合检验：用统计控制把"大模型什么都强"和"能力真正耦合"分开。**

如果只看 ToM accuracy 和 pragmatics accuracy 的相关性，很容易把"大模型样样强"误读成两种能力共享机制。作者用三条递进的预测堵住这个解释漏洞：P1 要求 48 个模型上 ToM accuracy 与 pragmatics accuracy 正相关；P2 要求在 Bayesian beta regression 里控制住模型家族、规模、训练类型、数据集类型后，"任务属于 ToM 还是语用"这个标签本身不再显著预测 accuracy；P3 要求用 ToM accuracy 预测语用 accuracy，比用 BLiMP/SNLI 代表的一般语言能力预测得更好。P2 的作用是把模型规模、微调这些显然的混杂因素纳入统计模型吃掉，P3 则进一步证明 ToM 和语用之间存在超出一般语言能力的特殊联系——只有这两条一起成立，"社会认知能力耦合"才不至于被简单解释掉。

**2. 基于认知神经科学 localizer 的 ToM 子网络定位：把人脑功能定位的思路搬进 transformer。**

要谈"共享机制"，先得在模型内部找出和 ToM 相关的单元。作者借鉴人类 fMRI 的 functional localizer 思路，对每个单元 $(l,i)$ 比较 target 刺激集合（需要心理状态推理）和 control 刺激集合（表面形式尽量匹配但不需要 ToM）在该单元上的激活差异，并用 Welch's $t$-test 得到统计量。这里有两种 localizer：simple localizer 直接合并所有 target/control 条件；conjunctive localizer 取所有 target-control 条件对的最小统计量，类似神经科学里的 minimum statistic，专门寻找跨条件都稳定激活的单元。显著单元若超过全模型单元数的 1%，就只保留绝对统计量最高的 1%。为了不把 ToM 窄化成单一的 false-belief 模板，作者用 ATOMS 框架覆盖信念、意图、欲望、情绪、知识、感知和非字面交流等细分面向，并从人类 fMRI 已验证材料出发合成 1400 条 localizer 刺激，构造出 LatentBeliefs、CommunicativeIntent、GameBeliefs、MoralIntent 四套 suite。

**3. ToM 子网络的因果消融与跨域迁移验证：把"共享机制"变成可证伪的因果预测。**

定位只能说明相关激活，要确认这批单元真的支撑 ToM 和语用，必须动手消融。作者对每个模型、每套 localizer 都构造两种对照消融：一种把 ToM target 选出的关键单元置零，另一种把同等数量、最不活跃的控制单元置零，随后重新评测 ToM、语用、BLiMP、SNLI，比较 accuracy 相对完整模型的下降幅度。这个设计把"功能整合"这个抽象假说翻译成了一组可观测的因果预测：如果整合成立，ToM 子网络消融不仅应损害 ToM 任务，也应损害语用任务，且这种损害要强于控制消融，同时不应同样强地破坏一般语言能力——三个方向上的结果若一致，就比单纯的行为相关性更接近"同一批单元跨任务复用"的结论。

### 损失函数 / 训练策略
本文没有训练新语言模型，核心是评价、定位和消融。模型预测全部通过候选答案条件 log-probability scoring 完成，并对答案 token 长度做平均归一化。统计分析主要使用 Bayesian beta regression 和 leave-one-out cross-validation；机制分析中，定位阶段用 target-control 激活差异筛单元，消融阶段将选中单元激活置零。

## 实验关键数据

### 主实验
行为实验支持 ToM 与语用推理存在紧密关系，但这种关系并不是简单地“所有大模型都更强”。最关键的是，ToM accuracy 比一般语言能力更能解释语用 accuracy，这使论文的论点比普通相关性分析更扎实。

| 问题 | 设置 | 关键结果 | 论文解释 |
|------|------|----------|----------|
| ToM 与语用能力是否相关 | 48 个模型上的平均 ToM accuracy vs. pragmatics accuracy | $r=0.68$, $p=1.24 \times 10^{-7}$ | 中等强度且显著的正相关，支持 P1 |
| 控制模型因素后，任务域是否仍重要 | Bayesian beta regression 控制 family、size、type、dataset type | domain 系数 $\beta=-0.03[-0.74,0.67]$ | ToM/语用标签本身不再提供可信额外解释，支持 P2 |
| ToM 是否比一般语言能力更能预测语用 | 比较含 ToM accuracy 的 M1 与含 BLiMP/SNLI accuracy 的 M0 | M1 显著优于 M0，报告差异 $ELPD=-16.1$, $p=0$ | ToM 表现对语用表现的预测更强，支持 P3 |
| ToM 内部哪些方面更有解释力 | ATOMS 七类特征的 128 个回归模型比较 | 最佳模型含 intentions、desires、emotions、percepts、non-literal communication；相对 baseline $ELPD=-58.80$, $p=0$ | ToM 不是均质能力，percepts 等细分维度对表现差异尤其关键 |

功能定位实验表明，定位出的单元分布不是随机的，不同 localizer 在层上的分布也不同。CommunicativeIntent 的活跃单元更集中在最后几层，LatentBeliefs 更偏中后层；LB + CI conjunctive localizer 更保守，很多模型中显著单元少于 1%。10-fold cross-validation 显示 all、MoralIntent、GameBeliefs 的泛化最强，通常有至少 9 个 fold 显著；simple CommunicativeIntent 和 LatentBeliefs 也较稳定；conjunctive suite 的泛化相对弱一些。

### 消融实验
消融结果是论文最重要的机制证据：ToM 子网络消融不仅损害 ToM，也损害语用任务，而对一般语言基准没有可信下降。这符合“共享社会推理机制”而非“只破坏通用语言处理”的解释。

| 预测 | 检验内容 | 全局分析结果 | 是否支持 |
|------|----------|--------------|----------|
| P1.1 | ToM 子网络消融会降低 ToM 任务表现 | $\beta=0.25[0.14,0.35]$ | 支持 |
| P1.2 | ToM 消融对 ToM 的影响强于控制消融 | $\beta=0.06[0.02,0.11]$ | 支持 |
| P2.1 | ToM 子网络消融会降低语用任务表现 | $\beta=0.30[0.20,0.39]$ | 支持 |
| P2.2 | ToM 消融对语用的影响强于控制消融 | $\beta=0.07[0.02,0.12]$ | 支持 |
| P3.1 | ToM 子网络消融不会可信降低一般语言能力 | $\beta=0.13[-0.10,0.35]$ | 支持“无可信下降” |
| P3.2 | 消融对语用的影响强于一般语言能力 | $\beta=0.17[-0.07,0.41]$ | 证据不足 |

更细粒度地看，并不是所有 localizer 都同样贡献结果。P1.1、P2.1、P3.1 大体在多数 localizer 上成立，但 P1.2 和 P2.2 主要由 LatentBeliefs simple、LatentBeliefs conjunctive 的边缘结果，以及 GameBeliefs 驱动。也就是说，信念、感知、欲望和情绪相关的 ToM 子网络似乎最稳定地支持 ToM 与语用任务。

| 分析对象 | 观察到的现象 | 含义 |
|----------|--------------|------|
| LatentBeliefs / GameBeliefs | 消融后 ToM 和语用表现都更明显下降 | 与信念、感知、欲望、情绪相关的单元可能是共享机制的主要来源 |
| CommunicativeIntent | 层分布很清晰，但因果消融不总符合预期 | “非字面交流” localizer 找到的活跃单元未必就是最关键的语用支撑机制 |
| entity tracking | 某些关键消融对 entity tracking 的影响接近 ToM/语用任务 | 共享机制可能包含实体状态追踪，而不完全是纯粹社会心理表征 |
| 模型规模比较 | 大中小模型在消融效应上无可信差异 | 功能整合不只是大模型容量带来的简单现象，但仍需更多模型验证 |

### 关键发现
- 行为层面，ToM 与语用 accuracy 显著相关，且 ToM accuracy 比 BLiMP/SNLI 更能预测语用任务表现，说明两类能力之间的联系不能完全归因于一般语言能力。
- 机制层面，ToM localizer 找到的子网络对 ToM 与语用任务都有因果影响，尤其是 LatentBeliefs 和 GameBeliefs 相关子网络更稳定。
- ATOMS 细分分析很有价值：percepts、intentions、desires、emotions 和 non-literal communication 能解释 ToM benchmark 差异，提示未来不应把 ToM 当作单一总分。
- 论文也诚实展示了复杂性：CommunicativeIntent localizer 的因果结果不完全符合直觉，entity tracking 可能是共同底层因素之一，P3.2 也没有得到可信支持。

## 亮点与洞察
- 最大亮点是把“LLM 是否有社会世界模型”变成了一个可操作的功能整合问题。论文没有直接声称模型理解人心，而是问 ToM 和语用推理是否共享可定位、可消融的机制，这个表述更克制也更可检验。
- localizer 设计很有启发：作者不是只用传统 false-belief，而是用 LatentBeliefs、CommunicativeIntent、GameBeliefs、MoralIntent 覆盖多个 ToM 面向，并引入 conjunctive analysis 来寻找跨条件稳定单元。这种方法可以迁移到道德推理、情绪理解、多智能体规划等其他功能性能力研究中。
- 行为统计和机制消融互相补位。行为相关性告诉我们能力之间“像是一起变化”，消融实验则进一步显示同一批单元被破坏后两个能力都会受损，使论文比单纯 benchmark paper 更接近可解释性研究。
- 论文最有意思的地方是把“社会认知”拆出了层次：有些 localizer 在激活分布上看起来漂亮，却不一定有强因果作用；有些共享影响可能来自 entity tracking 这类更基础机制。这提醒我们，社会世界模型可能不是一个干净模块，而是由心理状态表征、实体追踪、语言推断和任务格式共同拼出的功能网络。

## 局限与展望
- 作者承认，所有结论都依赖于数据集和 localizer 是否真的测到了 ToM、语用推理以及 ATOMS 中的细分类别。如果 benchmark 本身混入了启发式线索，那么定位出的也可能是解题技巧而非心理状态表征。
- 消融方法比较粗：把选中单元置零能说明这些单元有因果作用，但不能告诉我们具体信息如何流动，也不能区分单元是编码信念、实体状态、答案格式还是任务难度。后续可以结合 activation patching、attribution patching、causal tracing 和 circuit discovery 做更细粒度验证。
- localizer 刺激虽然基于人类神经科学材料并用 PCA 检查了合成数据质量，但合成版没有经过人类 fMRI 验证。因此它们更像是受神经科学启发的模型探针，不能直接推出模型与人脑有相同组织方式。
- 评测集中在英语，且 ToM benchmark 中 false-belief 类任务比例较高。真实交际里 true-belief、文化规范、长期互动和多轮对话很重要，未来应扩展到多语言、多文化和动态交互场景。
- 由于部分数据集已公开多年，近期模型可能在预训练或指令微调中接触过相关材料。论文使用合成 localizer 缓解了一部分污染风险，但行为 benchmark 的污染仍然是开放问题。

## 相关工作与启发
- **vs AlKhamissi et al. 2025a**: 该工作把认知神经科学式 functional localization 引入 LLM，并研究语言网络；本文沿用这一思路，但用更大的 ToM localizer 数据集和更细的 ToM 分类，得到比先前小数据实验更强的 ToM 定位与因果证据。
- **vs Hu et al. / Ma et al. / Sap et al. 等 ToM 与语用 benchmark**: 这些工作主要回答模型在社会推理或语用任务上表现如何；本文进一步问表现背后的机制是否共享，并用统计控制和消融实验连接两个任务域。
- **vs Saxe & Kanwisher / Hauptman et al. 的人类神经科学研究**: 人类研究发现 ToM 网络与非字面语言理解存在重叠；本文把这种“共享功能网络”的假设迁移到语言模型，但保持了谨慎，不把模型机制直接等同于人脑机制。
- **vs 机制可解释性中的 belief representation / circuit analysis**: 相关工作常关注某类信念表征或具体 circuit；本文的粒度更粗，重点是跨任务功能子网络。它的启发是，社会认知可解释性可以先用 localizer 找候选区域，再用更精细 circuit 工具追踪具体变量。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 把 ToM、语用推理、认知神经科学 localizer 和 LLM 消融连接起来，问题设定很有辨识度；不过核心工具仍是现有 functional localization 的扩展。
- 实验充分度: ⭐⭐⭐⭐☆ 行为评测覆盖 48 个模型和大量数据集，机制实验也覆盖 20 个模型与多套 localizer；不足是合成 localizer 与部分 benchmark 的生态有效性还需要复现。
- 写作质量: ⭐⭐⭐⭐☆ 假说、预测和统计检验组织清楚，局限写得诚实；但 Bayesian contrast 和 ELPD 的符号解释对普通读者略有门槛。
- 价值: ⭐⭐⭐⭐☆ 对“LLM 是否具有社会世界模型”的讨论提供了更可检验的路径，也为社会认知能力的机制可解释性研究给出了一套可复用范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/interpretability/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)

</div>

<!-- RELATED:END -->
