---
title: >-
  [论文解读] Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall
description: >-
  [ACL 2026][可解释性][量化] 这篇论文系统评估 GPTQ、AWQ、BitsAndBytes 等权重量化对 LLM 事实知识回忆的影响，发现量化通常会造成信息损失并削弱知识调用，尤其伤害较小模型和未饱和关系，但 8-bit / BitsAndBytes 往往能较好保留能力…
tags:
  - "ACL 2026"
  - "可解释性"
  - "量化"
  - "事实知识回忆"
  - "知识神经元"
  - "隐式多跳推理"
  - "模型压缩"
---

# Through a Compressed Lens: Investigating The Impact of Quantization on Factual Knowledge Recall

**会议**: ACL 2026  
**arXiv**: [2505.13963](https://arxiv.org/abs/2505.13963)  
**代码**: 无公开代码 / 待确认  
**领域**: 可解释性 / 模型压缩 / 事实知识回忆  
**关键词**: 量化、事实知识回忆、知识神经元、隐式多跳推理、模型压缩  

## 一句话总结
这篇论文系统评估 GPTQ、AWQ、BitsAndBytes 等权重量化对 LLM 事实知识回忆的影响，发现量化通常会造成信息损失并削弱知识调用，尤其伤害较小模型和未饱和关系，但 8-bit / BitsAndBytes 往往能较好保留能力，个别量化甚至会提升多跳事实回忆。

## 研究背景与动机
**领域现状**：量化是 LLM 部署中最常用的压缩手段之一。通过把权重或激活从高精度浮点数压到 8-bit、4-bit 甚至更低，模型可以更省显存、更快推理，也更容易在有限硬件上落地。

**现有痛点**：已有工作已经研究量化对多语言、偏见、公平性、校准、对齐和上下文学习的影响，但对事实知识回忆的分析仍然不足。事实知识回忆不是简单的下游 accuracy，它关心模型能否从参数记忆中调出实体、关系和组合事实，是问答、推理和知识密集型任务的底层能力。

**核心矛盾**：量化看似只是数值精度降低，但 LLM 的事实知识可能以分散的神经元、层间表征和隐式推理路径存储。若某些关键神经元贡献分数被压低，表面 perplexity 或一般 benchmark 下降不明显，也可能已经损伤了特定事实的调用链。

**本文目标**：作者希望回答三个问题：量化会让模型忘掉多少事实；这种损失发生在模型内部哪些层和神经元；在需要先回忆桥接实体、再完成两跳推理的场景中，量化到底破坏哪一步。

**切入角度**：论文没有只做黑盒性能对比，而是把 factual knowledge recall 拆成两个互补视角：一是 LRE 上的一跳事实记忆与知识神经元归因，二是 TwoHop-Fact 上的 latent multi-hop reasoning，观察桥接实体召回、输出分布一致性和最终答案准确率。

**核心 idea**：用“能力指标 + 可解释性归因 + 多跳内部路径”三层证据来判断量化是否真正损伤 LLM 的事实知识回忆，而不是只看压缩后总体准确率。

## 方法详解
论文研究的是 post-training weight-only quantization，不重新训练模型，也不引入新的压缩算法。它把 full-precision 模型作为参照，把不同量化方法和 bit-width 的模型放到同一套 factual knowledge recall 诊断流程中。

### 整体框架
实验分为两条主线。第一条是 knowledge memorization analysis：在 LRE 数据集上用一跳事实查询测试模型能否直接回忆对象实体，例如给定 subject-relation 后生成正确 object；随后用知识神经元归因方法追踪哪些 neuron 对正确答案 token 的 log-probability 有最大贡献，并观察这些贡献在量化后如何变化。

第二条是 latent multi-hop reasoning analysis：在 TwoHop-Fact 上构造两跳事实链，例如 $e_1 \xrightarrow{r_1} e_2$ 和 $e_2 \xrightarrow{r_2} e_3$。模型既要回忆桥接实体 $e_2$，又要沿第二个关系得到 $e_3$。作者比较 full-precision 与 quantized 模型在 $r_1(e_1)$、$r_2(e_2)$ 和 $r_2(r_1(e_1))$ 三类准确率上的差异，并结合 EntRec、CnstScore 等内部表征指标分析量化影响。

被评测模型包括 Llama3-8B、Qwen2.5-7B 和 Qwen2.5-14B；量化方法包括 GPTQ、AWQ、BitsAndBytes 的 4-bit / 8-bit 版本。由于公开 checkpoint 可获得性不同，不是每个模型都有所有量化配置。

### 关键设计

**1. 事实记忆与知识神经元联合分析：把宏观掉点和具体哪个 neuron 被压低对上号。**

只看 LRE 上的事实回忆 accuracy，没法分辨量化是在全局轻微扰动，还是直接削弱了承载事实的关键 neuron。论文先统计 full 与 quantized 模型在一跳查询上的回忆准确率，再给每个 neuron 算一个 contribution score——即该 neuron 对正确答案 token log-probability 的提升量。关键一步是取 full-precision 模型 top-300 feed-forward neurons 里最低的那个贡献作为阈值 $\tau$，再数量化模型里还剩多少 neuron 超过 $\tau$。超阈 neuron 变少，就把“输出准确率下降”和“内部高贡献 neuron 被压低”直接连了起来，让宏观性能和内部表示损失互相印证，而不是停在一个笼统的 accuracy 上。

**2. 层级归因定位信息损失位置：找出量化具体伤了哪几层。**

量化通常按矩阵或层统一压缩，但事实知识并不是均匀铺在所有层里。论文比较 attention sublayers 和 feed-forward sublayers 的 aggregate contribution score drop，结果不同架构差异明显：Qwen2.5-7B 上最后两层的下降最显著，Llama3-8B 则更多落在中后层，而且末层可能反而补偿性上升。这种层级画像一方面提示哪些层对低 bit 更敏感，另一方面也解释了为什么同一种量化算法换个模型家族表现就不一致——事实存储的位置本身就因架构而异。

**3. 隐式多跳推理拆解：分清量化到底卡在桥接实体、第二跳关系还是最终组合。**

多跳失败常被笼统说成“推理能力差”，但 TwoHop-Fact 把它拆成三个能分别测量的环节：$r_1(e_1)$ 测桥接实体 $e_2$ 的回忆，$r_2(e_2)$ 测第二跳事实，$r_2(r_1(e_1))$ 测完整两跳组合。再配上两个内部指标——EntRec 衡量 hidden representation 是否真的召回了桥接实体，CnstScore 衡量两跳 prompt 与对应一跳 prompt 的输出分布是否一致——就能看出量化主要伤的是 first-hop 的事实召回，还是后续的组合路径，而不是只丢一个最终准确率说不清原因。

### 损失函数 / 训练策略
本文没有训练新模型，所有实验都在公开 full-precision 与量化 checkpoint 上进行。量化方法属于 PTQ 范畴，其中 GPTQ 使用 Hessian 近似进行二阶误差补偿，AWQ 通过处理 activation outliers 保护低 bit 权重，BitsAndBytes 提供高效 integer quantization kernel。实验使用 A100/H100；作者报告 neuron-level 和 layer-level attribution 可在 10 小时内完成，LMHR 实验平均约 30 小时。

## 实验关键数据

### 主实验
LRE 一跳事实记忆结果显示，量化整体会降低 factual knowledge recall，但降幅依赖模型大小和量化方法。Qwen2.5-14B 的 GPTQ4 是最极端失败案例，准确率从 73.08% 掉到 25.20%；相对地，bib8 和 GPTQ8 基本贴近 full precision。Llama3-8B 的 4-bit / 8-bit 量化也有 0.67 到 6.23 个百分点的损失。

| 模型 | Full | bib4 | bib8 | GPTQ4 | GPTQ8 | AWQ | 主要观察 |
|------|------|------|------|-------|-------|-----|----------|
| Qwen2.5-7B | 63.25 | 60.72 | 63.01 | 60.10 | 63.22 | 60.60 | 4-bit 与 AWQ/GPTQ4 稳定掉点，8-bit 接近 full |
| Qwen2.5-14B | 73.08 | 70.33 | 73.06 | 25.20 | 73.03 | 70.61 | GPTQ4 严重崩塌，bib8/GPTQ8 几乎无损 |
| Llama3-8B | 77.62 | 72.19 | 76.95 | - | 71.39 | 71.83 | Llama3 对多种量化都有明显但非灾难性下降 |

知识神经元分析给出了机制解释。Qwen2.5-7B 和 Llama3-8B 中，量化后超过 full-precision top-300 阈值的 neuron 数量减少，且 Qwen2.5-7B 的最后两层 attention/FFN contribution score 下降尤其明显。作者还发现，full 模型本来尚未饱和的 relation 更容易在量化后掉点，说明“脆弱知识”比已经稳定掌握的知识更怕数值扰动。

| 分析层面 | Qwen2.5-7B 现象 | Llama3-8B 现象 | 含义 |
|----------|-----------------|----------------|------|
| Top-300 neuron | 量化后超过阈值的高贡献 neuron 变少 | 同样减少，但层分布不同 | 关键事实 neuron 的贡献被压低 |
| Layer-wise drop | 最后两层下降最明显 | 中后层下降，末层可能补偿性上升 | 不同架构事实存储位置不同 |
| Relation sensitivity | 未饱和 relation 掉点更严重 | 类似趋势 | 弱掌握事实更易被量化扰动 |
| 方法差异 | GPTQ4/AWQ/bib4 更伤，bib8/GPTQ8 更稳 | 各方法差异较小但普遍下降 | bit-width 不是唯一因素，量化算法也重要 |

### 消融实验
TwoHop-Fact 结果表明，量化对 first-hop bridge entity 的影响最大，论文主文报告第一跳最高可下降 30.08%，第二跳平均退化仅 4.25%；最终两跳准确率的退化和 bridge entity 预测能力高度相关，Spearman 相关系数为 0.93。值得注意的是，量化并不总是坏事：Llama3-8B 在 GPTQ8 和 AWQ 下的多跳指标反而大幅高于 full。

| 模型 / 方法 | $r_1(e_1)$ ↑ | $r_2(e_2)$ ↑ | $r_2(r_1(e_1))$ ↑ | 关键结论 |
|-------------|--------------|--------------|--------------------|----------|
| Qwen2.5-7B full | 25.03 | 39.07 | 20.61 | 基线 |
| Qwen2.5-7B bib4 / bib8 | 25.01 | 39.02 | 20.61 | 几乎完全保留 two-hop 表现 |
| Qwen2.5-7B AWQ | 22.07 | 38.89 | 18.05 | 第一跳下降带动最终答案下降 |
| Qwen2.5-14B full | 35.23 | 40.45 | 24.72 | 更大 Qwen 模型基线更强 |
| Qwen2.5-14B bib4 / bib8 | 35.16 | 40.56 | 24.76 | 最终两跳略高于 full |
| Qwen2.5-14B AWQ | 24.69 | 35.61 | 21.80 | 第一跳损失很大 |
| Llama3-8B full | 7.79 | 21.39 | 4.45 | full 在该设置下很弱 |
| Llama3-8B GPTQ8 | 23.62 | 40.73 | 20.94 | 量化反而显著提升多跳准确率 |
| Llama3-8B AWQ | 22.35 | 37.79 | 19.56 | 同样出现量化诱导提升 |

### 关键发现
- 量化通常造成事实知识信息损失，但不是线性地“bit 越低越差”。bib8/GPTQ8 常常几乎无损，bib4 在 Qwen 的两跳推理中也很稳。
- 小模型更脆弱。同一 Qwen2.5 家族中，7B 的量化损失更明显；14B 对 bib8/GPTQ8 更稳，但 GPTQ4 失败非常严重。
- 第一跳桥接实体是多跳 FKR 的瓶颈。最终两跳准确率和 $r_1(e_1)$ 的相关性高，说明很多错误不是第二步推理失败，而是第一步没召回中间实体。
- 量化效果高度异质。不同模型、层、relation 和方法都有不同模式，Llama3-8B 甚至出现量化后 EntRec/CnstScore 在深层高于 full 的现象。
- BitsAndBytes 是论文中最稳的实用选择。作者总结 bib 系列在 FKR 保留上整体优于 GPTQ4/AWQ，尤其在 Qwen2.5 上表现稳定。

## 亮点与洞察
- 论文把“量化影响能力”进一步细化为“量化如何改变事实知识的内部存储和调用”。这种分析比只看任务 accuracy 更有诊断价值。
- first-hop bottleneck 的发现很实用。很多多跳任务表面像推理问题，其实第一步事实召回决定了后续路径是否可走。
- 量化偶尔提升 FKR 是一个值得深挖的现象。作者推测可能来自正则化效应或量化噪声，它提示压缩不只是破坏，也可能改变模型的内部路径选择。
- 结果对部署有温和但重要的提醒：常见 PTQ 大多不会让 FKR 完全崩掉，但在知识密集任务中，不能只用通用 benchmark 判断量化是否安全。

## 局限与展望
- 实验只使用英文数据集，无法说明多语言事实知识是否同样稳健。量化已知会影响多语言能力，跨语言 FKR 可能更脆弱。
- 模型范围限制在 Llama3-8B、Qwen2.5-7B 和 Qwen2.5-14B。更大模型、MoE、DeepSeek/Mistral 等家族可能有不同知识存储结构。
- 作者只比较 4-bit 和 8-bit，未覆盖 1-bit、2-bit 等更激进压缩，也没有系统研究 weight-activation quantization、KV cache compression 或 QAT。
- 知识神经元和贡献分数本身是解释性假设，不等于完整机制解释。量化后表征可能重分布，top neuron 下降未必涵盖所有补偿路径。
- TwoHop-Fact 的结果中 Llama3 full 较弱而 quantized 变强，需要更多控制实验确认是量化噪声、checkpoint 差异，还是评测设置带来的偶然收益。

## 相关工作与启发
- **vs Namburi et al. 的 compression cost**: 早期工作关注压缩对参数知识的影响，本文更进一步加入 neuron/layer 归因和 latent multi-hop reasoning，能定位损失来自哪里。
- **vs Singh and Sajjad 的量化解释研究**: 后者关注量化如何改变模型内部行为，本文聚焦 factual knowledge recall，把解释性分析绑定到具体知识任务。
- **vs Yang et al. 的 latent multi-hop reasoning**: Yang 等提出检查 LLM 是否在内部执行多跳推理的方法，本文把该方法用于比较 full 与 quantized 模型的推理路径差异。
- **对后续研究的启发**: 可以为知识密集型应用建立“量化前后 FKR 回归测试”，尤其检测未饱和关系、桥接实体召回和最后几层 contribution score，而不是只跑通用问答集。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 不是提出新量化算法，而是把 FKR、知识神经元和多跳内部路径结合起来做诊断，问题切得很准。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖三类模型、三种 PTQ 方法、两类数据和多层解释分析，但模型家族和语言范围仍有限。
- 写作质量: ⭐⭐⭐⭐☆ 主线清楚，结论克制；部分表格和图在附录中较分散，需要读者来回对照。
- 价值: ⭐⭐⭐⭐☆ 对需要量化部署知识密集型 LLM 的团队很有参考意义，尤其提醒不要只看平均下游分数。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation](interpretable_traces_unexpected_outcomes_investigating_the_disconnect_in_trace-b.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](../../ACL2025/interpretability/an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)

</div>

<!-- RELATED:END -->
