---
title: >-
  [论文解读] Multilingual Unlearning in LLMs: 转移、动力学与可逆性
description: >-
  [ICML 2026][LLM安全][LLM 遗忘] 本文把 TOFU 遗忘基准扩到 5 种语言系统研究「跨语言遗忘转移」，发现遗忘强度随语言族/书写系统亲缘关系而变，且只动用了后段语言特化解码层、几乎不改前中段共享语义空间…
tags:
  - "ICML 2026"
  - "LLM安全"
  - "LLM 遗忘"
  - "跨语言转移"
  - "表征空间"
  - "转向向量"
  - "可逆遗忘"
---

# Multilingual Unlearning in LLMs: 转移、动力学与可逆性

**会议**: ICML 2026  
**arXiv**: [2606.03291](https://arxiv.org/abs/2606.03291)  
**代码**: https://github.com/MLCY1/multilingual-unlearning-in-llms  
**领域**: LLM 安全 / 隐私 / 多语言 LLM  
**关键词**: LLM 遗忘, 跨语言转移, 表征空间, 转向向量, 可逆遗忘

## 一句话总结
本文把 TOFU 遗忘基准扩到 5 种语言系统研究「跨语言遗忘转移」，发现遗忘强度随语言族/书写系统亲缘关系而变，且只动用了后段语言特化解码层、几乎不改前中段共享语义空间，因此能用一个推理时的转向向量恢复 Qwen 上 50%、Gemma 上 90% 的被遗忘知识——说明现有 LLM 遗忘本质是「表面抑制」而非真擦除。

## 研究背景与动机

**领域现状**：LLM 训练吸收的大量数据可能包含敏感事实，加上 GDPR「被遗忘权」的合规要求，催生了不重训而抹除特定知识的「LLM unlearning」研究。主流方法（GA、NPO、DPO 风格）都是在微调过的模型上加修改目标，鼓励模型在 forget 集上不再吐露目标内容。

**现有痛点**：(1) 现有评测几乎只在英文上做，多语言场景下「遗忘究竟转移到了什么程度」无人系统刻画——而真实部署里同一条敏感事实经常以多语形式重复出现；(2) 即使在单语言下，已有工作零星显示「遗忘像抑制信号」，但缺少机制定位（在哪些层？是否语言无关？）和无须重学的可逆性证据。

**核心矛盾**：如果多语言遗忘只动了「语言特化解码层」，那共享语义空间里的知识就还在，攻击者用另一种语言提问、或在推理时反向 steering 就能把它拽回来；如果它真正改变了「跨语言概念空间」，那遗忘的安全保证就强得多。两种情形对部署风险评估完全不同，但既往工作没区分。

**本文目标**：(i) 系统刻画跨语言遗忘转移随语言族/书写系统/预训练覆盖率的规律；(ii) 用机制可解释性定位遗忘动作发生在哪些层；(iii) 用一个简单的推理时转向向量验证遗忘是否可逆，并测它跨语言的传递性。

**切入角度**：把 TOFU（200 个虚构作者各 20 个 QA）翻译到 5 种语言（EN/CH/DE/RU/TU），三轴正交受控——共享语族 vs 共享书写 vs 都不共享；分别在某语言微调、再在某语言遗忘、再在某语言询问，得到 $5\times 5\times 5$ 的转移矩阵；并用 NLI 而非 lexical overlap 评估语义等价。

**核心 idea**：把 fine-tuned 和 unlearned 模型在同一题上的隐表征之差当作「遗忘方向」（steering vector），加权地反向 inject 回 forward pass。如果这个方向是「语言无关的抑制方向」，那它在任何语言下都能恢复知识——这正是论文要验证的假说。

## 方法详解

### 整体框架

实验流程分四步：(1) 用同一份 TOFU 双语数据在某 $\mathcal{L}_{FT}$ 上 LoRA 微调，得到 $f_{\text{ft}}$；(2) 用 DPO 风格 unlearning 目标 $J_{UN}$ 在 $\mathcal{L}_{\text{unl}}$ 上抹除 1% forget 作者，得到 $f_{\text{un}}$；(3) 在 $\mathcal{L}_Q$ 上用 NLI 评估 forget/retain 准确性，得到转移矩阵；(4) 抽取每层隐表征做余弦相似度分析 + 构造转向向量做可逆性实验。

所有实验在 Qwen2.5-7B 和 Gemma2-9B 上跑，遗忘目标为 $\arg\min_\theta \frac{1}{|\mathcal{L}_{\text{unl}}|} \sum_{\ell} (\mathbb{E}_{D_\ell^{\text{forget}}} J_{\text{forget}} + \lambda \mathbb{E}_{D_\ell^{\text{retain}}} J_{\text{retain}})$，其中 $J_{\text{forget}}$ 用 DPO 偏好「IDK 拒答」胜过「真实答案」。

### 关键设计

1. **三轴受控的多语言遗忘转移矩阵**:

    - 功能：把「语言关系如何影响遗忘转移」这个开放问题拆成三个可观测维度。
    - 核心思路：选 5 种语言覆盖「同族 + 同书写」（EN/DE）、「同族 + 异书写」（EN/RU）、「异族 + 同书写」（EN/TU）、「都不共享」（EN/CH）四种组合；对每种 $(\mathcal{L}_{FT}, \mathcal{L}_{\text{unl}}, \mathcal{L}_Q)$ 三元组报告 NLI 分数随 unlearning 的变化（更负代表更强遗忘），并用 5 次随机 forget 集抽样估方差。
    - 设计动机：以往工作只测英语单点，无法分清「书写系统效应」「语族效应」「预训练覆盖率效应」哪个起作用；这个 $5\times 5\times 5$ 矩阵让所有三类效应独立可观察，是后续机制分析的实验基石。

2. **跨语言提示作为「输出端」诊断**:

    - 功能：判断模型究竟是「不知道答案」还是「知道但解码不出来」。
    - 核心思路：查询用语言 $q$ 但要求模型用微调语言 $\ell$ 回答，记录性能增益 $\Delta_{\ell \leftarrow q}$。若 $\Delta > 0$ 大，说明知识在共享空间里完好，只是解码受语言绑定；论文进一步算出 $\Delta_{\ell \leftarrow q}$ 与遗忘转移矩阵中对应单元的相关系数（Pearson $r=0.50$，Spearman $\rho=0.60$，均显著），把「共享语义空间」和「转移强度」直接挂钩。
    - 设计动机：解码绑定假说只能间接观测；用「换语言作答」直接打开瓶颈，证明遗忘的损害是经过共享空间传到下游解码层。

3. **遗忘方向 = 表征差，推理时 steering 验证可逆性**:

    - 功能：把「遗忘是抑制」从假设升级为机制级证据，并定量恢复被遗忘的知识。
    - 核心思路：对同一 forget 问题在 $f_{\text{ft}}$ 和 $f_{\text{un}}$ 上取层 $l$ 的最终 token 隐状态差 $d^{(l)} = h_{\text{ft}}^{(l)} - h_{\text{un}}^{(l)}$，沿这个方向在 $f_{\text{un}}$ 的 forward pass 里 inject 一个加权扰动；若沿单一方向 + 跨语言都能恢复，则遗忘是「语言无关的表面抑制」。逐层余弦相似度分析显示，$f_{\text{un}}$ 在前中段几乎与 $f_{\text{ft}}$ 重合，分歧集中在最后若干解码层。
    - 设计动机：以往「可逆性」证据要么靠 brief relearning（仍然要数据），要么靠答案前缀诱导（需先知道答案）；本工作给出的 single direction inference-time steering 既无须 forget 数据、也无须答案，并能跨语言迁移，这是对「LLM 遗忘是否真的擦除」最强力的反例。

### 评估指标

用多语 NLI 模型（xlm-roberta-large-xnli）判断生成答案 $\hat y$ 与 ground truth $y$ 是否互蕴，避开 lexical overlap 在多语下的虚高/虚低；native speaker 在 50 样本上校验了 NLI 的可靠性。

## 实验关键数据

### 主实验：跨语言遗忘转移（Qwen2.5-7B）

| FT \ Unlearn | EN 查询 | CH 查询 | DE 查询 | RU 查询 | TU 查询 |
|--------------|---------|---------|---------|---------|---------|
| EN / EN | **-90** | -4 | -7 | -9 | -4 |
| EN / CH | -7 | **-8** | +1 | -5 | -3 |
| EN / DE | **-17** | -6 | **-4** | -5 | -4 |
| DE / EN | -13 | -4 | **-41** | -7 | 0 |
| TU / EN | -10 | -2 | -1 | -6 | **-55** |
| CH / TU | -1 | **+6** | -4 | -4 | 0 |

所有数字为 NLI 分数相对 fine-tuned base 的绝对降幅（更负=遗忘更强）。可以看到：(1) 同族同书写转移最强（EN→DE、EN→EN）；(2) 高覆盖语言遗忘（EN/CH）转移强于低覆盖（DE/RU/TU）；(3) 弱语言上的遗忘仍能反向影响强语言（TU/EN cell -55）。

### 跨语言提示增益 $\Delta_{\ell \leftarrow q}$

| FT \ Query | EN | CH | DE | RU | TU |
|------------|----|----|----|----|----|
| EN | — | +29 | +61 | +30 | +27 |
| CH | +11 | — | +10 | +12 | +12 |
| DE | +33 | +22 | — | +5 | +18 |
| RU | +20 | +8 | +15 | — | +7 |
| TU | +33 | +11 | +22 | +17 | — |

显著的正增益证明知识在共享语义空间里完好，缺的是语言特化解码；与遗忘转移矩阵的相关系数 Pearson $r=0.50$、Spearman $\rho=0.60$（均 $p<0.05$）。

### 可逆性实验：单一 steering 方向恢复多少知识

| 模型 | 恢复率（forget NLI 反弹） | 跨语言传递？ | 是否需要 forget 数据 |
|------|---|---|---|
| Qwen2.5-7B | $\approx 50\%$ | 是 | **否** |
| Gemma2-9B | $\approx 90\%$ | 是 | 否 |

### 关键发现

- **语族 + 书写系统双重影响**：控制书写后，EN→RU（同族异写）仍比 EN→CH（都不共享）强；控制语族后，EN→TU（同写异族）仍比 EN→CH 强。两轴都独立贡献。
- **不对称转移**：高覆盖语言（EN、CH）做遗忘源更强力，低覆盖（DE/RU/TU）相对弱——与「模型在主导语言锚定的共享空间里推理」假说吻合。
- **「我不会答」仍然能转移遗忘**：在 TU 微调模型上 EN 查询 base 只有 11% NLI，但在 EN 上遗忘后 TU 查询竟然下降 55%——验证共享概念空间假说。
- **层定位**：$f_{\text{un}}$ 与 $f_{\text{ft}}$ 在前中段余弦相似度几乎一致，仅最后几层显著分歧——遗忘的破坏面集中在「概念→语言特化输出」这一步。
- **可逆性**：Gemma 上 90% 的恢复率意味着「unlearning」对 Gemma 几乎是 cosmetic 的；Qwen 上 50% 也足够构成实质安全风险。

## 亮点与洞察

- **首个系统化的多语言遗忘转移图谱**：把语族、书写、覆盖率三轴解耦做出 $5\times 5\times 5$ 转移矩阵，给后续工作提供了清晰的对照基准。
- **机制定位 + 行为级证据闭环**：从隐表征逐层分析定位「后段解码层是抑制发生地」，再用跨语言提示在输出端验证，最后用 steering 把假说做成实操，证据链非常完整。
- **可逆性实验拆穿「遗忘」幻觉**：不靠 relearning、不靠答案前缀、仅用一个推理时方向就能恢复，且跨语言传递——这是对当前 unlearning 安全主张最直接的反例，比起 brief relearning 攻击更轻、危害更大。
- **NLI 评测**：避开 lexical overlap 在跨语言下的失真，对未来多语生成评估有方法论意义。
- **对抗视角的实际意义**：表明在多语模型部署「被遗忘权」时，仅做英文遗忘几乎等于没遗忘，必须把所有可能查询语言一起覆盖；甚至 steering 攻击让现有方法基本失效。

## 局限与展望

- **任务面窄**：只测 TOFU 类合成传记知识，对真正的「敏感事实」「PII」「版权文本」等情形未必同构；不同知识可能存在层级分布不同。
- **方法面窄**：只覆盖 DPO/GA/NPO 三类基于继续微调的方法，对 representation misdirection 类（如 RMU）或参数定位类方法（如 ROME-style）尚未系统验证。
- **5 种语言仍是抽样**：缺少低资源语言（如非洲、东南亚语系），可能错失「极低覆盖语言遗忘几乎不转移」这种潜在重要案例。
- **steering 方向恢复率差异（Qwen 50% vs Gemma 90%）解释不足**：可能与训练数据多语比例、模型架构、对齐流程都相关，但未深入。
- **真实威胁模型缺失**：steering 攻击假设攻击者能拿到 $f_{\text{ft}}$ 和 $f_{\text{un}}$ 两份检查点，部分场景（API 服务）不一定成立；下一步该测能否仅用 query 黑盒反推方向。

## 相关工作与启发

- **vs 单语 LLM unlearning（GA/NPO/DPO 系列）**：首次把这些方法搬到多语场景并指出转移不均衡、对齐攻击更危险。
- **vs Hu et al. 2025 等单语 suppression 假说**：从「单语经验观察」升级到「多语机制证据 + 可逆性证据」，并用 layer-wise 分析定位到具体层段。
- **vs Wendler et al. 2024 / Lim et al. 2025 多语 LLM 共享语义空间论**：把这些「正向」表征理论用作「负向」的安全分析工具，思路可推广到其他跨语言鲁棒性问题。
- **vs Seyitoğlu et al. 2024（steering 检索匿名概念）**：他们利用模型已有世界知识；本文反过来用 fine-tuned/unlearned 表征差构造方向，避免需要广义先验，方法更通用。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 第一次系统化解耦多语言遗忘转移的三类影响因素，并用 single-direction 推理时 steering 给出强力可逆性证据。
- 实验充分度: ⭐⭐⭐⭐⭐ 双模型 × 5 语言 × 三种遗忘目标 × NLI/层分析/steering 三种验证视角，关键结论全有消融。
- 写作质量: ⭐⭐⭐⭐ 数学符号清晰，但 5×5×5 矩阵颜色编码对纸面阅读不友好，部分关键 cell 在表里要回查才能 follow。
- 价值: ⭐⭐⭐⭐⭐ 直接挑战当前 LLM unlearning 的安全主张，对合规部署和后续防御研究都是必读，开源代码降低复现门槛。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs](../../AAAI2026/llm_safety/fedp2eft_federated_learning_to_personalize_peft_for_multilingual_llms.md)
- [\[ACL 2026\] CAP: Controllable Alignment Prompting for Unlearning in LLMs](../../ACL2026/llm_safety/cap_controllable_alignment_prompting_for_unlearning_in_llms.md)
- [\[ICML 2026\] Efficient DP-SGD for LLMs with Randomized Clipping](efficient_dp-sgd_for_llms_with_randomized_clipping.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)
- [\[ICML 2026\] Position: Uncertainty Quantification in LLMs is Just Unsupervised Clustering](position_uncertainty_quantification_in_llms_is_just_unsupervised_clustering.md)

</div>

<!-- RELATED:END -->
