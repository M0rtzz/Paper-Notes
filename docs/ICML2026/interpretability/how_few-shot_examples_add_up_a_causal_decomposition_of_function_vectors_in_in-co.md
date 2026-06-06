---
title: >-
  [论文解读] How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning
description: >-
  [ICML 2026][可解释性][function vector] 本文从 prompt 粒度因果分析 n-shot prompt 的 function vector（FV）形成机制，证明 FV 可线性叠加为各 example 子 FV 的加权和，且权重由 FV-head attention 决定…
tags:
  - "ICML 2026"
  - "可解释性"
  - "function vector"
  - "in-context learning"
  - "线性叠加"
  - "注意力机制"
  - "QK/V 因果分解"
---

# How Few-Shot Examples Add Up: A Causal Decomposition of Function Vectors in In-Context Learning

**会议**: ICML 2026  
**arXiv**: [2605.16591](https://arxiv.org/abs/2605.16591)  
**代码**: 无  
**领域**: 可解释性 / In-Context Learning 机制 / Function Vector  
**关键词**: function vector, in-context learning, 线性叠加, attention 重加权, QK/V 因果分解

## 一句话总结
本文从 prompt 粒度因果分析 n-shot prompt 的 function vector（FV）形成机制，证明 FV 可线性叠加为各 example 子 FV 的加权和，且权重由 FV-head attention 决定；并通过 2×2 QK/V 因果干预表明，contextualization 主要通过 QK 路径（而非 V）让模型把注意力集中到无歧义的 demonstration 上，从而提升 FV 质量。

## 研究背景与动机

**领域现状**：function vector（Todd et al. 2024；Hendel et al. 2023）已被确立为 in-context learning（ICL）的因果机制——在一组 FV-head 上的平均激活构成一个可注入的"任务方向"，把它加到 0-shot prompt 的 residual stream 即可恢复 few-shot 行为。Bakalova et al. (2025) 进一步把 ICL 拆成两阶段：低层让每个 example 互相吸收上下文（contextualization），中层把这些表示聚合到最后一个 token（aggregation）。

**现有痛点**：FV 是怎么从 n 个 few-shot example 中"长出来"的，至今缺一个 prompt 级的机械解释。具体而言：(i) FV 是不是每个 example 独立贡献再相加？还是必须经过非线性融合？(ii) contextualization 究竟通过什么通道改善 ICL——是改写每个 example 的 Value 内容，还是改写 Query/Key 之间的路由（attention 分配）？(iii) 当 prompt 里既有信息丰富的 example 又有歧义 example 时，模型用什么机制把权重压在前者上？

**核心矛盾**：现有理论文献（线性回归 ICL、softmax 检索）把 ICL 当作"基于 query 和 key 的相似度检索"，但实际 LLM 的 FV-head 注意力又强烈受 recency bias 主导，且 example 之间的 contextualization 显然在改变最终 FV——单一的"query-driven retrieval"图像并不足以解释。

**本文目标**：(1) 给出 FV 在 prompt 级别的可加性证据（表示+因果）；(2) 找出 contextualization 影响 FV 的具体通道；(3) 验证 contextualization 是通过 attention 重加权把质量提上去的，而不是简单地往 Value 里塞更多信息。

**切入角度**：作者引入 **uncontextualized 消融**——通过 attention edge ablation 把跨 example 的注意力边清零，但保留 example 内和到最后一个 token 的注意力——作为"无上下文化"的反事实基线，与完整模型对比即可干净地隔离 contextualization 的因果贡献。再叠加 OLS 拟合 + Q/K/V patching + Shapley 分解，三种工具串起来形成可证伪的因果链。

**核心 idea**：用 **"线性叠加 + attention 重加权 + QK/V 因果分解"** 三件套，把"few-shot prompt → FV"这个黑箱拆成可验证的加法结构，并定位重加权效应主要发生在 Query–Key 路由通道。

## 方法详解

### 整体框架
作者把 FV 的形成视作 FV-head 在最后一个 separator 位置上的注意力聚合过程：每个 example 通过 attention 把自己的 Value 写到最后 token 的 residual stream，所有 FV-head 输出加起来就是 $v_{FV}(p)$。整个分析由三层组成：

1. **表示层**：对一批 prompt 提取 $v_{FV}$ 与每个 example 的子 FV $v_i$，用 OLS 拟合 $v_{FV}\approx \sum_i w_i v_i$，看可加性是否成立。
2. **因果层**：把 OLS 重构出的 $\hat v_{FV}=\sum_i w_i v_i$ 注入到 0-shot prompt，对比它与真实 $v_{FV}$ 的 injection accuracy，验证可加性在因果上也成立。
3. **机制层**：用 Q/K/V patching + 2×2 factorial 干预把 contextualization 的效应拆到 QK 通道和 V 通道，并用 Shapley 值量化两者贡献。

模型覆盖 gemma-2-{2b,9b,27b}、Llama-3.2-{1B,3B}、Llama-3.1-8B-Instruct；任务覆盖 CC/PS/PC/PP/EN-FR 等 10 个 task 家族，其中带 -A 后缀的是"歧义版"（混入与多个 candidate mapping 兼容的 ambiguous example，与无歧义 example 比例固定为 2:1，作为压力测试）；n-shot 取 3/5/10。

### 关键设计

1. **Per-prompt sub-FV + OLS 线性叠加**：

    - 功能：把 n-shot prompt 的 FV 分解为各 example 独立贡献之和。
    - 核心思路：通过 attention mask 让最后一个 token $t_{n+1}$ 只能 attend 到第 $i$ 个 example 和 query $x_{n+1}$，得到该 example 的子 FV $v_i$；然后跨 prompt 用 OLS 拟合全局权重 $w_i$，即 $v_{FV}\approx \sum_{i=1}^n w_i v_i+\varepsilon$。权重在一个 batch 上拟合，因此描述的是"位置 $i$ 的 example 在平均意义上贡献多少"，而不是 per-prompt 的拟合。表示层（cosine、$R^2$）和因果层（注入 $\hat v_{FV}$ 后的准确率比 ratio）都要验证。
    - 设计动机：先把 FV 形成机制从"未知非线性"压到"可解释的加性结构"。一旦可加性成立，后面对 contextualization 的研究就可以等价转化成"研究权重 $w_i$ 如何变化"，从而把全局 FV 问题归约到 attention 分配问题。

2. **Uncontextualized 消融 + attention 边遮罩**：

    - 功能：构造一个干净的反事实基线，让 example 之间互不传递信息，但 example 内部和到最终 token 的信息流保持原样。
    - 核心思路：通过 attention edge ablation，把跨 prompt component（不同 example 之间）的注意力权重清零，同时保留 (i) 同一 component 内部的注意力、(ii) 到最后一个 token 的注意力。这样所有跨 example 的"contextualization"被切断，但 per-example 编码和 aggregation 步骤本身不被破坏。
    - 设计动机：直接对比 contextualized vs uncontextualized 两个模型，差异就只能归因于 contextualization 这一个因素——这是因果干预，不是 correlation。它给出了 contextualization 效应的 ground truth 测量平台，也让后续 Q/K/V patching 有了可移植的"上下文化版/未上下文化版"激活源。

3. **2×2 QK vs V 因果分解 + Shapley 值**：

    - 功能：把 contextualization 对 FV 质量的总收益 $G=F(1,1)-F(0,0)$ 拆解到 QK 通道（attention 路由）和 V 通道（聚合内容）。
    - 核心思路：把 FV-head 上的 contextualization 状态写成两个二值变量 $(QK, V)\in\{unc,ctx\}^2$，通过 Q/K/V patching 在 FV-head 上独立替换 Query/Key 或 Value 的激活源，得到 4 个配置 $F(0,0), F(0,1), F(1,0), F(1,1)$。对每个配置评估最大 FV injection accuracy，再算 QK 和 V 各自的 Shapley 值 $\phi_{QK}, \phi_V$（即两条路径在所有联立组合下的平均边际贡献）。
    - 设计动机：传统消融只能告诉你"去掉 X 掉了多少"，无法分离"X 起作用是因为它改了 attention 路由还是改了内容"。Shapley 分解把这两条物理上耦合的通道在因果效应层面解耦，从而能精确回答"contextualization 改善 FV 主要走哪条路"。结论是 QK 通道贡献更一致、更显著（尤其在 ambiguous 任务上），V 通道贡献则在不同任务间高度异质。

### 损失函数 / 训练策略
本文不训练模型，所有分析都在 frozen 预训练 LLM 上做因果干预。OLS 用闭式解全局拟合；FV injection 的 $(\ell,\alpha)$ 通过 layer/scale sweep 选最大准确率；entropy 用归一化注意力熵 $\hat H=-\sum p_i\log p_i/\log n\in[0,1]$ 量化 attention 平滑度；超参细节在 Appendix E.2/E.3。

## 实验关键数据

### 主实验
（在 gemma-2-2b 上代表性结果，10 个任务、3/5-shot、contextualized 与 uncontextualized 各一份；完整结果见 Appendix F/H。）

| 验证维度 | 指标 | 结果 | 说明 |
|---|---|---|---|
| FV 线性叠加（表示层） | 平均 cosine($v_{FV}, \hat v_{FV}$) | $\geq 0.925$ | OLS 重构的 FV 在所有 10 个任务、3/5-shot、ctx/unc 设置下都高度对齐 |
| FV 线性叠加（表示层） | 平均 $R^2$ | $\geq 0.875$ | 同上 |
| FV 线性叠加（因果层） | $Acc_{max}(\hat v_{FV}) / Acc_{max}(v_{FV})$ | 接近 1（ctx 设置） | 注入重构 FV 能回收真实 FV 的大部分因果效应 |
| 鲁棒性扩展 | 20-shot 线性性 | 仍然显著（CC/PP-A 上 cosine/$R^2$/causal ratio 均保持） | 加性结构不是 short-prompt artifact |
| 歧义任务上的注意力集中 | 对 unambiguous example 注意力占比 | unc 32% → ctx 61%（10-shot, PRESENT-PAST-A） | contextualization 把注意力锐化到信息丰富 example |
| 正常任务上的注意力分布 | 归一化熵 $\hat H$ 变化 $\Delta\hat H$ | $\approx 0$（保持 $>0.95$） | contextualization 在普通任务里只做位置再平衡，不锐化 |
| 正常任务的位置中心漂移 | $\Delta C$（10-shot） | $-0.4\sim -0.6$ | attention 质量从靠后位置向前移，缓解 recency bias |
| 歧义任务上的熵下降 | $\Delta\hat H$（3→10 shot） | $-0.08\sim -0.15$ | 与正常任务方向相反——歧义场景下 contextualization 是"选择机制" |

### 消融实验
| 配置 | FV injection accuracy | 注意力分配（amb vs unamb） | 结论 |
|---|---|---|---|
| Unintervened（PRESENT-PAST-A） | 0.52 | 0.94（unamb 占比高） | 完整 contextualized baseline |
| Ablate examples only | 0.19 | unamb 占比剧降，amb 反升 | example 内容决定"该选谁"——example-driven reweighting |
| Ablate $x_{n+1}$ only | 0.39 | unamb 占比基本不变，总 attention mass 下降 | query 主要决定"对 example 整体的注意强度"，不决定偏好 |
| CAP-A 任务下 ablate $x_{n+1}$ | 比 ablate examples 掉得更多 | 偏好被 $x_{n+1}$ 破坏 | 例外：在 capitalization 任务里 query 自身携带消歧信号 |
| 2×2 因果分解（Shapley） | $\phi_{QK}$ 整体显著为正 | — | 改 QK 路由 → FV 质量上升一致 |
| 2×2 因果分解（Shapley） | $\phi_V$ 任务间异质，部分接近 0 甚至为负 | — | 改 V 内容 → 不稳定，未必有增益 |

### 关键发现
- **可加性 + 因果性双重成立**：FV 不仅在几何上能用 sub-FV 的线性组合逼近，注入实验也证明这种线性组合在因果上几乎等价于真实 FV。这把 ICL 的复杂聚合机制压缩成了"加权求和"这一极其简洁的模型。
- **contextualization 在两种 prompt 下行为对立**：普通 prompt 上它"摊平"注意力（抗 recency），歧义 prompt 上它"锐化"注意力（选 informative example）。同一个机制根据任务可识别度切换角色，呈现 selection vs balancing 的二象性。
- **QK 通道是主因，V 通道是辅料**：Shapley 分解在所有 6 个评估模型上一致显示 QK 贡献更稳健、V 贡献更异质，这把"contextualization 提升 ICL"的归因从模糊的"上下文化让表示更好"细化为"上下文化主要改了路由"。
- **example-driven vs query-driven 的二分**：在大多数歧义任务里，破坏 example 比破坏 query 对 FV 质量伤害更大，说明 robust ICL 的核心机制是 example 之间互相比较后由 example 端"自荐"，而非 query 主动检索；CAP-A 是少数例外。

## 亮点与洞察
- **"sub-FV + OLS"是简单却致命的拆解工具**：用全局拟合权重 $w_i$ 而不是 per-prompt 拟合，让 FV 形成具有"位置贡献可比"的内在结构；这套拆解可以无缝迁移到任何"激活在多个 token 上聚合"的 circuit 分析（CoT、retrieval-augmented attention head 等）。
- **uncontextualized 消融是 attention edge ablation 的精准应用**：相比简单地把整个 layer 或整个 head 关掉，按边粒度切断"跨 component 注意力"既保留了模型的执行能力，又干净地隔离了一个特定的语义通道——这种"剪边不剪点"的消融模式值得在 mech interp 圈里推广。
- **2×2 factorial + Shapley 是把 QK 与 V 这对耦合通道分离的标准范式**：很多 attention 解释工作把 QK 和 V 混在一起讨论，本文证明它们对下游任务质量的贡献完全可以差异巨大，未来分析 induction head、reasoning head 时都应该照搬这种 factorial 思路。
- **"歧义 prompt"是测 attention 选择性的好压力测试**：在普通任务上 recency 主导，差异看不出来；只要把 ambiguous/unambiguous 例子按 2:1 混入，几乎所有 attention 效应都被放大成可测信号。这种"刻意制造冲突"的数据构造法对评估其他 attention 机制（routing、MoE、retrieval）同样适用。

## 局限与展望
- **任务局限于 token-级映射**：所有 10 个任务都是 $x\to y$ 的简短映射（country/capital、翻译、动词变形等），FV 这个概念本身就是基于这种简单任务定义的；线性叠加性是否延伸到 chain-of-thought、多轮对话、长上下文推理等更复杂场景，本文没有答案。
- **uncontextualized 消融可能 over-restrictive**：边粒度遮罩同时关掉了"良性的早期 example 信息流"和"含混的跨 example 干扰"，差异并不能完全归到"有用的"contextualization；可以考虑分层、分头的 selective ablation 来进一步细分。
- **Shapley 计算只考虑 QK/V 两条通道的 2×2**：实际 FV-head 之间还有 inter-head 协作、MLP 后处理等通路，本文都被并入"环境"，可能低估或高估特定通道的真实贡献；扩展到更高阶 factorial 是自然的下一步。
- **CAP-A 这个反例说明 query-driven 路径在某些任务确实占主导**：作者将其归因于"capitalization 任务的 query 本身就是消歧信号"，但缺少更系统的分类——什么任务族会触发 query-driven、什么会触发 example-driven，目前仍是经验性的。

## 相关工作与启发
- **vs Todd et al. 2024 / Hendel et al. 2023（FV 的提出）**：他们建立了 FV 这个概念并把它当作"任务方向"使用，但 FV 在 prompt 级是怎么形成的、是不是 per-example 可加的，他们没回答。本文把 FV 的形成机制下推到 per-example 粒度，并用因果干预验证可加性。
- **vs Bakalova et al. 2025（contextualization vs aggregation 二阶段假说）**：他们提出 ICL 有 contextualization 与 aggregation 两阶段，但 contextualization 的"功能"仍然不清楚。本文用 2×2 QK/V 干预精确指认 contextualization 主要通过 QK 路由发挥作用，把这个二阶段假说从描述层升格到机制层。
- **vs ICL 的线性回归理论（Mahankali et al. 2024；Von Oswald et al. 2023 等）**：理论文献用单层线性 attention 推出 $\hat y_{n+1}=\eta\sum y_i x_i^Tx_{n+1}$ 这种"相似度检索"形式，本质上是 query-driven。本文实证发现真实 LLM 在歧义场景下反而是 example-driven，与该理论的隐含图像产生张力，提示理论需要纳入"example 之间互相竞争"这一维度。
- **vs Dragutinovi´c et al. 2025（softmax attention 的 kernel 检索）**：同样属于 query-driven 检索图像，本文给出的 CAP-A vs 其他任务的对比提示——query-driven 和 example-driven 可能是任务依赖的两种 regime，而非单一描述。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把 FV 形成机制下推到 prompt 级线性叠加，并用 2×2 QK/V Shapley 分解一锤定音指出 QK 路由是 contextualization 主因。
- 实验充分度: ⭐⭐⭐⭐ 6 个模型 × 10 个任务 × 3/5/10-shot × ctx/unc 全交叉，OLS/causal injection/Shapley/Q-K-V patching 多种工具互证；缺一点对长 prompt 与复杂推理任务的延伸。
- 写作质量: ⭐⭐⭐⭐ Figure 1 的三联图把核心 claim 讲得很清楚，section 之间的因果链推进顺畅；公式与术语密度较高，初读门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 给 ICL/FV 这条 mech interp 主线提供了 prompt 级的可加性 + 机制定位，方法学（sub-FV、edge ablation、QK/V factorial）可直接迁移到其他 attention circuit 分析。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Letting Tutor Personas Speak Up for LLMs: Learning Steering Vectors from Dialogue via Preference Optimization](../../ACL2026/interpretability/letting_tutor_personas_speak_up_for_llms_learning_steering_vectors_from_dialogue.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](singular_vectors_of_attention_heads_align_with_features.md)
- [\[ICML 2026\] Dissecting Multimodal In-Context Learning: Modality Asymmetries and Circuit Dynamics in modern Transformers](dissecting_multimodal_in-context_learning_modality_asymmetries_and_circuit_dynam.md)
- [\[ICML 2025\] On the Power of Context-Enhanced Learning in LLMs](../../ICML2025/interpretability/on_the_power_of_context-enhanced_learning_in_llms.md)

</div>

<!-- RELATED:END -->
