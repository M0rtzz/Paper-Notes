---
title: >-
  [论文解读] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
description: >-
  [CVPR 2026][强化学习][ECRD] 提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据…
tags:
  - "CVPR 2026"
  - "强化学习"
  - "ECRD"
  - "visual grounding"
  - "hallucination mitigation"
  - "training-free"
  - "evidence pool"
---

# See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

**会议**: CVPR 2026  
**arXiv**: [2602.21497](https://arxiv.org/abs/2602.21497)  
**代码**: [GitHub](https://github.com/uuuuZYC/See-It-Say-It-Sorted)  
**领域**: 强化学习  
**关键词**: ECRD, visual grounding, hallucination mitigation, training-free, evidence pool

## 一句话总结

提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据，无需训练即可在多个LVLM上显著减少视觉幻觉、提升推理准确率。

## 研究背景与动机

**大视觉语言模型（LVLM）**已经能生成长链式思维（CoT）推理，但存在一个根本性问题：**推理-感知漂移**。在长文本解码过程中，模型需要平衡图像、增长的文本上下文和指令三个竞争性上下文。随着上下文变长，微妙但关键的视觉线索容易被语言先验淹没。一旦某个中间推理步骤偏离了视觉证据，即使后续推理逻辑上正确，最终答案也会错误——这就是**视觉幻觉传播**。

**现有方案**主要是通过RL训练模型学会"用图像思考"——让模型学习何时放大/裁剪图像并将裁剪区域重新注入推理上下文。代表工作如PixelReasoner和DeepEyes。但这类方法有三个痛点：（1）需要大量标注数据和奖励设计，训练成本高；（2）策略与特定backbone紧密耦合，难以迁移；（3）反复编码裁剪区域带来严重推理延迟。

**本文的切入角度**根本不同：不在训练时学习何时看图，而是在**测试时**用视觉证据监督每个推理步骤。核心idea是将解码过程重构为一系列"证据驱动的token选择"：维护一个文本形式的证据池，在每步解码时与模型的原始分布协商，用不确定性信号触发新证据的获取。

## 方法详解

### 整体框架

ECRD 把幻觉抑制从"训练时学会看图"挪到了"解码时用证据盯着每个 token"。它在一个**冻结**的 LVLM 外面套一层轻量监督逻辑，不改模型权重，而是逐 token 介入采样：先用 knee truncation 把 base 分布里那条又长又平的尾巴砍掉、只留几个真正有竞争力的候选；再让分布监督器拿当前证据池给这几个候选重新打分、和模型原始分布协商出最终概率；只有当协商完仍分不清谁对谁错时，才唤醒视觉决策器回到图像里抠一条新证据补进池子。整条链路上证据始终以**文本**形式累积，因此后面的 token 能直接复用前面看到的东西，而不必反复把图像裁剪重新编码一遍。

### 关键设计

**1. Knee truncation：先把候选集收敛到"值得协商"的那几个。**

LVLM 的输出分布常常是一个尖峰加一条很长的低概率尾巴，如果对全词表做证据重加权，既贵又会被噪声 token 干扰。ECRD 在每步先按概率排序找到分布的"膝点"——曲率骤降的位置——把候选数 $k^*$ 截到这里，只保留头部真正有歧义的几个 token。这一步本身就有收益（消融里"仅 knee truncation"已带来部分提升），更关键的是它把后续昂贵的证据协商和视觉决策都限定在小候选集上，让整个框架的开销可控。

**2. Distribution Supervisor：用证据池给候选 token 重新打分，并按模型自信度决定听谁的。**

这是抑制幻觉的核心环节，针对的就是"语言先验淹没视觉线索"这个痛点。对证据池里的每条证据 $\mathcal{E}$，它计算候选 token $w$ 在该证据前缀各位置上的平均生成概率

$$q_\mathcal{E}(w) = \frac{1}{L}\sum_{j=1}^{L}p_{\text{VLM}}(w \mid e_{<j}),$$

跨所有证据求平均再 softmax，得到一个由证据诱导的分布 $r_i$——它表达的是"如果只看证据，下一个 token 该是谁"。接着关键的一步是**自适应混合**：直接拿 base 分布的最大概率 $p_{(1)}$ 当混合权重 $\alpha_i = p_{(1)}$，

$$p_i^{\text{mix}} = \alpha_i\, p_i + (1-\alpha_i)\, \tilde{r}_i.$$

含义很直白——base 分布越尖锐（$p_{(1)}$ 大）说明模型越自信，就越保留它的原始行为；base 分布越平（$p_{(1)}$ 小）说明模型在犹豫、正是幻觉高发处，就把更多权重交给证据。这套"自信不插手、犹豫才强干预"的规则不需要任何可学习超参数，也解释了为什么在 OCR 这类模型本就 confident 的子任务上 ECRD 不会拖后腿——权重自动退化成几乎纯 base 分布。

**3. Visual Decider：协商仍分不清时，才回图像里抠一条文本微证据。**

重加权之后如果候选里仍有两个 token 难分伯仲，就说明现有证据不够、需要新观察。触发条件是 $k^* > 1$ 且混合后 top-2 的概率差 $\Delta_i \leq \delta$（$\delta$ 为不确定性阈值）。此时调用基于 Qwen2.5-VL-3B 的 GRIT 决策器，喂给它图像、文本前缀的尾部和当前候选集，让它输出两样东西：当下该选的 token $w^*$，以及一条人类可读的微证据句子 $\mathcal{E}_i$。关键在于证据是**文本句子**而不是像素裁剪——它被追加进证据池后能被后续任意 token 直接引用，既让干预轻量、可验证，又彻底避开了 RL 方法反复编码裁剪区域的延迟开销。

**4. Dynamic Evidence Pool：证据按推理需求增长，而非一开始喂满。**

证据池初始化只放一条全局图像描述 $d_{\text{global}}$ 提供广覆盖，之后**只在不确定性触发时**才增长：$E_{i+1} \leftarrow E_i \cup \{\mathcal{E}_i\}$。每条新证据语义上对应图像的某个局部子视图，但统一以文本存储、在 token 空间里组合复用。这样早期某一步的视觉消歧能直接惠及后面的推理，而无需重新处理像素——消融里"固定全局描述、不动态增长"只拿到中等效果，正说明按需积累的微证据才是减少幻觉的主力。

### 一个完整示例

设输入是一张菜单图，问"图中标价 $12 的那道菜是辣的吗"。模型解码到关键 token 时：base 分布给出 spicy/mild 两个候选概率接近，$p_{(1)}$ 偏低——典型的视觉漂移信号。**第一步**，knee truncation 把候选从词表截到 $\{$spicy, mild$\}$ 两个；分布监督器拿池中那条全局描述重加权，但全局描述没提辣度，协商后 top-2 margin 仍 $\leq \delta$。**第二步**，触发视觉决策器，GRIT 回到图像定位 $12 那一行，输出 token "mild" 并写回一条微证据"the $12 dish is labeled non-spicy"，追加进池子。**第三步**，解码继续，后面再遇到与这道菜相关的 token 时，分布监督器直接复用刚写入的这条文本证据、无需再编码图像，模型不确定度下降、$\alpha_i$ 回升，干预自动减弱。整条推理链就这样在"自信处放手、犹豫处补证据"之间动态切换，证据池随推理链按需从 1 条长到若干条。

### 损失函数 / 训练策略

完全免训练。框架包裹在冻结的 LVLM 外层，GRIT 决策器也直接用现成预训练模型，没有任何梯度更新。唯一的超参数是不确定性阈值 $\delta$：调大它触发次数减少、更快但补证据更保守，调小则更频繁回看图像、更准但更慢，由此可灵活控制 accuracy-latency 的 trade-off。

## 实验关键数据

### 主实验

| 模型 | TreeBench提升 | RH-Bench RH-AUC提升 | 备注 |
|------|-------------|---------------------|------|
| Qwen2.5-VL-7B + ECRD | +10.9 Overall | - | 属性+17.2, 物理+17.4 |
| Qwen2.5-VL-32B + ECRD | +6.1 Overall | - | 持续有效 |
| Qwen2.5-VL-72B + ECRD | +7.7 Overall | - | 大模型也受益 |
| LLaVA-OneVision-7B + ECRD | +6.2 Overall | - | 跨backbone |
| LLaVA-OneVision-72B + ECRD | +6.4 Overall | - | 跨backbone |
| InternVL3-8B + ECRD | +6.4 Overall | - | 跨backbone |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅knee truncation | 部分提升 | 候选集限制本身有帮助 |
| +Distribution Supervisor | 显著提升 | 证据协商是核心 |
| +Visual Decider | 最优 | 按需证据获取进一步减少幻觉 |
| 固定全局描述(无动态增长) | 中等 | 说明动态证据的重要性 |

### 关键发现

- ECRD在感知类任务（Attribute、Material、Physical）和推理类任务（Containment、Comparison）上均有提升，但感知任务提升更大，说明视觉grounding是主要收益来源
- 在已经很强的模型（Qwen2.5-VL-72B）上仍有7.7%提升，说明即使大模型也存在推理-感知漂移
- 在已经confident的子任务（如OCR）上ECRD不会造成性能下降——因为自适应混合权重自动退化为保留base分布
- ECRD使开源LVLM在多个任务上显著缩小了与GPT-4o/Gemini等私有模型的差距

## 亮点与洞察

- 自适应混合权重 $\alpha_i = p_{(1)}$ 的设计极简但有效：不需要学习的超参数，完全基于base分布的置信度自动调整干预强度。这种"模型自信时不干预、模型犹豫时强干预"的原则在其他模型纠错场景中也很有启发性。
- 用文本而非像素作为证据表示是关键设计选择——保持了模型原生的token空间，避免了反复图像编码，同时让证据可在整个链条中复用。

## 局限与展望

- Visual Decider（GRIT/Qwen2.5-VL-3B）本身也可能产生幻觉，当前框架没有对决策器输出做验证
- 不确定性阈值 $\delta$ 需要预设，不同任务/模型的最优阈值可能不同
- 每次触发决策器需要额外一次LVLM前向，对频繁触发的场景推理延迟可能显著增加

## 相关工作与启发

- **vs PixelReasoner/DeepEyes**: 这些方法需要RL训练学习zoom/crop策略，ECRD完全免训练且跨backbone迁移，代价是不能学到task-specific的观察策略
- **vs VDGD**: ECRD是VDGD的显著升级——将单一静态描述替换为动态增长的证据池，将logit覆盖替换为概率协商，保留了base模型在高置信步骤的行为

## 评分

- 新颖性: ⭐⭐⭐⭐ 分布协商+动态证据池+不确定性触发的完整设计很有originality
- 实验充分度: ⭐⭐⭐⭐ 6个backbone×多个benchmark，跨模型泛化验证充分
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，与VDGD的对比清晰
- 价值: ⭐⭐⭐⭐ 免训练即插即用对实际部署价值大，多个开源模型均可受益

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](../../ICML2026/reinforcement_learning/perceptual_flow_network_for_visually_grounded_reasoning.md)
- [\[ACL 2026\] Visually-Guided Policy Optimization for Multimodal Reasoning](../../ACL2026/reinforcement_learning/visually-guided_policy_optimization_for_multimodal_reasoning.md)
- [\[ICLR 2026\] Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](../../ICLR2026/reinforcement_learning/divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](../../ICML2026/reinforcement_learning/cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[CVPR 2026\] Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)

</div>

<!-- RELATED:END -->
