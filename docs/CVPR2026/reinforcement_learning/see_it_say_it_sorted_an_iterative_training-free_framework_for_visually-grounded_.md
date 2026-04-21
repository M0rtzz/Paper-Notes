---
title: >-
  [论文解读] See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs
description: >-
  [CVPR 2026][ECRD] 提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据，无需训练即可在多个LVLM上显著减少视觉幻觉、提升推理准确率。
tags:
  - CVPR 2026
  - ECRD
  - visual grounding
  - hallucination mitigation
  - training-free
  - evidence pool
---

# See It, Say It, Sorted: An Iterative Training-Free Framework for Visually-Grounded Multimodal Reasoning in LVLMs

**会议**: CVPR 2026  
**arXiv**: [2602.21497](https://arxiv.org/abs/2602.21497)  
**代码**: [GitHub](https://github.com/uuuuZYC/See-It-Say-It-Sorted)  
**领域**: 多模态推理  
**关键词**: ECRD, visual grounding, hallucination mitigation, training-free, evidence pool

## 一句话总结

提出Evidence-Constrained Reweighting Decoding（ECRD）框架：在LVLM解码时维护动态文本证据池，通过分布协商重加权候选token，不确定时自动调用轻量视觉决策器提取微证据，无需训练即可在多个LVLM上显著减少视觉幻觉、提升推理准确率。

## 研究背景与动机

**大视觉语言模型（LVLM）**已经能生成长链式思维（CoT）推理，但存在一个根本性问题：**推理-感知漂移**。在长文本解码过程中，模型需要平衡图像、增长的文本上下文和指令三个竞争性上下文。随着上下文变长，微妙但关键的视觉线索容易被语言先验淹没。一旦某个中间推理步骤偏离了视觉证据，即使后续推理逻辑上正确，最终答案也会错误——这就是**视觉幻觉传播**。

**现有方案**主要是通过RL训练模型学会"用图像思考"——让模型学习何时放大/裁剪图像并将裁剪区域重新注入推理上下文。代表工作如PixelReasoner和DeepEyes。但这类方法有三个痛点：（1）需要大量标注数据和奖励设计，训练成本高；（2）策略与特定backbone紧密耦合，难以迁移；（3）反复编码裁剪区域带来严重推理延迟。

**本文的切入角度**根本不同：不在训练时学习何时看图，而是在**测试时**用视觉证据监督每个推理步骤。核心idea是将解码过程重构为一系列"证据驱动的token选择"：维护一个文本形式的证据池，在每步解码时与模型的原始分布协商，用不确定性信号触发新证据的获取。

## 方法详解

### 整体框架

ECRD在冻结的LVLM外层包一个轻量级监督框架：每步解码时，（1）用knee truncation从base分布中选出top-k候选token；（2）Distribution Supervisor用证据池构建evidence-induced分布并与base分布协商混合；（3）如果协商后仍不确定，触发Visual Decider从图像中提取新的文本微证据加入池中。

### 关键设计

1. **Distribution Supervisor（分布监督器）**:

    - 功能：在每步解码时，利用当前证据池对候选token进行evidence-induced重加权，与base模型的分布协商出最终选择
    - 核心思路：对证据池中每条证据 $\mathcal{E}$，计算候选token $w$ 在证据前缀各位置上的平均概率 $q_\mathcal{E}(w) = \frac{1}{L}\sum_{j=1}^{L}p_{\text{VLM}}(w|e_{<j})$，跨证据平均后做softmax得到evidence-induced分布 $r_i$。然后与base分布按 $\alpha_i = p_{(1)}$（base分布最大概率）做自适应混合：$p_i^{\text{mix}} = \alpha_i p_i + (1-\alpha_i)\tilde{r}_i$
    - 设计动机：当base分布sharp时（$p_{(1)}$大），说明模型自信，应保留原始行为；当base分布diffuse时（$p_{(1)}$小），说明可能出现幻觉，应让证据获得更大权重。这种自适应混合无需超参数调节

2. **Visual Decider（视觉决策器）**:

    - 功能：当分布协商后仍不确定时，从图像中提取与当前推理上下文相关的微证据
    - 核心思路：触发条件为 $k^* > 1$ 且混合后top-2 margin $\Delta_i \leq \delta$。决策器用GRIT（基于Qwen2.5-VL-3B）接收图像、文本前缀尾部和候选集，输出（1）选择的token $w^*$（2）一条人类可读的微证据句子 $\mathcal{E}_i$，追加到证据池中
    - 设计动机：证据以文本而非像素形式存储，后续token可直接引用先前微观察而无需重新编码图像裁剪。这使得干预轻量且可验证，同时避免了RL方法中反复编码裁剪区域的开销

3. **Dynamic Evidence Pool（动态证据池）**:

    - 功能：累积式维护与推理链相关的视觉微证据集合
    - 核心思路：初始化为一条全局图像描述 $d_{\text{global}}$，之后仅在不确定性触发时按需增长——$E_{i+1} \leftarrow E_i \cup \{\mathcal{E}_i\}$。每条证据语义上对应图像的某个子视图，但以文本形式存储
    - 设计动机：全局描述提供广覆盖但不是唯一证据源，后续微证据按推理需求精确积累。证据在token空间中组合复用，使后续步骤受益于早期视觉消歧而无需重新处理像素

### 损失函数 / 训练策略

完全免训练。框架包裹在冻结的LVLM外层，GRIT决策器也使用现成预训练模型。唯一的超参数是不确定性阈值 $\delta$，通过调节它可以灵活控制accuracy-latency的trade-off。

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

## 相关论文

- [Divide, Harmonize, Then Conquer It: Shooting Multi-Commodity Flow Problems with Multimodal Language Models](../../ICLR2026/reinforcement_learning/divide_harmonize_then_conquer_it_shooting_multi-commodity_flow_problems_with_mul.md)
- [Do It for HER: First-Order Temporal Logic Reward Specification in Reinforcement Learning](../../AAAI2026/reinforcement_learning/do_it_for_her_first-order_temporal_logic_reward_specification_in_reinforcement_l.md)
- [Diving into Self-Evolving Training for Multimodal Reasoning](../../ICML2025/reinforcement_learning/diving_into_self-evolving_training_for_multimodal_reasoning.md)
- [Seeing is Improving: Visual Feedback for Iterative Text Layout Refinement](seeing_is_improving_visual_feedback_for_iterative_text_layout_refinement.md)
- [Anticipatory Planning for Multimodal AI Agents](anticipatory_planning_for_multimodal_ai_agents.md)

<!-- RELATED:END -->
