---
title: >-
  [论文解读] KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing
description: >-
  [CVPR2026][幻觉检测][多模态幻觉缓解] 提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。
tags:
  - "CVPR2026"
  - "幻觉检测"
  - "多模态幻觉缓解"
  - "KV-Cache平滑"
  - "注意力熵"
  - "指数移动平均"
  - "免训练推理"
---

# KVSmooth: Mitigating Hallucination in Multi-modal Large Language Models through Key-Value Smoothing

**会议**: CVPR2026  
**arXiv**: [2602.04268](https://arxiv.org/abs/2602.04268)  
**代码**: 待确认  
**领域**: 幻觉检测  
**关键词**: 多模态幻觉缓解, KV-Cache平滑, 注意力熵, 指数移动平均, 免训练推理

## 一句话总结

提出KVSmooth，一种免训练的即插即用方法，通过注意力行熵引导的自适应指数移动平均（EMA）对KV-Cache进行平滑，有效抑制多模态大语言模型（MLLM）在解码过程中因sink token引发的语义漂移与幻觉生成，在LLaVA-1.5上将CHAIR_S从41.8降至18.2（降幅56%），同时F1从77.5提升至79.2。

## 背景与动机

1. **多模态幻觉普遍存在**：MLLM在图像描述、VQA等任务中，常生成与视觉内容不一致的对象、属性或关系，严重影响可信部署。
2. **视觉依赖长程衰减**：随解码序列增长，早期视觉token的影响力在隐藏表示中逐渐衰减，导致生成文本逐步偏离图像内容。
3. **累积语义漂移**：生成早期的微小不准确随时间累积放大，使生成描述与视觉事实之间的差距不断扩大。
4. **Sink token加剧幻觉**：注意力聚集在少数"聚合token"上，这些token通过全局平均产生的隐藏状态偏离视觉上下文，系统性地抬高幻觉对象的logit分数。
5. **现有方法存在精度-召回权衡**：微调方法需大量数据和计算资源；对比解码方法计算开销大；注意力重新分配方法常在抑制幻觉的同时也压制了正确描述的对象，牺牲召回率。
6. **缺乏对sink token致幻机制的深入理解**：此前工作聚焦于减少sink token的出现或削弱其注意力权重，但未解释其为何触发幻觉。

## 方法详解

### 整体框架

KVSmooth 是一种**免训练、即插即用**的推理时方法，要治的是 MLLM 解码越说越偏、逐渐脱离图像的幻觉问题。作者把矛头指向 sink token——注意力过度聚集到少数"聚合 token"上，它们由全局平均产生的隐藏状态偏离视觉上下文，系统性地抬高幻觉对象的 logit。KVSmooth 的做法是在解码时对 KV-Cache 的 Key 和 Value 施加一个自适应 EMA 平滑，让隐藏状态的演化更平稳、压住 sink 带来的语义漂移。整条 pipeline 是一个嵌在逐 token 解码里的闭环：先用**注意力行熵**实时诊断当前 token 的 sink 强度，再据此对 **KV-Cache 做 EMA 平滑**，平滑的力度由**熵引导的自适应系数**动态决定，平滑后的 KV 写回缓存、参与下一个 token 的生成。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["输入：图像 + 指令"] --> B["MLLM 逐 token 解码<br/>（作用于第 3-31 层）"]
    B --> C["注意力行熵<br/>实时度量该 token 的 sink 强度"]
    C --> D["KV-Cache EMA 平滑<br/>对 Key、Value 做指数移动平均（贝叶斯 MAP 推导）"]
    D --> E["熵引导的自适应系数<br/>行熵百分位 → λ → 裁剪，sink 越强平滑越大"]
    E --> F["抑制语义漂移<br/>输出忠实于图像的描述"]
    E -.平滑后 KV 写回，下一 token.-> B
```

### 关键设计

**1. 注意力行熵：一个实时刻画 sink 强度的指标**

以往工作要么减少 sink token、要么削弱它的注意力，却没解释它为何触发幻觉。作者先用三个观察把因果链抠清楚：真实对象的 logit 均值单调递减、方差稳定，而幻觉对象的 logit 均值持续上升伴随方差增加（Obs1，幻觉候选在隐藏表示里积累了不稳定）。据此作者提出**注意力行熵**作为 sink 强度的实时度量：高行熵 token 的注意力分布均匀（扩散），其隐藏状态近似历史平均、与大多数状态角距离小，因而在后续步骤吸引不成比例的高注意力、形成注意力 sink（Obs2，行熵与传统列求和指标的余弦相似度集中在 0.79）。更关键的是 Obs3：幻觉对象的 logit 排名与行熵正相关——token 注意力越均匀（高熵）、幻觉对象 logit 越高，GT 对象则相反，这揭示了 sink token 通过全局平均系统性抬高幻觉分数的直接因果。

**2. KV-Cache 上的 EMA 平滑：把"理想轨迹应平滑"变成贝叶斯推断**

基于"理想解码轨迹应平滑"的假设，将隐藏状态转移建模为高斯随机游走 $h_t = h_{t-1} + \epsilon_t$，通过贝叶斯 MAP 估计推导出

$$\hat{h_t} = (1-\lambda_t) o_t + \lambda_t h_{t-1}$$

这恰好是 EMA 形式。关键选择是同时对 Key 和 Value 施加 EMA、而非直接改隐藏状态，因为这能在最大程度上正则化 logit 的均值和方差、实现最强的幻觉抑制，同时不损害召回；消融显示仅平滑隐藏状态会让召回严重下降。

**3. 熵引导的自适应系数：让 sink 越强的 token 平滑越多**

固定平滑系数会一刀切，KVSmooth 让平滑量随 sink 强度走。它在每一层计算当前 token 的注意力行熵 $z_t^l$、维护一个长度 $M$ 的 FIFO 队列，再用行熵的**百分位排名**确定平滑系数 $\hat{\lambda}_t^l = k/M$，高熵 token 获得更大的平滑系数；再以超参 $\lambda_{\text{ref}}$ 为中心、在 $[\lambda_{\text{ref}}-0.2, \lambda_{\text{ref}}+0.2]$ 内裁剪，稳定生成并保留表示多样性。最终对指定层 $l$ 的 token $x_t$，Key 与 Value 按

$$\hat{K_t^l} = (1-\tilde{\lambda}_t^l) K_t^l + \tilde{\lambda}_t^l K_{t-1}^l$$
$$\hat{V_t^l} = (1-\tilde{\lambda}_t^l) V_t^l + \tilde{\lambda}_t^l V_{t-1}^l$$

更新；实现上应用于第 3-31 层，FIFO 队列长度 15，$\lambda_{\text{ref}}$ 为 LLaVA-1.5/MiniGPT-4/InstructBLIP 分别设为 0.9/0.5/0.7。消融表明自适应系数比固定系数进一步把 LLaVA-1.5 的 CHAIR_S 从 36.2 降到 18.2。

## 实验关键数据

### CHAIR基准（图像描述幻觉）

| 方法 | LLaVA-1.5 CHAIR_S↓ | LLaVA-1.5 F1↑ | MiniGPT-4 CHAIR_S↓ | MiniGPT-4 F1↑ | InstructBLIP CHAIR_S↓ | InstructBLIP F1↑ |
|------|---------------------|----------------|---------------------|----------------|------------------------|-------------------|
| Baseline | 41.8 | 77.5 | 31.8 | 69.9 | 61.4 | 71.6 |
| PAI | 22.6 | 75.5 | 24.6 | 71.0 | 63.4 | 71.1 |
| OPERA | 44.2 | 78.6 | 27.4 | 69.4 | 68.0 | 69.2 |
| MiddleLayer | 17.8 | 75.9 | 24.6 | 71.2 | 75.0 | 67.2 |
| **KVSmooth** | **18.2** | **79.2** | **17.0** | **71.7** | **42.2** | **75.1** |

- LLaVA-1.5上CHAIR_S降幅56%，F1反而从77.5→79.2，**唯一同时提高精确率和召回率的方法**。
- 在MiniGPT-4上CHAIR_S也从31.8→17.0（降幅47%）。

### Object HalBench

在LLaVA-1.5上CHAIR_SR从45.3%降至16.7%（降幅63.1%），在三个模型上分别实现63.1%/40.3%/41.6%的句子级幻觉率降低。

### 消融实验

| 平滑位置 | LLaVA-1.5 Cs↓ | F1↑ |
|----------|----------------|-----|
| 仅注意力输出 $o_t$ | 33.8 | 74.7 |
| 仅Key $K_t$ | 35.6 | 79.4 |
| **Key+Value** | **18.2** | **79.2** |

- 同时平滑Key和Value效果最优；仅平滑隐藏状态导致召回率严重下降。
- 自适应系数（Ada.）相比固定系数进一步将LLaVA-1.5的CHAIR_S从36.2降至18.2，验证了熵引导机制的精准识别能力。

## 亮点

- **免训练即插即用**：无需微调或修改模型参数，推理时直接作用于KV-Cache，具天然的通用性。
- **理论与实证双重驱动**：从贝叶斯MAP估计推导出EMA形式，三个观察提供了清晰的因果解释链（logit分歧→行熵sink→熵-幻觉耦合）。
- **打破精度-召回困境**：PR曲线分析表明KVSmooth是唯一在大幅降低幻觉的同时保持甚至提升F1的方法。
- **提出sink度概念**：行熵作为连续实时sink指标，比传统列求和更高效，无需多步回溯。
- **广泛验证**：覆盖3个模型（LLaVA-1.5/MiniGPT-4/InstructBLIP）× 4个基准（CHAIR/OPOPE/AMBER/Object HalBench），结果一致。

## 局限与展望

- **超参数模型相关**：$\lambda_{\text{ref}}$ 对不同模型需分别调优（0.9/0.5/0.7），缺乏自动选择方案。
- **仅评估7B模型**：未验证在更大规模（13B/70B）或更新架构（如Qwen2.5-VL）上的表现。
- **生成长度限制**：最大生成512 token，未探讨超长生成场景下EMA是否持续有效。
- **层范围固定**：应用于3-31层为经验设定，未给出系统的层选择准则。
- **仅关注对象幻觉**：未评估属性幻觉、关系幻觉等更细粒度的幻觉类型。
- **高熵不总是坏的**：在语义转折点高熵可能是合理的上下文切换，一刀切的平滑存在理论上的信息损失风险。

## 与相关工作的对比

| 类别 | 代表方法 | 核心思路 | KVSmooth优势 |
|------|---------|----------|-------------|
| 对比解码 | VCD, OPERA | 噪声增强视图做对比/回溯惩罚 | 计算开销更低，无需多次前向传播 |
| 注意力重分配 | PAI, SPARC, MiddleLayer | 增强视觉token注意力 | 不牺牲召回率，PR曲线更优 |
| 微调对齐 | POVID, RLHF | 用偏好数据微调 | 免训练，无需额外数据 |
| KV-Cache修剪 | PruneHal | 删除冗余视觉token | 保留信息更完整，通过平滑而非删除调节 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 行熵sink度概念新颖，从贝叶斯视角推导EMA平滑具理论优雅性
- 实验充分度: ⭐⭐⭐⭐ — 3模型4基准+详尽消融+PR分析，但缺乏大模型和更多幻觉类型评估
- 写作质量: ⭐⭐⭐⭐ — 观察-方法-实验逻辑链清晰，数学推导简洁易懂
- 价值: ⭐⭐⭐⭐ — 轻量免训练方案有很强实用性，为推理时幻觉缓解提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] InEx: Hallucination Mitigation via Introspection and Cross-Modal Multi-Agent Collaboration](../../AAAI2026/hallucination/inex_hallucination_mitigation_via_introspection_and_cross-mo.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](../../ACL2025/hallucination/activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[CVPR 2026\] MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization](mod-dpo_towards_mitigating_cross-modal_hallucinations_in_omni_llms_using_modalit.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/hallucination/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)

</div>

<!-- RELATED:END -->
