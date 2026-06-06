---
title: >-
  [论文解读] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification
description: >-
  [CVPR 2026][LLM安全][大视觉语言模型] 提出注意力失衡（Attention Imbalance）概念来解释 LVLM 中的对象幻觉现象，并设计轻量级解码时干预方法 AIR，通过跨模态注意力重新分配和方差约束投影正则化矫正注意力失衡，在四个 LVLM 上将幻觉率最高降低 35.1%…
tags:
  - "CVPR 2026"
  - "LLM安全"
  - "大视觉语言模型"
  - "对象幻觉"
  - "注意力失衡"
  - "解码时干预"
  - "注意力矫正"
---

# Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification

**会议**: CVPR 2026  
**arXiv**: [2603.24058](https://arxiv.org/abs/2603.24058)  
**代码**: 无  
**领域**: 医学图像  
**关键词**: 大视觉语言模型, 对象幻觉, 注意力失衡, 解码时干预, 注意力矫正

## 一句话总结

提出注意力失衡（Attention Imbalance）概念来解释 LVLM 中的对象幻觉现象，并设计轻量级解码时干预方法 AIR，通过跨模态注意力重新分配和方差约束投影正则化矫正注意力失衡，在四个 LVLM 上将幻觉率最高降低 35.1%，同时提升通用能力最高达 15.9%。

## 研究背景与动机

1. **领域现状**：大视觉语言模型（LVLM）在跨模态理解任务上表现优异，但对象幻觉（生成图像中不存在的物体描述）严重损害了模型在自动驾驶、医学影像等高风险场景的可靠性。
2. **现有痛点**：现有方法分三类——视觉指令微调（高训练成本）、后处理技术（额外推理开销）、对比解码（稳定性和泛化性有限）。更根本的问题是，幻觉的根因分析仍然不充分。
3. **核心矛盾**：LVLM 复杂的训练流程和架构阻碍了可解释性分析，现有从视觉信息交互、位置编码、异常 token 等角度的研究未能提供全面的理解。
4. **本文目标** （1）提供一个定量框架来解释幻觉的注意力机制根因；（2）基于此设计无需训练的轻量级干预方法。
5. **切入角度**：作者通过系统性实验发现注意力分配失衡——包括模态间和 token 间——与对象幻觉存在强因果相关。
6. **核心 idea**：幻觉源于注意力失衡，矫正幻觉敏感注意力头的跨模态和 token 级失衡即可有效缓解。

## 方法详解

### 整体框架

AIR 是一种纯推理时的解码干预方法，不需要额外训练。它在每个解码步骤中对幻觉敏感注意力头执行两步操作：（1）模态平衡注意力重新分配——当文本注意力超过阈值时，降低文本 token 权重、提升视觉 token 权重；（2）方差约束投影正则化——通过零迹投影、Frobenius 能量保持和收缩正则化使注意力分布更均匀。

### 关键设计

1. **注意力失衡定义（MAI + TAI）**:

    - 功能：定量描述注意力分配的失衡程度
    - 核心思路：MAI（Modality-wise Attention Imbalance）定义为两种模态接收的总注意力之比 $\text{MAI}(M_p, M_q) = A_{M_p}/A_{M_q}$，值远大于1表示 $M_p$ 主导。TAI（Token-wise Attention Imbalance）定义为 token 接收的注意力比例与其信息贡献比例之比，值远大于1表示过度关注。
    - 设计动机：为幻觉的注意力根因提供可量化的度量框架。实验发现幻觉敏感头的 MAI 高达 5.1（而不敏感头仅 1.5），且高 TAI token 出现后 15 个 token 内几乎必然产生幻觉。

2. **模态平衡注意力重新分配（Modality-Balanced Attention Reallocation）**:

    - 功能：矫正幻觉敏感头中过度偏向文本模态的注意力
    - 核心思路：在每个解码步，对幻觉敏感注意力头计算文本 token 接收的累积注意力 $V^{\text{text}}$。若超过阈值 $\tau_{\text{text}}$，则对文本 token 权重乘以 $\lambda \in [0,1]$（抑制），对视觉 token 权重乘以 $\gamma > 1$（放大）。默认 $\lambda=0.1$, $\gamma=3.5$。
    - 设计动机：幻觉敏感头继承了基础语言模型的注意力模式（余弦相似度 0.81 vs 不敏感头的 0.69），过度关注文本而忽略视觉信息。有针对性地矫正这些头可保留模型正常功能。

3. **方差约束投影正则化（Variance-Constrained Projection Regularization）**:

    - 功能：抑制注意力过度集中在少数 token 上
    - 核心思路：三步操作：（a）对 $W_{\text{QK}}$ 按其谱能量自适应缩放；（b）零迹投影 $\hat{A} = A - \frac{\text{tr}(A)}{L}I$ 去除自对齐偏置；（c）Frobenius 能量归一化保持量级后，收缩正则化 $A^* = (1-\beta)\tilde{A} + \beta \cdot \text{mean}(\tilde{A}) \cdot \mathbf{1}$ 使分布更均匀。
    - 设计动机：TAI 分析表明幻觉前总有某个 token 获得过度集中的注意力（如 `<0x0A>` token 的 TAI 值达 98），正则化可平滑注意力分布，预防后续幻觉传播。

### 损失函数 / 训练策略

- AIR 是纯推理时方法，**不需要任何训练**
- 幻觉敏感头通过擦除归因法（erasure-based attribution）选取——逐一移除注意力头观察幻觉概率变化，选择影响最大的 top-20 个头
- 超参数：$\tau_{\text{text}}=0.3$, $\lambda=0.1$, $\gamma=3.5$, $\xi=0.01$, $\beta=0.3$

## 实验关键数据

### 主实验

CHAIR 幻觉评估（Max New Tokens=256）：

| LVLM | 指标 | AIR (Ours) | 最优基线 (AD-HH) | 改进 |
|------|------|------|----------|------|
| LLaVA-1.5 | $C_S$ ↓ | **28.8** | 35.2 | -18.1% |
| MiniGPT-4 | $C_S$ ↓ | **21.3** | 32.8 | -35.1% |
| InstructBLIP | $C_S$ ↓ | **30.1** | 36.0 | -16.4% |
| Shikra | $C_S$ ↓ | **30.3** | 36.9 | -17.9% |

MM-Vet 通用能力：

| LVLM | AIR Overall | Greedy Overall | 提升 |
|------|------------|----------------|------|
| LLaVA-1.5 | **32.0** | 27.6 | +15.9% |
| MiniGPT-4 | **22.0** | 20.0 | +10.0% |

### 消融实验

| 配置 | $C_S$ ↓ | $C_I$ ↓ | MM-Vet ↑ | 说明 |
|------|---------|---------|----------|------|
| Greedy (baseline) | 51.8 | 13.7 | 27.6 | 无干预 |
| R-only（仅重分配） | 32.1 | 9.9 | 30.5 | 文本抑制+视觉放大有效 |
| P-only（仅投影正则） | 38.4 | 11.2 | 29.8 | 均匀化注意力有效 |
| Full AIR | **28.8** | **8.6** | **32.0** | 两者互补，最佳 |

### 关键发现

- 注意力重分配贡献更大（$C_S$ 从 51.8 降到 32.1），说明跨模态失衡是幻觉主因
- AIR 独特之处在于**同时减少幻觉并提升通用能力**——其他方法（如 AD-HH）抗幻觉但通用能力下降 14.8%
- 幻觉敏感头主要集中在模型中间层，与先前研究一致
- 高 TAI token 与幻觉的共现现象在四个 LVLM 上一致出现，说明注意力失衡是通用的幻觉根因
- 幻觉存在"雪球效应"——一个幻觉词会触发后续更多幻觉

## 亮点与洞察

- **因果链路清晰**：从 TAI/MAI 定义 → 共现验证 → 头级别归因 → 继承假说验证，形成了完整的幻觉因果分析链。这不仅是方法论贡献，更是对 LVLM 可解释性的重要推进。
- **零训练开销**：AIR 完全在推理时操作，不引入额外参数或训练成本，极具实用性。
- **发现幻觉敏感头继承基座 LM 模式**：这一发现暗示 LVLM 的视觉对齐训练未能充分改变某些注意力头的纯文本偏好，为未来训练策略改进提供了方向。

## 局限与展望

- 幻觉敏感头的选取需要预先通过擦除法分析，增加了部署前的准备工作
- $\tau_{\text{text}}$、$\lambda$、$\gamma$ 等超参数可能需要针对不同模型调整
- 仅验证了 7B 级别模型，更大模型（70B+）上的效果未知
- 可探索将 AIR 的洞察融入训练阶段，设计注意力平衡的微调目标

## 相关工作与启发

- **vs VCD (ICLR24)**: VCD 通过对比不加/加视觉输入的输出分布来削弱语言先验，但在某些 LVLM 上反而加剧幻觉（LLaVA-1.5 上 $C_S$ 从 51.8 升至 59.4）。AIR 直接操作注意力权重更精准。
- **vs OPERA (ICML24)**: OPERA 通过惩罚过度关注摘要 token 来缓解幻觉，但仅关注 token 级别。AIR 同时解决模态级和 token 级失衡。
- **vs AD-HH**: 前最优基线，但通用能力下降 14.8%。AIR 在幻觉缓解上更强且通用能力反而提升，说明注意力矫正是更正确的干预方向。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 注意力失衡概念及 MAI/TAI 定义是全新贡献，从可解释性角度推导出干预方法的因果链路非常优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 四个 LVLM、三个基准、七个基线、详细消融和超参分析
- 写作质量: ⭐⭐⭐⭐⭐ 数学定义严谨，分析递进清晰，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 同时解决幻觉和通用能力退化问题，无需训练即可部署，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in LVLMs](hulluedit_subspace_editing_hallucination.md)
- [\[ACL 2026\] Hallucination Detection in LLMs with Topological Divergence on Attention Graphs](../../ACL2026/llm_safety/hallucination_detection_in_llms_with_topological_divergence_on_attention_graphs.md)
- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](../../ACL2026/llm_safety/generating_effective_cot_traces_for_mitigating_causal_hallucination.md)
- [\[ACL 2026\] Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs](../../ACL2026/llm_safety/aligning_with_your_own_voice_self-corrected_preference_learning_for_hallucinatio.md)
- [\[ACL 2025\] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](../../ACL2025/llm_safety/cracking_hallucination_vhd.md)

</div>

<!-- RELATED:END -->
