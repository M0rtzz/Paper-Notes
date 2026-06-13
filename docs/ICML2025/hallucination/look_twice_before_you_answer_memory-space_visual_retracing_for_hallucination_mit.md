---
title: >-
  [论文解读] Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models
description: >-
  [ICML2025][幻觉检测][MLLM] 提出 MemVR 解码范式，将视觉 token 作为补充证据通过 FFN 的 key-value memory 机制重新注入到中间触发层，以"再看一次"的方式缓解 MLLM 幻觉问题，不引入额外推理开销。 - MLLM（如 LLaVA）在生成过程中对视觉信息产生"遗忘"…
tags:
  - "ICML2025"
  - "幻觉检测"
  - "MLLM"
  - "hallucination"
  - "visual retracing"
  - "FFN key-value memory"
  - "decoding strategy"
---

# Look Twice Before You Answer: Memory-Space Visual Retracing for Hallucination Mitigation in Multimodal Large Language Models

**会议**: ICML2025  
**arXiv**: [2410.03577](https://arxiv.org/abs/2410.03577)  
**代码**: [GitHub](https://github.com/1zhou-Wang/MemVR)  
**领域**: 幻觉检测  
**关键词**: MLLM, hallucination, visual retracing, FFN key-value memory, decoding strategy

## 一句话总结
提出 MemVR 解码范式，将视觉 token 作为补充证据通过 FFN 的 key-value memory 机制重新注入到中间触发层，以"再看一次"的方式缓解 MLLM 幻觉问题，不引入额外推理开销。

## 研究背景与动机
- MLLM（如 LLaVA）在生成过程中对视觉信息产生"遗忘"，文本解码器随推理深入越来越依赖文本 token
- **实验发现一**：放大图像特征比放大文本特征对性能影响更大，说明文本解码器更"偏向文本"
- **实验发现二**：幻觉 token 在中间和深层产生时具有更高的不确定性（entropy 高）
- **实验发现三**：仅补充视觉信息（而非文本或两者兼有）效果最佳
- 现有方案的局限：
    - 对比解码（VCD）需双轮推理，延迟翻倍
    - 注意力干预（OPERA）延迟为基础方法的 3.66 倍
    - Fine-tuning 方法需额外数据和训练成本

## 方法详解

### 核心思想：Look-Twice Mechanism
当模型在中间层表现出高不确定性时，重新注入视觉 token 作为"补充证据"来纠正预测。

### FFN 的 Key-Value Memory 视角
将 FFN 重新解释为 key-value 记忆检索：
$$\text{FFN}(\mathbf{x}) = \sum \phi(\langle \mathbf{x}, \mathbf{k}_i \rangle) \cdot \mathbf{v}_i$$

### Visual Retracing（VR）
在第 $l$ 层的 FFN 中注入视觉记忆：
$$\text{FFN}^{(l)}(\mathbf{x} \propto \mathbf{z}_v) = \alpha \underline{\Delta} + (1-\alpha)\text{FFN}^{(l)}(\mathbf{x})$$

其中视觉检索项：
$$\underline{\Delta}(\mathbf{z}_v | \mathbf{x}) = \sum_{i=1}^{N_v} \phi(\langle \mathbf{x}, \mathbf{z}_{v,i} \rangle) \cdot \mathbf{z}_{v,i}$$

- $\alpha \in [0,1]$：注入比例，与图像复杂度成正比
- $N_v \ll D$（如 256 vs 11008），计算开销可忽略

### 触发层选择
- **静态 VR**：固定某一中间层
- **动态 VR**：根据不确定性分数动态选择触发层，哪层不确定性高就在哪层注入

## 实验关键数据

| 方法 | 延迟 (ms/token) | POPE ↑ | MME ↑ | CHAIR ↓ |
|------|-----------------|--------|-------|---------|
| Greedy | 65.71 (1.00×) | 基准 | 基准 | 基准 |
| VCD | 144.62 (2.20×) | +小幅 | 负面 | +小幅 |
| OPERA | 240.59 (3.66×) | +中等 | 负面 | +中等 |
| MemVR | 68.32 (1.04×) | **+7.0%** | **+32.2** | **-15.6%** |

- MemVR 仅增加 4% 延迟，而 VCD 增加 120%、OPERA 增加 266%
- 内存占用仅增加 1%（14345 vs 14257 MB）
- 在 POPE、MME、CHAIR 等多个基准上全面领先
- 跨多个 MLLM（LLaVA-1.5、Qwen-VL、GLM4V 等）通用有效

## 亮点与洞察
- **极简高效**：plug-and-play，无需训练、无需外部数据，零参数引入
- **理论洞察独到**：从 FFN = key-value memory 的视角出发，将 VR 解释为信息再检索
- **同时提升幻觉抑制和通用能力**：是唯一一个在两个维度上同时正向的方法（Table 2）
- **模态不平衡分析**：系统验证了幻觉根源在于视觉信息在深层被遗忘

## 局限与展望
- $\alpha$ 需要手动调整或启发式设定，缺乏自适应选择机制
- 仅在 image-text 场景验证，video/audio 多模态场景待探索
- 动态 VR 需要额外计算不确定性，虽然开销小但增加了方法复杂度
- 对视觉 token 数量极少的模型（如高压缩率视觉编码器）效果待验证
- 触发层的选择策略对性能影响较大，不同模型架构可能需不同策略
- 未分析在长文本生成场景下（如详细图像描述），VR 的效果是否稳定
- 与 RAG、fine-tuning 等方法的结合效果未探索
- 对于多图输入场景（如多轮对话），如何选择注入哪张图的视觉 token 是开放问题

### 补充分析
- 不确定性度量使用归一化熵：$u = \sum -p_i \log p_i / \log N$
- LLaVA-1.5-7B 为 32 层 Transformer，视觉 token 数为 256
- MemVR 的关键创新在于不修改 logits（与 CD 方法不同），而是直接修改 hidden states
- 实验表明中间层（第 14-18 层）的不确定性对幻觉预测最敏感
- 方法可直接推广到任何基于 Transformer decoder 的 MLLM

## 相关工作与启发
- **VCD**（Leng et al., 2024）：加噪视觉输入做对比解码，但翻倍推理成本
- **OPERA**（Huang et al., 2024）：注意力矩阵干预，高延迟
- **DoLa**（Chuang et al., 2023）：层间对比解码，对视觉幻觉无效
- **PAI**（Liu et al., 2024）：提出关注图像更多，与本文发现一致
- 启发：将 FFN 视为 memory 的观点为 MLLM 内部机制理解提供了新视角

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (look-twice + FFN memory 的解释框架非常新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (8个benchmark + 多模型 + 效率分析 + GPT-4o评估)
- 写作质量: ⭐⭐⭐⭐⭐ (motivation分析扎实，图表清晰)
- 价值: ⭐⭐⭐⭐⭐ (真正实用的方法，性能效率兼得)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Look Carefully: Adaptive Visual Reinforcements in Multimodal Large Language Models for Hallucination Mitigation](../../ICLR2026/hallucination/look_carefully_adaptive_visual_reinforcements_in_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](../../NeurIPS2025/hallucination/causalllava_causal_disentanglement_for_mitigating_hallucinat.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](../../ACL2025/hallucination/reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2025\] Visual Evidence Prompting Mitigates Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/visual_evidence_prompting.md)
- [\[ICLR 2026\] Dynamic Multimodal Activation Steering for Hallucination Mitigation in Large Vision-Language Models](../../ICLR2026/hallucination/dynamic_multimodal_activation_steering_for_hallucination_mitigation_in_large_vis.md)

</div>

<!-- RELATED:END -->
