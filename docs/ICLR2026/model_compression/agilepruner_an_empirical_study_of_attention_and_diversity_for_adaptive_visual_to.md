---
title: >-
  [论文解读] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs
description: >-
  [ICLR 2026][模型压缩][剪枝] 通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner…
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "剪枝"
  - "注意力机制"
  - "diversity"
  - "hallucination"
---

# AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs

**会议**: ICLR 2026  
**arXiv**: [2603.01236](https://arxiv.org/abs/2603.01236)  
**代码**: [https://cvsp-lab.github.io/AgilePruner](https://cvsp-lab.github.io/AgilePruner)  
**领域**: 模型压缩  
**关键词**: visual token pruning, attention, diversity, hallucination, adaptive pruning

## 一句话总结
通过 erank（有效秩）和注意力熵的系统性实证分析，揭示了视觉 token 剪枝中注意力方法和多样性方法的互补特性——注意力方法抑制幻觉但覆盖有限，多样性方法覆盖全面但易引入幻觉——并据此提出基于图像复杂度自适应切换剪枝策略的 AgilePruner，在 9 个 benchmark 上表现稳健。

## 研究背景与动机
**领域现状**：LVLM 的视觉 token 数量大量冗余（数百个），导致推理效率低。现有剪枝方法分两派：注意力方法（保留高注意力 token）和多样性方法（保留特征最分散的 token），也有混合策略。

**现有痛点**：各种方法优劣不清——(1) 多样性方法实际保留了多少多样性？(2) 多样性和幻觉是什么关系？(3) 不同类型图像分别适合哪种策略？这些问题缺乏系统研究。

**核心矛盾**：注意力方法在简单图像上好但覆盖不足，多样性方法在复杂图像上好但容易幻觉。没有一种方法是普遍最优的。

**本文目标** 通过实证研究揭示两种范式的本质行为差异，并据此设计自适应剪枝策略。

**切入角度**：用 erank 量化特征多样性、用注意力熵量化注意力集中度，作为分析工具和自适应切换的依据。

**核心 idea**：根据图像的注意力熵自适应地在注意力和多样性剪枝之间切换。

## 方法详解

### 整体框架
这篇工作要回答的问题是：视觉 token 剪枝里"保留高注意力的 token"和"保留最分散的 token"这两派到底各自在做什么、什么时候该用哪一派。它先用两个可量化的探针——erank（有效秩，量化被保留 token 集合的特征多样性）和注意力熵（量化注意力的集中/分散程度）——把各类方法解剖一遍，得到"多样性会换来幻觉""不同复杂度图像偏好不同策略"两条核心观察；然后把这两条观察直接落成一个 training-free 的自适应剪枝器：按注意力降序遍历 token，用一个相似度阈值剔除冗余，而这个阈值随图像的注意力熵动态调整。整条链路是"先做实证诊断、再让诊断结论驱动机制"，因此前两个关键设计其实是分析工具，第三个才是真正部署的算法。

### 关键设计

**1. erank 多样性探针：用有效秩戳穿"号称多样"的方法到底有多多样。**

各家多样性方法都声称自己保留了更分散的 token，但缺一把尺子去验证。这里对被保留 token 的嵌入矩阵做 SVD，把奇异值分布的熵定义为有效秩 erank，奇异值越均摊（信息越分散在多个方向）erank 越高。量出来的结果很说明问题：DivPrune(21.84) ≫ VisPruner(14.35) ≈ VisionZip(14.02) ≫ PruMerge+(10.91)，也就是不少打着"多样性"旗号的方法实际多样性并不高。更关键的是把 erank 和幻觉率对照后发现两者强正相关——erank 最高的 DivPrune 在 CHAIR 上 $C_S$=57.4，而注意力派方法只有 ~45。这把"多样性=好"的朴素假设直接推翻，为后面的设计埋下伏笔。

**2. 图像复杂度依赖性：把"该用哪派"归因到图像本身。**

既然没有一派普遍最优，那决定胜负的就该是图像。用注意力熵和 erank 一起刻画图像复杂度后规律很清晰：简单图像注意力熵低、erank 低，信息集中在少数几个 token 上，此时注意力方法精准命中要害、覆盖不足的缺点也无所谓，所以更好；复杂图像注意力熵高、erank 高，信息摊在很多 token 上，注意力方法会漏掉关键区域，多样性方法的全覆盖反而占优。实验侧的对照印证了这点——偏简单的 ScienceQA 上注意力方法更优，偏复杂的 POPE 上多样性方法更优。这条观察把"选策略"这件事变成了"读图像复杂度"，是自适应机制能成立的前提。

**3. 注意力熵自适应剪枝：把上面两条观察合成一个可跑的算法。**

部署的算法本身刻意做得简单：把 token 按注意力分数降序排列依次遍历，对每个候选 token 算它与已保留集合的最大相似度，超过阈值 $\tau$ 就判为冗余丢弃。点睛之处在于 $\tau$ 不是固定的，而是随注意力熵自适应——低熵（简单图像）用高阈值，相似度要很高才剔除，于是更激进地保留注意力高的 token，行为偏向注意力派；高熵（复杂图像）用低阈值，稍微相似就剔除，于是逼着保留下来的 token 彼此分散，行为偏向多样性派。这样一个阈值就把"简单图靠注意力、复杂图靠多样性"的结论编码进了一条遍历逻辑里，既不需要训练，也不需要为两派单独维护两套流程。

### 损失函数 / 训练策略
无需训练（training-free），整套方法只在推理阶段对视觉 token 做剪枝。

## 实验关键数据

### 主实验（9 个 benchmark 平均表现）

| 方法 | 类型 | POPE | ScienceQA | MME | CHAIR $C_S$↓ |
|------|------|------|-----------|-----|-------------|
| FasterVLM | 注意力 | - | 较好 | - | 45.4 |
| DivPrune | 多样性 | 较好 | - | - | 57.4 |
| PruMerge+ | 混合 | - | - | - | 45.2 |
| **AgilePruner** | **自适应** | **稳健** | **稳健** | **稳健** | **低** |

### 消融实验（注意力 vs 多样性比例 on CHAIR）

| 注意力比例 R | $C_S$↓ | $C_I$↓ | Recall↑ | erank |
|-------------|--------|--------|---------|-------|
| 0 (纯多样性) | 57.4 | 18.0 | 76.4 | 21.14 |
| 0.25 | 50.8 | 16.8 | 74.5 | 14.98 |
| 0.50 | 46.2 | 14.5 | 73.7 | 14.38 |
| 0.75 | 45.2 | 14.1 | 70.5 | 13.58 |

### 关键发现
- **多样性 ↔ 幻觉正相关**：增加注意力token比例从0→0.75，$C_S$ 从57.4降到45.2，但 recall 从76.4降到70.5——trade-off 清晰
- 在 LLaVA-1.5-13B、LLaVA-NeXT-7B、Qwen2.5-VL-7B 上观察到相同趋势，说明发现是模型无关的
- 将图像复杂度自适应策略应用到现有混合方法上后，性能一致提升，验证了实证发现的泛化性

## 亮点与洞察
- **"多样性导致幻觉"的反直觉发现**：以前认为保留更多样的 token 总是好的，本文揭示了这并非如此——保留更多样但注意力低的 token 反而容易引入虚假信息
- **erank 作为分析工具**：用有效秩来量化 token 集合的特征多样性是一个简洁有效的度量，可复用到其他需要评估 token 选择质量的场景
- **简单有效的自适应策略**：不需要复杂设计，仅用注意力熵做阈值调节就能获得稳健的跨场景表现

## 局限与展望
- 自适应阈值的设定仍依赖超参数，不同模型可能需要调整
- erank-幻觉关系的因果性未完全建立（是否是因为保留了特定类型的 token 而非纯粹的多样性？）
- 主要在 7B/13B 模型上验证，更大模型（70B+）上的行为未知
- 对视频理解、高分辨率多 patch 场景的分析不足

## 相关工作与启发
- **vs VisionZip/FasterVLM (注意力方法)**: AgilePruner 分析了它们在复杂图像上的不足并用多样性补充
- **vs DivPrune (多样性方法)**: 揭示了其高幻觉风险，并通过注意力信号约束来缓解
- **vs PruMerge+/VisPruner (混合方法)**: 证明将图像复杂度自适应策略应用到这些方法上能一致提升性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 实证分析深入，多样性-幻觉关系是新发现
- 实验充分度: ⭐⭐⭐⭐⭐ 9个benchmark、CHAIR幻觉分析、多模型验证、细致消融
- 写作质量: ⭐⭐⭐⭐ 分析驱动的叙事清晰，图表丰富
- 价值: ⭐⭐⭐⭐ 为 token pruning 提供了实证基础和实用指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression](../../ACL2025/model_compression/gist_token_context_compression.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](../../ICCV2025/model_compression/fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[ICLR 2026\] Why Attention Patterns Exist: A Unifying Temporal Perspective Analysis](why_attention_patterns_exist_a_unifying_temporal_perspective_analysis.md)
- [\[ICLR 2026\] Token Distillation: Attention-Aware Input Embeddings for New Tokens](token_distillation_attention-aware_input_embeddings_for_new_tokens.md)
- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](../../AAAI2026/model_compression/sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)

</div>

<!-- RELATED:END -->
