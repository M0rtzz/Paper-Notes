---
title: >-
  [论文解读] Dynamic Token Reweighting for Robust Vision-Language Models
description: >-
  [CVPR 2026][多模态][VLM safety] 提出Dtr（Dynamic Token Reweighting），首个通过优化VLM的KV缓存来防御多模态越狱攻击的推理时防御方法，通过定义"反向安全偏移"（RSS）来识别导致安全退化的视觉token，动态调整其权重以恢复模型的安全对齐能力，同时保持良性任务性能。
tags:
  - CVPR 2026
  - 多模态
  - VLM safety
  - jailbreak defense
  - KV cache optimization
  - token reweighting
  - refusal direction
---

# Dynamic Token Reweighting for Robust Vision-Language Models

**会议**: CVPR 2026  
**arXiv**: [2505.17132](https://arxiv.org/abs/2505.17132)  
**代码**: [GitHub](https://github.com/TanqiuJiang/DTR)  
**领域**: 多模态VLM  
**关键词**: VLM safety, jailbreak defense, KV cache optimization, token reweighting, refusal direction

## 一句话总结
提出Dtr（Dynamic Token Reweighting），首个通过优化VLM的KV缓存来防御多模态越狱攻击的推理时防御方法，通过定义"反向安全偏移"（RSS）来识别导致安全退化的视觉token，动态调整其权重以恢复模型的安全对齐能力，同时保持良性任务性能。

## 研究背景与动机

**领域现状**：大型视觉语言模型（VLM）通过整合视觉与语言能力实现了强大的多模态推理，但引入视觉模态也带来了新的安全漏洞——多模态越狱攻击利用视觉-文本交互绕过安全护栏。

**现有痛点**：微调阶段方案（RLHF安全对齐）计算昂贵且依赖标注数据；推理阶段方案要么需要迭代提示（高开销），要么依赖图像转文本（信息丢失严重）。最近的分布偏移修正方法（ShiftDC、CoCA）需要安全参考，而参考通常通过有损的图像转文本获得。

**核心矛盾**：准确量化视觉模态引起的"安全偏移"需要对比有图/无图两种状态，但获取准确的纯文本对应物（text-only counterpart）本身就是一个有信息损失的过程。

**本文目标**：设计一种无需安全参考数据、无需图像转文本、计算开销极低的推理时越狱防御。

**切入角度**：不测量"加入图片后偏移了多少"，而是测量"通过调整视觉token权重能把偏移反向推回多少"——这个"反向安全偏移"（RSS）可以直接区分越狱查询和良性查询。

**核心 idea**：越狱攻击将查询从"被拒绝"优化到"被接受"，因此也可以被反向优化回去——而良性查询没有这个可逆性。

## 方法详解

### 整体框架
给定查询 $\mathbf{x} = \mathbf{x}_{txt} \| \mathbf{x}_{img}$，对每个视觉token分配缩放因子 $\alpha_i \in [0,1]$，通过梯度下降优化 $\alpha$ 使最后一层激活沿refusal direction的投影最小化（即反向推动安全偏移），同时约束偏离原始激活的距离保持良性性能。优化结束后，低权重token直接驱逐，用调整后的KV缓存运行推理。

### 关键设计

1. **反向安全偏移（Reversal Safety-Relevant Shift, RSS）**

    - 功能：绕过图像转文本，直接通过优化量化安全偏移的可逆性
    - 核心思路：定义 $\Delta^*_{safe}(\mathbf{x}) = \max_{\alpha} \frac{(f(\mathbf{x}) - f(\mathbf{x}(\alpha))) \cdot \mathbf{d}_{ref}}{\|\mathbf{d}_{ref}\|}$，即通过调整视觉token权重，沿refusal direction能达到的最大反向偏移。越狱查询的RSS远大于良性查询（因为攻击本质上就是沿refusal direction优化的结果，天然可逆）
    - 设计动机：避免了图像转文本的信息损失和额外VLM开销，同时为攻击者创造了根本性的两难——增加对抗token重要性会增大RSS，降低对抗token重要性则无法越狱

2. **动态Token重加权优化**

    - 功能：优化视觉token的缩放向量以同时实现安全恢复和性能保持
    - 核心思路：$\alpha^* = \arg\min_{\alpha} \left[\frac{f(\mathbf{x}(\alpha)) \cdot \mathbf{d}_{ref}}{\|\mathbf{d}_{ref}\|} + \lambda \|f(\mathbf{x}) - f(\mathbf{x}(\alpha))\|_2 \right]$，第一项最小化沿refusal direction的投影（恢复安全），第二项保持与原始激活的距离（保持性能），$\lambda$ 平衡两者
    - 设计动机：单纯最小化安全偏移会破坏良性性能，距离约束确保对良性查询影响极小

3. **Early Stopping + Token Eviction**

    - 功能：仅需3-4步优化 + 驱逐低权重token提升效率
    - 核心思路：越狱查询的loss在前几步快速下降，无需等待收敛；权重低于阈值 $\beta$ 的token直接从KV缓存驱逐——视觉token本身高度冗余，驱逐反而加速推理
    - 设计动机：优化步数极少+token驱逐=推理时间甚至比基线更短

4. **Refusal Direction的鲁棒性**

    - 功能：仅需32对harmful/harmless文本提示即可稳定提取refusal direction
    - 核心思路：从AdvBench采样32条有害提示、AlpacaEval采样32条无害提示，计算最后层激活均值差作为refusal direction。实验证明该方向跨语言、跨攻击类型、跨数据集稳定
    - 设计动机：refusal direction捕获的是模型级别的内在属性而非数据特定伪影，因此小样本就够了

### 损失函数 / 训练策略
完全推理时方法，无需训练。优化器用AdamW，学习率0.01，$\lambda=0.1$，默认3-4步梯度下降。

## 实验关键数据

### 主实验

**LLaVA-LLaMA2-7B 攻击成功率（ASR↓，越低越好）**

| 防御方法 | HADES-S | HADES-S+A | HADES-S+T+A | MM-Safety-S | MM-Safety-T | JailBreak-Style |
|---------|---------|-----------|-------------|-------------|-------------|-----------------|
| 无防御 | 31.4% | 44.9% | 56.9% | 70.0% | 72.7% | 34.0% |
| AdaShield | 7.5% | 5.5% | 17.6% | 8.2% | 4.5% | 8.5% |
| ShiftDC | 20.0% | 32.9% | 16.8% | 10.9% | 5.5% | 25.5% |
| CoCA | 23.6% | 20.8% | 35.7% | 24.3% | 26.3% | 8.5% |
| **Dtr** | **8.9%** | **4.8%** | **15.9%** | **3.6%** | **3.6%** | **6.4%** |

### 消融实验

| 配置 | ASR↓ | MM-Vet↑ | 推理时间 |
|------|------|---------|---------|
| Dtr完整 | ~5% | ~35 | ~7s |
| w/o 距离约束 ($\lambda=0$) | ~3% | ~28 | ~7s |
| w/o token eviction | ~5% | ~35 | ~9s |
| 仅eviction无reweighting | ~15% | ~34 | ~5s |
| 基线(无防御) | ~45% | ~35 | ~6s |

### 关键发现
- Dtr在几乎所有攻击类型上都达到最低ASR，且在MM-Safety-S上从70%→3.6%，降幅最大
- Dtr保持甚至提升了推理效率——因为token eviction减少了KV缓存大小
- 距离约束 $\lambda$ 对良性性能至关重要，去掉后MM-Vet从35掉到28
- Refusal direction仅需32对样本即可稳定工作，跨域泛化性强
- 攻击者面临根本两难：增强对抗token→RSS增大→更容易被检测到

## 亮点与洞察
- **攻击者两难困境**是最深刻的贡献——不是在具体攻防上博弈，而是在根本层面证明了攻击和可检测性之间的trade-off
- **首个将KV cache优化用于安全的工作**——将效率优化（token eviction）和安全防御（token reweighting）统一到同一个优化框架中
- **RSS替代图像转文本**的设计巧妙——不问"图片带来了多少偏移"，而问"调整token能把偏移推回多少"，完美绕过了信息损失问题

## 局限与展望
- 每次推理都需要3-4步梯度优化，虽然快但仍有开销
- Refusal direction假设安全概念在激活空间中是线性的，对非常复杂的安全场景可能不够
- 仅在图像+文本的VLM上验证，视频/音频多模态场景未知
- 驱逐阈值 $\beta$ 需要调优

## 相关工作与启发
- **vs AdaShield**: 迭代提示检查图像安全，计算开销大；Dtr直接在KV cache层面操作
- **vs ShiftDC**: 需要图像转文本获取安全参考，有信息损失；Dtr用RSS绕过这一需求
- **vs CoCA**: 在解码logit层面修正偏移，Dtr在更底层的KV cache操作，效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ RSS概念和KV cache安全优化都是首创，攻击者两难困境的分析很深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 5个VLM、3个攻击benchmark、多类攻击、自适应攻击、消融全面
- 写作质量: ⭐⭐⭐⭐⭐ 从问题定义到理论分析到实验，逻辑链完整清晰
- 价值: ⭐⭐⭐⭐⭐ 对VLM安全部署有直接实用价值，开创KV cache安全优化方向

<!-- RELATED:START -->

## 相关论文

- [DTR: Dynamic Token Reweighting for Robust Vision-Language Models](dynamic_token_reweighting_for_robust_visionlanguag.md)
- [On Token's Dilemma: Dynamic MoE with Drift-Aware Token Assignment for Continual Learning of Large Vision Language Models](on_tokens_dilemma_dynamic_moe_with_drift-aware_token_assignment_for_continual_le.md)
- [Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)
- [HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [V2Drop: Variation-aware Vision Token Dropping for Faster Large Vision-Language Models](v2drop_variation_aware_token_dropping.md)

<!-- RELATED:END -->
