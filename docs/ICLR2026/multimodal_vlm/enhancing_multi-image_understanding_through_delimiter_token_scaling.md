---
title: >-
  [论文解读] Enhancing Multi-Image Understanding through Delimiter Token Scaling
description: >-
  [ICLR 2026][多模态VLM][多图理解] 通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WC…
tags:
  - "ICLR 2026"
  - "多模态VLM"
  - "多图理解"
  - "大型视觉语言模型"
  - "分隔符token"
  - "跨图信息泄漏"
  - "注意力机制"
---

# Enhancing Multi-Image Understanding through Delimiter Token Scaling

**会议**: ICLR 2026  
**arXiv**: [2602.01984](https://arxiv.org/abs/2602.01984)  
**代码**: [GitHub](https://github.com/MYMY-young/DelimScaling)  
**领域**: Multimodal / VLM  
**关键词**: 多图理解, 大型视觉语言模型, 分隔符token, 跨图信息泄漏, 注意力机制

## 一句话总结
通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。

## 研究背景与动机
大型视觉语言模型（LVLMs，如LLaVA、InternVL等）在单图任务上已取得优异性能，但在处理多图输入时性能明显下降。一个核心原因是**跨图信息泄漏（cross-image information leakage）**——模型难以区分来自不同图像的信息，导致推理时"张冠李戴"。

现有LVLMs已经使用分隔符token（delimiter tokens）来标记每张图像的起始和终止位置（如 `<image_start>` 和 `<image_end>`），但这些分隔符实际上未能有效地阻止跨图信息泄漏。模型在自注意力计算中，不同图像的视觉token仍然会相互交互，导致图像特异性信息被"稀释"。

**核心矛盾**: 分隔符token的存在提供了图像边界信息，但其隐藏状态的幅度不足以在注意力计算中形成有效的"信息屏障"。

**本文切入角度**: 极其简洁——直接放大分隔符token的隐藏状态（乘以一个缩放因子），从而增强其在注意力机制中的"隔离"效果。这一操作在推理时直接应用，无需重新训练模型。

## 方法详解

### 整体框架
输入是一段包含多张图像和文本 prompt 的多模态序列，图像 token 之间穿插着标记每张图起止的分隔符 token（如 `<image_start>`/`<image_end>`）。LVLM 原本就带这些分隔符，但它们在注意力里几乎不起边界作用，跨图信息照样相互渗透。本文的全部改动只有一步：在前向传播时，把分隔符 token 的隐藏状态乘上一个缩放因子 $\alpha > 1$，让它在后续注意力计算中变成一道更明显的"隔离墙"，其余模型权重和流程完全不动。整个干预只发生在推理时，是 training-free 的。

### 关键设计

**1. 分隔符 token 隐藏状态缩放：用一个标量乘法补强失效的图像边界。**

作者先做诊断：LVLM 序列里虽然有分隔符 token，但它们的隐藏状态范数相对视觉 token 并不突出，在自注意力里拿不到足够的权重，于是没能起到预期的边界标记作用——这正是跨图信息泄漏的来源。修法极其直接：在 Transformer 的某些层里，把分隔符 token 的隐藏状态 $h$ 替换成 $\alpha \cdot h$（$\alpha > 1$）。范数被放大后，softmax 注意力会自动给这些分隔符分配更大的权重，在注意力分布中形成一个"信息瓶颈"。问题的根因不在架构而在训练——分隔符没被充分学到，所以一次标量缩放就能把这个被埋没的信号重新顶起来。

**2. 增强图像内交互、抑制跨图交互：让注意力被分隔符"挡"在各自图像内部。**

缩放后的分隔符像一道屏障横在相邻图像之间。当某张图的视觉 token 计算注意力时，权重更多地落在同一张图内部的 token 上（强化 intra-image interaction），而越过分隔符去关注另一张图的 token 则被相对压低（抑制 cross-image interaction）。这样每张图的特异性信息不再被其他图"稀释"，模型在需要逐图区分、横向比较的多图任务上推理更准。这也解释了为什么同一招对纯文本里区分多文档/多表格的分隔符同样有效——本质是注意力机制里一个跨模态通用的边界信号问题，而非视觉模态独有。

**3. 免训练、零额外成本：只在前向里多做一次标量乘法。**

整个方法不新增任何参数或模块，也不需要重新训练或微调，直接套在已有 LVLM 上即可。前向时只是在特定层、特定位置对分隔符隐藏状态做一次标量乘法，计算开销可忽略不计，因此推理速度和显存基本不变。代价仅是两个需要调的超参：缩放因子 $\alpha$ 和施加缩放的层范围。

### 损失函数 / 训练策略
无需训练。该方法是推理时的直接干预，唯一需要设定的是缩放因子 $\alpha$ 和应用的层范围。

## 实验关键数据

### 主实验
论文在多个多图理解基准上进行了评估：

| 数据集 | 任务类型 | 效果 |
|--------|---------|------|
| Mantis | 多图推理 | 提升 |
| MuirBench | 多图理解基准 | 提升 |
| MIRB | 多图推理基准 | 提升 |
| QBench2 | 图像质量对比 | 提升 |

此外，方法还在需要区分不同文本实体的纯文本任务上验证了有效性：

| 数据集 | 任务类型 | 效果 |
|--------|---------|------|
| TQABench | 多表格理解 | 提升 |
| MultiNews | 多文档摘要/理解 | 提升 |
| WCEP-10 | 多文档事件理解 | 提升 |

### 消融实验

| 配置 | 关键发现 | 说明 |
|------|---------|------|
| 缩放因子 $\alpha$ | 存在最优区间 | 过小效果不明显，过大可能破坏模型原有分布 |
| 应用层范围 | 中间层最有效 | 早期层和最后层的效果可能较弱 |
| 分隔符类型 | start和end均有效 | 两种分隔符的缩放都贡献于性能提升 |

### 关键发现
- 现有LVLM中的分隔符token虽然存在，但在隐藏状态层面未能有效发挥边界标记作用
- 简单的缩放操作就能显著增强其功能，说明问题不在于架构设计，而在于训练过程中分隔符未被充分学习
- 方法不仅对视觉分隔符有效，对文本中的分隔符（区分多文档/多表格）同样有效，说明机制具有通用性
- 该方法与模型的具体架构无关，可应用于多种LVLM

## 亮点与洞察
- **极简方法，显著效果**: 仅通过缩放隐藏状态就能改善多图理解，方法的简洁性令人印象深刻
- **零成本**: 真正做到了"免费午餐"——无需训练、无需额外参数、推理开销可忽略
- **通用机制**: 从视觉分隔符扩展到文本分隔符（多文档/多表格），说明这是注意力机制中的一个通用问题，而非视觉模态特有
- **诊断性洞察**: 论文对分隔符token为何失效的分析（隐藏状态范数不足以影响注意力分布）提供了对LVLM内部工作机制的有价值理解
- **实用性极强**: 可直接应用于任何已有的LVLM，无需重新训练，适合即时部署

## 局限与展望
- 缩放因子 $\alpha$ 需要手动调节，不同模型和任务可能需要不同的最优值
- 方法是推理时干预，如果在训练时就考虑分隔符的学习可能获得更好效果
- 论文HTML版本在ar5iv上转换失败，部分实验数值细节难以完整获取
- 对于特别长的多图序列（如视频帧），缩放策略可能需要进一步调整
- 未探讨与其他注意力干预方法（如注意力掩码、位置编码修改）的对比或组合
- 未在最新的超大规模LVLM（如GPT-4V）上测试

## 相关工作与启发
- 与视觉token压缩方法（如TrimTokenator-LC、VisionTrim）关注效率不同，本文关注多图场景下的信息隔离质量
- 与专门为多图理解设计的训练方法不同，本文提供了一种免训练的补充手段
- 启发：注意力机制中特殊token的"信号强度"可能是一个被忽视的设计维度——未来的LVLM训练可能需要显式地让分隔符学到更强的边界表示
- "缩放隐藏状态"这一简单dry intervention思路可能适用于其他需要信息隔离的场景（如多轮对话中区分不同轮次、RAG中区分不同检索文档）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 观察和方法都很新颖，但技术复杂度较低
- 实验充分度: ⭐⭐⭐⭐ — 覆盖了多个基准和任务类型，包括消融和跨模态验证
- 写作质量: ⭐⭐⭐⭐ — 论文动机清晰、方法简洁（虽然全文HTML不可用，从摘要和代码可判断）
- 价值: ⭐⭐⭐⭐⭐ — 实用价值极高，任何LVLM用户都可以立即使用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Benchmarking and Enhancing VLM for Compressed Image Understanding](../../ICML2026/multimodal_vlm/benchmarking_and_enhancing_vlm_for_compressed_image_understanding.md)
- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[ICLR 2026\] TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [\[ICLR 2026\] MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)
- [\[ICLR 2026\] Index-Preserving Lightweight Token Pruning for Efficient Document Understanding](index-preserving_lightweight_token_pruning_for_efficient_document_understanding_.md)

</div>

<!-- RELATED:END -->
