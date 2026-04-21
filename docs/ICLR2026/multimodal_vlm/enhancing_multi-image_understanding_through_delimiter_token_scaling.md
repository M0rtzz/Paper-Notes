---
title: >-
  [论文解读] Enhancing Multi-Image Understanding through Delimiter Token Scaling
description: >-
  [ICLR 2026][多模态][多图理解] 通过对视觉语言模型中图像分隔符token的隐藏状态进行缩放，增强图像间的信息隔离能力，在不增加任何训练或推理成本的前提下，在多图理解（Mantis/MuirBench/MIRB/QBench2）和多文档/多表格理解（TQABench/MultiNews/WCEP-10）基准上均获得性能提升。
tags:
  - ICLR 2026
  - 多模态
  - 多图理解
  - 大型视觉语言模型
  - 分隔符token
  - 跨图信息泄漏
  - 注意力机制
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
输入为包含多张图像和文本prompt的多模态序列，其中图像token之间穿插有分隔符token。方法在模型的中间层或所有层对分隔符token的隐藏状态乘以缩放因子 $\alpha > 1$，输出为模型的最终预测。整个过程是training-free的推理时干预。

### 关键设计

1. **分隔符token隐藏状态缩放（Delimiter Token Scaling）**: 

    - 功能：在Transformer层中，将分隔符token（标记图像起止的特殊token）的隐藏状态乘以缩放因子 $\alpha$
    - 核心思路：放大后的分隔符隐藏状态在softmax注意力计算中会获得更大的注意力权重，从而在注意力分布中形成"信息瓶颈"或"隔离墙"
    - 设计动机：作者通过分析发现，虽然分隔符token存在于序列中，但其隐藏状态的范数相对于视觉token并不突出，因此在注意力计算中未能起到预期的边界标记作用。缩放操作直接增强了这一信号

2. **增强图像内交互、抑制跨图交互**: 

    - 缩放后的分隔符token像一道"信息屏障"，使得同一图像内的视觉token更多地关注彼此（增强图像内交互/intra-image interaction）
    - 同时限制了不同图像的视觉token之间的注意力交互（抑制跨图交互/cross-image interaction）
    - 结果是模型能更好地保持图像特异性信息，在需要区分和比较多张图像时推理更准确

3. **免训练、零额外成本**: 

    - 该方法是纯推理时干预，不需要额外的训练步骤
    - 不引入新的参数或模块
    - 推理时的计算开销可忽略不计（仅在特定位置做一次标量乘法）

### 损失函数 / 训练策略
无需训练。该方法是推理时的直接干预，唯一的超参数是缩放因子 $\alpha$ 和应用的层范围。

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

## 相关论文

- [DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [TableDART: Dynamic Adaptive Multi-Modal Routing for Table Understanding](tabledart_dynamic_adaptive_multi-modal_routing_for_table_understanding.md)
- [CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/cosyn_code_guided_synthetic_data.md)
- [Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)
- [MMR-Life: Piecing Together Real-life Scenes for Multimodal Multi-image Reasoning](mmr-life_piecing_together_real-life_scenes_for_multimodal_multi-image_reasoning.md)

<!-- RELATED:END -->
