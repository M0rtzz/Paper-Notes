---
title: >-
  [论文解读] Watermarking Autoregressive Image Generation
description: >-
  [NeurIPS 2025][LLM安全][自回归图像生成] 首次将 LLM 水印技术（KGW green/red scheme）适配到自回归图像生成模型的 token 层，识别并解决了关键挑战——反向循环一致性（RCC）不足…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "自回归图像生成"
  - "水印"
  - "反向循环一致性"
  - "Token 级水印"
  - "LLM 水印适配"
---

# Watermarking Autoregressive Image Generation

**会议**: NeurIPS 2025  
**arXiv**: [2506.16349](https://arxiv.org/abs/2506.16349)  
**代码**: [https://github.com/facebookresearch/wmar](https://github.com/facebookresearch/wmar)  
**领域**: 图像生成 / AI 水印  
**关键词**: 自回归图像生成, 水印, 反向循环一致性, Token 级水印, LLM 水印适配

## 一句话总结

首次将 LLM 水印技术（KGW green/red scheme）适配到自回归图像生成模型的 token 层，识别并解决了关键挑战——反向循环一致性（RCC）不足，通过 tokenizer-detokenizer 微调和水印同步层实现了鲁棒的、具有理论保证的图像水印检测。

## 研究背景与动机

自回归图像生成模型（DALL-E、Chameleon、RAR 等）通过将图像离散化为 token 序列后用 Transformer 生成，已成为扩散模型的重要替代。然而，其输出的溯源追踪尚无有效方案。

**现有水印方案的不足**：
- **后处理水印**（修改像素）：模型无关但对对抗性攻击脆弱，缺乏理论 p-value 保证
- **扩散模型水印**：专为扩散生成设计，不适用于自回归模型
- **LLM 水印**（KGW）：在文本 token 上效果好，但从未被适配到图像 token

**核心挑战——反向循环一致性（RCC）**：LLM 水印检测需要将生成内容 re-tokenize 后检查 green token 比例。对文本，BPE tokenizer 的 RCC 很高（token match ≈0.995）。但对**图像 VQ tokenizer**，生成的 token → 解码为图像 → 重新编码回 token，约 1/3 的 token 会改变（TM ≈ 0.66）。加上 JPEG 压缩等变换后更是降至 0.31，几何变换（翻转、旋转）则降至接近 0。这是因为：
1. VQ tokenizer 训练目标是前向循环一致性（FCC），解码后的图像 off-manifold
2. 空间敏感性导致语义保持的编辑也会改变大部分 token

## 方法详解

### 整体框架

1. 生成时：直接在自回归 token 序列上应用 KGW 水印（对 green token logit 加 $\delta$）
2. 检测时：图像 → re-tokenize → 统计 green token 数量 → 计算 p-value
3. 核心改进：(a) 微调 detokenizer/encoder 提升 RCC；(b) 水印同步层应对几何变换

### 关键设计

1. **RCC 微调**（Section 3.1）：

    - 保持编码器 $E$、量化器 $Q_C$、码本 $C$ 不变（避免重训自回归模型）
    - 仅微调解码器 $D$ 和编码器副本 $E'$（$E'$ 仅用于检测）
    - **RCC 损失**：$\mathcal{L}_{RCC}(s) = \mathbb{E}_{a \sim \mathcal{A}} \| \hat{z} - E'(a(D(\hat{z}))) \|_2^2$，目标是让解码-编码循环后的 soft latents 逼近原始 hard latents $\hat{z} = C_s$
    - 训练时随机采样数据增强（JPEG、亮度、微小旋转等），使 RCC 对 valuemetric 变换也鲁棒
    - **正则化**：$\mathcal{L}_{reg} = \|D(\hat{z}) - D_0(\hat{z})\|_2^2 + \mathcal{L}_{LPIPS}$，保持解码质量不退化
    - 总损失：$\mathcal{L} = \mathcal{L}_{RCC} + \lambda \cdot \mathcal{L}_{reg}$

2. **水印同步层**（Section 3.2）：

    - 几何变换（翻转、旋转）会彻底打乱 token 对应关系，RCC 微调无法解决
    - 方案：利用 localized watermark [Sander et al.] 在图像四象限嵌入 4 个固定 32-bit 同步消息
    - 检测时：遍历旋转角度网格，找到最佳分离四个消息的正交线对，由此估计并反转几何变换
    - 反转后再运行原始 token 级水印检测器获取 p-value

3. **跨模态联合检测**：

    - 对混合模态输出（如 Chameleon 的图文交织），对各 sample 的 score $S^{(i)}$, $T^{(i)}$, $h^{(i)}$ 求和，去重后统一计算 p-value
    - 跨文本和图像 token 的联合检测进一步提升检测置信度

### 损失函数 / 训练策略

在 50,000 张 ImageNet 训练图像的 token 上训练 10 epochs。Taming: 22h/16 V100；Chameleon: 2.5h/8 H200；RAR-XL: 0.5h/8 H200。水印参数 $\delta=2$, $\gamma=0.25$。

## 实验关键数据

### 主实验（TPR@1% FPR）

| 变体 | 无变换 | Valuemetric | Geometric | 对抗攻击 | 神经压缩 |
|------|--------|------------|-----------|---------|---------|
| Base | 0.99 | 0.26 | 0.01 | 0.43 | 0.48 |
| FT | 1.00 | 0.45 | 0.01 | 0.70 | 0.71 |
| FT+Augs | 1.00 | **0.92** | 0.01 | 0.70 | 0.79 |
| FT+Augs+Sync | 0.98 | 0.83 | **0.82** | 0.69 | 0.80 |

RCC 微调将 valuemetric 鲁棒性从 0.26 提升到 0.92；同步层将 geometric 鲁棒性从 0.01 提升到 0.82。

### 消融实验（Token Match 和生成质量）

| 配置 | Token Match (原始) | Token Match (JPEG Q=25) | FID |
|------|---------|------|------|
| 原始 tokenizer | 0.66 | 0.31 | 16.7 |
| FT | >0.80 | ~0.55 | ≤16.7 |
| FT+Augs | >0.80 | ~0.70 | ≤16.7 |
| FT+Augs+Sync | >0.80 | ~0.70 | 17.3 |

微调显著提升 token match，FID 几乎不变（水印不损害生成质量）。

### 关键发现
- RCC 是水印鲁棒性的核心瓶颈：原始 VQ tokenizer 的 TM 仅 0.66，微调后超过 0.80
- 微调不仅提升 valuemetric 鲁棒性，还意外地提升了对神经压缩和扩散纯化攻击的鲁棒性
- 同步层解决了几何变换这一根本性挑战，但对 valuemetric 鲁棒性有轻微tradeoff
- 对比后处理方法（CIN、MBRS、Trustmark、WAM）：本文方法在扩散纯化和神经压缩上更鲁棒
- 三个模型（Taming、Chameleon、RAR-XL）上结论一致，证明方法的通用性

## 亮点与洞察
- **RCC 问题的发现和解决**是本文最大的贡献：精确诊断了 LLM 水印技术迁移到图像 token 的核心障碍
- 微调方案极其轻量（仅 decoder 和 encoder 副本），不需要重训自回归模型
- 跨模态统一检测的 p-value 计算保持了理论严谨性（二项分布检验）
- 同步层的思路（用辅助信号估计变换 → 反转变换 → 检测水印）具有通用性

## 局限与展望
- 同步层假设裁剪保留一个角落，对任意裁剪需更复杂的同步模式
- 同步层和 valuemetric 鲁棒性之间存在 tradeoff（同步信号被破坏导致错误反转）
- 仅研究零比特水印（检测有无），未探索多比特消息嵌入
- 对 VAR 等非标准自回归架构的适用性有待验证

## 相关工作与启发
- **vs KGW (LLM 水印)**：直接迁移但识别并解决了 RCC 挑战，实现水印从文本到图像 token 的跨模态扩展
- **vs 扩散模型水印（Tree-Ring 等）**：不同范式——扩散模型在潜空间注入水印，本文在 token 序列上注入
- **vs 后处理方法（Trustmark、WAM）**：后处理方法在 valuemetric 鲁棒性上更强，但对扩散纯化和神经压缩很脆弱

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次研究自回归图像生成水印，RCC 问题的发现和解决方案均为原创
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个模型、多种攻击（valuemetric/geometric/对抗/压缩）、与后处理方法对比
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，挑战分析深入，实验全面
- 价值: ⭐⭐⭐⭐⭐ 为快速发展的自回归图像生成领域填补了水印溯源的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ImageSentinel: Protecting Visual Datasets from Unauthorized Retrieval-Augmented Image Generation](imagesentinel_protecting_visual_datasets_from_unauthorized_retrieval-augmented_i.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)
- [\[NeurIPS 2025\] Learning to Watermark: A Selective Watermarking Framework for Large Language Models via Multi-Objective Optimization](learning_to_watermark_a_selective_watermarking_framework_for_large_language_mode.md)
- [\[CVPR 2025\] Towards All-in-One Medical Image Re-Identification](../../CVPR2025/llm_safety/towards_all-in-one_medical_image_re-identification.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
