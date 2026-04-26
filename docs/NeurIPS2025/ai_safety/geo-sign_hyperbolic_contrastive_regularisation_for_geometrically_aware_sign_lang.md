---
title: >-
  [论文解读] Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation
description: >-
  [NeurIPS 2025][AI安全][手语翻译] Geo-Sign 提出将骨架特征投影到 Poincaré 球模型的双曲空间中，通过双曲对比损失正则化 mT5 语言模型，使其感知手语运动的层次结构，仅用骨架数据就在 CSL-Daily 上超越了基于 RGB 的 SOTA 方法（BLEU-4 +1.81, ROUGE-L +3.03）。
tags:
  - NeurIPS 2025
  - AI安全
  - 手语翻译
  - 双曲几何
  - Poincaré球
  - 对比学习
  - 骨架表示
---

# Geo-Sign: Hyperbolic Contrastive Regularisation for Geometrically Aware Sign Language Translation

**会议**: NeurIPS 2025  
**arXiv**: [2506.00129](https://arxiv.org/abs/2506.00129)  
**代码**: [GitHub](https://github.com/ed-fish/geo-sign)  
**领域**: ai_safety  
**关键词**: 手语翻译, 双曲几何, Poincaré球, 对比学习, 骨架表示

## 一句话总结
Geo-Sign 提出将骨架特征投影到 Poincaré 球模型的双曲空间中，通过双曲对比损失正则化 mT5 语言模型，使其感知手语运动的层次结构，仅用骨架数据就在 CSL-Daily 上超越了基于 RGB 的 SOTA 方法（BLEU-4 +1.81, ROUGE-L +3.03）。

## 研究背景与动机

**领域现状**：手语翻译 (SLT) 近年转向利用大型语言模型（如 T5 变体）处理视觉特征。大多数 SOTA 方法依赖 RGB 视频输入和大型视觉编码器（如 DINO-ViT），计算成本高且存在隐私问题。

**现有痛点**：骨架表示通过时空图卷积 (ST-GCN) 提取后，被投影到欧几里得空间供语言模型处理。然而欧几里得空间中，大尺度的手臂运动主导嵌入范数，压缩了手指关节等细粒度动作的区分度。例如"水"的 ASL 手势需要区分手指 W 形 + 点下巴（叶节点动作）和手臂外展（枝节点动作），在平坦空间中二者嵌入会混淆。

**核心矛盾**：手语运动具有天然的树状层次结构（躯干→手臂→手腕→手指），但欧几里得空间的多项式体积增长无法有效编码这种层次。

**本文目标**：增强骨架表示的几何性质，使其自然尊重手语运动学的层次结构。

**切入角度**：双曲空间的体积呈指数增长 $V_H(r) \propto e^{(d-1)r}$，天然适合编码树状层次——边界附近距离放大区分细粒度差异，中心附近类欧几里得适合语义级表示。

**核心 idea**：将骨架部件特征投影到可学习曲率的 Poincaré 球中，用双曲对比损失对齐姿态-文本嵌入，正则化语言模型以感知运动层次。

## 方法详解

### 整体框架

输入 2D 骨架关键点（RTM-Pose 提取）→ 按身体部位分组（身体/左手/右手/面部）→ 各部位 ST-GCN 提取特征 → 两条分支：(1) 拼接投影到 mT5 编码器（欧几里得主分支）；(2) 时间平均池化后投影到 Poincaré 球（双曲正则化分支）。最终损失 = $\alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{hyp\_reg}$。

### 关键设计

1. **双曲投影层 (Hyperbolic Projection)**：

    - 功能：将欧几里得部件特征 $\bar{\mathbf{f}}_p$ 映射到 Poincaré 球
    - 核心公式：$\mathbf{h}_p = \exp_{\mathbf{0}}^c(s_p \mathbf{W}^p \bar{\mathbf{f}}_p)$，其中指数映射 $\exp_{\mathbf{0}}^c(\mathbf{v}) = \tanh(\frac{\sqrt{c}\|\mathbf{v}\|_2}{2}) \frac{\mathbf{v}}{\frac{\sqrt{c}}{2}\|\mathbf{v}\|_2}$
    - 可学习缩放标量 $s_p$ 控制各部件在双曲空间中的"深度"——大运动幅度的手臂靠近原点，细粒度的手指靠近边界
    - 设计动机：利用双曲空间边界区域的距离放大效应区分细粒度手指动作

2. **加权 Fréchet 均值聚合 (Pooled 策略)**：

    - 功能：将多个部件嵌入 $\{\mathbf{h}_p\}$ 聚合为全局姿态嵌入 $\boldsymbol{\mu}_\text{pose}$
    - 核心思路：权重 $w_p \propto \exp(d_{\mathbb{B}_c}(\mathbf{0}, \mathbf{h}_p))$，离原点越远（越靠近边界、越细粒度）的部件权重越大
    - 迭代算法：在切空间中做加权对数映射求和，再用指数映射返回流形
    - 设计动机：Fréchet 均值是双曲空间中几何上正确的平均操作，比欧几里得平均更准确

3. **双曲注意力对齐 (Token 策略)**：

    - 功能：每个姿态部件 $\mathbf{h}_p$ 作为 Query，对文本 token 嵌入做双曲注意力，生成部件特定的上下文向量 $\mathbf{c}_p$
    - Key 变换：$\mathbf{k}_t = (\mathbf{M} \otimes \mathbf{v}_t) \oplus \mathbf{b}$（Möbius 仿射变换）
    - 注意力分数：$s_{pt} = -d_{\mathbb{B}_c}(\mathbf{h}_p, \mathbf{k}_t)$（负测地距离）
    - 上下文向量：$\mathbf{c}_p = \mu_{\mathcal{B}_c}(\{\mathbf{v}_t\}, \{\alpha_{pt}\})$（双曲加权中点）
    - 设计动机：允许不同身体部位关注不同的文本 token，实现更精细的姿态-语义对齐

4. **双曲对比损失 (Geometric Contrastive Loss)**：

    - 核心公式：$\mathcal{L}_\text{hyp\_pair}(\mathbf{p}_i, \mathbf{t}_i) = -\log \frac{\exp(-d_{\mathbb{B}_c}(\mathbf{p}_i, \mathbf{t}_i)/\tau)}{\sum_{j=1}^B \exp(-d_{\mathbb{B}_c}(\mathbf{p}_i, \mathbf{t}_j)/\tau + m \cdot \mathbb{I}(i \neq j))}$
    - 可学习温度 $\tau$ 和可学习 margin $m$
    - 测地距离：$d_{\mathbb{B}_c}(\mathbf{u}, \mathbf{v}) = \frac{2}{\sqrt{c}} \text{artanh}(\sqrt{c}\|(-\mathbf{u}) \oplus_c \mathbf{v}\|_2)$

### 训练策略
- 总损失：$\mathcal{L}_\text{total} = \alpha \cdot \mathcal{L}_{CE} + (1-\alpha) \cdot \mathcal{L}_{hyp\_reg}$，$\alpha$ 在训练中动态调整，初期偏重双曲正则化
- 欧几里得参数用 AdamW，双曲参数（曲率 $c$、流形参数）用 Riemannian Adam
- 曲率 $c$ 端到端可学习（初始 $c=1.5$），在 log 空间优化

## 实验关键数据

### 主实验：CSL-Daily 手语翻译

| 方法 | 模态 | B-1 | B-4 | ROUGE-L |
|------|------|-----|-----|---------|
| Uni-Sign (Pose) | 骨架 | 53.86 | 25.61 | 54.92 |
| Uni-Sign (Pose+RGB) | 骨架+RGB | 55.08 | 26.36 | 56.51 |
| Geo-Sign (Euclidean Token) | 骨架 | 54.02 | 25.98 | 53.93 |
| Geo-Sign (Hyperbolic Pooled) | 骨架 | 55.80 | 27.17 | 57.75 |
| **Geo-Sign (Hyperbolic Token)** | **骨架** | **55.89** | **27.42** | **57.95** |
| CV-SLT (Gloss-based) | RGB | 58.29 | 28.94 | 57.06 |

### 消融：对比策略与几何空间

| 配置 | B-4 | ROUGE-L | 说明 |
|------|-----|---------|------|
| Uni-Sign (Pose, 无正则化) | 25.61 | 54.92 | 基线 |
| Euclidean Pooled | 25.72 | 55.57 | 欧几里得对比有帮助 |
| Euclidean Token | 25.98 | 53.93 | token对齐改善BLEU |
| Hyperbolic Pooled | 27.17 | 57.75 | 双曲 vs 欧几里得: +1.45/+2.18 |
| **Hyperbolic Token** | **27.42** | **57.95** | 最佳配置 |

### 关键发现
- 双曲空间 vs 欧几里得空间：Hyperbolic Token 相比 Euclidean Token 在 B-4 上提升 +1.44, ROUGE-L 提升 +4.02，证明双曲几何的贡献显著
- 仅用骨架数据的 Geo-Sign 超越了 Uni-Sign (Pose+RGB)，ROUGE-L 上甚至首次超越 SOTA gloss-based 方法 CV-SLT (57.95 vs 57.06)
- 在 How2Sign (ASL) 和 WLASL2000 (孤立手语识别) 上也有一致改善，说明方法的语言无关性
- Token 策略优于 Pooled 策略（更精细的部件-文本对齐），但 Pooled 更高效

## 亮点与洞察
- **几何与语言学的精妙契合**：手语的树状运动层次结构天然匹配双曲空间的指数体积增长，这是"用对了数学工具"的典范
- **可学习曲率**：$c$ 端到端优化让模型自适应调节"放大镜"的倍率——更负的曲率 $\kappa = -c$ 更强调细粒度区分
- **仅骨架超越 RGB**：在隐私保护的前提下（骨架天然去标识）实现了超越 RGB 方法的性能，对实际部署（如公共场所手语翻译）有重要意义

## 局限与展望
- 双曲计算（指数映射、对数映射、Möbius 加法）的数值稳定性需要高精度浮点（float32），可能影响训练效率
- Token 策略的内存开销随序列长度增长显著，在超长手语视频上可能受限
- 仅在两个手语数据集 (CSL-Daily, How2Sign) 上验证，缺少更多语言/数据集的评估
- 曲率参数 $c$ 的初始值 (1.5) 和超参数 $d_\text{hyp}=256$ 的选择缺乏理论指导

## 相关工作与启发
- **vs Uni-Sign (Gong et al.)**：Uni-Sign 是本文的基础架构（共享 ST-GCN 预训练），Geo-Sign 在其上只增加了双曲正则化分支，证明了几何先验的价值
- **vs Sign2GPT / FLa-LLM**：这些方法依赖大型视觉编码器 + LLM，Geo-Sign 只用轻量骨架输入就达到竞争力，效率更高
- **vs 双曲动作识别 (Franco et al.)**：已有工作将双曲空间用于一般动作识别，但 Geo-Sign 首次将其用于端到端手语翻译，并引入双曲注意力和 Fréchet 均值聚合

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 双曲几何应用于手语翻译的想法新颖且动机充分，Token 策略的双曲注意力是原创设计
- 实验充分度: ⭐⭐⭐⭐ 消融充分证明了双曲空间的贡献，但数据集有限
- 写作质量: ⭐⭐⭐⭐ 数学严谨，图示直觉，但双曲几何基础部分较密集
- 价值: ⭐⭐⭐⭐ 对手语翻译社区有直接价值，双曲正则化的思路可迁移到其他层次化动作理解任务

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Angular Gradient Sign Method: Uncovering Vulnerabilities in Hyperbolic Networks](../../AAAI2026/ai_safety/angular_gradient_sign_method_uncovering_vulnerabilities_in_h.md)
- [\[NeurIPS 2025\] FairContrast: Enhancing Fairness through Contrastive Learning and Customized Augmentation](faircontrast_enhancing_fairness_through_contrastive_learning_and_customized_augm.md)
- [\[NeurIPS 2025\] Incentivizing Time-Aware Fairness in Data Sharing](incentivizing_time-aware_fairness_in_data_sharing.md)
- [\[NeurIPS 2025\] MARS: A Malignity-Aware Backdoor Defense in Federated Learning](mars_a_malignity-aware_backdoor_defense_in_federated_learning.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)

<!-- RELATED:END -->
