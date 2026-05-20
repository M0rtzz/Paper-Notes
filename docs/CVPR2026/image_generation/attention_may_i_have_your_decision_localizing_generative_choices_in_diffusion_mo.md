---
title: >-
  [论文解读] Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models
description: >-
  [CVPR 2026][图像生成][扩散模型] 本文通过线性探针（linear probing）发现扩散模型中**隐式决策**（如未指定性别时默认生成男性）主要由自注意力层而非交叉注意力层控制，并基于此提出 ICM 方法，仅在少量关键自注意力层上进行干预即可实现 SOTA 的去偏见效果，同时最小化图像质量退化…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "扩散模型"
  - "注意力机制"
  - "Debiasing"
  - "Linear Probing"
  - "Implicit Decision"
---

# Attention, May I Have Your Decision? Localizing Generative Choices in Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2604.06052](https://arxiv.org/abs/2604.06052)  
**代码**: [https://github.com/kzaleskaa/icm](https://github.com/kzaleskaa/icm)  
**领域**: 图像生成 / 扩散模型可解释性  
**关键词**: Diffusion Model Interpretability, Self-Attention, Debiasing, Linear Probing, Implicit Decision

## 一句话总结
本文通过线性探针（linear probing）发现扩散模型中**隐式决策**（如未指定性别时默认生成男性）主要由自注意力层而非交叉注意力层控制，并基于此提出 ICM 方法，仅在少量关键自注意力层上进行干预即可实现 SOTA 的去偏见效果，同时最小化图像质量退化。

## 研究背景与动机

**领域现状**：文生图扩散模型功能强大但内部机制不透明。当提示不够具体时（如"一张人的照片"未指定性别），模型必须做出隐式决策来填充缺失细节。

**隐式决策的双面性**：
   - 无害情况：生成"花"时自行决定颜色和形状
   - 有害情况：生成"医生"默认为男性（性别偏见）、"美国总统"只生成特定人物（代表性偏差）

**现有定位方法的局限**：
   - 主流方法（如 prompt injection/activation patching）通过在特定层注入不同文本提示来测量层的影响
   - 但这些方法本质上依赖显式文本条件——这**掩盖了模型在缺乏显式条件时的内部决策机制**
   - 已有研究认为交叉注意力层负责语义信息整合，因此大多数干预瞄准交叉注意力

**核心假设**：隐式决策的计算机制是**局部化的**，且与显式文本条件的机制**不同**。

**切入角度**：用线性探针直接在中间激活上训练分类器，无需 prompt 工程即可定位决策层，避免了 prompt injection 的混淆效应。

## 方法详解

### 整体框架
ICM（Implicit Choice-Modification）分为三步：
1. 使用通用提示生成图像并收集中间层激活
2. 外部分类器对生成图像进行伪标注
3. 训练线性探针定量评估每层的属性可分性 → 选择最关键层 → 定向干预

### 关键设计

1. **基于线性探针的层选择**：

    - 定义概念 $\mathcal{C}$（如性别）和互斥选项 $\mathcal{A} = \{a_1, ..., a_K\}$（如male, female）
    - 用通用提示 $p_{gen}$（"a photo of a person"）生成 $N$ 张图像
    - 提取每层 $l$、每时间步 $t$ 的激活 $H_{l,t}^{(i)}$（平均池化）
    - 用 FairFace 等外部分类器对生成图像伪标注
    - 对每 $(l, t)$ 训练逻辑回归：$f_{l,t}: \mathbb{R}^{d_l} \to \mathcal{A}$
    - 分类准确率高的层 = 属性可分性强 = 隐式决策的关键位置
    - **核心发现**：自注意力层的探针准确率显著高于交叉注意力层，中间块附近达到峰值

2. **激活引导（Activation Steering）**：

    - 利用逻辑回归的权重向量作为引导方向：$s_{\ell,t} = \hat{w}_{\ell,t} / \|\hat{w}_{\ell,t}\|_2$
    - 生成时修改选定层的激活：$H_{\ell,t}' = H_{\ell,t} + \alpha \cdot s_{\ell,t}$
    - $\alpha$ 控制干预强度
    - **为什么有效**：权重向量定义了最大类可分性方向，沿此方向移动可直接改变隐式决策

3. **选择性微调**：

    - 仅对选定的自注意力层应用 LoRA 微调
    - 生成特定属性的图像（如"年轻医生"），但以通用提示（"医生"）作为训练条件
    - 标准扩散损失：$\mathcal{L} = \mathbb{E}_{\epsilon,t}[\|\epsilon - \hat{\epsilon}_\theta(x_t, p, t)\|_2^2]$

### 关键洞察：自注意力 vs 交叉注意力

| 层类型 | 角色 | 隐式决策相关性 |
|--------|------|---------------|
| 交叉注意力 | 整合文本条件的语义信息 | 低——当提示不含属性 token 时，交叉注意力无从得知属性 |
| 自注意力 | 传播和固化随机线索 | **高**——将初始噪声中的独立线索（如发型 + 口红）统一为连贯生成 |

作者的假设：自注意力层通过传播和巩固来自初始噪声的随机线索来做出隐式决策。

## 实验关键数据

### 主实验（SD v1.5 去偏见）

| 方法 | Gender FD↓ | Age FD↓ | Race FD↓ | FID↓ | CLIP-T↑ |
|------|------------|---------|----------|------|---------|
| Original | 0.564 | 0.752 | 0.558 | 120.06 | 0.6155 |
| DIFFLENS | **0.046** | **0.049** | 0.401 | 112.83 | 0.6090 |
| Finetuning | 0.050 | 0.746 | 0.198 | 161.47 | 0.6095 |
| **ICM (Steering)** | 0.087 | 0.133 | **0.266** | 122.08 | **0.6140** |
| **ICM (Finetuning)** | 0.535 | 0.681 | 0.449 | 143.98 | 0.6189 |

### 消融实验：自注意力 vs 交叉注意力引导

| 干预目标 | Gender FD↓ | Age FD↓ | Race FD↓ | FID↓ |
|----------|------------|---------|----------|------|
| Cross-attn steering | 0.365 | 0.612 | 0.428 | 118.39 |
| **Self-attn steering** | **0.085** | **0.273** | **0.298** | 118.31 |
| Cross-attn finetuning | 0.535 | 0.681 | 0.449 | 143.98 |
| **Self-attn finetuning** | **0.463** | **0.770** | **0.524** | 139.04 |

### 关键发现
- **自注意力层引导在所有属性上都大幅优于交叉注意力层引导**（Gender FD: 0.085 vs 0.365）
- ICM Steering 实现了最好的质量-公平性权衡：FD 下降的同时 FID/CLIP-T 几乎不受影响
- 仅干预 2-4 个关键自注意力层 > 干预所有层（后者导致伪影和质量退化）
- 通用提示和显式提示产生的激活分布**可被线性分离**（测试准确率 56-89%），证实两种生成机制确实不同
- 方法可推广到 SDXL（70 层）和 SANA（Transformer 架构，20 层）

## 亮点与洞察
- **"隐式决策的位置在自注意力层"这一发现**意义重大，挑战了以往"所有语义信息都在交叉注意力"的认知
- **精准而非全面的干预**：仅改动 2-4 层就能实现 SOTA 效果，比修改所有层或整个中间块更高效
- 线性探针方法避免了 prompt injection 的混淆效应，提供了更纯粹的内部机制视角
- 实验设计严谨：通过训练探针区分隐式/显式生成来验证两种机制的差异

## 局限与展望
- Steering 强度 $\alpha$ 需要针对每个场景逐个调整，缺乏自动化
- 对于多属性同时去偏（如性别+年龄+种族）的效果未充分探讨
- 线性探针假设属性在激活空间中是线性可分的，对更复杂的概念可能不成立
- 从去偏推广到更一般的隐式概念控制还需更多验证

## 相关工作与启发
- DIFFLENS 在特定指标上更好，但使用稀疏自编码器方法复杂度更高
- 与 LLM 中的"知识定位"研究（如 ROME, activation patching）方法论类似
- 对 prompt injection 方法提出了质疑——显式注入不等于模型的内部决策机制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "自注意力层是隐式决策中心"的发现是重要贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 三种属性 × 三种架构 × 多种干预方式，分析非常深入
- 写作质量: ⭐⭐⭐⭐⭐ 叙事优雅，从假设到验证到应用的逻辑链完整
- 价值: ⭐⭐⭐⭐ 对理解扩散模型内部机制有基础贡献，去偏应用实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] HAM: A Training-Free Style Transfer Approach via Heterogeneous Attention Modulation for Diffusion Models](ham_a_training-free_style_transfer_approach_via_heterogeneous_attention_modulati.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](../../AAAI2026/image_generation/countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[ICCV 2025\] Attention to Neural Plagiarism: Diffusion Models Can Plagiarize Your Copyrighted Images!](../../ICCV2025/image_generation/attention_to_neural_plagiarism_diffusion_models_can_plagiarize_your_copyrighted_.md)
- [\[AAAI 2026\] Melodia: Training-Free Music Editing Guided by Attention Probing in Diffusion Models](../../AAAI2026/image_generation/melodia_training-free_music_editing_guided_by_attention_probing_in_diffusion_mod.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)

</div>

<!-- RELATED:END -->
