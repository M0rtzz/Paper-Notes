---
title: >-
  [论文解读] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking
description: >-
  [CVPR2026][人体理解][图像水印] 提出 AdvMark 两阶段解耦防御框架：Stage 1 Encoder Adversarial Training（EAT）将水印图像移入 non-attackable 区域抵御对抗攻击，Stage 2 直接图像优化抵御失真+再生攻击并保留对抗鲁棒性，在 9 种水印方法 ×10 种攻击上分别提升失真/再生/对抗准确率 29%/33%/46%，且图像质量最优。
tags:
  - CVPR2026
  - 人体理解
  - 图像水印
  - 对抗鲁棒性
  - 扩散再生攻击
  - 解耦训练
  - 对抗训练
  - 图像质量
---

# AdvMark: Decoupling Defense Strategies for Robust Image Watermarking

**会议**: CVPR2026  
**arXiv**: [2602.20053](https://arxiv.org/abs/2602.20053)  
**代码**: 无  
**领域**: human_understanding  
**关键词**: 图像水印, 对抗鲁棒性, 扩散再生攻击, 解耦训练, 对抗训练, 图像质量

## 一句话总结
提出 AdvMark 两阶段解耦防御框架：Stage 1 Encoder Adversarial Training（EAT）将水印图像移入 non-attackable 区域抵御对抗攻击，Stage 2 直接图像优化抵御失真+再生攻击并保留对抗鲁棒性，在 9 种水印方法 ×10 种攻击上分别提升失真/再生/对抗准确率 29%/33%/46%，且图像质量最优。

## 研究背景与动机

**领域现状**：深度学习图像水印（DL watermarking）通过 encoder 将信息嵌入图像、decoder 提取信息，已成为版权保护和内容溯源的核心技术。近年来攻击手段不断升级，形成三重威胁。

**三重威胁**：
   - **对抗攻击（Adversarial Attack）**：如 WEvade，通过微小扰动使 decoder 提取错误信息，攻击后图像视觉上无变化
   - **再生攻击（Regeneration Attack）**：利用扩散模型对水印图像加噪再去噪，有效"洗掉"水印
   - **失真攻击（Distortion Attack）**：如 JPEG 压缩、高斯模糊、裁剪等传统图像处理操作

**联合训练（JAT）的两大问题**：
   - **问题 1**：decoder 对抗训练导致 clean accuracy 下降——为了在对抗样本上也能正确解码，decoder 被迫扩展决策边界，反而在干净图像上精度降低
   - **问题 2**：同时训练三种攻击收敛慢效果差——三种攻击的梯度方向冲突，优化 landscape 复杂，联合训练难以同时满足所有防御需求

**核心洞察**：对抗攻击与失真/再生攻击本质不同。对抗攻击利用模型决策边界的弱点（model-specific），而失真/再生攻击是信号层面的破坏（model-agnostic）。应该解耦防御策略而非联合训练

**核心 idea**：两阶段解耦——先用 EAT 让 encoder 把图像"推入"non-attackable 区域，再用直接图像优化处理失真和再生攻击

## 核心问题
如何同时防御对抗攻击、再生攻击和失真攻击三重威胁，避免联合训练的梯度冲突和 clean accuracy 下降？

## 方法详解

### 整体框架
AdvMark 采用两阶段解耦设计：Stage 1 EAT 专注对抗鲁棒性，通过微调 encoder（而非扩展 decoder 边界）将水印图像移入安全区域；Stage 2 直接优化 encoded image 抵御失真和再生攻击，用约束保留 Stage 1 的对抗鲁棒性。

### 关键设计

1. **Stage 1: Encoder Adversarial Training (EAT)**:

    - 功能：构造 defender-tailored 对抗样本，主要微调 encoder 使水印图像远离对抗攻击可达区域
    - 核心思路：
        - **对抗样本构造**（Eq.2）：$\min_{\delta} |0.5 - l(\text{clamp}(D(x_w + \delta), 0, 1), m)|$，寻找最容易使 decoder 输出接近 0.5（最大不确定性）的扰动 $\delta$，这些是 defender-tailored 对抗样本
        - **Encoder 为主的更新策略**：将对抗样本反馈给 encoder，让 encoder 学习将水印图像嵌入到远离决策边界的安全区域。Decoder 仅在 bit accuracy $< \tau_1$ 时条件更新一次
    - 设计动机：传统对抗训练（AT）同时更新 encoder 和 decoder，decoder 扩展决策边界虽能容纳对抗样本但牺牲 clean accuracy。EAT 反其道而行——不扩大边界，而是让 encoder 把图像"搬到"边界够不到的地方
    - 关键区别：EAT 中 encoder 是主要被训练的对象，decoder 基本冻结

2. **Stage 2: Direct Image Optimization**:

    - 功能：对 Stage 1 输出的水印图像 $x_{w1}$ 进一步优化得到 $x_{w2}$，使其同时抵御失真和再生攻击
    - 核心思路：
        - **优化目标**：直接在像素空间优化 $x_{w2}$（不更新网络参数），使 $x_{w2}$ 经过失真/再生攻击后 decoder 仍能正确提取水印
        - **Constrained Image Loss**：约束 $x_{w2}$ 与 $x_{w1}$ 的偏移量，使优化后的图像不偏离 Stage 1 建立的 non-attackable 区域，从而保留对抗鲁棒性。论文提供了理论保证：在 $\|x_{w2} - x_{w1}\| \leq \epsilon$ 约束下，Stage 1 的对抗鲁棒性以高概率保持
        - **Quality-aware Early-stop**：不使用固定 $\epsilon$-ball 投影（会导致图像质量不均），而是监控图像质量指标（PSNR/SSIM），在质量下降到阈值时提前停止优化
    - 设计动机：失真/再生攻击是 model-agnostic 的信号破坏，用 encoder 训练效果有限；直接优化像素更直接高效，且通过约束保留 Stage 1 的对抗防御成果

3. **两阶段解耦的理论保证**:

    - 功能：证明 Stage 2 优化不会破坏 Stage 1 的对抗鲁棒性
    - 核心思路：若 $x_{w1}$ 在对抗攻击半径 $r$ 内是安全的，且 $\|x_{w2} - x_{w1}\| \leq \epsilon$，则 $x_{w2}$ 在半径 $r - \epsilon$ 内仍是安全的
    - 设计动机：解耦两阶段需要保证后一阶段不破坏前一阶段的成果，理论保证使框架可靠

### 训练与推断流程
- **Stage 1**：在对抗样本上迭代训练 encoder（K 步 PGD 攻击 + encoder 更新），decoder 条件冻结
- **Stage 2**：固定 encoder/decoder，直接优化 $x_{w2}$ 的像素值（梯度下降），quality-aware early-stop
- **推断时**：正常执行 encoder 嵌入 → Stage 2 优化 → 输出最终水印图像

## 实验关键数据

### 主实验——9 种水印方法 ×10 种攻击

| 防御策略 | 失真攻击 Acc (%) | 再生攻击 Acc (%) | 对抗攻击 Acc (%) | PSNR ↑ | SSIM ↑ |
|---------|----------------|----------------|----------------|--------|--------|
| 无防御 (Baseline) | ~60-70 | ~50-60 | ~20-30 | 最高 | 最高 |
| JAT (联合训练) | ~65-75 | ~55-65 | ~40-50 | 较低 | 较低 |
| AT + Distortion | ~70-78 | ~58-68 | ~45-55 | 低 | 低 |
| **AdvMark (Ours)** | **+29%** | **+33%** | **+46%** | **最高** | **最高** |

### 消融实验

| 配置 | 对抗 Acc | 失真 Acc | 再生 Acc | 图像质量 |
|------|---------|---------|---------|---------|
| Stage 1 only (EAT) | 高 | 中 | 中 | 高 |
| Stage 2 only (DIO) | 低 | 高 | 高 | 中 |
| JAT (联合训练) | 中 | 中 | 中 | 低 |
| EAT + 标准 AT (非 EAT) | 中 | — | — | 低 |
| EAT + DIO w/o constraint | 低 | 高 | 高 | 中 |
| **AdvMark (EAT + constrained DIO)** | **高** | **高** | **高** | **高** |

### 关键发现
- **EAT vs 标准 AT**：标准 AT 扩展 decoder 边界导致 clean BA 从 ~99% 降至 ~92%；EAT 保持 clean BA ~98-99% 的同时对抗鲁棒性更强
- **约束的重要性**：去掉 Stage 2 的 image constraint 后，对抗 Acc 显著下降，验证了理论分析
- **Quality-aware early-stop vs ε-ball 投影**：early-stop 在相同 Acc 下 PSNR 平均高 1-2 dB
- **泛化性**：在 9 种不同架构的水印方法上均带来提升，说明 AdvMark 是即插即用的通用框架
- **对抗攻击提升最显著（+46%）**：说明 EAT 的"移入安全区域"策略比"扩展边界"更有效

## 亮点与洞察
- **"移入安全区域 vs 扩展边界"**：这是全文最核心的洞察。传统 AT 让 decoder 包容更多，EAT 让 encoder 把图像送到安全的地方。类比：与其让房子抗震（改 decoder），不如把房子建在没地震的地方（改 encoder）
- **解耦策略的思想深度**：对抗攻击是 model-specific（利用决策边界弱点），失真/再生是 model-agnostic（信号破坏）。两类攻击本质不同，防御策略也应解耦——这是问题理解驱动的设计
- **理论 + 实践的完整链条**：先理论证明约束下鲁棒性保持，再用 quality-aware early-stop 实践落地，理论指导工程
- **通用框架**：即插即用于 9 种已有水印方法，说明方法的通用性和实用价值

## 局限与展望
- Stage 2 的直接图像优化需要额外推断时间（每张图像优化数十步），实时场景可能受限
- Quality-aware early-stop 的阈值需要针对不同应用场景设定，不完全免调参
- 理论保证基于 $\|x_{w2} - x_{w1}\| \leq \epsilon$ 的假设，实际优化可能超出此范围
- 仅在图像水印上验证，视频水印、音频水印等其他模态的适用性待探索
- 对抗攻击类型以 WEvade 为主，更多样化的自适应攻击测试可增强可信度

## 相关工作与启发
- **vs RivaGAN/StegaStamp 等水印方法**: 这些方法的 encoder-decoder 训练不考虑对抗鲁棒性，AdvMark 作为通用后处理可即插即用提升它们的鲁棒性
- **vs 联合对抗训练 (JAT)**: JAT 同时训练三种攻击导致梯度冲突和 clean accuracy 下降；AdvMark 解耦两阶段各自优化，效果和质量均更优
- **vs DiffPure 等扩散净化方法**: DiffPure 用扩散模型净化对抗样本，但这恰好是水印面临的再生攻击。AdvMark 需要同时防御扩散模型作为攻击者的场景
- **启发**：多类型攻击防御的解耦思想可推广到其他安全场景（如多模态对抗防御、联邦学习鲁棒性）

## 评分
- 新颖性: ⭐⭐⭐⭐ EAT "移入安全区域"的思路新颖，两阶段解耦设计有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 9 种方法 ×10 种攻击的大规模对比极为充分，消融细致
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，"扩展边界 vs 移入安全区域"的对比叙事清晰
- 价值: ⭐⭐⭐⭐ 即插即用的通用框架，对水印防御实践有直接指导意义

<!-- RELATED:START -->

## 相关论文

- [RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [Two is Better than One: Efficient Ensemble Defense for Robust and Compact Models](../../CVPR2025/human_understanding/two_is_better_than_one_efficient_ensemble_defense_for_robust_and_compact_models.md)
- [DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [Bilevel Layer-Positioning LoRA for Real Image Dehazing](bilevel_layer-positioning_lora_for_real_image_dehazing.md)
- [Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)

<!-- RELATED:END -->
