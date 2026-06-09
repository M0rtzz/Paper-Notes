---
title: >-
  [论文解读] PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing
description: >-
  [ICCV 2025][LLM评测][图像去雾] 提出物理引导的雾迁移网络PHATNet，通过将大气散射模型（ASM）扩展到潜空间来解耦和迁移雾模式，生成域自适应的微调数据集，使去雾模型在测试时有效适应未见过的真实世界雾场景。
tags:
  - "ICCV 2025"
  - "LLM评测"
  - "图像去雾"
  - "域适应"
  - "雾迁移"
  - "大气散射模型"
  - "解耦学习"
---

# PHATNet: A Physics-guided Haze Transfer Network for Domain-adaptive Real-world Image Dehazing

**会议**: ICCV 2025  
**arXiv**: [2507.14826](https://arxiv.org/abs/2507.14826)  
**代码**: [GitHub](https://github.com/pp00704831/PHATNet)  
**领域**: LLM评测  
**关键词**: 图像去雾, 域适应, 雾迁移, 大气散射模型, 解耦学习

## 一句话总结

提出物理引导的雾迁移网络PHATNet，通过将大气散射模型（ASM）扩展到潜空间来解耦和迁移雾模式，生成域自适应的微调数据集，使去雾模型在测试时有效适应未见过的真实世界雾场景。

## 研究背景与动机

图像去雾旨在去除图像中的雾霾伪影。现有去雾模型面临关键瓶颈：

**合成-真实域差距**：基于ASM合成的训练数据无法准确模拟真实雾分布

**跨域性能下降**：即使在真实配对数据上训练的模型，在未见域的真实雾图像上性能也大幅下降

**现有域适应方案不足**：GAN方法存在模式坍塌和训练不稳定问题，无法捕获区域特异性退化模式（如非均匀雾）

**核心洞察**：提取雾模式通常比恢复无雾内容更容易。雾通常形成平滑、均匀、半透明的层状覆盖物，在ASM参数域中更可预测。因此可以设计"先解耦雾模式，再迁移到源域干净图像"的策略。

## 方法详解

### 整体框架

PHATNet是一个灵活的域适应框架：给定目标域雾图和源域无雾图，PHATNet将目标域雾模式迁移到源域干净图像上，生成域特异的配对微调数据集。然后用该数据集离线微调去雾模型，实现测试时域适应。整个适应过程离线完成，不增加推理延迟。

### 关键设计

1. **参数化雾解耦与迁移模块（PHDT）**: 核心双分支网络：

    - **顶分支（雾解耦）**：从雾图 $I^H$ 中解耦两类雾相关特征
        - 大气光编码器（ALE）：提取大气光特征 $F^{AL} = \exp(-\text{ALE}(I^H)) \in \mathbb{R}^{128}$
        - 透射图编码器（TME）：提取透射图特征 $F^{TM} = \exp(-\text{TME}(I^H)) \in \mathbb{R}^{H/8 \times W/8 \times 128}$
        - 通过 $e^{-x}$ 归一化到 $[0,1]$，保持与ASM物理解释的一致性
    - **底分支（内容提取）**：内容编码器（CE）从无雾图 $I^C$ 提取内容特征 $F^J$
    - **ASM引导融合**：在潜空间按ASM公式融合：

    $F^I = F^J \times F^{TM} + F^{AL} \times (1 - F^{TM})$

   关键区别是在潜空间操作，使透射图特征成为通道级注意力图，对场景深度变化具有不变性，从而避免鬼影伪影。最后通过Rehazing Encoder（RE）生成最终雾迁移图像。

2. **多尺度雾迁移**: PHATNet采用3级多尺度PHDT结构，从粗到细处理非均匀雾：

    $I^O = \text{PHDT}(I^H, I^C) + \text{UP}(\text{PHDT}(I^{H\downarrow}, I^{C\downarrow}) + \text{UP}(\text{PHDT}(I^{H\downarrow\downarrow}, I^{C\downarrow\downarrow})))$

   3级结构在消融中被证明最优（PSNR 16.25→16.68→16.95→17.07）。

3. **ASM参数域数据增强**: 生成雾迁移图像后，可在ASM参数域进一步增强：

    - 对 $F^{TM}$ 应用gamma校正调节雾密度（增强/减弱）
    - 垂直翻转 $F^{TM}$ 创建额外变体
    - 每对雾/无雾图可生成 $M \times N$ 个训练样本

### 损失函数 / 训练策略

- **雾迁移一致性损失（$\mathcal{L}_{HTC}$）**：衡量雾迁移图像 $I^O$ 与原始雾图 $I^H$ 的差异（当两者使用相同场景时）

  $$\mathcal{L}_{HTC} = \sum_{s=1}^{3} \|I^O_{i,s} - I^H_{i,s}\|_1$$

- **内容泄漏损失（$\mathcal{L}_{CL}$）**：用无雾图像作为雾源输入PHATNet，输出应等于目标无雾图

  $$\mathcal{L}_{CL} = \sum_{s=1}^{3} \|I^O_{i,j,s} - I^C_{j,s}\|_1$$

  这巧妙地避免了GAN训练的不稳定性，同时防止ALE和TME捕获内容特征。

- 总损失：$\mathcal{L}_{total} = \mathcal{L}_{HTC} + \mathcal{L}_{CL}$
- Adam优化器，学习率 $10^{-4}$ → $10^{-7}$ (cosine annealing)，1000 epochs，分辨率 $1600 \times 1200$

## 实验关键数据

### 主实验

**Setting1（源域：NH-Haze20 → 目标域）平均PSNR (dB)**：

| 去雾模型 | Baseline | +PHATNet | 提升 |
|----------|----------|----------|------|
| FocalNet | 16.25 | 17.07 | +0.82 |
| Dehamer | 16.33 | 17.29 | +0.96 |
| MITNet | 14.83 | 16.31 | +1.48 |
| SGDN | 14.84 | 16.77 | +1.93 |

**Setting2（源域：HD-NH-Haze → 目标域）平均PSNR (dB)**：

| 去雾模型 | Baseline | +PHATNet | 提升 |
|----------|----------|----------|------|
| FocalNet | 15.38 | 16.22 | +0.84 |
| Dehamer | 15.95 | 16.61 | +0.66 |
| MITNet | 12.44 | 15.79 | +3.35 |
| SGDN | 12.99 | 16.17 | +3.18 |

PHATNet在所有去雾模型和所有目标域上均带来显著提升，MITNet和SGDN提升最为明显（Setting2平均超3dB）。

### 消融实验

| 组件分析 | PSNR |
|----------|------|
| Baseline（无PHATNet） | 16.25 |
| +CNN (Concatenate) | 16.62 |
| +ALE only | 16.46 |
| +TME only | 16.80 |
| +ALE + TME (完整PHDT) | 17.07 |

| Content-Leakage Loss | NH-Haze21 | HD-NH-Haze | DenseHaze | I-Haze | O-Haze |
|----------------------|-----------|------------|-----------|--------|--------|
| Baseline | 16.45 | 14.76 | 14.80 | 16.13 | 19.10 |
| w/o $\mathcal{L}_{CL}$ | 16.68 | 15.48 | 15.14 | 16.37 | 18.04 |
| w/ $\mathcal{L}_{CL}$ | 16.90 | 15.87 | 15.60 | 16.96 | 20.01 |

- O-Haze（稀疏雾）中 $\mathcal{L}_{CL}$ 影响最大（18.04→20.01），因为稀疏雾场景更容易受内容干扰
- 与竞争方法比较：PHATNet (17.07 dB) > D4+ (16.69) > CNN (16.62) > FocalNet baseline (16.25) > HTFANet (16.21) > PTTD (16.02) > TMD (15.71)
- PHATNet参数26M，生成一张 $1600 \times 1200$ 雾迁移图仅需0.153秒

### 关键发现

- ASM在潜空间操作比在图像域操作更有效：避免了因场景深度变化导致的鬼影伪影
- 内容泄漏损失对稀疏雾场景尤为重要
- 离线微调模式不增加推理延迟，适合实际部署
- 3级多尺度PHDT结构最优

## 亮点与洞察

- **物理模型+深度学习的完美结合**：ASM不再仅用于图像域的简单线性变换，而是作为潜空间特征融合的归纳偏置
- **"提取雾比恢复内容更容易"的洞察朴素但有力**：这一观察使得域适应的切入点从"学习去雾"转向"学习迁移雾"
- **$\mathcal{L}_{CL}$ 设计巧妙**：用无雾图像作为雾源输入，输出应等于无雾目标——简单但有效地防止内容泄漏
- **通用框架**：可搭配任意去雾模型使用，已验证4种SOTA模型均有效

## 局限与展望

- 对真实世界雾分布的先验知识有限，参数域增强仍受制于此
- 训练需要源域的配对真实雾/无雾数据，完全无配对场景尚未探索
- 仅在7个数据集上验证，更多元的真实场景（如工业废气、沙尘暴）待测试
- ASM本身的局限性（如不适用于极厚雾或有色雾）可能传递到PHATNet

## 相关工作与启发

- 与GAN方法（HTFANet、D4+）对比体现了PHDT相对于vanilla ASM的优势
- TMD和PTTD的域适应方法在线推理，PHATNet的离线方案更实用
- 可启发其他恢复任务（如去雨、去雪）的域适应策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将ASM扩展到潜空间的思路新颖，物理+学习结合得当
- **实验充分度**: ⭐⭐⭐⭐⭐ 7个数据集、4个去雾模型、2种设置、完整消融、竞争方法对比
- **写作质量**: ⭐⭐⭐⭐ 方法描述清楚，损失函数设计有直觉解释
- **价值**: ⭐⭐⭐⭐ 通用的去雾域适应框架，对实际部署有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ForCenNet: Foreground-Centric Network for Document Image Rectification](forcennet_foreground-centric_network_for_document_image_rectification.md)
- [\[ICCV 2025\] A Real-world Display Inverse Rendering Dataset](a_real-world_display_inverse_rendering_dataset.md)
- [\[ACL 2025\] RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios](../../ACL2025/llm_evaluation/rulearena_rule_guided_reasoning.md)
- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](../../ACL2025/llm_evaluation/editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)
- [\[ACL 2025\] TripTailor: A Real-World Benchmark for Personalized Travel Planning](../../ACL2025/llm_evaluation/triptailor_a_real-world_benchmark_for_personalized_travel_planning.md)

</div>

<!-- RELATED:END -->
