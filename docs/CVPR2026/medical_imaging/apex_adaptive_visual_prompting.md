---
title: >-
  [论文解读] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation
description: >-
  [CVPR 2026][医学图像][视觉提示] 提出 APEX（Adaptive Prompt EXtraction），通过从可学习 prompt 记忆中自适应检索输入特定的 visual prompt（而非为每个域固定一个 prompt），结合低频特征对比学习增强域间区分能力…
tags:
  - "CVPR 2026"
  - "医学图像"
  - "视觉提示"
  - "域适应"
  - "域泛化"
  - "医学图像分割"
  - "低频特征对比学习"
---

# From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation

**会议**: CVPR 2026  
**arXiv**: [2604.17455](https://arxiv.org/abs/2604.17455)  
**代码**: [https://github.com/cetinkayaevren/apex/](https://github.com/cetinkayaevren/apex/)  
**领域**: 医学图像  
**关键词**: 视觉提示, 域适应, 域泛化, 医学图像分割, 低频特征对比学习

## 一句话总结

提出 APEX（Adaptive Prompt EXtraction），通过从可学习 prompt 记忆中自适应检索输入特定的 visual prompt（而非为每个域固定一个 prompt），结合低频特征对比学习增强域间区分能力，显著提升医学图像分割在已见域和未见域上的泛化性能。

## 研究背景与动机

**领域现状**：Visual Prompting (VP) 作为域适应方法在医学图像分割中颇受关注。VP 方法在输入图像空间添加可学习参数，优化后可以将目标域数据映射到预训练模型可处理的空间，由于不修改原始模型参数，天然避免了灾难性遗忘。

**现有痛点**：当前 VP 方法（如 VPT、FVP、A2XP）为每个目标域优化单一 prompt 并统一应用于所有图像。这存在两个根本限制：(1) 忽略域内变异性——同一设备采集的不同图像在采集设置和病理特征上可能差异显著，单一 prompt 过于粗糙；(2) 忽略域间变异性——固定于某域优化的 prompt 难以适配来自不同设备或机构的数据，对未见域的泛化能力有限。

**核心矛盾**：表达力与泛化性的 trade-off——粗粒度的域级 prompt 表达力不足但易于优化，细粒度的输入级 prompt 表达力强但需要更精巧的检索机制。

**本文目标**：设计一种自适应 prompt 提取框架，能够根据每张输入图像的特征动态组合最合适的 prompt，同时保证对已见域和未见域的泛化性。

**切入角度**：医学图像的域偏移主要来源于全局外观变化（对比度、亮度、色调），这些变化编码在频率域的低频分量中。如果能从低频信息中提取域区分性特征，就能精确检索匹配的 prompt。

**核心 idea**：构建包含多样 prompt 向量的记忆库，用基于低频频谱的域特征编码器查询记忆库，通过加权组合得到输入特定的 prompt，同时用低频特征对比学习 (LFC) 增强域间区分性。

## 方法详解

### 整体框架

输入图像经 FFT 变换提取低频振幅分量，送入 APEX 模块（域特征编码器 → prompt 记忆查询 → prompt 解码器），生成输入特定的 prompt。prompt 以元素乘法应用于原始低频振幅，经 IFFT 重建后送入冻结的分割模型。整个优化过程中只有 APEX 的参数被更新，分割模型参数完全冻结。

### 关键设计

1. **自适应 Prompt 记忆检索**:

    - 功能：根据输入图像的域特征动态组合最优 prompt
    - 核心思路：域特征编码器 $E^D$ 从低频振幅中提取 K 维特征向量 $z_m^n$。Prompt 记忆 $B \in \mathbb{R}^{J \times K}$ 包含 J 个可学习 prompt 向量（正交初始化）。计算 $z_m^n$ 与每个 $b_j$ 的余弦相似度得到寻址向量 $a_m^n$，加权求和 $z_m^{\prime n} = \sum_j a_{m,j}^n \cdot b_j$ 得到最终 prompt 特征，经解码器 $D^P$ 生成空间 prompt
    - 设计动机：加权组合让系统能在特征级别灵活组合已有知识，产生超越任何单一存储 prompt 的适应能力。正交初始化促进记忆槽的多样性

2. **低频特征对比学习 (LFC)**:

    - 功能：增强域特征编码器的域间区分性和域内聚类性
    - 核心思路：在域特征 $z_m^n$ 上加辅助投影头得到 $z_m^{n,aux}$。对比损失 $\mathcal{L}_{LFC}$ 拉近同域样本特征、推远异域样本特征，温度参数 $\tau$ 控制相似度缩放。辅助投影头仅训练时使用，推理时丢弃
    - 设计动机：域特征编码器需要同时捕获域间差异（跨域区分）和域内变异（同域内的精细差别）。对比学习通过聚类同域特征来学习域的共享特性，同时分离不同域来捕获差异

3. **低频空间 Prompt 应用**:

    - 功能：在不破坏解剖结构的前提下实现有效的域适应
    - 核心思路：prompt 仅作用于频率域的低频振幅分量（元素乘法），高频分量和相位信息保持不变
    - 设计动机：医学图像的域偏移主要反映在低频（对比度、亮度等全局外观），而高频和相位编码了精细解剖细节和空间布局。在低频空间做 prompt 既能有效适配域偏移，又保护了分割所需的结构信息

### 损失函数 / 训练策略

总损失 $\mathcal{L}_{total} = \mathcal{L}_{Seg} + \mathcal{L}_{LFC}$。分割损失使用 Dice + Cross-Entropy。记忆 B 通过梯度反传更新。域特征编码器同时受两个损失优化。

## 实验关键数据

### 主实验

| 任务 | Backbone | 域类型 | Source Only | VPAD(最强基线) | APEX |
|------|---------|--------|------------|--------------|------|
| 息肉分割 | UNet | 已见域 Avg | 81.43 | 82.39 | **83.75** |
| 息肉分割 | UNet | 未见域 Avg | 55.16 | 56.31 | **58.03** |
| 息肉分割 | PraNet | 已见域 Avg | 81.44 | 82.45 | **83.75** |
| OC/OD | UNet | 已见域 Avg | 85.08 | 85.88 | **88.43** |
| OC/OD | UNet | 未见域 Avg | 73.46 | 76.76 | **82.57** |

### 消融实验

| 配置 | Dice(已见) | Dice(未见) | 说明 |
|------|-----------|-----------|------|
| APEX (Full) | 最优 | 最优 | 完整方法 |
| w/o LFC | 下降 | 显著下降 | 域间区分性不足 |
| w/o Memory (单一 prompt) | 下降 | 显著下降 | 回退到传统 VP |
| 固定 prompt (非自适应) | 下降 | 下降 | 无法处理域内变异 |

### 关键发现

- 在未见域上的提升尤为显著，例如 OC/OD 任务的 RIM-ONE-r3 域上 UNet 提升 41.44% DICE
- LFC 对未见域泛化贡献最大——表明域特征的区分性是泛化的关键
- APEX 作为即插即用模块，在 5 种不同 backbone 上均一致提升，证明了方法的通用性
- 记忆槽数 J=150 在多数设置下表现最优

## 亮点与洞察

- 从"一域一 prompt"到"一图一 prompt"的范式升级思路清晰且有效。记忆+检索的机制让系统在有限的训练域上学到的 prompt 组件可以自由组合，从而泛化到未见域
- 在低频振幅空间做 prompt 是很有洞察力的设计——既利用了频域的物理可解释性（低频=全局外观=域偏移主因），又保护了分割任务最需要的高频结构信息
- 方法的即插即用特性使其在临床实际中非常实用，任何现有分割模型都可以搭配 APEX 提升跨域性能

## 局限与展望

- 需要多个域的数据来训练 APEX，在域数量很少时记忆多样性可能不足
- 低频 prompt 假设域偏移主要在低频，对于高频域偏移（如不同分辨率）可能效果有限
- 未验证在 3D 医学图像（CT/MRI 体数据）上的有效性
- 改进方向：引入在线 prompt 记忆更新机制以适应测试时遇到的新域

## 相关工作与启发

- **vs VPT/FVP/A2XP**: 传统 VP 方法使用固定域级 prompt，APEX 使用输入级自适应 prompt，在未见域上优势尤为明显
- **vs VPTTA**: 同样使用记忆结构但专门设计用于测试时适应，问题设置不同
- **vs 域适应微调**: 微调会修改模型参数导致灾难性遗忘，APEX 完全不动模型参数

## 评分

- 新颖性: ⭐⭐⭐⭐ 自适应 prompt 检索 + 低频对比学习的组合有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 两个任务、5 种 backbone、4 种对比方法、已见/未见域全覆盖
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，方法描述详细
- 价值: ⭐⭐⭐⭐⭐ 即插即用的域泛化方案对医学图像分割非常实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[CVPR 2026\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)

</div>

<!-- RELATED:END -->
