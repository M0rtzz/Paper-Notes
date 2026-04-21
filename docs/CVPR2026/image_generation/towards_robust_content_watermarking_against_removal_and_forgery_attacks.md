---
title: >-
  [论文解读] Towards Robust Content Watermarking Against Removal and Forgery Attacks
description: >-
  [CVPR 2026][图像生成][内容水印] 提出实例特定双侧检测水印方法 ISTS，通过根据图像语义动态选择水印注入时间和位置来抵抗去除攻击和伪造攻击，并设计双侧检测机制抵御反向潜在表示攻击，在三种去除攻击和三种伪造攻击的平均和最坏情况下均达到 SOTA 鲁棒性。
tags:
  - CVPR 2026
  - 图像生成
  - 内容水印
  - 扩散模型
  - 去除攻击
  - 伪造攻击
  - 实例特定水印
---

# Towards Robust Content Watermarking Against Removal and Forgery Attacks

**会议**: CVPR 2026  
**arXiv**: [2604.06662](https://arxiv.org/abs/2604.06662)  
**代码**: 无  
**领域**: 图像生成 / 数字水印  
**关键词**: 内容水印, 扩散模型, 去除攻击, 伪造攻击, 实例特定水印

## 一句话总结

提出实例特定双侧检测水印方法 ISTS，通过根据图像语义动态选择水印注入时间和位置来抵抗去除攻击和伪造攻击，并设计双侧检测机制抵御反向潜在表示攻击，在三种去除攻击和三种伪造攻击的平均和最坏情况下均达到 SOTA 鲁棒性。

## 研究背景与动机

1. **领域现状**：内容水印（如 Tree-Ring）在文本到图像扩散模型中被广泛研究，通过在生成过程中将身份标记嵌入潜在空间来验证图像来源。这些方法对常见图像变换（旋转、裁剪、压缩等）具有良好鲁棒性。
2. **现有痛点**：最近研究（Müller et al., Yang et al., Jain et al.）揭示现有水印在去除攻击和伪造攻击下极其脆弱——去除后检测 AUC 降至 0.1 以下（如 Gaussian-Shading），伪造后 AUC 接近 1.0（轻松伪造水印）。这意味着水印既可被抹除也可被伪造，严重威胁版权保护可靠性。
3. **核心矛盾**：现有方法使用静态、单一类型的水印模式（如 Tree-Ring 固定在傅里叶空间中心注入环形图案），这种一致性无意中泄露了水印的结构特征，使攻击者可以利用代理模型提取/复制水印。
4. **本文目标**：如何设计一种对去除攻击和伪造攻击都具有鲁棒性的水印方案？
5. **切入角度**：关键洞察是"静态水印 = 信息泄露"。如果每张图像的水印模式和注入参数都不同，攻击者就无法从单张或少量参考图像中提取通用水印特征。
6. **核心idea**：实例特定的动态水印（基于语义选择注入时间和位置）+ 双侧检测（同时检查正反潜在表示，封堵反向优化攻击路径）。

## 方法详解

### 整体框架

生成阶段：给定文本 prompt → 先生成非水印图像 → 用 CLIP 编码器提取语义特征 → 通过预训练的语义选择器映射到水印参数 $(t, l)$（注入时间步和频域坐标）→ 执行前 $T-t$ 步 DDIM → 在步 $t$ 的频域坐标 $l$ 处注入水印 → 完成剩余去噪得到水印图像。检测阶段：对可疑图像提取语义特征 → 恢复参数 $(t, l)$ → DDIM 反演到步 $t$ → 在坐标 $l$ 提取水印区域 → 双侧检测判定。

### 关键设计

1. **实例特定动态水印（动态模式+动态时间步）**:

    - 功能：使每张图像的水印模式和注入位置各不相同，阻断攻击者的特征提取
    - 核心思路：首先生成非水印图像并用 CLIP 编码为语义向量，然后通过 K-Means 聚类+分类器构建参数选择器 $f = \phi \circ h \circ g$，将语义特征映射到具体的注入时间步 $t$ 和频域坐标 $l$。由于水印注入对语义影响极小，水印/非水印图像的语义特征高度一致，检测时可从可疑图像准确恢复相同参数。动态模式打断了"从参考图像提取水印模式"的攻击路径，动态时间步使基于梯度的优化攻击无法追溯到正确的注入步。
    - 设计动机：静态水印下，攻击者用一张参考图像即可提取水印模式并伪造（Müller et al.），或通过多张水印图像取平均提取共同模式（Yang et al.）。动态化后，不同图像的水印特征相互抵消，平均失效。

2. **双侧检测（Two-Sided Detection）**:

    - 功能：封堵通过将潜在表示优化到水印模式反方向来去除水印的攻击路径
    - 核心思路：传统单侧检测度量 $d = \frac{1}{|M|} \sum_{i \in M} |W_i - \mathcal{F}(z_T)_i|$，仅检查水印匹配方向。攻击者将水印潜在表示优化到反方向使检测失败。双侧检测取正反两方向的最小值：$d = \min\{\frac{1}{|M|}\sum|W_i - \mathcal{F}(z_T)_i|, \frac{1}{|M|}\sum|W_i + \mathcal{F}(z_T)_i|\}$。对非水印图像（标准高斯分布），符号翻转不影响分布，检测度量不变；对水印图像，无论正反方向都能被捕获。
    - 设计动机：Müller et al. 的去除攻击正是利用了单侧检测的漏洞，将水印推到反方向。双侧检测以极低成本（仅多计算一次距离取 min）就堵住了这个攻击面。

3. **语义参数选择器训练**:

    - 功能：建立从图像语义到水印参数的确定性映射
    - 核心思路：(1) 从 prompt 集合生成非水印图像 → (2) CLIP 提取特征向量 → (3) K-Means 聚类到 $N$ 个类别 → (4) 通过预定义的取模映射 $\phi$ 将类别标签映射到 $(t, l)$ 参数 → (5) 训练分类器 $h$ 完成特征→类别的映射。最终选择器 $f = \phi \circ h \circ g$（$g$ 为 CLIP 编码器）。
    - 设计动机：需要一个确定性映射确保生成和检测时恢复相同参数。K-Means 聚类自然地将语义相近的图像分配到同一参数组，保证一致性。

### 损失函数 / 训练策略

- 参数选择器训练仅需一次 K-Means 聚类 + 简单分类器训练
- 水印注入和检测无需额外训练，直接利用预训练扩散模型
- 使用 Stable-Diffusion-2-1-base 模型，100 对图像评估对抗攻击，1000 对评估非对抗场景

## 实验关键数据

### 主实验（去除攻击鲁棒性）

| 水印方法 | 原始 AUC | Imp-Removal | Avg-Removal | 平均 AUC | 最坏 AUC |
|---------|----------|-------------|-------------|---------|---------|
| Tree-Ring | 1.000 | 0.267 | 0.527 | 0.589 | 0.267 |
| Gaussian-Shading | 1.000 | 0.000 | 0.371 | 0.457 | 0.000 |
| ROBIN | 1.000 | 0.082 | 0.742 | 0.595 | 0.082 |
| SEAL | 1.000 | 0.508 | 0.959 | 0.752 | 0.508 |
| **ISTS (Ours)** | 1.000 | **0.821** | **0.990** | **0.936** | **0.821** |

### 消融实验

| 配置 | Imp-Removal AUC | Imp-Forgery AUC | 说明 |
|------|----------------|-----------------|------|
| 完整 ISTS | **0.821** | **0.634** | 三个组件协同 |
| w/o 动态模式 | 0.71 左右 | 0.72 | 固定模式易被伪造 |
| w/o 动态时间步 | 降低 | 降低 | 梯度攻击可追溯 |
| w/o 双侧检测 | 0.71 左右 | 持平 | 反向潜在攻击有效 |

### 关键发现

- **Imp-Removal 是最强去除攻击**：几乎所有现有方法 AUC 降至 0.7 以下，ISTS 仍保持 0.821（提升 20%+）
- **伪造攻击下 ISTS 最优**：平均 AUC 0.686（越低越好），最坏情况 0.949，均优于所有基线
- **动态模式对抗伪造贡献最大**：去掉后 Imp-Forgery AUC 从 0.62 升至 0.72（更易被伪造）
- **双侧检测对抗去除贡献最大**：去掉后 Imp-Removal AUC 从 0.82 降至 ~0.71
- **图像质量无损**：PSNR、SSIM、LPIPS 与 ROBIN（最佳质量基线）相当，CLIP-Score 保持一致
- **常规图像变换鲁棒性**：平均 AUC 0.974（vs Tree-Ring 0.975），最坏情况 0.933（vs Tree-Ring 0.928），与最佳基线持平

## 亮点与洞察

- **"静态=泄露"的深刻洞察**：虽然黑盒攻击者名义上不知道水印算法，但静态水印的一致性模式实际上赋予了攻击者额外先验。这个观察揭示了安全设计中"实现细节可成为侧信道"的通用原理。
- **双侧检测的极简优雅**：仅需多算一次距离取 min（几乎零额外开销），就封堵了反向优化攻击路径。这种"对称化检测度量"的思路对其他基于距离的安全检测方案也有借鉴价值。
- **语义一致性保证检测可靠性**：利用水印注入对图像语义影响极小这一物理特性，确保从水印图像中恢复的参数与生成时一致。这是将对抗鲁棒性与功能正确性解耦的巧妙设计。

## 局限与展望

- 需要为每张图像先生成一张非水印版本来提取语义特征，使生成成本翻倍
- 参数选择器依赖 CLIP 语义一致性假设，极端图像编辑后可能打破这一假设
- 仅在 Stable-Diffusion-2-1-base 上验证，未测试 SDXL、FLUX 等更新模型
- K-Means 聚类数 $N$ 和参数映射 $\phi$ 的选择缺乏理论指导
- 评估使用 100 对图像，样本量偏小，统计显著性可能受限
- 可探索自适应动态策略（如根据攻击检测信号调整后续水印参数）

## 相关工作与启发

- **vs Tree-Ring**: Tree-Ring 在频域中心注入固定环形图案，静态模式在三种去除攻击下 AUC 降至 0.267-0.527；ISTS 动态化后保持 0.821-0.990
- **vs SEAL**: SEAL 使用 SimHash 控制去噪随机性，对伪造有一定抵抗（AUC 0.703），但对图像旋转、模糊、裁剪等常规变换极其脆弱（最坏 AUC 0.523）；ISTS 兼顾对抗攻击和常规鲁棒性
- **vs RingID**: RingID 常规鲁棒性极佳（最坏 AUC 0.953），但伪造攻击下几乎完全失效（Imp-Forgery AUC=1.0）；ISTS 在两方面均有保障

## 评分

- 新颖性: ⭐⭐⭐⭐ 实例特定水印+双侧检测的组合有效解决了现有水印的根本脆弱性，思路清晰且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 覆盖三种去除+三种伪造攻击+六种图像变换，平均/最坏情况分析全面；但样本量偏小且模型单一
- 写作质量: ⭐⭐⭐⭐ 问题定义和方法动机阐述清晰，算法伪代码规范，威胁模型严谨
- 价值: ⭐⭐⭐⭐ 首次系统性地解决内容水印的去除+伪造双重威胁，对生成式 AI 的版权保护有实际意义

<!-- RELATED:START -->

## 相关论文

- [Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)
- [Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusionbased_image_man.md)

<!-- RELATED:END -->
