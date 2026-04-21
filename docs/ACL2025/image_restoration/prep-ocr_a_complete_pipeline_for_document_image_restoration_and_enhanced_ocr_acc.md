---
title: >-
  [论文解读] PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy
description: >-
  [ACL 2025][图像恢复][文档图像修复] 提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。
tags:
  - ACL 2025
  - 图像恢复
  - 文档图像修复
  - OCR
  - 合成数据
  - 历史文档数字化
  - ByT5
---

# PreP-OCR: A Complete Pipeline for Document Image Restoration and Enhanced OCR Accuracy

**会议**: ACL 2025  
**arXiv**: [2505.20429](https://arxiv.org/abs/2505.20429)  
**代码**: https://github.com/NikoGuan/PreP-OCR (有)  
**领域**: 文档图像处理 / OCR  
**关键词**: 文档图像修复, OCR后纠错, 合成数据, 历史文档数字化, ByT5

## 一句话总结
提出 PreP-OCR 两阶段流水线：先用合成退化数据训练的 ResShift 模型修复历史文档图像（多方向 patch 提取+中值融合），再用 ByT5 做 OCR 后语义纠错，在 13,831 页真实历史文档上降低 CER 63.9-70.3%。

## 研究背景与动机

**领域现状**：大量历史文档在数字化过程中面临退化问题（模糊、噪声、墨渍、老化），严重影响 OCR 文本提取质量。现有方法主要分为图像预处理（对比度增强、去噪）和 OCR 后纠错两条技术路线，但很少将两者系统性地结合。

**现有痛点**：(1) 图像修复可以改善视觉质量但无法消除所有 OCR 错误——形态歧义依然存在；(2) OCR 后纠错依赖于可预测的错误模式，但修复不充分时错误类型多样且不可预测；(3) LLM-based OCR（如 GPT-4o）虽然原始准确率高，但存在输出不稳定和"幻觉"问题。

**核心矛盾**：真实退化文档的配对数据（退化图+干净图）几乎不可能获取，无法直接训练监督式修复模型。

**本文目标**：如何用纯合成数据训练出能泛化到真实历史文档的图像修复模型？如何将图像修复与语义纠错协同工作？

**切入角度**：用程序化方式模拟文档退化（随机顺序施加噪声/模糊/形态操作），训练图像修复模型；再用 OCR 错误分布合成训练对训练 ByT5 纠错模型。

**核心 idea**：合成数据驱动的两阶段 pipeline——图像修复解决像素级歧义，文本纠错解决语义级错误。

## 方法详解

### 整体框架
输入退化历史文档图像 → 图像修复（ResShift + 多方向 patch 提取 + 中值融合）→ OCR（Tesseract）→ 后纠错（ByT5）→ 输出清洁文本。两阶段分别处理视觉歧义和语义错误。

### 关键设计

1. **合成退化数据生成**:

    - 功能：从纯文本出发，渲染为干净文档图像，再施加多种退化操作生成配对训练数据
    - 核心思路：先用多种字体渲染文本（随机缩进、字符偏移、旋转、弯曲、行距变化），生成干净 base image。然后**随机顺序**施加退化操作：随机噪声、分辨率降低、高斯模糊、黑/白斑块、线条、纹理叠加、墨渍、形态学膨胀/腐蚀。预定义 4 个噪声级别，10% 做 Otsu 二值化
    - 设计动机：**随机操作顺序**是关键——相同的操作以不同顺序产生不同退化效果，极大增加数据多样性。共生成 10 万对图像用于训练

2. **多方向 Patch 提取与中值融合**:

    - 功能：从 4 个方向扫描图像提取 256×256 patch，修复后取中值融合
    - 核心思路：从左上→右下、右上→左下、左下→右上、右下→左上 4 个方向各扫描一遍，stride=128。每个 patch 修复后丢弃外围 64 像素边框，仅保留中央 128×128 区域。最终每个像素由 4 次独立预测的中值合成：$\hat{I}[r,c,\chi] = \text{median}(\hat{I}_k[r,c,\chi] | k \in \{1,2,3,4\})$
    - 设计动机：patch 边界处修复质量差（PSNR 低于中心区域），多方向扫描使同一区域在不同 patch 中处于不同位置，中值融合抑制异常值/伪影

3. **ByT5 OCR 后纠错**:

    - 功能：训练 ByT5-base 模型将 OCR 错误文本纠正为正确文本
    - 核心思路：从 ICDAR 2017 提取 OCR 错误分布（如 "m" → {"n": 0.001, "rn": 0.002, ...}），按分布向干净文本注入错误，生成合成训练对。ByT5 做 byte-level 序列到序列翻译
    - 设计动机：byte-level tokenization 天然适合处理历史文档中的罕见字符和变音符号。合成数据方法避免了对大量真实 OCR 错误标注数据的依赖

### 损失函数 / 训练策略
- 图像修复：ResShift 在 90k 合成图像对上训练，256×256 patch
- 后纠错：ByT5-base 在约 89 万合成文本对上微调，每条最长 512 字符
- 推理时：Multi-Median-64 配置，每页处理约 45 秒（RTX 4090）

## 实验关键数据

### 主实验（英语，Tesseract OCR）

| Pipeline | CER%(全部) | CER%(去异常) | 相对Raw降幅 |
|----------|-----------|-------------|------------|
| Raw（原始图像） | 5.91% | 5.87% | - |
| Pre（图像修复） | 2.81% | 1.99% | -66.1% |
| **PreP（修复+纠错）** | **2.00%** | **1.30%** | **-77.9%** |

### 跨语言泛化（Tesseract，英语训练的修复模型）

| 语言 | Raw CER | Pre CER | PreP CER | 总降幅 |
|------|---------|---------|----------|--------|
| 英语 | 5.91% | 2.81% | 2.00% | -66.2% |
| 法语 | 5.16% | 2.89% | 1.53% | -70.3% |
| 西班牙语 | 7.12% | 3.42% | 2.57% | -63.9% |

### 消融实验（图像修复配置，英语 CER%）

| 配置 | ResShift | DiffIR | MIMO-UNet+ |
|------|----------|--------|------------|
| Single-0（无border裁剪） | 4.43% | 3.77% | 4.65% |
| Single-64（裁边64px） | 3.17% | 3.12% | 3.70% |
| Multi-Median-64 | **2.81%** | **2.94%** | **3.65%** |

### 关键发现
- **两阶段互补**：图像修复主要解决字符形态歧义（CER 5.91→2.81），后纠错处理系统性 OCR 错误（2.81→2.00），各有不可替代的作用
- **GPT-4o 反而不适合此 pipeline**：GPT-4o 的"幻觉"（生成看似合理但错误的文本）会被后纠错模型误判为正确，导致纠错后 CER 反而上升。传统 Tesseract 的字符级错误更可预测更易纠正
- **跨语言零迁移有效**：英语训练的 ResShift 模型直接用于法语/西班牙语，CER 降幅达 44-52%，因为拉丁字母结构相似
- **Patch 边界效应显著**：丢弃外围 64px 使 CER 从 4.43% 降到 3.17%；多方向扫描+中值融合进一步降到 2.81%

## 亮点与洞察
- **"合成数据训练→真实数据泛化"的成功案例**：100% 合成数据训练的模型在 13,831 页真实历史文档上表现优异，随机操作顺序是保证泛化的关键
- **多方向 patch 融合的中值操作**：简单优雅地解决了 patch 边界伪影问题，类似多视角共识机制
- **"传统 OCR 优于 LLM-OCR"的反直觉发现**：在 pipeline 场景下，Tesseract 的可预测错误模式使其与后纠错模型配合更好，而 GPT-4o 的幻觉难以纠正

## 局限与展望
- 仅验证了拉丁字母系统（英/法/西），非拉丁文字（阿拉伯语、中文等）未测试
- 合成退化操作的参数范围基于经验设定，可能无法覆盖所有真实退化类型
- 后纠错模型的错误分布来自 ICDAR 2017，可能不完全匹配 LLM-based OCR 的错误模式
- 推理速度较慢（每页 45 秒），大规模数字化项目需要考虑效率
- 修复模型可能处理不好极端非常规字体

## 相关工作与启发
- **vs 纯后纠错方法**: 纯后纠错缺少视觉信息，对严重退化文档效果有限。PreP-OCR 先修复图像提升 OCR 输入质量，再做语义纠错
- **vs 扩散模型修复**: DiffIR 在 patch 级 AMP 最高（25.64 dB），但全页 OCR CER 不如 ResShift，说明 patch 级指标不完全对应任务级性能
- **vs GPT-4o OCR**: GPT-4o 原始 CER 最低（2.34%），但不适合 pipeline 组合，因为幻觉难以被下游纠错

## 评分
- 新颖性: ⭐⭐⭐ 各组件（图像修复、合成数据、后纠错）都有先例，创新在于系统性整合和多方向融合策略
- 实验充分度: ⭐⭐⭐⭐⭐ 13,831 页真实数据，3 种语言，6 种修复模型，3 种 OCR 系统，消融全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，问题形式化规范
- 价值: ⭐⭐⭐⭐ 对历史文档数字化有直接实用意义，完整开源

<!-- RELATED:START -->

## 相关论文

- [Beyond the Ground Truth: Enhanced Supervision for Image Restoration](../../CVPR2026/image_restoration/beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [ShiftLUT: Spatial Shift Enhanced Look-Up Tables for Efficient Image Restoration](../../CVPR2026/image_restoration/shiftlut_spatial_shift_enhanced_look-up_tables_for_efficient_image_restoration.md)
- [Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](../../ECCV2024/image_restoration/contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [DarkIR: Robust Low-Light Image Restoration](../../CVPR2025/image_restoration/darkir_robust_low-light_image_restoration.md)
- [UniRes: Universal Image Restoration for Complex Degradations](../../ICCV2025/image_restoration/unires_universal_image_restoration_for_complex_degradations.md)

<!-- RELATED:END -->
