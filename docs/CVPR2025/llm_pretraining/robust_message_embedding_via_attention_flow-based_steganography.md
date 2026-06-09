---
title: >-
  [论文解读] Robust Message Embedding via Attention Flow-Based Steganography
description: >-
  [CVPR 2025][预训练][消息嵌入] 本文提出RMSteg（Robust Message Steganography）框架，首次将Transformer注意力机制集成到归一化流网络中（AttnFlow），配合可逆QR码转换和可逆Token融合模块，实现了高质量、高容量且鲁棒的消息-图像隐写…
tags:
  - "CVPR 2025"
  - "预训练"
  - "消息嵌入"
  - "归一化流"
  - "注意力机制"
  - "QR码"
  - "鲁棒隐写"
---

# Robust Message Embedding via Attention Flow-Based Steganography

**会议**: CVPR 2025  
**arXiv**: [2405.16414](https://arxiv.org/abs/2405.16414)  
**代码**: 待确认  
**领域**: 图像隐写术  
**关键词**: 消息嵌入, 归一化流, 注意力机制, QR码, 鲁棒隐写

## 一句话总结

本文提出RMSteg（Robust Message Steganography）框架，首次将Transformer注意力机制集成到归一化流网络中（AttnFlow），配合可逆QR码转换和可逆Token融合模块，实现了高质量、高容量且鲁棒的消息-图像隐写，隐写图像即使经过打印-拍照等极端扭曲仍可准确解码。

## 研究背景与动机

**领域现状**：图像隐写术将秘密信息隐藏在宿主图像中，生成视觉上不可区分的隐写图像。基于深度学习的方法（特别是归一化流/可逆神经网络）已在隐写质量和容量上取得显著进展，广泛用于版权保护、信息溯源等场景。

**现有痛点**：(1) 质量与鲁棒性不可兼得——现有鲁棒隐写方法（如StegaStamp、ChartStamp）可以抵抗打印-拍照等物理扭曲，但隐写图像质量差（可见伪影）且嵌入容量低（仅几十bit）。(2) CNN骨干的局限——现有归一化流方法使用CNN（如DenseNet）作为仿射耦合块的骨干，缺乏通道内特征融合能力，在追求鲁棒性时容易产生明显伪影。(3) 消息嵌入要求完全正确——不同于图像-图像隐写（允许一定误差），消息嵌入要求解码结果100%正确，这大幅增加了难度。

**核心矛盾**：高鲁棒性要求将信息"更深"地嵌入到图像中（修改更显著），但高质量要求修改"尽可能小"——二者天然矛盾。

**本文目标** 如何在保持高隐写质量的同时实现抗打印-拍照等极端物理扭曲的鲁棒消息嵌入？

**切入角度**：发现图像的token化表示天然适合隐写任务——token级别的特征交互比像素级CNN更抽象、更鲁棒，据此将Transformer注意力引入归一化流。

**核心 idea**：将注意力机制引入归一化流（AttnFlow）实现高质量隐写 + 可逆QR码转换适应宿主图像 + token融合提升质量。

## 方法详解

### 整体框架

RMSteg流水线：(1) 将秘密消息编码为QR码图像 $I_q$。(2) 可逆QR码转换（IQRT）——根据宿主图像 $I_h$ 的特征变换QR码，降低其黑白对比度以减少隐写伪影。(3) 可逆Token融合（ITF）——对tokenized QR码执行可逆矩阵变换。(4) AttnFlow——基于注意力仿射耦合块的归一化流模型执行信息隐匿。(5) 训练时加入扭曲模拟模块。解码过程为逆向流程。

### 关键设计

1. **注意力仿射耦合块（Attention Affine Coupling Block, AACB）**：
    - 功能：替代传统CNN骨干实现更高质量的可逆隐写变换
    - 核心思路：在每个耦合块中，将宿主图像tokens $T_h$ 和QR码tokens $T_q$ 作为两个分支。$T_h$ 分支通过自注意力块 $\phi(T_q)$ 和交叉注意力块 $\mathcal{C}(T_q, T_h^{(0)})$ 逐步融入QR码信息。$T_q$ 分支通过仿射变换 $T_q^{(i)} = \eta(T_h^{(i)}) + T_q^{(i-1)} \odot \exp(\rho(T_h^{(i)}))$ 被修改。交叉注意力引入宿主图像初始token $T_h^{(0)}$ 作为key/value，使QR码信息自适应地分布到与宿主图像兼容的区域
    - 设计动机：CNN骨干仅做通道内卷积，缺乏patch间的全局交互；注意力机制允许token间长距离依赖，使信息嵌入更平滑更均匀

2. **可逆QR码转换（Invertible QR Code Transition, IQRT）**：
    - 功能：预处理QR码使其更容易被隐写而不产生伪影
    - 核心思路：用一个轻量级可逆网络根据宿主图像的特征变换QR码的外观（颜色、亮度）。变换后的QR码不再是黑白的，而是根据宿主图像调整了亮度和色调，但仍保持足够的模块对比度以被QR扫描器识别。解码时用逆变换恢复。使用ArtCoder中的高斯卷积核约束确保变换后的QR码可识别
    - 设计动机：黑白QR码与自然图像的分布差异大，直接嵌入会产生明显伪影。预先将QR码"伪装"得更接近自然图像可以大幅降低伪影

3. **可逆Token融合（Invertible Token Fusion, ITF）**：
    - 功能：通过可学习矩阵变换进一步优化QR码token的分布
    - 核心思路：受GLOW中可逆1×1卷积启发，引入一个可学习矩阵 $\mathcal{M} \in \mathbb{R}^{N \times N}$（用Cholesky分解初始化为正交矩阵），对QR码token做矩阵乘法 $T_q' = \mathcal{M} \cdot T_q$。解码时用逆矩阵恢复。与GLOW的通道混合不同，ITF实现的是patch间的特征交互
    - 设计动机：实验发现仅引入这一小矩阵即可显著提升隐写质量，因为它学到了最优的patch级信息重分布策略

## 实验关键数据

### 关键发现

- RMSteg的隐写图像在PSNR/SSIM上显著优于StegaStamp和ChartStamp（PSNR提升约5-8dB）
- 在打印-拍照极端扭曲下，消息解码准确率接近100%
- 嵌入容量远超bit级方法：通过QR码可嵌入约196字节信息（约1500 bit）
- 消融实验证明IQRT贡献最大（~3dB PSNR提升），AACB和ITF各贡献约1-2dB
- 对JPEG压缩、高斯噪声、亮度变化等数字扭曲也有良好鲁棒性
- AACB的token级表示相比CNN的像素级表示能更均匀地分配嵌入信息，减少局部伪影

## 亮点与洞察

- **首次将Transformer引入归一化流隐写**：AACB的设计是核心创新，token级表示天然适合鲁棒隐写
- **QR码转换思路巧妙**：不修改隐写网络而修改输入，降低了任务难度
- **ITF极简有效**：一个可学习矩阵即可显著提升质量，说明patch级信息重分布很重要
- **实用价值高**：抗打印-拍照意味着可应用于实物版权保护场景
- 将token化表示与归一化流的结合为其他需要可逆变换的任务提供了新思路

## 局限与展望

- QR码的纠错上限限制了可嵌入信息量
- 训练时扭曲模拟可能无法覆盖所有真实场景（如非平面打印、强透视变形）
- 推理速度受归一化流层数影响，可能不适合实时应用
- 未来可探索将注意力流扩展到视频隐写和3D打印防伪
- 当前方法在低分辨率图像上的效果有待验证
- 在更多类型的QR码（如微型QR码）上的泛化性有待确认
- 多尺度特征融合可能进一步提升隐写质量
- 结合图像增强技术提升弱扭曲场景下的鲁棒性值得探索

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Scaling Embedding Layers in Language Models](../../NeurIPS2025/llm_pretraining/scaling_embedding_layers_in_language_models.md)
- [\[ICML 2025\] Benign Overfitting in Token Selection of Attention Mechanism](../../ICML2025/llm_pretraining/benign_overfitting_in_token_selection_of_attention_mechanism.md)
- [\[CVPR 2025\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ICML 2025\] Towards Robust Influence Functions with Flat Validation Minima](../../ICML2025/llm_pretraining/towards_robust_influence_functions_with_flat_validation_minima.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)

</div>

<!-- RELATED:END -->
