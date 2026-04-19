# Degradation-Aware Feature Perturbation for All-in-One Image Restoration

**会议**: CVPR 2025  
**arXiv**: [2505.12630](https://arxiv.org/abs/2505.12630)  
**代码**: https://github.com/TxpHome/DFPIR  
**领域**: 图像修复  
**关键词**: 全合一修复, 退化感知扰动, 通道混洗, 注意力掩码, CLIP引导

## 一句话总结
提出 DFPIR，通过退化引导的特征扰动——通道级扰动（退化引导的通道混洗在高维空间中重组特征）和注意力级扰动（top-K 掩码交叉注意力选择性传递信息）——将特征空间对齐到统一的参数空间，在 3/5 任务全合一修复中平均 PSNR 超越 InstructIR +0.45/+1.09 dB。

## 研究背景与动机

**领域现状**：全合一图像修复用单一模型处理多种退化。现有方法通过提示学习（PromptIR）、语言引导（InstructIR）或混合专家（MoCE-IR）实现，但不同退化的特征空间差异大。

**现有痛点**：不同退化类型的特征分布差异大，统一模型的参数空间难以同时适配所有退化的特征空间。需要将不同退化的特征空间"对齐"到统一参数空间。

**核心矛盾**：特征空间多样性（不同退化差异大）vs 参数空间统一性（单模型共享参数），需要一种机制在两者之间建立桥接。

**切入角度**：引入退化感知的特征扰动——不改变特征的信息量，只改变特征的组织方式（通道顺序、注意力分配），使不同退化的特征以统一方式呈现给共享参数。

**核心 idea**：CLIP 编码退化类型→引导通道混洗（重组特征通道顺序）+ top-K 注意力掩码（选择性传递信息），实现退化感知的特征空间对齐。

## 方法详解

### 关键设计

1. **通道级扰动（退化引导通道混洗）**：先将特征扩展到高维空间，用退化类型的 CLIP 嵌入生成混洗矩阵，在高维空间中重组通道顺序后投影回原维度。不同退化用不同混洗方式组织特征

2. **注意力级扰动（Top-K 掩码）**：在 DGPB（退化引导扰动块）的交叉注意力中，仅保留注意力分数最高的 top-K 位置，抑制与当前退化无关的信息传递

3. **Restormer 骨干 + CLIP 退化提示**：DGPB 置于编码器和解码器之间，用 CLIP 编码退化类型文本作为引导

## 实验关键数据

| 设置 | 方法 | 平均PSNR↑ | 去雾↑ |
|------|------|---------|------|
| 3任务 | InstructIR | 32.43 | 30.22 |
| 3任务 | **DFPIR** | **32.88** | **31.87** |
| 5任务 | InstructIR | 29.55 | 27.10 |
| 5任务 | **DFPIR** | **30.64** | **31.64** |

5 任务去雾提升 +4.54 dB（31.64 vs 27.10），说明特征扰动对难退化类型帮助最大。

## 评分
- 新颖性: ⭐⭐⭐⭐ 通道混洗+注意力掩码的扰动思路新颖
- 实验充分度: ⭐⭐⭐⭐ 3/5 任务+消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐ 对全合一修复有实用提升

<!-- RELATED:START -->

## 相关论文

- [ClearAIR: A Human-Visual-Perception-Inspired All-in-One Image Restoration](../../AAAI2026/image_restoration/clearair_a_human-visual-perception-inspired_all-in-one_image_restoration.md)
- [Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [EAMamba: Efficient All-Around Vision State Space Model for Image Restoration](../../ICCV2025/image_restoration/eamamba_efficient_all-around_vision_state_space_model_for_image_restoration.md)
- [MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](../../NeurIPS2025/image_restoration/modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion](../../ICCV2025/image_restoration/decouple_to_reconstruct_high_quality_uhd_restoration_via_active_feature_disentan.md)

<!-- RELATED:END -->
