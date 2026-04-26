---
title: >-
  [论文解读] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection
description: >-
  [CVPR 2026][图像分割][RGB-T显著性检测] 提出 RSONet 两阶段 RGB-T 显著性检测框架：先通过三支并行编码器-解码器生成区域引导图并基于相似度选择主导模态，再通过选择性优化模块融合双模态特征，在 VT5000/VT1000/VT821 上 MAE 达 0.020/0.014/0.021，超越 27 个 SOTA 方法。
tags:
  - CVPR 2026
  - 图像分割
  - RGB-T显著性检测
  - 模态选择
  - 区域引导
  - 选择性优化
  - Transformer
---

# RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection

**会议**: CVPR 2026  
**arXiv**: [2603.12685](https://arxiv.org/abs/2603.12685)  
**代码**: 无  
**领域**: 目标检测 / RGB-T 显著性检测  
**关键词**: RGB-T显著性检测, 模态选择, 区域引导, 选择性优化, SwinTransformer  

## 一句话总结

提出 RSONet 两阶段 RGB-T 显著性检测框架：先通过三支并行编码器-解码器生成区域引导图并基于相似度选择主导模态，再通过选择性优化模块融合双模态特征，在 VT5000/VT1000/VT821 上 MAE 达 0.020/0.014/0.021，超越 27 个 SOTA 方法。

## 研究背景与动机

**领域现状**：显著性目标检测（SOD）旨在像素级别识别场景中最吸引注意力的物体。随着深度学习发展，RGB-T SOD 利用热红外图像提供的温度信息弥补 RGB 在复杂场景下的不足，成为多模态显著性检测的活跃方向。

**现有痛点**：(1) RGB 图像在复杂背景/低对比度/暗光场景下检测困难；(2) 热红外图像虽不受光照影响，但可能因环境温度、材料属性等因素导致目标与背景不可区分；(3) 现有 RGB-T 融合方法（加法/乘法/拼接/注意力）隐式假设两模态同等重要，当信息质量差异大时会引入大量噪声。

**核心矛盾**：两个模态中显著区域分布不一致——一个模态可能包含准确的目标信息而另一个被噪声主导，等权融合会相互拉低质量。

**本文目标** 自适应判断哪个模态更可靠，让可靠模态主导融合过程，避免低质量模态的噪声干扰。

**切入角度**：先用一个"区域引导阶段"对 RGB、Thermal 和 RGB+T 分别预测引导图并比较相似度来选择主导模态，再在"显著性生成阶段"让主导模态引导融合。

**核心 idea**：在融合之前先做模态质量判断，让好的模态带着差的模态走，而非无差别混合。

## 方法详解

### 整体框架

两阶段设计。**区域引导阶段**：RGB/Thermal/RGB+T 三个并行的编码器-解码器分支（共享 SwinTransformer backbone），各生成引导图 $G^R$、$G^T$、$G^{RT}$，计算 $G^R$ 和 $G^T$ 分别与 $G^{RT}$ 的相似度来选择主导模态。**显著性生成阶段**：选择性优化（SO）模块根据相似度结果融合双模态特征，低层特征经 DDE 模块增强细节，高层特征经 MIS 模块挖掘位置线索，跨层连接生成最终显著图。

### 关键设计

1. **上下文交互（CI）模块 + 空间感知融合（SF）模块 + 相似度计算**
    - CI 模块采用层自适应卷积核策略：低层特征用 1×1/3×3/5×5/7×7 四分支并行卷积捕获多尺度上下文，中层去掉 7×7 分支，高层只保留 1×1/3×3——因为高层特征分辨率低，大卷积核反而引入背景噪声
    - 各分支之间有残差连接（前一分支输出加到当前分支输入），打破不同尺度特征的隔阂
    - SF 模块通过全局 max pooling + 1×1 conv + sigmoid 生成空间权重，对 CI 输出做乘加优化，逐层自上向下融合
    - 相似度计算：对三张引导图计算像素均值 $M^R$、$M^T$、$M^{RT}$，比较 $|M^R - M^{RT}|$ vs $|M^T - M^{RT}|$，差异更小的模态为主导模态
2. **选择性优化（SO）模块**
    - 双模态特征先与引导图 $G^{RT}$ 做乘加增强，初步抑制背景区域
    - 各做通道注意力（1×1 conv → GAP → sigmoid）进一步优化通道响应
    - 主导模态的空间注意力施加到非主导模态特征上完成跨模态优化，最终两路求和得融合输出
    - 根据主导模态不同有两种对称的融合路径（R→T 或 T→R）
3. **DDE（密集细节增强）+ MIS（互交互语义）**
    - DDE 用 4 分支空洞卷积（d=1,3,5,7）做密集连接（每分支输出加到后续所有分支输入），再接 4 个 VSS（Visual State Space）块捕获空间关系，处理低层特征保留边缘细节
    - MIS 用 3 主分支 × 3 子分支（d=1,2,3）的互交互结构处理高层特征：第一子分支输出加到其他两子分支输入，实现多尺度感受野交互，最终通道注意力融合

### 损失函数 / 训练策略

BCE + boundary IoU + F-measure 三项损失联合监督 5 个显著图（深监督）。SwinTransformer backbone 使用 ImageNet 预训练权重，RMSprop ($lr=1 \times 10^{-4}$)，输入分辨率 384×384，单卡 RTX 4080 训练。

## 实验关键数据

### 主实验

| 数据集 | MAE↓ | $F_\beta$↑ | $E_\xi$↑ | $S_\alpha$↑ |
|--------|------|----------|----------|------------|
| VT5000 | **0.020** | **0.910** | **0.926** | **0.963** |
| VT1000 | **0.014** | 0.923 | 0.946 | 0.972 |
| VT821 | 0.021 | 0.883 | 0.921 | 0.946 |

vs PATNet (KBS24)：VT5000 $F_\beta$ +3.4%，$E_\xi$ +1.2%  
vs ContriNet (TPAMI25)：VT5000 $F_\beta$ +3.6%，$S_\alpha$ +2.4%  
速度：~8.8 FPS（101.3M 参数），远慢于 CGFNet 52.3 FPS。

### 消融实验

| 变体 | VT5000 MAE↓ | VT5000 $F_\beta$↑ |
|------|-------------|-------------------|
| 完整 RSONet | 0.0197 | 0.9071 |
| SO 模块 → 简单加法 | 0.0208 | 0.8952 |
| SO 模块 → 拼接 | 0.0217 | 0.8857 |
| 去掉相似度引导（固定融合方向） | 0.0215 | 0.8896 |
| 去掉 DDE + MIS | 0.0217 | 0.8834 |
| SwinTransformer → ResNet50 | — | 0.8146 |

### 关键发现

- 相似度引导的模态选择贡献显著——固定融合方向 MAE 升 9.1%
- SO 模块优于所有简单融合策略（加法/乘法/拼接/CA）
- DDE 和 MIS 互补——同时去掉 MAE 升 10.2%，单独去掉效果也下降
- SwinTransformer 远优于 ResNet 系列，$F_\beta$ 差距高达 9pp

## 亮点与洞察

- 自适应模态选择思路新颖——根据每张图片的实际情况选择主导模态而非等权融合，对多模态融合任务有通用启发
- 层自适应卷积核设计合理——低层大感受野 + 高层小感受野适配特征分辨率特性
- 27 个对比方法的全面评估覆盖了 2021-2025 年的 RGB-T SOD 工作

## 局限与展望

- 8.8 FPS 速度过慢——三分支并行编码器和密集空洞卷积带来巨大计算开销，难以实时应用
- 相似度计算过于简单（全图像素值求和做标量比较），无法捕获空间分布差异——局部区域一个模态好另一个差的场景处理不了
- 极小/细长目标和双模态同时退化时可能失效
- 引导图质量本身依赖编码器-解码器的预测能力，在困难样本上可能产生错误引导

## 相关工作与启发

- **SAMSOD (Liu et al.)**：SAM-based RGB-T SOD，通过梯度去冲突处理模态不平衡，VT5000 MAE 0.021 vs 本文 0.020
- **Samba (CVPR25)**：纯 Mamba 框架做显著性检测，VT5000 $F_\beta$ 0.894 vs 本文 0.910
- **ContriNet (TPAMI25)**：三流分治汇流设计，VT5000 $F_\beta$ 0.878 vs 本文 0.910
- **模态选择策略可推广**到任何多模态融合任务（RGB-D/RGB-Event/多光谱等），核心思想是"先评估再融合"
- VSS 块在低层特征细节增强中表现好，值得在其他 dense prediction 任务中尝试

## 评分

- 新颖性: ⭐⭐⭐⭐ 区域引导模态选择有新意，但整体仍是 encoder-decoder + 注意力范式
- 实验充分度: ⭐⭐⭐⭐⭐ 27 个对比方法、3 个数据集、4 个指标、多维度消融
- 写作质量: ⭐⭐⭐ 方法描述详细但模块多、符号多，阅读门槛较高
- 价值: ⭐⭐⭐⭐ 在 RGB-T SOD 子领域有实用价值，模态选择思路可泛化

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportion-aware_dynamic_adaptive_salient_object_detection_network_.md)
- [\[CVPR 2026\] MPM: Mutual Pair Merging for Efficient Vision Transformers](mpm_mutual_pair_merging_for_efficient_vision_transformers.md)
- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] FCL-COD: Weakly Supervised Camouflaged Object Detection with Frequency-aware and Contrastive Learning](fcl-cod_weakly_supervised_camouflaged_object_detection_with_frequency-aware_and_.md)

<!-- RELATED:END -->
