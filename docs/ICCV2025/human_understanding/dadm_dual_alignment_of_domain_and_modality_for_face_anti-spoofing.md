---
title: >-
  [论文解读] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing
description: >-
  [ICCV 2025][人体理解][人脸反欺骗] 提出 DADM 框架，通过互信息掩码（MIM）模块和域-模态双对齐优化策略，同时解决多模态人脸反欺骗中的域内模态不对齐和域间模态不对齐问题，在四种协议下取得 SOTA 性能。
tags:
  - ICCV 2025
  - 人体理解
  - 人脸反欺骗
  - 多模态融合
  - 域泛化
  - 互信息
  - 不变风险最小化
---

# DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing

**会议**: ICCV 2025  
**arXiv**: [2503.00429](https://arxiv.org/abs/2503.00429)  
**代码**: [GitHub](https://github.com/)  
**领域**: 人体理解  
**关键词**: 人脸反欺骗, 多模态融合, 域泛化, 互信息, 不变风险最小化

## 一句话总结

提出 DADM 框架，通过互信息掩码（MIM）模块和域-模态双对齐优化策略，同时解决多模态人脸反欺骗中的域内模态不对齐和域间模态不对齐问题，在四种协议下取得 SOTA 性能。

## 研究背景与动机

多模态人脸反欺骗（FAS）利用 RGB、深度（Depth）和红外（Infrared）三种模态的互补信息来检测欺骗攻击。然而，现有的多模态 FAS 方法面临两个核心的对齐难题：

**域内模态不对齐（Intra-domain modality misalignment）**：不同攻击类型下各模态的防御能力差异悬殊。例如，深度模态对 3D 面具攻击几乎无防御力，而 RGB 模态可能对纸质打印攻击更敏感。简单的融合策略无法适应这种动态变化，甚至可能让某些"有害"模态拖累整体检测性能。

**域间模态不对齐（Inter-domain modality misalignment）**：引入额外模态可能加剧域偏移（domain shift），因为每个模态在不同数据集/设备之间都可能产生独立的分布漂移。传统 ERM 优化范式将所有源域混合在一起学习，无法保证在每个子域上都获得最优分类超平面，导致模型依赖虚假相关性。

先前工作 MMDG 使用基于 Monte Carlo dropout 的不确定性估计来识别不可靠模态，但存在较大随机性。本文从互信息最大化和不变风险最小化两个理论视角出发，提出更优雅的解决方案。

## 方法详解

### 整体框架

DADM 基于冻结的预训练 ViT-B/16 构建，对 RGB、Depth、Infrared 三种模态分别提取特征。在每个 ViT 层的 MHSA 输出后插入三个 MIM 模块（分别处理 RGB-D、RGB-I、D-I 模态对），通过互信息最大化实现域内模态对齐。同时引入 CDC-Adapter 捕获细粒度局部特征，以及双对齐优化策略进行域间模态对齐。仅 MIM 和 CDC-Adapter 的参数是可训练的。

### 关键设计

1. **互信息掩码模块（Mutual Information Mask, MIM）**:

    - 功能：为每对模态动态生成注意力掩码，增强可靠模态区域、抑制不可靠模态区域
    - 核心思路：将两个模态特征拼接后经轻量级交互卷积和掩码生成网络，产出两个 sigmoid 激活的掩码 $\mathbf{m}_{m1}, \mathbf{m}_{m2}$，用于重新加权原始特征：
    $\mathbf{z}_{\text{aligned}\_m1} = \mathbf{m}_{m1} \mathbf{z}_{m1}$
      然后将对齐后的特征做全局平均池化得到 MI tokens $z_{\text{mi1}}, z_{\text{mi2}}$，通过简化版 MINE 估计器最大化互信息：
    $\mathcal{L}_{\text{mi}} = -\left[\mathbb{E}_{p(z_{\text{mi1}}, z_{\text{mi2}})}\left[\frac{z_{\text{mi1}} + z_{\text{mi2}}}{2}\right] - \log\left(\mathbb{E}_{p(z_{\text{mi1}})p(z_{\text{mi2}})}\left[e^{\frac{z_{\text{mi1}} + z_{\text{mi2}}}{2}}\right]\right)\right]$
    - 设计动机：直接利用掩码重加权后的 MI tokens 作为得分函数的特例，避免了传统 MINE 需要额外独立神经网络估计得分函数的额外负担。掩码值高表示信息可靠区域，低值表示冗余或负面信息区域。

2. **MI 引导的梯度调制（ReGrad）**:

    - 功能：根据 MI tokens 的强度自适应调整每个 MIM 模块的梯度方向
    - 核心思路：当两个模态梯度方向冲突（点积 < 0）时，MI 值较低的模态梯度需要被投影修正；当梯度方向一致时，MI 值较高的模态获得更大权重。全部通过四种条件分支实现：
    $\text{ReGrad}(\mathbf{g}_1, \mathbf{g}_2) = \begin{cases} \mathbf{g}_1 + \frac{\mathbf{g}_1 \cdot \mathbf{g}_2}{\|\mathbf{g}_1\|_2^2} \mathbf{g}_1 \cdot \text{mi}_2, & \text{if } \mathbf{g}_1 \cdot \mathbf{g}_2 < 0, \text{mi}_1 < \text{mi}_2 \\ \cdots & \end{cases}$
    - 设计动机：不同于 MMDG 中基于不确定性的 ReGrad，本文基于互信息强度来衡量模态可靠性，更加确定性和稳定。

3. **域-模态双对齐优化策略**:

    - 功能：同时对齐子域分类超平面和模态间角度边际
    - 核心思路：
        - **超平面对齐**：采用 PG-IRM（Projected Gradient IRM）优化，使全局最优超平面在每个子域上也局部最优，即 $\beta^* \in \arg\min_\beta R^e(\phi, \beta), \forall e \in \mathcal{E}$
        - **角度边际对齐**：约束不同域中相同类别的模态特征之间角度余弦一致：
       $$\mathcal{L}_{\text{angle}} = \sum_{e_1 \neq e_2} \sum_{i \neq j} \mathbb{I}(y_i=1) \cdot \left(\frac{\mathbf{z}_i^{e_1} \cdot \mathbf{z}_i^{e_2}}{\|\mathbf{z}_i^{e_1}\| \|\mathbf{z}_i^{e_2}\|} - \tau_l\right)^2 + \cdots$$
      其中 $\tau_l=1.0$（活体样本严格对齐），$\tau_s=0.85$（欺骗样本适度松弛）
    - 设计动机：单模态 DG 方法只对齐分类超平面，但在多模态场景中，任何模态的显著域偏移（角度偏差）都会严重影响整体性能，因此需要额外约束模态间角度一致性。

### 损失函数 / 训练策略

总损失为三项之和：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{ce} + \lambda_{\text{mi}} \mathcal{L}_{\text{mi}} + \lambda_{\text{angle}} \mathcal{L}_{\text{angle}}$$

其中 $\lambda_{\text{mi}} = 0.1$，$\lambda_{\text{angle}} = 0.3$。使用 PG-IRM 优化总损失。推理阶段使用所有子域超平面的均值进行预测。

## 实验关键数据

### 主实验

Protocol 1（固定模态，LOO 跨域）各方法平均 HTER(%)↓ / AUC(%)↑：

| 方法 | 类型 | 平均HTER(%)↓ | 平均AUC(%)↑ | vs MMDG提升 |
|------|------|-------------|------------|------------|
| SSDG | DG | 34.22 | 70.20 | - |
| SSAN | DG | 29.15 | 77.22 | - |
| SA-FAS | DG | 28.77 | 78.18 | - |
| VP-FAS | FM | 25.45 | 81.08 | - |
| MMDG | MM-DG | 19.25 | 87.96 | - |
| **DADM (Ours)** | **MM-DG** | **13.63** | **92.96** | **HTER↓5.62, AUC↑5.00** |

### 消融实验

各组件贡献（Protocol 1 平均）：

| 配置 | HTER(%)↓ | AUC(%)↑ | 说明 |
|------|---------|---------|------|
| ViT + CE | 31.14 | 74.81 | 基线 |
| + U-Adapter + SSP (MMDG) | 24.54 | 83.14 | 再现先前SOTA |
| + MIM + CDC-Adapter | 21.75 | 86.17 | MIM 替代 U-Adapter |
| + MI-Guided ReGrad | 17.17 | 88.14 | MI 引导的梯度比不确定性引导更好 |
| + PG-IRM | 16.54 | 90.27 | 引入不变风险最小化 |
| + DADM（双对齐） | 14.31 | 92.05 | 加入角度边际对齐 |
| + MI Loss（完整DADM） | 13.63 | 92.96 | 加入互信息最大化损失 |

互信息损失对比：本文简化 MI 估计器比 MINE（14.40/92.13）和 InfoNCE（15.80/91.47）效果更好（13.63/92.96），且无额外网络开销。

### 关键发现

- 在 Protocol 2（缺失模态）和 Protocol 3（灵活模态）场景下，DADM 同样显著优于所有方法，表明双对齐策略提升了模型对模态缺失的鲁棒性
- 在 Protocol 4（有限源域）中，DADM 以较大优势领先，特别是 PS→CW 场景从 MMDG 的 36.60% HTER 降至 20.40%
- CDC-Adapter 比普通卷积 Adapter 提升约 1% 的 HTER，验证了中心差分卷积对捕捉欺骗痕迹的有效性

## 亮点与洞察

1. **问题分解精准**：将多模态域泛化 FAS 中的对齐挑战清晰分解为域内模态对齐和域间模态对齐两个子问题，分别用信息论和优化理论工具解决
2. **轻量高效的 MI 估计**：巧妙利用已有的 MI tokens 作为得分函数的特例，避免额外引入评分网络
3. **角度边际对齐思想新颖**：不同于只对齐分类超平面的传统方法，显式约束模态间角度一致性是处理多模态域偏移的关键

## 局限与展望

- 目前仅在 RGB+Depth+Infrared 三模态场景下验证，对更多模态组合（如近红外+热成像）的扩展性有待探索
- MIM 模块在每层需要三个实例（三对模态组合），模态数增加时计算量呈二次增长
- 角度边际的超参 $\tau_l, \tau_s$ 是手动设定的，可能需要针对不同数据集调优

## 相关工作与启发

- PG-IRM 优化框架为多模态域泛化提供了统一的理论视角，可推广到其他多模态任务
- MIM 模块的互信息掩码思想可应用于任何需要动态模态融合的场景，如自动驾驶中的多传感器融合
- 角度边际对齐可与对比学习结合，增强跨域表征学习

## 评分

- 新颖性: ⭐⭐⭐⭐ 双对齐框架和简化MI估计器有创新，但整体是对已有模块的组合
- 实验充分度: ⭐⭐⭐⭐⭐ 四种协议覆盖固定/缺失/灵活/有限源域场景，消融非常全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整，但符号较多读起来有一定门槛
- 价值: ⭐⭐⭐⭐ 在多模态FAS领域取得显著进步，双对齐思路对其他多模态DG任务有参考价值

<!-- RELATED:START -->

## 相关论文

- [PA-FAS: Towards Interpretable and Generalizable Multimodal Face Anti-Spoofing via Path-Augmented Reinforcement Learning](../../AAAI2026/human_understanding/pa-fas_towards_interpretable_and_generalizable_multimodal_face_anti-spoofing_via.md)
- [Towards Unified Representation of Invariant-Specific Features in Missing Modality Face Anti-Spoofing](../../ECCV2024/human_understanding/towards_unified_representation_of_invariant-specific_features_in_missing_modalit.md)
- [TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](../../ECCV2024/human_understanding/tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [GestureHYDRA: Semantic Co-speech Gesture Synthesis via Hybrid Modality Diffusion Transformer and Cascaded-Synchronized Retrieval-Augmented Generation](gesturehydra_semantic_co-speech_gesture_synthesis_via_hybrid_modality_diffusion_.md)
- [IDFace: Face Template Protection for Efficient and Secure Identification](idface_face_template_protection_for_efficient_and_secure_identification.md)

<!-- RELATED:END -->
