---
title: >-
  [论文解读] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation
description: >-
  [NeurIPS 2025][医学图像][3D医学图像分割] 提出Mamba-HoME架构，将层次化Soft MoE（HoME）与Mamba SSM结合，通过两级token路由机制实现局部-全局特征建模，在CT/MRI/US三种模态的3D医学图像分割任务上超越现有SOTA方法，同时保持线性计算复杂度。
tags:
  - NeurIPS 2025
  - 医学图像
  - 3D医学图像分割
  - Mamba
  - Mixture-of-Experts
  - 状态空间模型
  - 层次化路由
---

# Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation

**会议**: NeurIPS 2025  
**arXiv**: [2507.06363](https://arxiv.org/abs/2507.06363)  
**代码**: [github.com/gmum/MambaHoME](https://github.com/gmum/MambaHoME)  
**领域**: 医学图像  
**关键词**: 3D医学图像分割, Mamba, Mixture-of-Experts, 状态空间模型, 层次化路由

## 一句话总结
提出Mamba-HoME架构，将层次化Soft MoE（HoME）与Mamba SSM结合，通过两级token路由机制实现局部-全局特征建模，在CT/MRI/US三种模态的3D医学图像分割任务上超越现有SOTA方法，同时保持线性计算复杂度。

## 研究背景与动机

3D医学图像分割是计算机辅助诊断和介入治疗的核心任务，需要处理CT、MRI、超声等多种模态的数据。医学影像数据天然具有层次化结构：局部病变（如肿瘤）嵌套在更大的解剖结构（如器官）中，而器官又遵循全局的解剖排列规律。这种局部到全局的空间层次对分割性能至关重要。

然而现有方法存在明显不足：CNN虽然有线性复杂度但感受野有限，难以捕捉全局空间模式；Vision Transformer通过全局注意力机制建模长程依赖，但二次方复杂度使其在高分辨率3D数据上代价高昂。近年来Mamba（选择性状态空间模型）以线性复杂度捕获长程依赖，成为有前景的替代方案，但SSM并非天然具备自适应处理多样局部模式的能力。另一方面，Mixture-of-Experts（MoE）通过动态路由实现局部模式的高效管理，但将SSM的全局效率与MoE的局部自适应性相结合的工作几乎没有被探索过。

本文的核心切入点是：设计一种层次化的两级Soft MoE（HoME），第一级在局部组内路由token到专家提取局部特征，第二级在全局聚合跨组信息，将这一层次设计无缝嵌入Mamba架构中，实现对3D医学图像中局部-全局空间层次的高效建模。

## 方法详解

### 整体框架
Mamba-HoME采用U型编码器-解码器架构。编码器由stem层和多个级联的Mamba-HoME Block（Mamba-HoMEB）组成，解码器通过上采样和跳跃连接恢复分辨率。核心创新在于Mamba-HoMEB，它依次包含：门控空间卷积（GSC）→Mamba层→HoME层，并使用残差连接。

### 关键设计

1. **Hierarchical Soft Mixture-of-Experts (HoME)**: 

    - 功能：两级token路由层，先局部后全局处理特征
    - 核心思路：首先将输入序列划分为 $G_i$ 个组，每组 $K_i$ 个token。在**分组slot分配**阶段，每个组的token通过与可学习slot嵌入 $E_{\text{slots}}^{(i)} \in \mathbb{R}^{M_i \times d}$ 的点积和softmax归一化，软分配到专家slot上：$A_{b,g,k,m} = \frac{\exp(S_{b,g,k,m})}{\sum_{m'}\exp(S_{b,g,k,m'})}$。第一级 $E_{1,i}$ 个局部专家（FFN）处理各组内的slot表示，通过路由权重加权聚合；第二级 $E_{2,i}$ 个全局专家处理所有组拼接后的slot，实现跨组信息融合。
    - 设计动机：全局SMoE在长序列上计算成本为 $\mathcal{O}(N_i \cdot M_i)$，分组路由将峰值内存降低并保留局部性；两级设计在不显著增加计算量的前提下实现局部特征提取+全局上下文整合。

2. **Mamba-HoMEB（Mamba-HoME Block）**: 

    - 功能：将GSC、Mamba、HoME三个模块顺序集成
    - 处理流程：$x_i'^l = f_{\text{GSC}}(x_i^l)$，$\bar{x}_i^l = f_{\text{Mamba}}(f_{\text{Norm}}(x_i'^l)) + x_i'^l$，$x_i^{l+1} = f_{\text{HoME}}(f_{\text{Norm}}(\bar{x}_i^l)) + \bar{x}_i^l$。GSC模块先提取局部空间先验，展平为1D序列后送入Mamba层做长程建模，最后HoME层做层次化专家细化。
    - 设计动机：GSC捕捉局部空间先验、Mamba高效处理长序列、HoME提供层次化专家自适应——三者互补形成完整的局部到全局建模链条。

3. **Dynamic Tanh (DyT) 归一化**: 

    - 功能：替换Layer Normalization的高效归一化方法
    - 核心公式：$f_{\text{DyT}}(x) = w \cdot \tanh(\alpha \cdot x) + b$，其中 $w, b \in \mathbb{R}^d$ 为可学习参数，$\alpha$ 为共享标量
    - 设计动机：tanh的有界性天然稳定梯度，避免了LN中均值/方差计算的开销，在SSM架构中实现约6%的训练和推理加速，性能不降。

### 损失函数 / 训练策略
使用 $\mathcal{L}_{\text{DiceCE}}$ 损失函数（Dice + Cross Entropy），AdamW优化器，初始学习率1e-4配合余弦退火调度。支持两种配置：从头训练和在大规模CT/MRI数据集上有监督预训练后微调。预训练使用AbdomenAtlas 1.1（CT）和TotalSegmentator MRI数据集。

## 实验关键数据

### 主实验

**PANORAMA + In-house CT（胰腺分割）**

| 方法 | PDAC DSC(%) | 胰腺 DSC(%) | mDSC(%) | mHD95(mm) | GPU(G) |
|------|-------------|-------------|---------|-----------|--------|
| SegMamba | 49.7 | 88.5 | 76.0 | 8.6 | 10.1 |
| uC 3DU-Net | 52.0 | 88.2 | 76.6 | 8.4 | 13.6 |
| SuPreM* | 51.7 | 88.3 | 76.6 | 4.7 | 17.1 |
| **Mamba-HoME** | **54.8** | 88.3 | **77.5** | **4.8** | **11.1** |
| **Mamba-HoME*** | **56.7** | **88.5** | **78.2** | **4.3** | **11.1** |

**FeTA 2022（胎儿脑MRI）** 5折交叉验证

| 方法 | mDSC(%) | mHD95(mm) |
|------|---------|-----------|
| SegMamba | 85.9 | 3.5 |
| Hermes | 86.5 | 4.0 |
| **Mamba-HoME*** | **87.7** | **2.0** |

**MVSeg（3D超声）**

| 方法 | mDSC(%) | mHD95(mm) |
|------|---------|-----------|
| VSmTrans | 84.4 | 6.2 |
| Swin UNETR | 84.4 | 4.8 |
| **Mamba-HoME*** | **85.0** | **4.1** |

### 消融实验

| 配置 | PANO mDSC | AMOS mDSC | FeTA mDSC | 说明 |
|------|-----------|-----------|-----------|------|
| E1=[4,8,12,16], E2=[8,16,24,32] | 77.5 | 86.3 | 87.5 | 最优专家数配置 |
| E1=[8,16,24,48], E2=[8,16,24,48] | 76.3 | 86.2 | 87.2 | 更多专家反而不如 |
| K=[2048,1024,512,256] | 77.5 | 86.3 | 87.5 | 最优组大小 |
| K=[1024,512,256,128] | 77.2 | 86.1 | 87.4 | 组太小性能降低 |
| Slots S=4 | 77.5 | 86.3 | 87.5 | 最优slot数 |
| Slots S=1 | 76.2 | 85.9 | 87.3 | slot太少信息不足 |
| Layer Norm | 77.4 | 86.2 | 87.5 | 传统归一化 |
| Dynamic Tanh | 77.5 | 86.3 | 87.4 | 性能相当但快6% |

### 关键发现
- Mamba-HoME在CT/MRI/US三种模态上均达到SOTA，证明了跨模态泛化能力
- 专家数量并非越多越好，E1=[4,8,12,16]的配置在参数量最少的情况下取得最佳性能
- 预训练版本在所有数据集上进一步提升，尤其是在PDAC这种困难的小目标分割任务上提升显著（+1.9% DSC）
- DyT归一化在保持性能的同时加速训练/推理约6%

## 亮点与洞察
- 首次将Mamba SSM和层次化Soft MoE结合，创建了一种全新的3D医学图像分割范式。两级路由的设计理念——先局部专家处理后全局专家融合——与医学图像天然的层次化解剖结构高度匹配。
- GPU内存使用仅11.1G（相当于SegMamba的水平），即使参数量达170M也能有效控制推理开存，这得益于分组路由和线性复杂度的Mamba层。

## 局限与展望
- 参数量为170.1M，在所有对比方法中最大，推理速度比baseline慢约30%
- DyT在SSM上的加速效果相对有限（约6%），更激进的加速策略值得探索
- 预训练仅限CT和MRI两种模态，未包含超声数据的预训练

## 相关工作与启发
- **vs SegMamba**: Mamba-HoME在SegMamba基础上增加了HoME层和GSC模块，PANORAMA数据集上mDSC从76.0提升到77.5，证明层次化MoE对Mamba的增益
- **vs Swin UNETR**: 在多个数据集上持续超越这一经典Transformer方法，且GPU占用更低
- **vs SuPreM**: 即便SuPreM使用了大规模预训练，Mamba-HoME从头训练的性能也与之相当或更优

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次结合Mamba和层次化MoE，局部-全局两级路由设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集覆盖CT/MRI/US三种模态，消融实验非常全面
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，结构完整
- 价值: ⭐⭐⭐⭐ 为3D医学图像分割提供了一个高效且强大的新baseline

<!-- RELATED:START -->

## 相关论文

- [\[NeurIPS 2025\] DyG-Mamba: Continuous State Space Modeling on Dynamic Graphs](dyg-mamba_continuous_state_space_modeling_on_dynamic_graphs.md)
- [\[ICML 2025\] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts](../../ICML2025/medical_imaging/i2moe_interpretable_multimodal_interaction-aware_mixture-of-experts.md)
- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)

<!-- RELATED:END -->
