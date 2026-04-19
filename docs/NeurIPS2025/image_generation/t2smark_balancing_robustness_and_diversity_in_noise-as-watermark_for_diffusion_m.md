---
title: >-
  [论文解读] T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][图像水印] 提出 T2SMark，一种基于尾部截断采样（Tail-Truncated Sampling）的两阶段扩散模型水印方案，通过在高斯噪声的尾部区域嵌入水印比特、中心区域随机采样，首次在水印鲁棒性和生成多样性之间取得最优平衡。
tags:
  - NeurIPS 2025
  - 图像生成
  - 图像水印
  - 扩散模型
  - Noise-as-Watermark
  - 尾部截断采样
  - 鲁棒性-多样性平衡
---

# T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2510.22366](https://arxiv.org/abs/2510.22366)  
**代码**: [GitHub](https://github.com/0xD009/T2SMark)  
**领域**: 扩散模型 / 图像水印  
**关键词**: 图像水印, 扩散模型, Noise-as-Watermark, 尾部截断采样, 鲁棒性-多样性平衡

## 一句话总结

提出 T2SMark，一种基于尾部截断采样（Tail-Truncated Sampling）的两阶段扩散模型水印方案，通过在高斯噪声的尾部区域嵌入水印比特、中心区域随机采样，首次在水印鲁棒性和生成多样性之间取得最优平衡。

## 研究背景与动机

扩散模型生成的高保真图像引发了版权保护和 AI 生成内容溯源的迫切需求。Noise-as-Watermark（NaW）方法是一类有前景的水印技术：将水印信息编码为特定的标准高斯噪声向量作为生成初始噪声，提取时通过反演扩散过程恢复初始噪声来解码水印。

现有 NaW 方法面临**鲁棒性与多样性的根本矛盾**：
- **Gaussian Shading (GS)**：使用简单重复码获得高鲁棒性，但每个用户使用固定码字，生成多样性严重受限（LPIPS 仅 0.6446）
- **PRC-Watermark (PRCW)**：使用伪随机纠错码保持多样性（LPIPS 0.7074），但鲁棒性极弱，对抗条件下 TPR 仅 29.4%

作者的关键洞察：**靠近原点的高斯样本在噪声扰动下极易发生符号翻转**，导致比特错误。因此与其简单地用正/负号映射比特，不如只在更可靠的尾部区域嵌入信息。

## 方法详解

### 整体框架

T2SMark 包含两个核心创新：
1. **尾部截断采样（TTS）**：将高斯分布划分为三个区域——bit-0 区、bit-1 区和不确定区（中心区），仅在尾部嵌入水印
2. **两阶段架构**：第一段用固定主密钥加密随机会话密钥，第二段用会话密钥加密实际水印比特，引入可控随机性

### 关键设计

1. **尾部截断采样（TTS）**：给定噪声维度 $n$、截断阈值 $\tau$、水印长度 $m$，TTS 的核心操作是：

    - 计算尾部采样维度数 $k = 2\Phi(-\tau)n$ 和每比特子空间维度 $r = \lfloor k/m \rfloor$
    - 用密钥 $K$ 生成 $m$ 个伪随机正交支撑向量 $\{v_j\}_{j=1}^m$，作为比特编码超平面的法向量
    - 水印子空间（$w_i=1$）从尾部 $\mathcal{TN}(0,1; (-\infty,-\tau]\cup[\tau,\infty))$ 采样
    - 随机子空间（$w_i=0$）从中心 $\mathcal{TN}(0,1; [-\tau,\tau])$ 采样
    - 最终水印噪声：$z^w = w \odot |z| \odot \sum_j b_j v_j + (1-w) \odot z$
   
   设计动机：尾部样本距离判决边界更远，在 AWGN 扰动下符号翻转概率更低，本质上是用更少的编码维度换取更高的信噪比。

2. **投影解码**：提取时用相同密钥重建法向量集，通过点积投影恢复比特：
    $p_j = \langle \hat{z}^w, v_j \rangle, \quad \hat{b} = \text{sign}(p)$
   检测统计量使用 $L_1$ 范数 $l = \|p_k\|_1$，TTS 使投影幅度更大，提升检测可靠性。

3. **两阶段水印结构**：将 $n$ 维噪声分为 $n_k$ 和 $n_b$ 两段。第一段用主密钥 $K$ 编码随机会话密钥 $K_r$，第二段用 $K_r$ 编码实际水印比特 $b$。由于每次生成都使用不同的随机 $K_r$，整个噪声向量保持随机性，确保生成多样性。检测依赖第一段（避免二阶段误差传播），溯源需完整两阶段解码。

### 损失函数 / 训练策略

T2SMark 是 training-free 的水印方案，无需额外训练。关键超参数：
- 截断阈值 $\tau = 0.674$（对应高斯分布 25% 分位点）
- 会话密钥 16 bit
- 水印容量 256 bit
- DDIM 反演步数 10 步

## 实验关键数据

### 主实验

**SD v2.1 上的综合评估**

| 方法 | TPR (Clean/Adv) | Bit Acc (Clean/Adv) | 多样性 LPIPS↑ | CLIP Score | FID |
|------|:---:|:---:|:---:|:---:|:---:|
| GS | 1.000/0.998 | 1.000/0.9548 | 0.6446 | 0.3242 | 58.14 |
| PRCW | 1.000/0.294 | 0.6494/0.5024 | **0.7074** | 0.3218 | 56.90 |
| TRW | 1.000/0.907 | - | 0.6943 | 0.3210 | 58.27 |
| **T2SMark** | **1.000/0.998** | **1.000/0.9754** | 0.7069 | 0.3227 | 56.93 |

**SD v3.5M (DiT backbone) 上的泛化性**

| 方法 | TPR (Clean/Adv) | Bit Acc (Clean/Adv) | Det. Acc↓ | Diversity↑ |
|------|:---:|:---:|:---:|:---:|
| GS | 1.000/0.990 | 0.9994/0.9663 | 0.991 | 0.5176 |
| PRCW | 0.998/0.279 | 0.9920/0.6067 | **0.516** | 0.6096 |
| **T2SMark** | **1.000/0.985** | **1.000/0.9768** | 0.518 | **0.6102** |

### 消融实验

| 配置 | TPR (Clean/Adv) | Bit Acc (Clean/Adv) | Diversity↑ |
|------|:---:|:---:|:---:|
| w/o TTS | 1.000/0.996 | 0.9988/0.9307 | 0.6743 |
| w/ TTS | 1.000/0.998 | 1.000/0.9754 | 0.6746 |
| 容量=256 bit | 1.000/0.9754 | - | - |
| 容量=512 bit | 1.000/0.9437 | - | - |
| 容量=1024 bit | 0.9968/0.8789 | - | - |
| 会话密钥=16 bit | 1.000/0.998 | 1.000/0.9754 | 最佳平衡点 |
| 会话密钥=32 bit | 1.000/0.991 | 1.000/0.9481 | 仍可接受 |

### 关键发现

- TTS 带来的鲁棒性提升幅度显著（对抗 Bit Acc 从 93.07% 升至 97.54%），且对多样性的影响可忽略（差异 < 0.001）
- T2SMark 在图像质量上是唯一和 PRCW 并列满足 "无退化准则" 的方法（t 检测 CLIP 和 FID 均不显著）
- 不可检测性方面，T2SMark 的 Det. Acc（0.578）接近随机（0.5），远优于 GS（0.994）
- 反演步数从 5 到 100 对性能影响极小，可通过减少步数加速提取

## 亮点与洞察

- **三区域划分**的设计直觉简洁有力：高斯分布尾部天然信噪比高，是嵌入信息的理想位置
- 两阶段密钥分层的巧妙构思：会话密钥既是第一段的负载又是第二段的密钥，一石二鸟
- 用多维投影而非逐元素判决来解码，充分利用了高斯向量的连续结构信息
- 评估方法严谨：用 t 检验而非简单对比来评估视觉质量退化，是同类工作中少见的统计学规范

## 局限与展望

- 对高斯噪声攻击脆弱（$\sigma=0.1$ 即严重退化），这是 NaW 类方法的通病
- 依赖 ODE 可逆采样器（DDIM），不适用于不支持反演的扩散模型
- 缺乏对几何变换（旋转、透视变换）的抵抗机制
- 尾部采样引入的分布异常可能被专门训练的分类器检测到
- 与控制生成方法（ControlNet 等）可能存在冲突

## 相关工作与启发

- 与 Gaussian Shading 和 PRC-Watermark 形成直接对比，T2SMark 在两者之间找到帕累托最优
- Tree-Ring 在频域嵌入鲁棒模式但引入分布偏差，T2SMark 保持分布一致性
- 可启发将 TTS 思路推广到其他基于高斯采样的信息隐藏场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 尾部截断采样的思路新颖且有理论动机，两阶段密钥架构设计精巧
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖 UNet 和 DiT 两种骨干、9 种攻击类型、多种超参消融，评估全面系统
- **写作质量**: ⭐⭐⭐⭐ 方法阐述清晰，公式推导严谨，图表直观
- **价值**: ⭐⭐⭐⭐ 为扩散模型水印提供了实用且理论合理的解决方案，但应用范围限于 NaW 范式

<!-- RELATED:START -->

## 相关论文

- [GaussMarker: Robust Dual-Domain Watermark for Diffusion Models](../../ICML2025/image_generation/gaussmarker_robust_dual-domain_watermark_for_diffusion_models.md)
- [Golden Noise for Diffusion Models: A Learning Framework](../../ICCV2025/image_generation/golden_noise_for_diffusion_models_a_learning_framework.md)
- [Robustness in Both Domains: CLIP Needs a Robust Text Encoder](robustness_in_both_domains_clip_needs_a_robust_text_encoder.md)
- [Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [When Are Concepts Erased From Diffusion Models?](when_are_concepts_erased_from_diffusion_models.md)

<!-- RELATED:END -->
