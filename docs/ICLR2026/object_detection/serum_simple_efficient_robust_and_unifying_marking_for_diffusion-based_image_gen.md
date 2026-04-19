---
title: >-
  [论文解读] SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation
description: >-
  [ICLR 2026][目标检测][扩散模型水印] 提出SERUM水印方法，将唯一水印噪声添加到扩散模型初始噪声中，训练轻量检测器直接从生成图像识别水印（无需昂贵的DDIM反演），在多种攻击下达到最高检测率，且注入/检测极快，支持多用户场景。
tags:
  - ICLR 2026
  - 目标检测
  - 扩散模型水印
  - 轻量检测器
  - 噪声注入
  - 鲁棒性
  - 多用户
---

# SERUM: Simple, Efficient, Robust, and Unifying Marking for Diffusion-based Image Generation

**会议**: ICLR 2026  
**arXiv**: [2603.13396](https://arxiv.org/abs/2603.13396)  
**代码**: [GitHub](https://github.com/Hubizon/SERUM)  
**领域**: AI安全/水印  
**关键词**: 扩散模型水印, 轻量检测器, 噪声注入, 鲁棒性, 多用户

## 一句话总结
提出SERUM水印方法，将唯一水印噪声添加到扩散模型初始噪声中，训练轻量检测器直接从生成图像识别水印（无需昂贵的DDIM反演），在多种攻击下达到最高检测率，且注入/检测极快，支持多用户场景。

## 研究背景与动机

**领域现状**：扩散模型生成高度逼真图像→需要水印区分生成/真实内容。现有方法分两类：调参式(Stable Signature微调解码器)和免调式(Tree-Ring/GaussMarker向初始噪声加水印)。

**现有痛点**：(1) Stable Signature需大量训练且对高级攻击不鲁棒；(2) Tree-Ring/GaussMarker等免调方法鲁棒但检测依赖昂贵DDIM反演（$O(T)$步）；(3) 两者不可兼得——要么快但弱，要么强但慢。

**核心矛盾**：水印检测需要DDIM反演恢复初始噪声→计算昂贵→不适合大规模部署。

**切入角度**：不做DDIM反演——训练轻量外部检测器直接从生成图像中识别水印噪声的签名。既享受噪声注入的鲁棒性，又得到即时检测。

## 方法详解

### 整体框架
水印注入：$\eta' = \sqrt{1-\alpha}\eta + \sqrt{\alpha}A'$（加权混合随机噪声和水印噪声）→ 正常扩散生成。水印检测：训练轻量CNN在LDM潜空间中分类有/无水印。

### 关键设计

1. **水印注入**:

    - 功能：向初始扩散噪声添加归一化水印噪声
    - 核心思路：$A' = (A - \text{mean}(A))/\text{std}(A)$归一化保证低KL散度→高图像质量
    - 设计动机：归一化后 $\eta'$ 仍接近标准正态分布→证明比GaussMarker有更低KL散度

2. **轻量检测器**:

    - 功能：在LDM潜空间中训练二分类CNN
    - 核心思路：训练集=水印潜变量+干净潜变量，用增强的优先经验回放采样困难扰动
    - 损失：$\mathcal{L} = \mathcal{L}_w + \mathcal{L}_n$，各含干净/增强/预计算三项
    - 设计动机：在潜空间操作→输入维度小→训练和推理都快

3. **多用户支持**:

    - 功能：为每个用户分配唯一噪声模式子集
    - 核心思路：用户 $i$ 使用 $k$ 个噪声模式的组合，用户检测分数 $D_i(x) = \prod_{p \in S_i} d_p(x)$
    - 训练规模：$O(n^{1/k})$而非 $O(n)$

## 实验关键数据

### 检测鲁棒性（SDv1.4/2.0/2.1）
| 方法 | TPR@1%FPR(标准攻击) | TPR@1%FPR(去水印) | 注入速度 | 检测速度 |
|------|------------------|------------------|---------|---------|
| Stable Signature | 中 | 差 | 快 | 快 |
| Tree-Ring | 好 | 好 | 快 | **极慢**(DDIM) |
| GaussMarker | 好 | 好 | 快 | **极慢**(DDIM) |
| **SERUM** | **最优** | **最优** | **极快** | **极快** |

### 图像质量
| 方法 | FID↓ | CLIP Score↑ | 说明 |
|------|------|-------------|------|
| 无水印 | 基线 | 基线 | 参考 |
| **SERUM** | **接近基线** | **接近基线** | 质量几乎无损 |

### 关键发现
- SERUM在8种扰动和7种去水印攻击中几乎全部最高TPR
- 检测无需DDIM反演→比Tree-Ring/GaussMarker快几十倍
- 水印具有"放射性"——即使模型在水印图像上微调，输出仍可检测
- 多用户场景下用户间水印干扰可忽略

## 亮点与洞察
- **简洁统一两大范式**：噪声注入(免调方法的鲁棒性) + 外部检测器(调参方法的速度) = 最佳组合。这个想法直觉简单但之前无人尝试。
- **KL散度保证**：归一化水印噪声的KL散度理论保证比GaussMarker更低→图像质量更好的数学基础。
- **优先经验回放训练**：借鉴RL中的PER来采样"困难"增强→检测器自动聚焦弱点→不需要手动选择增强策略。

## 局限与展望
- 需要访问LDM编码器做检测——纯像素级检测未探索
- $\alpha$ 的选择需平衡检测率和图像多样性
- 多用户时组合数限制了用户上限
- 仅在SD系列验证，其他扩散模型(DALL-E/Imagen)效果未知

## 评分
- 新颖性: ⭐⭐⭐⭐ 噪声注入+外部检测器的组合虽简单但有效且之前未尝试
- 实验充分度: ⭐⭐⭐⭐⭐ 8种扰动+7种攻击+3个SD版本+多用户+放射性
- 写作质量: ⭐⭐⭐⭐ 方法清晰，理论保证到位
- 价值: ⭐⭐⭐⭐⭐ 对AI生成内容检测有直接实际部署价值

<!-- RELATED:START -->

## 相关论文

- [Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](../../CVPR2026/object_detection/adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](../../CVPR2025/object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [MCCD: Multi-Agent Collaboration-based Compositional Diffusion for Complex Text-to-Image Generation](../../CVPR2025/object_detection/mccd_multi-agent_collaboration-based_compositional_diffusion_for_complex_text-to.md)

<!-- RELATED:END -->
