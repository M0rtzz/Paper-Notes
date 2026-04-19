---
title: >-
  [论文解读] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection
description: >-
  [CVPR 2026][目标检测][扩散模型记忆] 提出 RAPTA（训练时区域感知提示增强）缓解扩散模型记忆化，以及 ADMCD（注意力驱动多模态拷贝检测）检测生成图像是否复制训练数据，两个模块互补形成端到端的记忆化缓解与检测框架。
tags:
  - CVPR 2026
  - 目标检测
  - 扩散模型记忆
  - 提示增强
  - 拷贝检测
  - 多模态融合
  - 版权保护
---

# Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection

**会议**: CVPR 2026  
**arXiv**: [2603.13070](https://arxiv.org/abs/2603.13070)  
**代码**: 无  
**领域**: 目标检测  
**关键词**: 扩散模型记忆, 提示增强, 拷贝检测, 多模态融合, 版权保护

## 一句话总结
提出 RAPTA（训练时区域感知提示增强）缓解扩散模型记忆化，以及 ADMCD（注意力驱动多模态拷贝检测）检测生成图像是否复制训练数据，两个模块互补形成端到端的记忆化缓解与检测框架。

## 研究背景与动机
文本到图像扩散模型（如 Stable Diffusion）可能记忆并复制训练图像，带来版权和隐私风险。现有方法的局限：

**推理时提示扰动**（如随机 token 插入、BLIP 改写、CLIP 嵌入加噪）：降低拷贝率但损害提示-图像对齐和生成质量，且不解决训练时记忆化

**单视角检测指标**（SSIM、SSCD、CLIP 余弦）：仅提供粗粒度信号，对部分拷贝或风格拷贝不鲁棒，依赖人工判断

**缺乏大规模标注的拷贝对数据集**

本文的核心观察：记忆化源于大模型容量 + 强文本-图像对齐 + 过度依赖训练时 caption-image 配对，因此应在训练时多样化提示以打破固定配对关系。

## 方法详解

### 整体框架
两个互补模块：
- **RAPTA**（训练时）：利用目标检测器生成区域感知的提示变体，随机采样一个变体作为训练条件
- **ADMCD**（推理时）：融合 patch 级、全局语义、纹理三种特征进行拷贝检测和类型分类

### 关键设计

1. **RAPTA（Region-Aware Prompt Augmentation）**：

    - 对训练图像 $I$ 运行预训练检测器（Faster R-CNN），获取高置信度区域 $(b_i, c_i, S_i)$
    - 将框中心离散化到 $3 \times 3$ 网格 $\mathcal{G}$ 得到位置 token（如 top-left, center 等）
    - 通过小型模板集 $\{T_j\}_{j=1}^{J}$ 实例化区域感知变体，如 "$p$, with a $\langle c \rangle$ in the $\langle \text{pos} \rangle$"
    - CLIP 一致性评分 $S_v = \cos(f_I, f_v)$ → 温度加权 $w_v = S_v^\gamma$ → 归一化为采样分布 $\pi(v)$
    - 每次迭代采样一个变体 $\tilde{p} \sim \pi(\cdot)$ 条件化去噪器，损失不变：$\mathcal{L}_{\mathrm{diff}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t, e)\|_2^2]$
    - 核心优势：语义锚定的多样性（基于检测区域），不引入语义漂移

2. **ADMCD（Attention-Driven Multimodal Copy Detection）**：

    - **三流特征提取**：ViT patch 级视觉描述子 $\mathbf{f}^{\mathrm{vis}}$、CLIP 全局语义描述子 $\mathbf{f}^{\mathrm{clip}}$、CNN 纹理描述子 $\mathbf{f}^{\mathrm{tex}}$
    - **注意力融合**：线性投影到共同维度 → 轻量 Transformer 编码器进行注意力融合 → $\ell_2$ 归一化得到融合向量 $\hat{\mathbf{f}}_{\mathrm{fus}}$
    - **两阶段决策规则**：
        - 拷贝判定：$S_{\mathrm{fus}} = \cos(\hat{\mathbf{f}}_{\mathrm{fus}}(G), \hat{\mathbf{f}}_{\mathrm{fus}}(R)) > \tau_1 = 0.938$
        - 拷贝类型：加权评分 $\bar{S} = 0.24 S_{\mathrm{vis}} + 0.38 S_{\mathrm{clip}} + 0.38 S_{\mathrm{tex}}$，$\bar{S} > \tau_2 = 0.970$ 为 Retrieve/Exact，否则为 Style 拷贝

3. **ADMCD 作为相似度度量**：融合相似度比单一指标更符合人类感知，在光度和几何扰动下更稳定。三流设计使得当某一线索不可靠时（如纹理匹配对 LPIPS、关键点稀疏对 ORB），其他线索可以补偿。

### 损失函数 / 训练策略
- RAPTA 使用标准扩散损失，仅改变条件输入（提示变体），无额外训练开销
- ADMCD 不需要任务特定训练数据，阈值和权重通过验证集网格搜索确定后固定
- 评估集：1200 对 query-reference（~25 retrieve/exact，~200 style，~1000 non-copy）

## 实验关键数据

### 主实验
| 方法 | Copy Rate ↓ | FID | CLIP Score | KID |
|------|------------|-----|------------|-----|
| DCR | 3.2 | 7.9 | 30.5 | 2.9 |
| LDM-T2I | 5.3 | 10.4 | 33.2 | 3.1 |
| SD2.1-base | 7.4 | 8.3 | 27.8 | 3.3 |
| **RAPTA (Ours)** | **2.6** | 8.1 | 23.1 | **1.6** |

RAPTA 拷贝率降低 18.8%~64.9%（相对），FID 和 KID 保持可比或更优。

### 消融实验（鲁棒性）
| 扰动类型 | ADMCD | DreamSim | SSCD | SSIM |
|----------|-------|----------|------|------|
| Original | 0.974 | 0.857 | 0.680 | 0.677 |
| Gaussian Noise | 0.923 | 0.781 | 0.594 | 0.504 |
| Salt & Pepper | 0.871 | 0.689 | 0.485 | 0.389 |
| Rotation 30° | 0.939 | 0.689 | 0.489 | 0.207 |

ADMCD 在所有噪声和几何扰动下保持最高且最稳定的相似度评分。

### 关键发现
- ADMCD 融合相似度在 0.871~0.974 范围内波动，远优于单一指标
- RAPTA 的 CLIP Score 较低（23.1 vs 27.8~33.2），反映抑制复制时文本-图像相似度的权衡
- 三流设计的互补性：ViT 提供空间锚点，CLIP 提供颜色/光照不变性，CNN 提供噪声/模糊鲁棒性

## 亮点与洞察
1. **训练时缓解 vs 推理时检测**的双管齐下策略，覆盖了记忆化问题的两个阶段
2. **区域感知模板**比随机扰动更有语义依据，通过检测器提供 grounded diversity
3. **零训练拷贝检测**：ADMCD 无需标注数据训练，阈值固定即可部署
4. **拷贝类型分类**（retrieve/exact vs style）提供比二分类更细粒度的判断

## 局限与展望
- 评估集仅 1200 对，retrieve/exact 仅约 25 对，样本量较小
- RAPTA 的 CLIP Score 下降说明多样化与对齐之间存在张力
- 模板集 $J$ 较小，更丰富的模板或 LLM 生成的变体可能进一步提升
- 仅在 LAION-10k 上评估，大规模数据上的效果待验证

## 相关工作与启发
- RAPTA 利用目标检测器为扩散模型提供结构化信息，类似 GLIGEN/ControlNet 的思路但目的不同
- ADMCD 的多流融合思路可推广到其他图像取证任务
- 对扩散模型版权保护研究有直接参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 训练时区域感知增强+多模态检测的组合新颖
- 实验充分度: ⭐⭐⭐ 评估集规模偏小，鲁棒性测试全面但主实验数据有限
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码完整
- 价值: ⭐⭐⭐ 对扩散模型安全有实际意义，但评估规模限制了说服力

<!-- RELATED:START -->

## 相关论文

- [CLCR: Cross-Level Semantic Collaborative Representation for Multimodal Learning](clcr_cross-level_semantic_collaborative_representation_for_multimodal_learning.md)
- [Prompt-Free Universal Region Proposal Network](prompt-free_universal_region_proposal_network.md)
- [CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization](cinesrd_leveraging_visual_acoustic_and_linguistic_cues_for_open-world_visual_med.md)
- [Bridging Pixels and Words: Mask-Aware Local Semantic Fusion for Multimodal Media Verification](bridging_pixels_and_words_mask-aware_local_semantic_fusion_for_multimodal_media_.md)
- [Parameter-Efficient Semantic Augmentation for Enhancing Open-Vocabulary Object Detection](parameter-efficient_semantic_augmentation_for_enhancing_open-vocabulary_object_d.md)

<!-- RELATED:END -->
