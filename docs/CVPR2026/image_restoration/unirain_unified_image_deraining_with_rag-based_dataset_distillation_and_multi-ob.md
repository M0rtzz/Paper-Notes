---
title: >-
  [论文解读] UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization
description: >-
  [CVPR 2026][图像恢复][image_restoration] 提出 UniRain，一个统一的去雨框架，通过 RAG 驱动的数据集蒸馏从 200 万+ 公开图像对中筛选高质量训练样本，结合非对称 MoE 架构和多目标自适应重加权优化策略，首次在单一模型中同时处理白天/夜晚的雨条纹和雨滴四种退化。
tags:
  - CVPR 2026
  - 图像恢复
  - image_restoration
  - deraining
  - mixture_of_experts
  - multi_objective_optimization
  - RAG
---

# UniRain: Unified Image Deraining with RAG-based Dataset Distillation and Multi-objective Reweighted Optimization

**会议**: CVPR 2026  
**arXiv**: [2603.03967](https://arxiv.org/abs/2603.03967)  
**代码**: 无  
**领域**: 图像恢复  
**关键词**: image_restoration, deraining, mixture_of_experts, multi_objective_optimization, RAG  

## 一句话总结

提出 UniRain，一个统一的去雨框架，通过 RAG 驱动的数据集蒸馏从 200 万+ 公开图像对中筛选高质量训练样本，结合非对称 MoE 架构和多目标自适应重加权优化策略，首次在单一模型中同时处理白天/夜晚的雨条纹和雨滴四种退化。

## 背景与动机

现有去雨方法面临两个核心挑战：

1. **数据质量不均**：直接混合所有合成与真实数据集（>200 万对）引入不准确的监督信号，反而损害模型收敛和泛化（实验证实直接混合甚至不如精心筛选的子集）。
2. **训练不平衡**：不同类型雨退化（白天雨条纹 DRS、白天雨滴 DRD、夜晚雨条纹 NRS、夜晚雨滴 NRD）的难度和收敛速度差异大，统一优化导致模型偏向简单类型而忽略困难类型。

## 方法详解

### 1. RAG-based Dataset Distillation

#### 检索阶段
从大规模语料库构建数据库，对每张真实雨图存储三元组 $(T_r, f_r, I_r)$（BLIP 文本描述、CLIP 视觉特征、图像）。

对查询图像执行三级层次化相似度匹配：
- **语义相似度**：$s_{txt}(q,r) = \|\phi_T(T_q) - \phi_T(T_r)\|_2$，选 Top-$K_1$
- **视觉特征相似度**：$s_{vis}(q,r') = \frac{f_q^\top f_{r'}}{\|f_q\|_2 \|f_{r'}\|_2}$，选 Top-$K_2$
- **结构相似度**：$s_{perc}(q,r'') = SSIM(I_q, I_{r''})$，选 Top-$K_3$

#### 生成阶段
将检索到的参考图像与查询图像组合，通过 3 个 VLM（InternVL2.5-8B、LLaVA-NeXT-7B、MobileVLM-3B）投票判断数据质量：

$$\hat{R}_q = \begin{cases} 1 & \text{if } \sum_{i=1}^3 \mathbb{I}(R_q^i = 1) \geq 2 \\ 0 & \text{otherwise} \end{cases}$$

最终从 200 万+ 图像对中蒸馏出 52,869 对高质量训练样本（仅保留约 2.6%）。

### 2. 多目标自适应重加权优化

通过滑动窗口线性回归估计每种退化类型的收敛斜率 $\alpha_i$，然后计算三个动态权重指标：

- **Type Balance Score (TBS)**：向收敛慢的类型倾斜
  $$\mathrm{TBS}_i(t) = \text{softmax}_i\left(K \frac{\alpha_i(t)}{\sum_i |\alpha_i(t)|}\right)$$

- **Type Stability Score (TSS)**：抑制发散类型的过高权重
  $$\mathrm{TSS}_i(t) = \text{softmax}_i\left(-N \frac{\alpha_i(t)}{\sum_{k=t-N+1}^t |\alpha_i(k)|}\right)$$

- **Adaptivity Factor (AF)**：动态调节 TBS 与 TSS 的比例
  $$AF(t) = \min\left(t \cdot \text{softmax}_t\left(-\frac{\tau t \cdot \alpha_{\max}(t)}{\sum_{i=1}^t \alpha_{\max}(i)}\right), 1\right)$$

最终权重：$\omega_i(t) = AF(t) \cdot TBS(t) + (1 - AF(t)) \cdot TSS(t)$

### 3. 非对称 MoE 架构

- **编码器（Soft-MoE）**：所有专家的输出通过连续权重加权聚合，全面保留多样化退化线索
  $$y_{en} = \sum_{i=1}^N \mathcal{R}_{soft}^i \otimes y_{en}^i$$

- **解码器（Hard-MoE）**：Top-k 路由选择性激活最相关专家，聚焦细粒度纹理重建
  $$y_{de} = \sum_{i=1}^N \mathcal{R}_{hard}^i \cdot y_{de}^i$$

## 实验结果

### 表1：RainRAG 数据集四种退化统一评测

| 方法 | DRS PSNR | DRD PSNR | NRS PSNR | NRD PSNR | 平均 PSNR↑ | 平均 SSIM↑ |
|------|----------|----------|----------|----------|------------|------------|
| Restormer | 28.45 | 23.36 | 33.92 | 25.85 | 27.89 | 0.8405 |
| MSDT | 28.60 | 23.31 | 34.56 | 25.28 | 27.94 | 0.8410 |
| NeRD-Rain | 28.11 | 23.30 | 33.88 | 25.31 | 27.65 | 0.8340 |
| URIR | 28.29 | 23.19 | 34.32 | 25.82 | 27.91 | 0.8425 |
| **UniRain** | **29.58** | **24.71** | **35.23** | **26.21** | **28.93** | **0.8515** |

### 表2：真实世界公开基准平均性能

| 方法 | 平均 PSNR↑ | 平均 SSIM↑ |
|------|------------|------------|
| NeRD-Rain | 27.81 | 0.8132 |
| URIR | 27.69 | 0.8061 |
| **UniRain** | **29.42** | **0.8222** |

UniRain 在所有四种退化类型和所有真实世界基准上均以显著优势领先，平均 PSNR 比 SOTA 高 ~1 dB。

## 亮点与创新

- **RAG + VLM 的数据蒸馏**思路新颖：将检索增强生成从 NLP 迁移到低层视觉数据筛选，仅保留 2.6% 数据反而提升性能
- **多目标自适应重加权**有效解决了混合训练中类型不平衡问题，TBS/TSS/AF 三级策略逻辑自洽
- **非对称 MoE** 编码端soft/解码端hard的设计符合直觉（探索 vs 聚焦）
- 首个覆盖白天+夜晚、雨条纹+雨滴的统一去雨框架
- 模型复杂度与竞争方法持平（126.5G FLOPs, 24.4M 参数）

## 不足与局限

- RAG 数据蒸馏流程需要多个 VLM 的推理，前期计算成本高
- VLM 质量评估的准确性依赖于 prompt 工程和 VLM 能力，可能存在偏差
- 多目标优化中的窗口大小 $N$ 和灵敏度参数 $\tau$ 需要手动调节
- 仅处理雨相关退化，未扩展到雾、雪等其他天气条件
- 夜晚雨滴（NRD）子集上性能改善相对较小（+0.39 dB），说明复杂退化仍有提升空间

## 评分

⭐⭐⭐⭐ — 问题定义有实际价值，RAG 数据蒸馏和多目标优化的组合逻辑清晰有效；统一框架的实用性强，但 RAG 流程的可扩展性和泛化到其他退化类型有待验证。

<!-- RELATED:START -->

## 相关论文

- [EVLF: Early Vision-Language Fusion for Generative Dataset Distillation](evlf_early_vision-language_fusion_for_generative_dataset_distillation.md)
- [Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_network_for_expansive_receptive_fields_in_l.md)
- [MRO: Enhancing Reasoning in Diffusion Language Models via Multi-Reward Optimization](../../NeurIPS2025/image_restoration/mro_enhancing_reasoning_in_diffusion_language_models_via_multi-reward_optimizati.md)
- [SD-PSFNet: Sequential and Dynamic Point Spread Function Network for Image Deraining](../../AAAI2026/image_restoration/sd-psfnet_sequential_and_dynamic_point_spread_function_netwo.md)

<!-- RELATED:END -->
