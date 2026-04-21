---
title: >-
  [论文解读] Golden Noise for Diffusion Models: A Learning Framework
description: >-
  [ICCV 2025][图像生成][噪声提示] 本文提出"噪声提示"（Noise Prompt）概念，设计了一个轻量级噪声提示网络（NPNet），通过 Re-denoise Sampling 收集 10 万对噪声数据训练 NPNet，将随机高斯噪声转化为承载语义信息的"黄金噪声"，作为即插即用模块提升 SDXL 等多种扩散模型的生成质量，仅增加 3% 推理时间。
tags:
  - ICCV 2025
  - 图像生成
  - 噪声提示
  - 黄金噪声
  - 扩散模型
  - 图像质量提升
  - 即插即用
---

# Golden Noise for Diffusion Models: A Learning Framework

**会议**: ICCV 2025  
**arXiv**: [2411.09502](https://arxiv.org/abs/2411.09502)  
**代码**: [GitHub](https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models)  
**领域**: 扩散模型/图像生成  
**关键词**: 噪声提示, 黄金噪声, 扩散模型, 图像质量提升, 即插即用

## 一句话总结
本文提出"噪声提示"（Noise Prompt）概念，设计了一个轻量级噪声提示网络（NPNet），通过 Re-denoise Sampling 收集 10 万对噪声数据训练 NPNet，将随机高斯噪声转化为承载语义信息的"黄金噪声"，作为即插即用模块提升 SDXL 等多种扩散模型的生成质量，仅增加 3% 推理时间。

## 研究背景与动机

文生图扩散模型中，图像质量由**文本提示**和**初始噪声**两个因素共同决定。文本提示的重要性已被广泛研究（prompt engineering），但噪声的角色长期被忽视。

**关键观察**：一些噪声天然比其他噪声更好（golden noise），能生成更高质量、语义对齐更好的图像。但如何系统地获得这些黄金噪声？

**现有噪声优化方法的局限**：
1. 难以泛化到不同数据集和模型
2. 在逆向过程中优化噪声引入显著时延
3. 需要修改原始 pipeline 的内部结构
4. 需要特定 subject token 计算损失

核心问题：**能否将获取黄金噪声formulate为机器学习问题，用一次前向推理高效预测？能否泛化到不同噪声、提示和模型？**

本文的核心 idea：**Noise Prompt = 对随机噪声施加由文本驱动的小扰动，将其转为黄金噪声**。

## 方法详解

### 整体框架

三阶段工作流：
- **Stage I**（数据收集）：通过 Re-denoise Sampling 生成噪声对，用人类偏好模型筛选
- **Stage II**（训练）：训练 NPNet 学习从源噪声到目标噪声的映射
- **Stage III**（推理）：NPNet 作为即插即用模块替换随机噪声为黄金噪声

### 关键设计

1. **Re-denoise Sampling 数据收集**:

    - 功能：构建大规模噪声提示数据集（NPD），包含 10 万对源噪声-目标噪声及对应文本
    - 核心思路：对初始噪声 $\mathbf{x}_T$ 先做一步 DDIM 去噪得到 $\mathbf{x}_{T-1}$，再用 DDIM-Inversion 反转回 $\mathbf{x}_T'$。由于 DDIM 和 DDIM-Inversion 使用不同的 CFG 尺度（$\omega_l > \omega_w$），$\mathbf{x}_T'$ 相比 $\mathbf{x}_T$ 携带了更多语义信息：
    $\mathbf{x}_T' = \text{DDIM-Inversion}(\text{DDIM}(\mathbf{x}_T))$
    - 数据筛选：使用 HPSv2 偏好模型，仅保留 $s_0 + m < s_0'$ 的高质量对
    - 设计动机：利用 CFG 尺度不一致性将语义信息"注入"到噪声中，AI 反馈确保数据集质量

2. **NPNet 架构设计（SVD 预测 + 残差预测）**:

    - 功能：从源噪声和文本提示预测黄金噪声
    - 核心思路包含两条路径：
   
      **奇异值预测**：基于 Davis-Kahan 定理的观察——源噪声和目标噪声的奇异向量高度相似，只需预测新的奇异值：
    $\mathbf{x}_T = U \times \Sigma \times V^T$
    $\tilde{\Sigma} = f(g(\phi(U, \Sigma, V^T)))$
    $\tilde{\mathbf{x}}_T' = U \times \tilde{\Sigma} \times V^T$
     
      **残差预测**：通过 ViT + UpSample/DownSample 预测源噪声和目标噪声的残差，并注入文本语义：
    $\mathbf{e} = \sigma(\mathbf{x}_T, \mathcal{E}(\mathbf{c}))$
    $\hat{\mathbf{x}}_T = \varphi'(\psi(\varphi(\mathbf{x}_T + \mathbf{e})))$
    - 设计动机：SVD 路径利用了噪声对之间的结构相似性，残差路径捕获文本相关的精细调整

3. **训练与推理**:

    - 功能：MSE 损失训练，推理时直接替换初始噪声
    - 核心思路：
    $\mathcal{L}_\text{MSE} = \text{MSE}(\mathbf{x}_T', \mathbf{x}_{T_{pred}}')$
    $\mathbf{x}_{T_{pred}}' = \alpha\mathbf{e} + \tilde{\mathbf{x}}_T' + \beta\hat{\mathbf{x}}_T$
      其中 $\alpha, \beta$ 为可学习参数
    - 设计动机：$\alpha$ 控制语义注入强度，$\beta$ 控制残差预测权重，自适应平衡两条路径

### 损失函数 / 训练策略

- 损失函数：MSE 损失（预测噪声 vs 目标噪声）
- 训练集：从 Pick-a-Pic 随机选取 10 万 prompt，每个配随机种子
- 训练配置：batch size 64，30 epochs

## 实验关键数据

### 主实验

| 模型/数据集 | 指标 | Standard | NPNet | 提升 |
|--------|------|------|----------|------|
| SDXL / Pick-a-Pic | HPSv2↑ | 28.48 | **28.68** | +0.20 |
| SDXL / Pick-a-Pic | ImageReward↑ | 58.01 | **65.01** | +7.00 |
| SDXL / Pick-a-Pic | CLIPScore↑ | 0.8204 | **0.8408** | +0.0204 |
| SDXL / DrawBench | ImageReward↑ | 62.21 | **70.67** | +8.46 |
| DreamShaper / Pick-a-Pic | HPSv2↑ | 32.12 | **32.69** | +0.57 |
| DreamShaper / Pick-a-Pic | ImageReward↑ | 98.09 | **106.74** | +8.65 |
| Hunyuan-DiT / HPD | ImageReward↑ | 99.22 | **108.29** | +9.07 |
| Hunyuan-DiT / HPD | MPS↑ | - | **52.87** | 胜率超50% |

### 消融实验

| 配置 | PickScore | HPSv2 | AES | ImageReward |
|------|---------|------|------|------|
| Standard（无 NPNet） | 21.69 | 28.48 | 6.0373 | 58.01 |
| NPNet w/o 奇异值预测 | 21.49 | 27.76 | 6.0164 | 49.03 |
| NPNet w/o 残差预测 | 21.83 | 28.55 | 6.0315 | 63.05 |
| NPNet w/o 数据筛选 | 21.73 | 28.46 | 6.0375 | 62.91 |
| **NPNet（完整）** | **21.86** | **28.68** | **6.0540** | **65.01** |

### 关键发现

- **奇异值预测是核心组件**：去掉后性能甚至低于 Standard，说明 SVD 结构先验至关重要
- **跨模型泛化性强**：SDXL 训练的 NPNet 仅需 600 样本微调即可用于 Hunyuan-DiT
- **跨采样器泛化**：DDIM 训练的 NPNet 在 7 种不同随机/确定性采样器上均有效
- **与其他方法正交**：可与 DPO、AYS 等方法叠加使用，进一步提升效果
- **仅 3% 额外推理时间**：NPNet 作为 plug-and-play 模块的开销极小

## 亮点与洞察

1. **概念创新**："Noise Prompt"类比"Text Prompt"是非常直觉且有洞察力的概念
2. **SVD 观察精彩**：发现噪声对的奇异向量高度相似这一先验，巧妙地简化了学习问题
3. **工程落地性强**：3% 时间开销 + plug-and-play + 跨模型泛化 = 极高的实用价值
4. **Re-denoise Sampling 的理论理解**：CFG 尺度不一致性注入语义信息的机制有理论支撑

## 局限与展望

- 提升幅度在某些指标上不大（如 PickScore ~0.1 的提升）
- 依赖特定的 HPSv2 等人类偏好模型进行数据筛选，可能引入偏差
- Re-denoise Sampling 数据收集本身需要大量计算（10 万对噪声需要大量扩散推理）
- 在少步推理模型（如 LCM 4 步）上的效果有待更多验证
- 黄金噪声的"为什么好"缺乏更深入的理论分析

## 相关工作与启发

- 与噪声调度（noise schedule）优化正交，两者可组合
- Re-denoise Sampling 思想可推广到其他需要"注入信息到潜空间"的场景
- 未来方向：直接用 RLHF 或 DPO 端到端优化 NPNet，避免两阶段数据收集

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Noise Prompt 概念新颖，SVD 结构先验独特，Re-denoise Sampling 巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 跨模型、跨数据集、跨采样器、正交实验、消融实验全面
- 写作质量: ⭐⭐⭐⭐ 思路清晰，但细节较多需要仔细追踪
- 价值: ⭐⭐⭐⭐⭐ 作为即插即用的质量提升模块，实用价值极高

<!-- RELATED:START -->

## 相关论文

- [VisualCloze: A Universal Image Generation Framework via Visual In-Context Learning](visualcloze_a_universal_image_generation_framework_via_visua.md)
- [Improved Noise Schedule for Diffusion Training](improved_noise_schedule_for_diffusion_training.md)
- [LoRAverse: A Submodular Framework to Retrieve Diverse Adapters for Diffusion Models](loraverse_a_submodular_framework_to_retrieve_diverse_adapters_for_diffusion_mode.md)
- [Joint Diffusion Models in Continual Learning](joint_diffusion_models_in_continual_learning.md)
- [T2SMark: Balancing Robustness and Diversity in Noise-as-Watermark for Diffusion Models](../../NeurIPS2025/image_generation/t2smark_balancing_robustness_and_diversity_in_noise-as-watermark_for_diffusion_m.md)

<!-- RELATED:END -->
