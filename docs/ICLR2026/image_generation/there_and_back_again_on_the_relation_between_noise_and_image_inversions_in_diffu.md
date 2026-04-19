---
title: >-
  [论文解读] There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models
description: >-
  [ICLR 2026][图像生成][DDIM反转] 深入分析 DDIM 反转的误差机制，发现潜在编码在平滑图像区域（如天空）呈现低多样性和高相关性，并追溯到反转初始步骤的噪声预测不准确，提出用正向扩散替代前几步反转的简单修复方案。
tags:
  - ICLR 2026
  - 图像生成
  - DDIM反转
  - 潜在编码
  - 噪声相关性
  - 平滑区域
  - 正向扩散修复
---

# There and Back Again: On the Relation between Noise and Image Inversions in Diffusion Models

**会议**: ICLR 2026  
**arXiv**: [2410.23530](https://arxiv.org/abs/2410.23530)  
**代码**: [GitHub](https://github.com/luk-st/taba)  
**领域**: 扩散模型 / 反转分析 / 图像编辑  
**关键词**: DDIM反转, 潜在编码, 噪声相关性, 平滑区域, 正向扩散修复

## 一句话总结

深入分析 DDIM 反转的误差机制，发现潜在编码在平滑图像区域（如天空）呈现低多样性和高相关性，并追溯到反转初始步骤的噪声预测不准确，提出用正向扩散替代前几步反转的简单修复方案。

## 研究背景与动机

扩散模型缺乏显式的低维潜在空间来编码数据的可编辑特征。DDIM 反转通过逆转去噪轨迹、将图像传输到其近似的初始噪声来部分解决这一问题。然而：

**反转误差的来源不明**：虽然已知 DDIM 反转产生的潜在不完全是高斯噪声，但其根本原因和表现形式未被系统研究

**现有改进方法的局限**：Null-text inversion、Renoise 等方法虽改善重建质量，但未真正解决潜在的非高斯性

**潜在编码可操作性差**：与原始噪声空间相比，反转潜在空间的插值和编辑质量更低

## 方法详解

### 分析框架

研究三个变量之间的关系：
- **噪声** $\mathbf{x_T}$：生成图像的高斯输入
- **样本** $\mathbf{x_0}$：扩散模型生成的图像
- **潜在编码** $\hat{\mathbf{x}}_T$：DDIM 反转的结果

### 核心发现 1：潜在编码偏离高斯分布

8×8 像素块内的 Pearson 相关系数分析：

| 模型 | 噪声相关 | 潜在相关 | 样本相关 |
|------|---------|---------|---------|
| ADM-32 | 0.039 | **0.382** | 0.964 |
| ADM-64 | 0.039 | **0.126** | 0.925 |
| IF | 0.039 | **0.498** | 0.936 |
| LDM | 0.039 | **0.045** | 0.645 |
| DiT | 0.041 | **0.103** | 0.748 |
| SDXL | 0.036 | **0.155** | 0.637 |

潜在编码的相关性远高于噪声，且在视觉上可以观察到图像结构模式。

### 核心发现 2：平滑区域是主要问题

将图像分为"平滑区域"（如天空、背景）和"非平滑区域"：

| 模型 | 平滑区域误差 | 非平滑区域误差 | 平滑区域标准差 | 非平滑区域标准差 |
|------|-----------|-------------|-------------|---------------|
| ADM-32 | **0.49** | 0.43 | **0.34** | 0.46 |
| IF | **0.56** | 0.40 | **0.46** | 0.72 |
| LDM | **0.13** | 0.03 | **0.45** | 0.59 |
| DiT | **0.12** | 0.06 | **0.43** | 0.54 |

平滑区域的反转误差更高、潜在多样性更低。

### 核心发现 3：问题源于初始反转步骤

通过分析去噪轨迹发现：
- 中间步骤 $x_t$ 在约50-70%的生成轨迹处开始趋近反转潜在 $\hat{\mathbf{x}}_T$
- 潜在编码保留了原始样本的某些特性
- **关键**：前几步反转的噪声预测对平滑区域特别不准确和不多样

线性插值路径分析 $\|(1-\lambda)\mathbf{x_T} + \lambda \hat{\mathbf{x}}_T - x_t\|_2$ 显示，生成轨迹逐步向反转潜在靠拢。

### 核心发现 4：DDIM 潜在空间可操作性差

**插值实验**：DDIM 潜在空间的球面插值质量低于原始噪声空间
**编辑实验**：在平滑图像区域进行的编辑效果尤为受限

### 核心发现 5：现有改进方法未解决根本问题

| 方法 | 重建改善 | 高斯性保持 |
|------|---------|----------|
| Null-text inversion | ✓ | ✗ |
| Renoise | ✓ | ✗ |
| DPM-Solver inversion | ✓ | ✗ |
| 正则化方法 | 部分 | 部分 |

### 提出的修复方案

**简单修复**：用前向扩散替代 DDIM 反转的前 $k$ 步

$$\text{前} k \text{步}：x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t$$
$$\text{后续步}：\text{标准 DDIM 反转}$$

**效果**：
- 成功去相关潜在编码
- 不降低重建质量
- 改善插值和编辑质量
- 特别有效改善平滑区域

## 实验

### 实验模型
7种扩散模型覆盖像素空间/潜在空间、有条件/无条件、U-Net/DiT架构

### 修复方案验证

| 策略 | 相关性↓ | 重建质量↑ | 插值质量↑ | 编辑质量↑ |
|------|--------|---------|---------|---------|
| 标准 DDIM 反转 | 高 | 基线 | 低 | 低 |
| + 正则化 | 中 | 基线 | 中 | 中 |
| + 前k步前向扩散 | **低** | 保持 | **高** | **高** |

### Flow Matching 扩展
相同的相关性和低多样性问题在 Flow Matching 模型中也存在。

### 消融实验

| 参数 | 效果 |
|------|------|
| 替代步数 k 增大 | 多样性提升但重建可能降低 |
| DDIM步数增加 | 误差减小但问题仍存在 |
| 不同模型架构 | 趋势一致 |

## 亮点

1. **系统性分析的深度**：从7个模型全面验证反转误差的规律
2. **平滑区域问题的发现**：精确定位误差的空间分布模式
3. **简单有效的修复方案**：仅替换前几步即可显著改善
4. **跨架构泛化**：U-Net、DiT、像素空间、潜在空间均适用
5. **对社区认知的校正**：现有改进方法并未真正解决潜在的非高斯性

## 局限性

1. 修复方案带来的前向扩散步骤引入了随机性，可能影响确定性重建
2. 最优替代步数 $k$ 需要根据模型调整
3. 分析主要基于 DDIM 采样器，对其他采样器的推广需进一步验证
4. 平滑区域的定义依赖于手动阈值 $\tau = 0.025$
5. 未提供理论解释为何前几步的噪声预测对平滑区域特别不准确

## 相关工作

- **DDIM 反转**：Song et al. (2021)、Dhariwal & Nichol (2021)
- **反转改进**：Null-text inversion (Mokady 2023)、Renoise (Garibi 2024)
- **图像编辑**：P2P (Hertz 2022)、SDEdit (Meng 2021)
- **Flow Matching**：Lipman et al. (2023)

## 评分

- **创新性**: ⭐⭐⭐⭐ — 系统性分析视角新颖，发现有深度
- **实用性**: ⭐⭐⭐⭐ — 简单修复方案可直接应用
- **实验**: ⭐⭐⭐⭐⭐ — 7个模型的全面对比和验证
- **写作**: ⭐⭐⭐⭐⭐ — 分析层层递进，逻辑清晰

<!-- RELATED:START -->

## 相关论文

- [On the Relation between Rectified Flows and Optimal Transport](../../NeurIPS2025/image_generation/on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [Does Semantic Noise Initialization Transfer from Images to Videos? A Paired Diagnostic Study](does_semantic_noise_initialization_transfer_from_images_to_videos_a_paired_diagn.md)

<!-- RELATED:END -->
