---
title: >-
  [论文解读] ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation
description: >-
  [CVPR 2026][图像生成][稀有概念生成] 提出 ADAPT 框架，通过注意力驱动的自适应 Prompt 调度（APS）、池化嵌入操控（PEM）和潜空间操控（LSM）三个零样本模块，确定性且语义对齐地控制从通用到罕见概念的生成过渡，在 RareBench 上显著超越 R2F 基线。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "稀有概念生成"
  - "提示学习"
  - "正交投影"
  - "注意力机制"
  - "文本-图像对齐"
---

# ADAPT: Attention Driven Adaptive Prompt Scheduling and InTerpolating Orthogonal Complements for Rare Concepts Generation

**会议**: CVPR 2026  
**arXiv**: [2603.19157](https://arxiv.org/abs/2603.19157)  
**代码**: 有 (论文中提到 Code is available)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 稀有概念生成, Prompt调度, 正交投影, 注意力机制, 文本-图像对齐

## 一句话总结

提出 ADAPT 框架，通过注意力驱动的自适应 Prompt 调度（APS）、池化嵌入操控（PEM）和潜空间操控（LSM）三个零样本模块，确定性且语义对齐地控制从通用到罕见概念的生成过渡，在 RareBench 上显著超越 R2F 基线。

## 研究背景与动机

文本到图像扩散模型在生成常见物体方面表现优秀，但面对训练数据中罕见或缺失的组合概念（如"香蕉形状的汽车"、"黑白棋盘鳄鱼"）时，已有属性绑定方法（SynGen、Attend & Excite）仍力不从心。R2F 方法通过 GPT-4o 构建"频繁-罕见"概念对并进行 Prompt 调度来缓解此问题，但存在三个核心缺陷：

**GPT-4o 的随机性导致方差大**：同一 prompt 的视觉细节等级输出不一致，导致调度不稳定

**固定停止点缺乏语义对齐**：线性映射的停止点是启发式的，无法随生成过程中的 token 语义进展自适应

**迭代切换带来的语义不连续**：在罕见和频繁 prompt 之间反复切换文本嵌入，难以提供一致的语义指导

ADAPT 的核心 idea 是：**用注意力分数的收敛行为替代 GPT-4o 的主观评判来决定概念切换时机，同时用正交分解在嵌入空间中解耦罕见语义，提供全程一致的生成引导**。

## 方法详解

### 整体框架

ADAPT 在 Stable Diffusion 3（MM-DiT）架构上引入三个互补的零样本控制模块，无需额外训练或微调：
- **APS（自适应 Prompt 调度）**：基于注意力分数确定最优停止点
- **PEM（池化嵌入操控）**：在 CLIP 池化嵌入层面提供一致的罕见语义引导
- **LSM（潜空间操控）**：在 Transformer 注意力层内注入属性特定的方向引导

### 关键设计

1. **自适应 Prompt 调度（APS）**：

    - **功能**：确定性地决定何时从频繁概念过渡到罕见概念
    - **核心思路**：构建两类重建 prompt——进展 prompt $y_{\text{prog}}$（逐步过渡）和目标 prompt $y_{\text{tar}}$（包含所有罕见概念），在去噪过程中交替使用。利用转换计数器 $P_{\text{trans}}$ 追踪已完成的概念替换数
    - **关键公式**：对目标 prompt 中每个 token 计算注意力响应分数 $z_i = \max(\mathbf{A}^c_{y_{\text{tar},i}})$，取剩余未转换概念数 $k = m - P_{\text{trans}}$ 对应的 top-k 分数 $s^{(k)}$；当 $s^{(k)} < \tau_s$ 时触发转换
    - **设计动机**：空间注意力在生成过程中逐渐收敛，区分罕见/频繁概念的 token 收敛最慢。利用这一收敛特性作为语义特征饱和的指示器，实现语义对齐的调度，**消除了对 GPT-4o 的依赖**

2. **池化嵌入操控（PEM）**：

    - **功能**：在整个生成过程中提供一致的、解耦的罕见语义引导
    - **核心思路**：将罕见 prompt 的池化嵌入 $\boldsymbol{c}_{r,\text{pool}}$ 正交投影到频繁嵌入 $\boldsymbol{c}_{f,\text{pool}}$ 上，提取罕见特有的语义方向：
    $\Delta_r = \boldsymbol{c}_{r,\text{pool}} - \frac{\boldsymbol{c}_{f,\text{pool}} \cdot \boldsymbol{c}_{r,\text{pool}}}{\|\boldsymbol{c}_{f,\text{pool}}\|^2} \cdot \boldsymbol{c}_{f,\text{pool}}$
    - **自适应权重**：通过余弦相似度驱动的 sigmoid 函数 $\delta(\gamma) = \frac{s}{1 + e^{-p(\gamma - \epsilon)}}$ 动态调节插值强度，最终嵌入为 $\boldsymbol{c}_{\text{pool}} = (1 - \lambda_{\text{pool}}) \cdot \boldsymbol{c}_{f,\text{pool}} + \lambda_{\text{pool}} \cdot \delta(\gamma) \cdot \Delta_r$
    - **设计动机**：R2F 在罕见/频繁 prompt 之间迭代切换嵌入导致语义不连续；PEM 通过正交分解提取罕见特有方向并自适应融合，提供稳定且解耦的全程引导

3. **潜空间操控（LSM）**：

    - **功能**：为语义差异较大的概念对提供属性级别的方向引导
    - **核心思路**：从 LLM 提取属性文本（如"made of steel"），计算属性嵌入在注意力层输出中的正交分量：
    $l'_\theta(x_t, \boldsymbol{c}_{\text{attr}}, t) = l_\theta(x_t, \boldsymbol{c}_{\text{attr}}, t) - \frac{l_\theta(x_t, \boldsymbol{c}_{\text{attr}}, t) \cdot l_\theta(x_t, \boldsymbol{c}_\phi, t)}{\|l_\theta(x_t, \boldsymbol{c}_\phi, t)\|^2} \cdot l_\theta(x_t, \boldsymbol{c}_\phi, t)$
    - 最终表示为 $\hat{l}_\theta = l_\theta(x_t, \tilde{\boldsymbol{c}}_t, t) + \lambda_{\text{attr}} \cdot l'_\theta(x_t, \boldsymbol{c}_{\text{attr}}, t)$
    - **设计动机**：当频繁和罕见 prompt 差异极大时（如"金属类人形体" vs "钢铁小丑"），PEM 的嵌入层操控不足；LSM 在特征层面注入更细粒度的属性引导

### 训练策略

ADAPT 是完全 **training-free** 的框架：
- 超参数：$\tau_s = 0.025$，$\lambda_{\text{pool}} = 0.3$，$(s, p, \epsilon) = (2.0, 100, 0.93)$，$\lambda_{\text{attr}} = 0.15$
- 推理步数：$T = 50$，使用固定随机种子 42
- 所有实验在单张 NVIDIA A6000 GPU 上完成

## 实验关键数据

### 主实验

在 RareBench 基准上使用 GPT-4o 评估文本-图像对齐性能：

| 方法 | Property | Shape | Texture | Action | Single Complex | Concat | Relation | Multi Complex | **Avg** |
|------|----------|-------|---------|--------|---------------|--------|----------|--------------|---------|
| SD3.0 | 49.4 | 76.3 | 53.1 | 71.9 | 65.0 | 55.0 | 51.2 | 70.0 | 61.5 |
| FLUX | 58.1 | 71.9 | 47.5 | 52.5 | 60.0 | 55.0 | 48.1 | 70.3 | 57.9 |
| Attend & Excite | 55.0 | 38.8 | 33.8 | 23.1 | 36.9 | 23.1 | 24.4 | 36.3 | 33.9 |
| R2F (SD3) | 89.4 | 79.4 | 81.9 | 80.0 | 72.5 | 70.0 | 58.8 | 73.8 | 75.7 |
| **ADAPT (Ours)** | **96.3** | **88.8** | **83.8** | **81.9** | **79.4** | **76.9** | **75.0** | **82.5** | **83.1** |

ADAPT 在所有类别上均超越 R2F，平均提升 **+7.4%**，其中 Single Shape +9.4，Multi Relation +16.2 提升最为显著。

### 消融实验

各模块的增量贡献（Table 2）：

| 方法 | Property | Shape | Action | Avg |
|------|----------|-------|--------|-----|
| R2F (SD3) | 89.4 | 79.4 | 80.0 | 75.7 |
| + PEM (w/o Adaptive) | 90.0 | 84.4 | 71.9 | 78.4 |
| + PEM | 92.5 | 91.3 | 69.4 | 79.8 |
| + PEM + LSM | 92.5 | 91.3 | 71.9 | 80.4 |
| + PEM + APS | 96.3 | 88.8 | 77.5 | 80.7 |
| + PEM + LSM + APS (Full) | 96.3 | 88.8 | 81.9 | **83.1** |

注意力分数提取策略对比（Table 4）：使用所有 token（不含 SOS）的策略最优（Avg 83.1），表明对全体 token 的注意力监控比仅监控名词或罕见短语更有效。

### 关键发现

- PEM 的自适应权重（基于余弦相似度）比固定权重提升 +1.4%，验证了自适应尺度的必要性
- APS 在 PEM 基础上额外带来约 +1% 的提升，且消除了 GPT-4o 依赖
- LSM 主要在语义差异大的概念对上发挥作用（如 Action/Texture 类别）

## 亮点与洞察

1. **注意力收敛即语义饱和**：发现空间注意力收敛可作为概念生成完成度的指示器，这一洞察具有普遍性
2. **正交投影解耦语义**：在 CLIP 嵌入空间用 Gram-Schmidt 正交化提取罕见特有方向，思路简洁优雅
3. **多层次操控互补**：PEM（嵌入层）+ LSM（特征层）+ APS（时间调度层）三个层面完整覆盖
4. **完全 training-free**：作为即插即用的推理增强方案，实用性强

## 局限与展望

- 依赖 SD3 架构的 MM-DiT 设计，对其他架构（如 UNet-based）的适配未验证
- 仍需 LLM（GPT-4o）进行概念映射和属性提取，只是消除了其在"视觉细节评分"上的依赖
- 超参数（$\tau_s$, $\lambda_{\text{pool}}$, $\lambda_{\text{attr}}$ 等）跨模型/跨任务的鲁棒性未充分讨论
- 计算开销（额外的注意力分数提取和正交投影）未量化

## 相关工作与启发

- **R2F**：本文的直接前身，提出频繁-罕见概念配对+调度的范式
- **Attend & Excite**：利用交叉注意力增强 token 绑定，但不适用于罕见概念
- **SynGen**：改善属性绑定但难以处理极端罕见组合
- **启发**：正交分解在嵌入空间中解耦语义的思路，可推广到其他需要精细概念控制的任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 正交解耦+注意力驱动调度的组合有创新性，但核心idea建立在R2F框架之上
- 实验充分度: ⭐⭐⭐⭐ RareBench单一基准上的全面消融，但缺少其他benchmark和用户研究的主文展示
- 写作质量: ⭐⭐⭐⭐ 动机清晰、方法叙述完整，公式推导清楚
- 价值: ⭐⭐⭐⭐ 对罕见概念生成有实际帮助，training-free特性增强了实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [\[CVPR 2026\] Precise Object and Effect Removal with Adaptive Target-Aware Attention](precise_object_and_effect_removal_with_adaptive_target-aware_attention.md)
- [\[ECCV 2024\] HybridBooth: Hybrid Prompt Inversion for Efficient Subject-Driven Generation](../../ECCV2024/image_generation/hybridbooth_hybrid_prompt_inversion_for_efficient_subject-driven_generation.md)
- [\[CVPR 2026\] OPRO: Orthogonal Panel-Relative Operators for Panel-Aware In-Context Image Generation](opro_orthogonal_panel-relative_operators_for_panel-aware_in-context_image_genera.md)
- [\[CVPR 2026\] Verify Claimed Text-to-Image Models via Boundary-Aware Prompt Optimization](verify_claimed_text-to-image_models_via_boundary-aware_prompt_optimization.md)

</div>

<!-- RELATED:END -->
