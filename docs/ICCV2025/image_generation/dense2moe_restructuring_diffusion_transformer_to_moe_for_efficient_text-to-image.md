---
title: >-
  [论文解读] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation
description: >-
  [ICCV 2025][图像生成][扩散模型] 首次提出将密集型扩散Transformer（DiT）转化为MoE稀疏结构的范式Dense2MoE，通过FFN替换为MoE层+Transformer块分组为MoB（Mixture of Blocks），配合多阶段蒸馏流水线，将FLUX.1的12B参数压缩至5.2B激活参数同时保持原始性能，全面超越剪枝方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "扩散模型"
  - "Mixture of Experts"
  - "模型压缩"
  - "知识蒸馏"
  - "FLUX"
---

# Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation

**会议**: ICCV 2025  
**arXiv**: [2510.09094](https://arxiv.org/abs/2510.09094)  
**代码**: 无  
**领域**: 图像生成  
**关键词**: 扩散模型, Mixture of Experts, 模型压缩, 知识蒸馏, FLUX

## 一句话总结

首次提出将密集型扩散Transformer（DiT）转化为MoE稀疏结构的范式Dense2MoE，通过FFN替换为MoE层+Transformer块分组为MoB（Mixture of Blocks），配合多阶段蒸馏流水线，将FLUX.1的12B参数压缩至5.2B激活参数同时保持原始性能，全面超越剪枝方法。

## 研究背景与动机

扩散Transformer（DiT）在文生图领域取得卓越性能，但模型规模急剧膨胀——FLUX.1有120亿参数，是SD1.5的13.8倍，带来巨大的推理开销。现有模型压缩主要依赖剪枝，但高压缩率下剪枝会导致严重的性能退化——因为剪枝减少了模型的总参数量，从根本上限制了模型容量。

本文的核心洞察：**能否在不减少总参数量的前提下，减少每次推理的激活参数量？** MoE架构允许不同输入激活不同参数子集，既减少计算量又保持模型容量。关键观察是：FFN占DiT总参数近50%——非常适合MoE化；DiT块的重要性随时间步和prompt显著变化——适合动态激活。

## 方法详解

### 整体框架

Dense2MoE包含两层稀疏化设计和三阶段蒸馏流水线：
- **MoE层**：替换FFN为MoE层（共享专家 + 多个普通专家），减少每个块内的激活参数
- **MoB组**：将连续Transformer块分组，动态路由选择性激活部分块，减少模型深度
- **蒸馏流水线**：增强初始化 → MoE蒸馏 → MoB蒸馏

### 关键设计

1. **FFN→MoE替换**: 用共享专家（expansion ratio $r_s$）和 $n$ 个普通专家（expansion ratio $r_n$ ）替换原始FFN。约束 $r = r_s + n \cdot r_n$ 保持总参数不变。推理时每个token仅选top-k个普通专家激活，激活扩展比 $r_a = r_s + k \cdot r_n$。默认配置：$r_s=1, r_n=0.25, n=12, k=2$，FFN激活参数压缩率为 $r_a/r = 1.5/4 = 37.5\%$（压缩62.5%）。前向过程：
    $y^{(t)} = \text{MLP}_s(x^{(t)}) + \sum_{i=1}^k g(x^{(t)},i) \cdot \text{MLP}_n^{(i)}(x^{(t)})$

2. **Mixture of Blocks (MoB)**: 将连续 $m$ 个Transformer块分组为MoB组，推理时仅激活相邻的 $\kappa$ 个块。路由器设计利用AdaLN的全局嵌入 $y$（融合了text和timestep条件）：
    $\text{TopK}(\alpha W_x[x^{(p)}, c^{(p)}] + (1-\alpha)W_y y, \kappa)$
   这使路由能显式感知文本和时间步条件，实现动态的深度压缩。

3. **三阶段蒸馏流水线**:

    - **增强初始化**: 用一阶Taylor度量 $\mathcal{I}_i = |\frac{\partial\mathcal{L}}{\partial w_i}w_i|$ 评估权重重要性，将重要权重分配给共享专家，其余均分给普通专家。再对仅含共享专家的模型做KD提升初始化质量
    - **MoE蒸馏**: 冻结共享专家，激活普通专家和门控网络，使用输出蒸馏损失 + 块特征损失 + 负载平衡损失训练
    - **MoB蒸馏**: 设计组特征损失——用原始模型中对应MoB组最后一个块的输出作为教师特征，与MoB组输出对齐

### 损失函数 / 训练策略

- **输出蒸馏损失**: $\mathcal{L}_{distill} = \mathbb{E}\|f_{tea} - f_{stu}\|_2^2$
- **块特征损失**: $\mathcal{L}_{feature} = \sum_{l=1}^L w_l \|f_{tea}^{(l)} - f_{stu}^{(l)}\|_2^2$，特征权重经归一化确保各层学习稳定
- **负载平衡损失**: 防止路由坍塌到少数专家，$\lambda_{balance} = 10^{-2}$
- 训练配置: 32×A100 GPU，全局batch size 64
- 训练数据: Laion-5B + Coyo-700M + JourneyDB
- 在FLUX.1 [dev]基础上构建4个稀疏度级别：L(5.2B)/M(4B)/S(3.2B)/XS(2.6B)

## 实验关键数据

### 主实验 (与剪枝方法对比)

| 模型 | 激活参数(B) | FLOPs(T) | GenEval↑ | DPG↑ | CLIP↑ | IR↑ |
|------|-----------|---------|---------|------|-------|-----|
| FLUX.1 [dev] | 11.90 | 66.00 | 0.6595 | 83.42 | 32.24 | 0.9656 |
| FLUX.1-Lite (剪枝) | 8.16 | 53.15 | 0.5229 | 79.00 | 31.79 | 0.8380 |
| FLUX-Mini (剪枝) | 3.18 | 17.37 | 0.3209 | 69.34 | 29.94 | 0.2151 |
| **FLUX.1-MoE-L** | **5.15** | **43.42** | **0.5702** | **81.63** | **31.39** | **0.8011** |
| **FLUX.1-MoE-S** | **3.19** | **26.43** | **0.4441** | **75.61** | **30.67** | **0.5942** |
| **FLUX.1-MoE-XS** | **2.64** | **20.26** | **0.4036** | **73.66** | **30.40** | **0.5076** |

MoE-L比FLUX.1-Lite少3B激活参数但性能更优；MoE-S/XS在相似参数下大幅超越FLUX-Mini。

### 消融实验 (MLP剪枝 vs MoE / 深度剪枝 vs MoB)

**MLP Pruning vs FFN-to-MoE**:

| 方法 | 激活扩展比 | GenEval | DPG |
|------|----------|---------|-----|
| Diff-Pruning (r=1.5) | 1.5 | 0.4113 | 72.23 |
| Diff-Pruning (r=2.0) | 2.0 | 0.4888 | 77.53 |
| **MoE (r=1.5)** | **1.5** | **0.5728** | **81.24** |

MoE在62.5%压缩率下仍优于剪枝50%压缩率的结果。

**Depth Pruning vs MoB (相同激活块数)**:

| 方法 | D块 | S块 | GenEval | DPG |
|------|-----|-----|---------|-----|
| Lite | 9 | 26 | 0.0926 | 41.62 |
| BK-SDM | 9 | 26 | 0.3450 | 66.86 |
| **MoB** | **9** | **26** | **0.4956** | **76.51** |

MoB在高压缩率下优势更加显著——Lite方法在深压缩后几乎崩溃。

### 关键发现

- **Taylor度量初始化显著提升共享专家质量**: 相比随机分割，各指标稳定提升
- **两阶段分离训练（共享→全MoE）优于联合训练**: 独立优化策略更佳
- **更多MoB组 = 更好的蒸馏效果**: 因为提供更多的组特征对齐监督层
- **专家特化分析**: 双流块中图像分支的MoE呈现空间特化（不同专家负责不同空间区域）；同prompt类别的专家选择模式相似；高噪声阶段专家选择更集中
- **支持动态Top-K**: 由于蒸馏流水线设计，不用额外训练即可动态调整激活专家数量。Top-K=0时只用共享专家也能生成合理图像，增加K则提升细节和真实感
- HyperFLUX加速与MoE兼容：MoE-L+8步采样仍保持强性能

## 亮点与洞察

- **范式创新**: 首次将Dense-to-MoE范式引入扩散模型，思路是"保容量减计算"vs剪枝的"减容量减计算"
- **MoB概念新颖**: 不同于token级MoE路由，MoB在特征级做块路由，天然适配扩散模型的多步去噪过程
- **三阶段蒸馏设计精巧**: Taylor初始化→共享专家蒸馏→MoE蒸馏→MoB蒸馏，逐步构建
- **对FLUX实用**: 12B→5.2B激活参数且性能不降，对实际部署有直接价值

## 局限与展望

- 虽然激活参数减少60%，但总参数量（9B）仍然很大，显存占用未必显著降低
- MoE推理需要专门的grouped GEMM kernel支持（如MegaBlocks），通用GPU上可能不如预期的加速
- 延迟改善相对有限（21.2s→17.8s，仅约16%），因为attention部分未压缩
- 仅在FLUX.1上验证，未在其他DiT（如SD3、PixArt）上实验
- 蒸馏训练需要32×A100，成本不低

## 相关工作与启发

- LLM领域的Dense-to-MoE工作（如MoE化LLM）为本文提供直接灵感
- Mixture of Depths方法让token自适应跳过层，与MoB思路相关但实现不同
- DiT-MoE在训练时就使用MoE扩展到16B，本文是从已有dense模型转化
- 步蒸馏方法（HyperFLUX）与模型压缩可正交组合

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次Dense-to-MoE用于扩散模型，MoB设计新颖
- 实验充分度: ⭐⭐⭐⭐ 与多种剪枝方法对比、详细消融、专家特化分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，可视化丰富
- 价值: ⭐⭐⭐⭐⭐ 为高效扩散模型开辟新范式，实用性和学术价值兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DICE: Staleness-Centric Optimizations for Parallel Diffusion MoE Inference](dice_staleness-centric_optimizations_for_parallel_diffusion_moe_inference.md)
- [\[ICCV 2025\] LiT: Delving into a Simple Linear Diffusion Transformer for Image Generation](lit_delving_into_a_simple_linear_diffusion_transformer_for_image_generation.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[CVPR 2025\] DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](../../CVPR2025/image_generation/dit-ic_aligned_diffusion_transformer_for_efficient_image_compression.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](../../NeurIPS2025/image_generation/sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
