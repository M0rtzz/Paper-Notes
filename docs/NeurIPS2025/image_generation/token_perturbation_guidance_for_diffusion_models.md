---
title: >-
  [论文解读] Token Perturbation Guidance for Diffusion Models
description: >-
  [NeurIPS 2025][图像生成][无训练引导] 提出 Token Perturbation Guidance（TPG），通过对扩散模型中间 token 表示进行保范数的 shuffling 扰动来构造负分数信号，实现无需训练的条件无关引导，在无条件生成中将 SDXL 的 FID 提升近 2 倍，在条件生成中接近 CFG 效果。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "无训练引导"
  - "Token 扰动"
  - "无条件生成"
  - "CFG 替代"
  - "扩散模型"
---

# Token Perturbation Guidance for Diffusion Models

**会议**: NeurIPS 2025  
**arXiv**: [2506.10036](https://arxiv.org/abs/2506.10036)  
**代码**: [https://github.com/TaatiTeam/Token-Perturbation-Guidance](https://github.com/TaatiTeam/Token-Perturbation-Guidance)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 无训练引导, Token 扰动, 无条件生成, CFG 替代, 扩散模型

## 一句话总结

提出 Token Perturbation Guidance（TPG），通过对扩散模型中间 token 表示进行保范数的 shuffling 扰动来构造负分数信号，实现无需训练的条件无关引导，在无条件生成中将 SDXL 的 FID 提升近 2 倍，在条件生成中接近 CFG 效果。

## 研究背景与动机

Classifier-Free Guidance（CFG）是现代扩散模型提升生成质量和条件对齐的核心技术，但存在两个根本限制：(1) 仅适用于条件生成，无法用于无条件生成；(2) 需要特定的训练策略（随机替换条件为空条件）。

已有若干免训练替代方案——SAG（自注意力引导）、PAG（扰动注意力引导）、SEG（平滑能量引导）——通过操作注意力层来构造引导信号，但改进幅度有限，尤其在 prompt 对齐和无条件生成质量上远不及 CFG。

核心观察：**PAG 和 SEG 在去噪早期步骤产生过度平滑的结果**，而早期步骤对建立全局结构和粗语义至关重要。如果这一阶段引导不足，模型可能永远无法从高层语义错配中恢复。这解释了为何现有方法在 prompt 对齐和生成质量上仅有边际改进。

TPG 的切入点：直接在扩散网络的中间 token 表示上施加**保范数的 shuffling 扰动**，构造比注意力操作更有效的引导信号。

## 方法详解

### 整体框架

令 H∈R^{B×N×C} 为去噪器的中间隐层表示。TPG 在每个时间步执行两次前向传播：(1) 标准传播得到正分数 s_θ⁺；(2) 在选定层应用 token shuffling 扰动后传播得到负分数 s_θ⁻。引导输出为：

s̃_θ = s_θ⁺ + γ(s_θ⁺ - s_θ⁻)

### 关键设计

1. **Token Shuffling 扰动**：

    - 在 token 维度应用置换矩阵 S∈R^{N×N} 得到 H' = S·H
    - Shuffling 满足三个关键性质：
        - **线性**：可表示为矩阵乘法，计算效率与 CFG 相当
        - **保范数**：S^T·S = I，是正交变换，保持 token 范数不变，避免内部协变量偏移
        - **破坏局部结构同时保持全局统计**：随机重排 token 位置
    - 每个时间步 t 和每层 k 使用**不同的独立随机置换矩阵** S_{k,t}

2. **为何 Shuffling 优于其他保范数扰动**：

    - 对比了 Sign Flip（翻转符号）、Hadamard 变换、Haar 随机正交变换
    - Shuffling 在 FID 上远优于其他：78.43 vs 118-120（5K 样本评估）
    - 原因：Hadamard 和 Haar 将所有 token 混合在一起，可能破坏有用信息；Sign Flip 信号太弱；Shuffling 随机重排但保留可恢复的全局结构

3. **与 CFG 行为的深度对比分析**：

    - **方向分析**：TPG 和 CFG 的引导向量在整个去噪轨迹上与真实噪声几乎正交（余弦值接近 0），而 PAG/SEG 在中间步骤出现强负对齐
    - **范数分析**：TPG 和 CFG 的引导项范数趋势几乎一致（从约 40 开始、后期陡增），而 PAG/SEG 始终保持低范数
    - **频域分析**：TPG 和 CFG 在低频段有轻微正偏向，其余频率保持正交；SEG 在中频出现负条纹，能量比 CFG/TPG 小两个数量级
    - 结论：TPG 在方向、频率内容上最接近 CFG 行为

4. **对 DiT/ViT 架构的兼容性**：

    - U-Net 架构的残差连接帮助恢复扰动 token，但 DiT 的连续 Transformer 层会让退化累积
    - 对于 SD3（DiT 架构）：仅 shuffle 少部分 token 并在每个 Transformer block 后立即 unshuffle
    - PAG 在 SD3 上甚至比 vanilla 更差（FID 138.08 vs 113.86），而 TPG 大幅提升（83.01）

### 损失函数 / 训练策略

- TPG 是**完全免训练**的方法，无需修改模型架构
- 引导尺度 γ 固定为 3.0
- 扰动仅应用于 U-Net 的下采样层（encoder 部分）
- 即插即用，仅需几行代码

## 实验关键数据

### 主实验（SDXL，30K 样本，MS-COCO 2014 验证集）

| 设置 | 方法 | FID↓ | sFID↓ | IS↑ | Aesthetic↑ | CLIP↑ |
|------|------|------|-------|-----|-----------|-------|
| 无条件 | Vanilla SDXL | 124.04 | 78.91 | 9.19 | 5.02 | - |
| 无条件 | PAG | 98.83 | 94.71 | 13.74 | 5.94 | - |
| 无条件 | SEG | 82.64 | 74.98 | 13.22 | 6.15 | - |
| 无条件 | **TPG** | **69.31** | **44.18** | **17.99** | 6.14 | - |
| 条件 | Vanilla SDXL | 48.97 | 43.71 | 22.10 | 5.37 | 27.47 |
| 条件 | CFG | **12.79** | **23.31** | **42.75** | **6.20** | **32.03** |
| 条件 | PAG | 20.49 | 28.78 | 34.66 | 6.11 | 29.67 |
| 条件 | SEG | 23.94 | 31.50 | 30.29 | 6.18 | 29.49 |
| 条件 | **TPG** | 17.77 | 24.32 | 34.89 | 6.12 | 30.15 |

### 消融实验（不同扰动方法，5K 样本）

| 方法 | FID↓ | IS↑ |
|------|------|-----|
| Vanilla（无扰动）| 131.57 | 9.21 |
| Sign Flip | 119.23 | 10.98 |
| Hadamard | 120.54 | 10.34 |
| Haar 随机正交 | 118.47 | 10.75 |
| Token Blurring（非保范数）| 157.67 | 6.70 |
| **Token Shuffling** | **78.43** | **18.26** |

### 关键发现

- 无条件生成：TPG 将 SDXL 的 FID 从 124.04 降至 69.31，接近 2 倍提升
- 条件生成：TPG (FID=17.77) 紧随 CFG (FID=12.79)，大幅超越 PAG (20.49) 和 SEG (23.94)
- 在 SD 2.1 无条件生成上同样取得最佳结果（FID 16.69 vs PAG 21.30 vs SEG 20.98）
- 保范数性质至关重要：Token Blurring（不保范数）反而比 vanilla 更差
- 引导尺度 γ=3 为最优，γ 过大（>4）导致 FID 退化
- 在 SD3（DiT 架构）上，PAG 完全失效但 TPG 仍显著有效（无条件 FID 83.01 vs Vanilla 113.86）

## 亮点与洞察

- **简洁而深刻的核心机制**：用 token shuffling 构造负分数——思路极简但效果卓越，体现了"局部结构破坏+全局统计保持"的引导本质
- **对引导机制的深入频域分析**：揭示了 CFG vs TPG vs PAG/SEG 的本质差异——引导向量应与噪声正交而非反向
- **条件无关的通用引导**：首次将 CFG 级别的引导效果扩展到无条件生成
- **跨架构兼容（U-Net + DiT）**：通过 shuffle-unshuffle 策略适配 DiT 架构

## 局限与展望

- 与 CFG 一样需要两次前向传播，采样时间翻倍
- 条件生成中仍落后于 CFG（FID 17.77 vs 12.79），差距约 5 个 FID 点
- 在极端分布外场景下引导项的效果可能受限
- 最佳引导尺度和层选择需要实验调整

## 相关工作与启发

- CFG 是条件生成的金标准，但本质上是推离无条件分布的分数
- Autoguidance 使用弱版本去噪器构造引导信号，但仍需训练
- PAG（用单位矩阵替换注意力图）和 SEG（高斯模糊注意力图）在实际效果和理论行为上都不如 TPG
- 启发：token 级扰动比注意力层操作能构造更有效的引导信号，为免训练引导方法开辟新方向

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Where and How to Perturb: On the Design of Perturbation Guidance in Diffusion and Flow Models](where_and_how_to_perturb_on_the_design_of_perturbation_guidance_in_diffusion_and.md)
- [\[NeurIPS 2025\] Entropy Rectifying Guidance for Diffusion and Flow Models](entropy_rectifying_guidance_for_diffusion_and_flow_models.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[ICML 2025\] ToMA: Token Merge with Attention for Diffusion Models](../../ICML2025/image_generation/toma_token_merge_with_attention_for_diffusion_models.md)
- [\[NeurIPS 2025\] SparseDiT: Token Sparsification for Efficient Diffusion Transformer](sparsedit_token_sparsification_for_efficient_diffusion_transformer.md)

</div>

<!-- RELATED:END -->
