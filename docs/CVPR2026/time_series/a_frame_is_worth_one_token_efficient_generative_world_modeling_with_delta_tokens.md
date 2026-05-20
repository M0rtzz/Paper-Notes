---
title: >-
  [论文解读] A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens
description: >-
  [CVPR 2026][时间序列][世界模型] 提出 DeltaTok 将连续帧的 VFM 特征差压缩为单个 delta token，配合 Best-of-Many 训练的 DeltaWorld 在单次前向传播中高效生成多样化未来预测，参数量仅为 Cosmos 的 1/35、FLOPs 仅为 1/2000…
tags:
  - "CVPR 2026"
  - "时间序列"
  - "世界模型"
  - "Delta token"
  - "视频预测"
  - "帧差压缩"
  - "Best-of-Many训练"
---

# A Frame is Worth One Token: Efficient Generative World Modeling with Delta Tokens

**会议**: CVPR 2026  
**arXiv**: [2604.04913](https://arxiv.org/abs/2604.04913)  
**代码**: [deltatok.github.io](https://deltatok.github.io)  
**领域**: 时序预测 / 世界模型  
**关键词**: 世界模型, Delta token, 视频预测, 帧差压缩, Best-of-Many训练

## 一句话总结
提出 DeltaTok 将连续帧的 VFM 特征差压缩为单个 delta token，配合 Best-of-Many 训练的 DeltaWorld 在单次前向传播中高效生成多样化未来预测，参数量仅为 Cosmos 的 1/35、FLOPs 仅为 1/2000，但在密集预测任务上表现更优。

## 研究背景与动机
**领域现状**：世界模型需要预测未来状态以支持自主决策（如自动驾驶）。未来固有地不确定，模型需要生成多个可能的未来。

**现有痛点**：
   - **判别式世界模型**：产出单一决定性预测，在不确定情况下坍缩为条件均值，无法捕获多样未来
   - **现有生成式世界模型**（如Cosmos）效率低下，因为：(i) 面向像素重建优化，而非语义理解；(ii) 需要多次前向传播生成单个假设；(iii) 未利用帧间时空冗余

**关键洞察**：自然视频中，连续帧之间差异是结构化的、通常低维的——背景静止，仅小部分场景变化。表示整帧为密集特征图导致大量冗余。

**核心idea**：只编码帧间变化（delta）而非整帧，将视频从3D时空表示压缩为1D时间序列。

## 方法详解

### 整体框架
冻结VFM（DINOv3）提取帧级特征 → DeltaTok编码器将相邻帧差压缩为单个delta token → DeltaWorld预测器在delta token序列上做生成预测 → DeltaTok解码器恢复空间特征图 → 下游任务头。

### 关键设计
1. **DeltaTok（帧差压缩器）**：

    - 编码器：$z_t = g(x_{t-1}, x_t, z_{\text{init}}) \in \mathbb{R}^D$，将前后帧特征压缩为单个delta token
    - 解码器：$\hat{x}_t = h(x_{t-1}, z_t)$，利用前帧+delta token重建当前帧特征
    - Token压缩率：$512 \times 512$ 分辨率下 $1024 \times$ 压缩（从 $32 \times 32 = 1024$ token到1个token）
    - 训练：MSE重建损失 $L_{\text{tok}} = \|x_t - \hat{x}_t\|^2$
    - 设计动机：delta天然具有低维性——预测"无变化"即保留前帧，模型只需学习变化的部分。这比压缩整帧容易得多，信息密度更高。

2. **Best-of-Many (BoM) 训练**：
   采样 $K$ 个高斯噪声查询：$q^k \sim \mathcal{N}(\mu, \Sigma)$，预测 $K$ 个未来假设，仅监督最接近GT的那个：
    $k^\star = \arg\min_k \sum_{h,w} \ell(x_{t+1,h,w}, \hat{x}^k_{t+1,h,w})$
    $L_{\text{BoM}} = \sum_{h,w} \ell(x_{t+1,h,w}, \hat{x}^{k^\star}_{t+1,h,w})$
    - 设计动机：不同噪声查询映射到不同未来模式。只需单次前向传播即可采样多个未来，避免扩散模型的迭代去噪。与delta token结合使 BoM 代价可忽略。

3. **DeltaWorld 完整管线**：

    - 预测器在delta token序列上工作：$\hat{z}_{t+1} = f(q^k, Z_{1:t}, T_{1:t}, \tau_{t+1})$
    - BoM损失直接在delta token空间计算，无需解码
    - 自回归rollout：逐步追加预测的delta token到上下文窗口
    - 第一帧用黑背景帧差表示绝对特征

### 损失函数 / 训练策略
- DeltaTok 单独训练 50K 迭代
- DeltaWorld 预测器训练 300K 迭代 + 5K 低学习率微调
- $K = 256$（BoM 训练时采样数），评估时采样20个
- VFM: DINOv3 ViT-B，预测器也用 ViT-B

## 实验关键数据

### 主实验

| 方法 | GFLOPs↓ | VSPW mIoU (Mid) | Cityscapes mIoU (Mid) | KITTI RMSE (Mid) |
|------|---------|------|------|------|
| DINO-world (判别式) | 5.8K | 47.9 | 49.8 | 4.07 |
| Cosmos-4B | 60M | 47.0 (44.5) | 49.1 (48.4) | 4.08 (4.14) |
| Cosmos-12B | 64M | 47.7 (45.5) | 53.3 (51.2) | 4.01 (4.14) |
| **DeltaWorld** | **31K** | **50.1 (46.7)** | **55.4 (51.3)** | **3.88 (4.17)** |

*括号内为mean，括号外为best-of-20*

### 消融实验（渐进式设计验证）

| 步骤 | GFLOPs | VSPW best(mean) | Cityscapes best(mean) | 说明 |
|------|--------|------|------|------|
| (0) 判别式基线 | 959 | 44.8 | 45.4 | 均值预测 |
| (1) +BoM | 12013 | 47.0 (39.4) | 46.8 (31.1) | best提升但mean崩 |
| (2) +帧压缩 | 6315 | 45.7 (40.3) | 42.7 (35.5) | 效率提升但精度不足 |
| (3) +Delta压缩 | 6721 | **46.8 (44.4)** | **48.7 (45.5)** | mean恢复到基线水平 |

### 关键发现
- DeltaWorld的best预测全面超越Cosmos（不论4B还是12B），FLOPs却仅为1/2000
- Delta压缩 vs 帧压缩：mean从35.5恢复到45.5（Cityscapes），证明delta的容量效率远超整帧
- Delta的自然先验：预测"无变化" = 保留前帧，模型不需要重新编码静态背景
- Best-of-Many的K增大 → best持续提升且不牺牲mean（K=64后mean稳定）
- 预测器在delta空间仅占推理FLOPs的0.5%

## 亮点与洞察
- **极端压缩+高质量**：512×512帧压缩为1个token（1024×），且可重建
- **单次前向传播生成多样未来**：彻底避免扩散模型的迭代去噪
- **delta先验优雅**：连续帧差的低维结构与"无变化即保留"完美匹配世界模型需求
- **mean恢复到判别式水平**是重要验证：多样性没有以牺牲合理性为代价

## 局限与展望
- 当场景变化剧烈（场景切换），delta token 可能不够用（虽可退化为绝对编码）
- 自回归rollout中误差可能累积
- 当前仅在15M参数级别验证，扩展到更大模型的效果待探索
- 侧重实验指标但缺少对生成多样性的定性分析

## 相关工作与启发
- Delta编码思想借鉴经典视频编码（帧间压缩），但首次将其与VFM特征空间结合
- Best-of-Many相比扩散模型的优势在于单次前向——这对实时系统意义重大
- DINO-world → DeltaWorld 的渐进扩展非常教科书式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Delta token化+BoM组合解决了高效生成式世界模型的核心要求
- 实验充分度: ⭐⭐⭐⭐⭐ 渐进消融+3数据集+效率分析非常透彻
- 写作质量: ⭐⭐⭐⭐⭐ 渐进式展示从判别到高效生成的路径，极其清晰
- 价值: ⭐⭐⭐⭐⭐ 为自动驾驶等场景提供了实用的多假设预测方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CoMeT: Collaborative Memory Transformer for Efficient Long Context Modeling](../../ACL2026/time_series/comet_collaborative_memory_transformer_for_efficient_long_context_modeling.md)
- [\[ICCV 2025\] I²-World: Intra-Inter Tokenization for Efficient Dynamic 4D Scene Forecasting](../../ICCV2025/time_series/i2-world_intra-inter_tokenization_for_efficient_dynamic_4d_scene_forecasting.md)
- [\[ICLR 2026\] T1: One-to-One Channel-Head Binding for Multivariate Time-Series Imputation](../../ICLR2026/time_series/t1_one-to-one_channel-head_binding_for_multivariate_time-series_imputation.md)
- [\[ACL 2026\] Jacobian Scopes: Token-Level Causal Attributions in LLMs](../../ACL2026/time_series/jacobian_scopes_token-level_causal_attributions_in_llms.md)
- [\[CVPR 2026\] PFGNet: A Fully Convolutional Frequency-Guided Peripheral Gating Network for Efficient Spatiotemporal Predictive Learning](pfgnet_a_fully_convolutional_frequency-guided_peripheral_gating_network_for_effi.md)

</div>

<!-- RELATED:END -->
