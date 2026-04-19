---
title: >-
  [论文解读] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization
description: >-
  [ICCV 2025][LLM对齐][视频定制化] 提出 MagicID 框架，通过构建身份偏好和动态偏好的混合视频对数据，并设计两阶段混合偏好优化（HPO）训练策略，首次将 DPO 应用于身份定制化视频生成，同时解决传统自重建训练导致的身份退化和动态减弱问题。
tags:
  - ICCV 2025
  - LLM对齐
  - 视频定制化
  - 身份一致性
  - 偏好优化
  - 扩散模型
  - 混合采样
---

# MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization

**会议**: ICCV 2025  
**arXiv**: [2503.12689](https://arxiv.org/abs/2503.12689)  
**代码**: [echopluto.github.io/MagicID-project](https://echopluto.github.io/MagicID-project/)  
**领域**: 视频生成 / 偏好对齐  
**关键词**: 视频定制化, 身份一致性, 偏好优化, 扩散模型, 混合采样

## 一句话总结

提出 MagicID 框架，通过构建身份偏好和动态偏好的混合视频对数据，并设计两阶段混合偏好优化（HPO）训练策略，首次将 DPO 应用于身份定制化视频生成，同时解决传统自重建训练导致的身份退化和动态减弱问题。

## 研究背景与动机

### 问题定义

视频身份定制化旨在给定少量用户参考图像，生成保持身份一致且具有显著动态效果的高保真视频。与图像定制化相比，视频定制化面临的核心难题在于参考输入是静态图像而非视频。

### 已有方法的不足

现有方法（MagicMe、DreamBooth）沿用图像定制化的自重建训练范式，即在参考图像上进行自重建学习来保持身份，但引入两个严重问题：

**身份退化随帧数增加而恶化**：参考图像本质上是单帧，与多帧视频之间存在时间分辨率的固有差异。自重建训练无法弥合这一域偏移，导致生成更多帧时身份一致性明显下降

**动态性随训练推进而减弱**：自重建仅重建静态图像，驱动模型生成越来越"静"的视频。实验显示，随着定制化训练步数增加，生成视频的动态度持续下降

### 核心 idea

用偏好优化替代自重建训练：构建"好/差"视频对，让模型直接学习生成身份一致且动态丰富的视频。通过混合采样策略分阶段解决身份和动态两个目标之间的冲突。

## 方法详解

### 整体框架

MagicID 包含三个阶段：
1. 初始 LoRA 微调（1000步，传统自重建）
2. 偏好数据构建（生成视频 + 评估奖励 + 混合配对选择）
3. 混合偏好优化训练（5000步，HPO 损失）

基座模型为 HunyuanVideo（T2V DiT 模型）。

### 关键设计

#### 1. **偏好视频数据生成（Preference Data Generation）**
- **功能**：构建包含身份一致性和动态度差异的视频对
- **核心思路**：从三个来源构建基础视频库 $\mathcal{B}$：
    - $V_t$：LoRA 微调后模型生成的视频（有身份信息但可能不完美）
    - $V_s$：原始 T2V 模型（无 LoRA）生成的视频（保留原始动态分布）
    - $V_{id}$：参考图像扩展的静态视频（身份完美但零动态）
- **设计动机**：仅用生成样本配对会限制模型学习不可见参考图像中的身份信息，加入静态视频作为身份"锚点"

#### 2. **定制化视频奖励（Customized Video Reward）**
- **功能**：用三个维度评估视频质量
- **核心思路**：
    - $R_{id}$（身份一致性）：用预训练 ArcFace 编码器计算与参考图像的面部相似度
    - $R_{dy}$（动态度）：用 RAFT 光流模型分析连续帧间的运动强度
    - $R_{sem}$（语义对齐）：用 VLM 评估视频与文本提示的语义匹配度
    - 所有分数归一化到1-10
- **设计动机**：身份、动态和语义三个维度互相制约，需要明确量化才能进行偏好配对

#### 3. **混合配对选择（Hybrid Pair Selection）**
- **功能**：分两阶段构建偏好视频对
- **核心思路**：
    - **阶段1（身份优先）**：从 $V_{id}$ 和 $V_s$ 中选择身份一致性差异大的视频对 $P_{id}$，对动态度差异容忍
    - **阶段2（动态优先）**：从 $V_s$ 和 $V_t$ 中使用 Pareto 前沿采样方法选择动态偏好对 $P_{dy}$
    - 使用非支配排序算法找到上/下 Pareto 前沿
    - 按身份一致性差异排序，保留 Top-100 对
    - 最终 $P = P_{dy} \cup P_{id}$
- **设计动机**：同时追求身份和动态两个目标存在冲突，Pareto 前沿采样确保选择的视频对在两个维度上都具有区分性

### 损失函数 / 训练策略

混合偏好优化（HPO）损失：

$$\mathcal{L}_{\text{HPO}}(\theta) = \mathbb{E}_{(v_0^w, v_0^l) \sim \mathcal{D}, t \sim \{1..T\}} \left[ \beta \log \sigma \left( \left(\|\epsilon_w - \epsilon_\theta(v_t^w, t)\|^2 - \|\epsilon_w - \epsilon_{\text{ref}}(v_t^w, t)\|^2\right) - \left(\|\epsilon_l - \epsilon_\theta(v_t^l, t)\|^2 - \|\epsilon_l - \epsilon_{\text{ref}}(v_t^l, t)\|^2\right) \right) \right]$$

- 通过 ELBO 和 Jensen 不等式将高维视频序列概率转化为噪声预测误差的比较
- 基座模型 HunyuanVideo + LoRA，AdamW 优化器，学习率2e-5
- 总训练6000步（初始1000步自重建 + 5000步 HPO）

## 实验关键数据

### 主实验

| 方法 | Face Sim.↑ | Dyna. Deg.↑ | T. Cons.↑ | CLIP-T↑ | FVD↓ |
|------|-----------|------------|-----------|---------|------|
| DreamBooth | 0.276 | 5.690 | 0.9922 | 25.83 | 1423.55 |
| MagicMe | 0.322 | 5.332 | 0.9924 | 25.42 | 1438.66 |
| IDAnimator | 0.433 | 10.33 | 0.9938 | 25.21 | 1558.33 |
| ConsisID | 0.482 | 9.26 | 0.9811 | 26.12 | 1633.21 |
| **MagicID** | **0.600** | **14.42** | **0.9933** | **26.28** | **1228.33** |

### 消融实验（混合配对选择）

| 配置 | Face Sim.↑ | Dynamic↑ | CLIP-T↑ | 说明 |
|------|-----------|----------|---------|------|
| 无偏好（自重建）| 0.276 | 5.690 | 25.83 | 基线 DreamBooth |
| + 身份偏好对 | 0.605 | 7.382 | 25.94 | 身份大幅提升 |
| + 身份 + 动态偏好对 | 0.600 | **14.42** | **26.28** | 动态翻倍提升 |

### 消融实验（奖励组合）

| ID | Dynamic | Semantic | Face↑ | Dynamic↑ | CLIP-T↑ |
|----|---------|----------|-------|----------|---------|
| ✓ | | | 0.598 | 6.332 | 24.92 |
| ✓ | ✓ | | 0.607 | 12.33 | 25.73 |
| ✓ | ✓ | ✓ | 0.600 | **14.42** | **26.28** |

### 关键发现

1. **偏好优化显著优于自重建**：Face Similarity 从0.276提升到0.600（+117%），Dynamic Degree 从5.69提升到14.42（+153%）
2. **两阶段混合策略的互补性**：身份偏好阶段专注提升 Face Sim，动态偏好阶段在不损害身份的前提下将动态度翻倍
3. **三个奖励维度各有贡献**：动态奖励对运动质量提升最大，语义奖励额外改善了提示跟随能力
4. **MagicID 不需要大规模视频训练数据**：与 IDAnimator 和 ConsisID（需要数千高质量人物视频）不同，MagicID 仅需少量参考图像

## 亮点与洞察

1. **首次将 DPO 应用于身份定制化视频生成**：清晰地诊断了自重建训练的两个核心缺陷（身份退化 + 动态减弱），并用偏好优化提供了原理性解决方案
2. **Pareto 前沿采样方法**：在多目标优化场景中选择偏好对的创新方法，确保两个维度的平衡
3. **静态视频作为身份锚点**：将参考图像扩展为静态视频加入偏好库的思路简单有效
4. **分析深入**：通过帧数-身份一致性和训练步数-动态度的定量分析，直观展示了自重建训练的问题

## 局限与展望

1. **仅支持单人定制化**：无法生成包含多个自定义身份的视频
2. **需要预训练的身份编码器**：ArcFace 的质量直接影响偏好数据的构建
3. **偏好数据构建成本不低**：需要生成大量候选视频并逐一评估奖励
4. **仅在 HunyuanVideo 上验证**：对其他 T2V 基座模型的泛化性未知
5. **生成61帧视频**：更长视频的身份保持能力需要进一步验证

## 相关工作与启发

- DreamBooth 和 MagicMe 代表了传统自重建方案的局限
- ConsisID 使用 face adapter 编码身份信息，但出现 copy-paste 伪影（不自然的运动+静态贴图效果）
- HuViDPO 将 DPO 引入通用 T2V 生成，本文将其扩展到具有特殊约束的定制化场景
- Pareto 前沿采样的思路可推广到其他多目标视频优化任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将 DPO 用于视频身份定制化，混合采样策略设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 定量+定性+用户研究+消融均有，但基线方法数量偏少
- **写作质量**: ⭐⭐⭐⭐ — 问题诊断清晰，两阶段框架逻辑自然
- **价值**: ⭐⭐⭐⭐ — 为视频定制化提供了新的训练范式，身份+动态的双重提升具有实际应用价值

<!-- RELATED:START -->

## 相关论文

- [Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](../../ACL2025/llm_alignment/probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](../../NeurIPS2025/llm_alignment/mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [D-Fusion: Direct Preference Optimization for Aligning Diffusion Models with Visually Consistent Samples](../../ICML2025/llm_alignment/d-fusion_direct_preference_optimization_for_aligning_diffusion_models_with_visua.md)
- [Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](../../ICLR2026/llm_alignment/uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [DenseDPO: Fine-Grained Temporal Preference Optimization for Video Diffusion Models](../../NeurIPS2025/llm_alignment/densedpo_finegrained_temporal_preference_optimization_for_vi.md)

<!-- RELATED:END -->
