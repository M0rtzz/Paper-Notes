---
title: >-
  [论文解读] SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding
description: >-
  提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。
tags:

---

# SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding

## 元信息
- **会议**: ICLR 2026
- **arXiv**: [2503.06437](https://arxiv.org/abs/2503.06437)
- **代码**: [https://github.com/Concarne2/SEED](https://github.com/Concarne2/SEED)
- **领域**: 其他
- **关键词**: brain decoding, evaluation metrics, fMRI, semantic similarity, visual attention, human evaluation

## 一句话总结
提出 SEED（Semantic Evaluation for Visual Brain Decoding），一个结合 Object F1、Cap-Sim 和 EffNet 三个互补指标的组合评估度量，在与人类评估的对齐度上显著超越现有所有指标。

## 研究背景与动机
- 视觉脑解码（从 fMRI 重建视觉刺激）取得显著进展，最新模型在现有百分比指标上接近满分，看似问题已解决。
- **但仔细审视**：重建图像常常丢失关键语义元素（如泰迪熊变成猫），现有指标却给出高分，误导研究。
- **现有评估的三大问题**：
  1. **池依赖性**：二路识别指标（AlexNet、CLIP 等）依赖比较池，不同模型的比较不公平
  2. **难度不足**：二路识别任务太简单，近期模型已接近完美
  3. **缺乏人类一致性**：基于抽象特征的指标与人类直觉偏差大

## 方法详解

### 整体框架：受人类视觉注意力启发
人类视觉注意力是两阶段过程：
- **第一阶段**：并行处理基本特征（颜色、方向、亮度）→ 对应 EffNet 等卷积模型
- **第二阶段**：聚焦注意力绑定特征为连贯物体 → 现有指标缺失此阶段

SEED 集成三个互补指标模拟完整视觉感知：

### 指标 1：Object F1（模拟物体导向注意力）
使用开放词汇图像 grounding 模型（MM-Grounding-DINO）检测 82 类物体：

$$\text{Object Recall}_t = \frac{\text{GT 和重建中共有的类别数}}{\text{GT 中的类别数}}$$

$$\text{Object Precision}_t = \frac{\text{GT 和重建中共有的类别数}}{\text{重建中的类别数}}$$

通过阈值 $t$ 从 0 到截断值滑动取平均，消除阈值超参：
$$\text{Object F1} = \frac{2}{\text{Object Recall}^{-1} + \text{Object Precision}^{-1}}$$

### 指标 2：Cap-Sim（模拟特征绑定过程）
用图像标注模型（GIT）生成描述，再比较描述的语义相似度：

$$\text{Cap-Sim} = \cos(e_{\text{text}}(c(I_{GT})), e_{\text{text}}(c(I_{recon})))$$

其中 $e_{\text{text}}$ 用 Sentence Transformer，$c$ 用 GIT。捕获物体属性（姿态、颜色）、背景等 Object F1 遗漏的语义。

### 指标 3：EffNet（捕获全局结构）
$$\overline{\text{EffNet}} = \text{corr}(e_{\text{img}}(I_{GT}), e_{\text{img}}(I_{recon}))$$

使用 ImageNet 预训练 EfficientNet，捕获更全局和结构性的场景特征。

### SEED 组合
$$\text{SEED} = \frac{\text{Object F1} + \text{Cap-Sim} + \overline{\text{EffNet}}}{3}$$

三个指标互补：Object F1 检查关键物体存在性，Cap-Sim 捕获高层语义细节，EffNet 捕获全局结构。

### 人类评估数据收集
- 22 名评估者对 1,000 对 GT-重建图像对进行 5 分 Likert 量表评分
- ICC(2, n) = 0.84 (p=0)，表明高度评估者间一致性
- 数据开源发布

## 实验关键数据

### 主实验：与人类评估的对齐度（NSD + MindEye2）

| 指标 | 配对准确率 | Kendall τ | Pearson r |
|------|----------|----------|----------|
| PixCorr | 53.8% | .075 | .117 |
| SSIM | 54.5% | .090 | .112 |
| AlexNet(2) | 55.0% | .185 | .187 |
| AlexNet(5) | 49.5% | .236 | .258 |
| Inception | 63.8% | .330 | .475 |
| CLIP | 66.4% | .368 | .436 |
| EffNet | 78.0% | .559 | .748 |
| SwAV | 69.7% | .394 | .576 |
| Object F1 | 75.8% | .516 | .708 |
| Cap-Sim | 73.8% | .477 | .666 |
| **SEED** | **81.0%** | **.621** | **.813** |

> SEED 在所有三个人类对齐指标上都显著领先，配对准确率 81%、Pearson r 0.813。

### 跨数据集验证（GOD + Mind-Vis）

| 指标 | 配对准确率 | Kendall τ | Pearson r |
|------|----------|----------|----------|
| CLIP | 62.6% | — | — |
| EffNet | ~70% | — | — |
| Object F1 | ~68% | — | — |
| **SEED** | **~73%** | — | **最优** |

> SEED 的优势在不同数据集和模型组合上保持一致。

### 关键发现
1. 大多数常用指标（PixCorr、SSIM、AlexNet）与人类评估几乎不相关
2. EffNet 是现有最好的单一指标（Pearson 0.748），但 SEED 进一步提升到 0.813
3. Object F1 和 Cap-Sim 各自与人类评估的相关性也很高
4. 用 SEED 重新评估 SOTA 模型发现：即使"近完美"分数的模型也经常混淆关键物体
5. 基于描述的相似度评估（Cap-Sim）此前从未被提出，尽管概念简单

## 亮点与洞察
- **揭示评估盲区**：动摇了"脑解码已近解决"的错觉
- **神经科学启发**：两阶段视觉注意力 → Object F1 + Cap-Sim
- **人类评估基准**：1,000 对 × 22 人评估数据开源，为后续研究提供标准
- **Cap-Sim 新颖性**：最简单的想法（比较图像描述）竟从未有人做过

## 局限性
- SEED 仅关注语义相似度，不评估低级视觉质量（如纹理、颜色精度）
- Object F1 受限于检测模型能识别的 82 个物体类别
- Cap-Sim 依赖图像标注模型的质量（可能产生幻觉描述）
- 等权平均三个指标是否最优未做深入分析

## 相关工作
- **脑解码模型**: MindEye (Scotti et al., 2023/2024), NeuroPictor (Huo et al., 2024), BrainDiffuser (Ozcelik et al., 2023)
- **图像质量评估**: SSIM (Wang et al., 2004), FID, LPIPS
- **开放词汇检测**: Grounding DINO (Zhao et al., 2024)
- **图像标注**: GIT (Wang et al., 2022)

## 评分
- 新颖性: ⭐⭐⭐⭐ — Cap-Sim 新颖，问题定义和解决思路清晰
- 理论深度: ⭐⭐⭐ — 以经验驱动为主，缺乏理论分析
- 实验充分性: ⭐⭐⭐⭐⭐ — 大规模人类评估 + 多指标全面对比 + 跨数据集验证
- 实用价值: ⭐⭐⭐⭐⭐ — 直接改善脑解码评估标准，人类数据开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICCV 2025\] Beyond Brain Decoding: Visual-Semantic Reconstructions to Mental Creation Extension Based on fMRI](../../ICCV2025/medical_imaging/beyond_brain_decoding_visualsemantic_reconstructions_to_ment.md)
- [\[ICLR 2026\] Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model](brain-semantoks_learning_semantic_tokens_of_brain_dynamics_with_a_self-distilled.md)
- [\[NeurIPS 2025\] Zebra: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding](../../NeurIPS2025/medical_imaging/zebra_towards_zero-shot_cross-subject_generalization_for_universal_brain_visual_.md)

</div>

<!-- RELATED:END -->
