---
title: >-
  [论文解读] NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images
description: >-
  [CVPR 2026 (Workshop)][图像恢复][雨滴去除] NTIRE 2026第二届日夜双焦点雨滴去除挑战赛总结报告：基于Raindrop Clarity真实数据集（14,139训练/407验证/593测试），168支队伍参赛，17支提交有效方案，冠军AIIA-Lab以MSDT骨干+伪GT精修流水线取得35.24分最佳成绩。
tags:
  - "CVPR 2026 (Workshop)"
  - "图像恢复"
  - "雨滴去除"
  - "双焦点图像"
  - "日夜场景"
  - "图像复原竞赛"
  - "NTIRE"
---

# NTIRE 2026 The Second Challenge on Day and Night Raindrop Removal for Dual-Focused Images

**会议**: CVPR 2026 (Workshop)  
**arXiv**: [2604.10634](https://arxiv.org/abs/2604.10634)  
**代码**: [Competition Page](https://www.codabench.org/competitions/12808/)  
**领域**: 图像复原 / 去雨滴  
**关键词**: 雨滴去除, 双焦点图像, 日夜场景, 图像复原竞赛, NTIRE

## 一句话总结

NTIRE 2026第二届日夜双焦点雨滴去除挑战赛总结报告：基于Raindrop Clarity真实数据集（14,139训练/407验证/593测试），168支队伍参赛，17支提交有效方案，冠军AIIA-Lab以MSDT骨干+伪GT精修流水线取得35.24分最佳成绩。

## 研究背景与动机

图像去雨滴是低层视觉的基础任务，直接影响自动驾驶、监控等下游应用。然而现有数据集存在关键局限：(1) 多数仅覆盖白天场景；(2) 很少同时包含"雨滴聚焦"和"背景聚焦"两种对焦模式；(3) 真实配对数据极度稀缺。Raindrop Clarity数据集填补了这一空白，提供日间/夜间+双焦点的真实退化图像。延续第一届挑战赛的成功，第二届调整了数据划分方式（14,139训练/407验证/593测试），旨在构建更强大的实用基准。

与第一届相比，本届在数据划分上更注重验证集和测试集在不同场景之间的平衡分布，并吸引了更多参赛队伍(168支 vs 第一届)。

## 方法详解

### 整体框架

本文是竞赛总结报告，汇总了17支参赛队伍的方案。核心评价指标为综合分数 Score = 10×PSNR(Y) + 10×SSIM(Y) − 5×LPIPS。

### 关键设计

1. **MSDT骨干主导**：冠军(AIIA-Lab)和亚军(raingod)均选择MSDT作为骨干网络，通过超参优化+长期训练保留多个强checkpoint，再基于场景级融合构造伪GT进行二阶段精修
2. **场景级伪GT策略**：多支顶尖队伍采用"同一场景的多张图像融合→生成伪GT→微调"的三阶段流水线，利用同场景多视角的一致性提升去雨滴效果
3. **多样化骨干探索**：参赛方案覆盖了Restormer、Histoformer、NAFNet、AdaIR、扩散模型等多种架构，展示了图像复原领域的方法多样性

### 训练策略

- 冠军方案分两阶段训练：第一阶段用混合退化数据训练MSDT 200 epochs (256×256 patches)；第二阶段用场景融合生成的伪GT以更小学习率微调
- 多数方案使用L1 + SSIM + FFT/感知损失的组合
- 测试阶段普遍采用滑动窗口推理、多模型集成、checkpoint选择等工程技巧
- 轻量级方案(如Cidaut AI，仅2.95M参数)也达到了竞争力水平

## 实验关键数据

### 主实验

| 排名 | 队伍 | 综合分 | PSNR↑ | SSIM↑ | LPIPS↓ | 参数量 |
|------|------|--------|-------|-------|--------|--------|
| 1 | AIIA-Lab | 35.24 | 28.34 | 0.827 | 0.273 | 16.6M |
| 2 | raingod | 35.22 | 28.28 | 0.826 | 0.264 | 16.6M |
| 3 | BUU_CV | 35.04 | 28.15 | 0.822 | 0.267 | 26.9M |
| 4 | RetinexDualV2 | 33.86 | 27.24 | 0.806 | 0.289 | 4.8M |
| 5 | ULR | 33.75 | 27.06 | 0.797 | 0.255 | 593M |
| 9 | Cidaut AI | 31.95 | 25.84 | 0.765 | 0.309 | 2.95M |
| 17 | BITssvgg | 30.94 | 25.14 | 0.750 | 0.338 | 16.9M |

### 关键发现

- 前三名成绩非常接近（差距<0.2分），MSDT骨干被验证为最有效的基础架构
- ULR队伍取得最佳LPIPS(0.255)，说明感知质量并非与信号保真度完全一致
- 轻量方案(GU-day Mate, 2.14M参数)也能达到32.9分，展示了性能-效率权衡的多样性
- 伪GT构造+场景融合是本届最核心的通用策略

## 亮点与洞察

- 场景级一致性利用是本届竞赛最关键的技巧：同一场景的多张图像天然提供了互补信息，中值/均值融合可以有效压制随机雨滴
- 相比第一届，方案整体质量大幅提升，但真正的创新点集中在测试时适配(test-time adaptation)而非训练阶段的架构创新
- RetinexDualV2引入物理先验（残差雨强度掩码）是少数从物理角度建模雨滴的方案

## 局限与展望

- 所有方案都假设同一场景有多张图像可用于融合，但实际应用中可能只有单张输入
- 夜间场景的去雨效果仍明显弱于白天，需要更多夜间训练数据
- 缺少对极端退化（如大面积覆盖雨滴）的专项评估
- 未来可探索视频序列中的时域一致性去雨
- 训练数据规模有限（仅14,139张），更大规模的数据集可能带来进一步提升

## 相关工作与启发

- MSDT、Restormer、NAFNet等主流复原骨干在雨滴任务上依然有效
- 伪GT+自监督微调的范式可迁移至其他图像复原竞赛（去雾、去模糊等）
- 频域注意力（如RetinexDualV2的频率分支）是值得深入研究的方向
- Raindrop Clarity数据集是目前唯一同时覆盖日夜+双焦点退化的真实配对基准
- 扩散模型方案(NTR)表现不如传统复原模型，可能因数据规模不足以发挥其潜力

## 方法汇总

| 方案 | 骨干 | 特色 | 使用额外数据 |
|------|------|------|-------------|
| AIIA-Lab | MSDT | 多checkpoint选择+场景融合+伪GT精修 | 否 |
| raingod | MSDT | UAV-Rain1k增强+中值滤波伪GT | 是 |
| BUU_CV | STRRNet+Restormer | 矩形/方形patch互补+加权集成 | 是 |
| RetinexDualV2 | Retinex双分支 | 物理先验(残差雨强度掩码)+Mamba注意力 | 否 |
| Cidaut AI | NAFNet | 双注意力模块(空间+频率) | 否 |

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 2 | 竞赛总结报告，方法创新主要在参赛队伍 |
| 技术深度 | 3 | 详细记录了17种方案的训练/测试策略 |
| 实验充分性 | 4 | 基准完善，多指标评测，168支队伍参与 |
| 写作质量 | 3 | 结构清晰，但部分方案描述较为简略 |
| 实用价值 | 4 | 真实数据基准+多样化方案，实用性强 |

**总评**：作为竞赛报告，全面记录了去雨滴领域的最新进展和最佳实践，伪GT+场景融合策略具有较高的迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] NTIRE 2026 The 3rd RAIM Challenge: AI Flash Portrait (Track 3)](ntire_2026_ai_flash_portrait_challenge.md)
- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[CVPR 2026\] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2025\] Tokenize Image Patches: Global Context Fusion for Effective Haze Removal in Large Images](../../CVPR2025/image_restoration/tokenize_image_patches_global_context_fusion_for_effective_haze_removal_in_large.md)

</div>

<!-- RELATED:END -->
