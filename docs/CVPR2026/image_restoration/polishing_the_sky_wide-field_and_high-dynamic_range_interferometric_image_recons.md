---
title: >-
  [论文解读] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction
description: >-
  [CVPR 2026][图像恢复][射电干涉成像] 在 POLISH 框架基础上提出 POLISH+ 和 POLISH++，通过分块训练-拼接策略和基于 arcsinh 的非线性变换，实现宽视场（12,960×12,960 像素）和高动态范围（$\sim 10^6$）条件下的射电干涉图像重建与超分辨率…
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "射电干涉成像"
  - "深度学习去卷积"
  - "超分辨率"
  - "高动态范围"
  - "强引力透镜"
---

# POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction

**会议**: CVPR 2026  
**arXiv**: [2603.09162](https://arxiv.org/abs/2603.09162)  
**代码**: 无（基于 POLISH 框架扩展）  
**领域**: 射电天文图像重建 / 图像去卷积  
**关键词**: 射电干涉成像, 深度学习去卷积, 超分辨率, 高动态范围, 强引力透镜

## 一句话总结

在 POLISH 框架基础上提出 POLISH+ 和 POLISH++，通过分块训练-拼接策略和基于 arcsinh 的非线性变换，实现宽视场（12,960×12,960 像素）和高动态范围（$\sim 10^6$）条件下的射电干涉图像重建与超分辨率，并首次展示深度学习方法可超分辨强引力透镜系统。

## 研究背景与动机

射电干涉成像通过天线阵列合成大口径实现高分辨率天文成像，其核心是求解图像去卷积问题。即将建成的 **DSA-2000**（1650 根天线）将以超过 80 Tb/s 的数据吞吐量产出超过 $10,000 \times 10,000$ 像素、动态范围 $\sim 10^6$ 的图像。

现有方法的不足：

**CLEAN 算法**：标准方法，但分辨率受限于 PSF 尺度，无法超分辨；对复杂形态源恢复不佳

**RML 优化方法**：迭代求解计算代价高，不适合 DSA 的实时需求

**现有深度学习方法**：
   - 图像尺寸小（< 1,024 像素）、动态范围低（< $10^3$）
   - 仅测试简单高斯源，未应对复杂天体形态（如强透镜）
   - 未考虑 PSF 不匹配（校准误差）

核心挑战：(1) 宽视场带来的超高维度 (2) 真实天空的极高动态范围。

## 方法详解

### 整体框架

这篇工作要解决的是即将上线的 DSA-2000 阵列带来的两道硬约束：图像大到 12,960×12,960 像素、动态范围高到 $\sim 10^6$，而现有方法要么分辨率受限（CLEAN），要么算不动（RML），要么只在千像素小图上验证过。作者沿用 POLISH 的端到端思路——用一个基于 WDSR 的 CNN 直接学习从低分辨率脏图到高分辨率干净天空图的映射，正向退化模型为 $I_{\text{dirty}} = [k * (I_{\text{true}} + n)]{\downarrow_s}$（$k$ 是 PSF，$s=2$ 为下采样因子）——但在此之上做了两次关键升级：POLISH+ 把训练和推理改成分块处理，让网络能在巨幅图上跑起来；POLISH++ 再叠一层 arcsinh 非线性变换，专治高动态范围下暗源被亮源淹没的问题。

### 关键设计

**1. 分块训练与拼接：让网络吃得下 $10^8$ 像素的全视场图**

GPU 显存装不下一整张上亿像素的天空图，所以没法直接端到端训练。作者把每张 12,960×12,960 的全视场图切成 $J$ 个互不重叠的 324×324 小块，各自配成训练对，18 张全视场图就这样裂解出 28,800 个训练对（覆盖约 600 万颗可检测星系），推理时也是各块独立预测、再拼回全视场。这里有个不能忽略的细节：从全视场脏图里裁出来的 patch，并不等同于一张干净的小图——它身上带着**跨块污染**，也就是相邻亮源 PSF 旁瓣甩进来的非局部伪影。正因如此，网络被迫学会的不只是局部去卷积，还包括识别并扣除这些来自块外的旁瓣，这和直接在孤立小图上训练是两回事。

**2. Arcsinh 非线性变换：把跨六个数量级的强度压到同一量级，救回暗源**

真实天空的亮度从最暗到最亮能横跨 $10^4 \sim 10^6$。如果直接在原始强度空间算 $\ell_1$ 损失，梯度会被极少数极亮源垄断，暗源几乎得不到监督、恢复质量很差。作者借天文里常用的 arcsinh 拉伸把强度先压缩再训练：

$$\text{AsinhStretch}(x; a) = \frac{\operatorname{arcsinh}(x/a)}{\operatorname{arcsinh}(1/a)}$$

它在大值处近似对数、把跨多个数量级的像素压回相近量级，又能同时容纳正负值（干涉脏图本就可能为负），天然契合干涉成像。训练目标整个搬到变换空间里算：

$$\theta^* = \arg\min_\theta \frac{1}{NJ}\sum_{i,j} \|\text{G}_\theta(\text{AsinhStretch}(I_{\text{dirty}}^{[j]}; a_d)) - \text{AsinhStretch}(I_{\text{true}}^{[j]}; a_t)\|_1$$

推理时再用逆变换 $\text{AsinhStretch}^{-1}$ 把结果还原回原始强度尺度。消融里这一步把召回率又拉高约 4%，主要受益的正是此前被压制的暗源。

**3. 鲁棒性与适应性：理想 PSF 训练的模型也能扛校准误差、还能快速换场景**

实际观测中 PSF 永远和理想值有偏差（校准误差），所以作者额外验证了两件事。一是鲁棒性：模型只在理想 PSF 上训过，但面对随机扰动的 PSF（扰动幅度 $\gamma \in [0, 30]$）时，PSNR 虽有下降，重建结果在视觉上仍保持一致，说明它没有死记某个固定 PSF。二是适应性：换一种观测条件时不必从头训，微调 11 个 epoch 就能收敛，而从头训要 57 个 epoch，提速 5 倍以上——这对需要频繁适配不同巡天配置的真实部署很关键。

### 损失函数 / 训练策略

- 损失：$\ell_1$ 损失（在 AsinhStretch 变换空间中）
- 优化器：Adam，lr 0.0001，batch 12
- $a_{\text{dirty}} = a_{\text{true}} = 0.1$
- 训练数据：T-RECS 天空模拟，18 张训练图 + 5 张测试图
- 噪声：高斯噪声 $\sigma = 1\mu$Jy

## 实验关键数据

### 主实验

| 方法 | Precision↑ | Recall↑ | F1↑ | 主轴 FWHM RMSE↓ | 副轴 FWHM RMSE↓ |
|------|-----------|---------|-----|-----------------|-----------------|
| CLEAN | 0.3612 | 0.2220 | 0.2750 | 1.0046″ | 0.7862″ |
| POLISH | 0.5560 | 0.4612 | 0.5042 | 0.9642″ | 0.3219″ |
| POLISH+ | 0.8744 | 0.5751 | 0.6938 | 0.4335″ | 0.1889″ |
| POLISH++ | **0.8433** | **0.6142** | **0.7107** | 0.4654″ | 0.2056″ |

注：POLISH++ F1 较 CLEAN 提升 159%，形状估计精度提升 2 倍以上。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| POLISH (全图训练) | F1=0.5042 | 基线 |
| POLISH+ (分块训练) | F1=0.6938 | 分块策略大幅提升 |
| POLISH++ (分块+arcsinh) | F1=0.7107 | 非线性变换进一步提升召回率 +4% |
| PSF 扰动 γ=0→30 | PSNR 下降但视觉一致 | 对校准误差具有鲁棒性 |
| 微调 vs 从头训练 | 11 vs 57 epochs | 5× 加速 |

### 关键发现

- **超分辨率能力**：POLISH++ 可准确估计 PSF 尺度以下（≈3.3″）的星系形状参数，CLEAN 在此尺度下完全失效
- **强透镜发现**：在 POLISH++ 超分辨图像上训练的透镜发现 CNN，可将可发现的透镜从 PSF 3 倍分辨率极限降低到 PSF 尺度附近，预期 DSA 巡天的强透镜产量增加约 **10 倍**
- **动态范围**：POLISH++ 成功处理 $\sim 10^6$ 动态范围，比现有 DL 方法（< $10^3$）高 3 个数量级
- CLEAN 的通量估计仍优于 POLISH（基于模型的方法保持绝对通量标定）

## 亮点与洞察

1. **真实部署导向**：不追求小图上的 benchmark PSNR，而是针对 DSA 的实际需求（12,960×12,960 像素、$10^6$ 动态范围）设计
2. **跨块污染的发现**：分块后脏图 patch 包含邻域亮源的 PSF 伪影，这是独特的领域洞察
3. **科学应用价值**：超分辨率直接赋能强引力透镜发现，将 DSA 的透镜产量提升 10 倍
4. **坦诚的局限分析**：明确指出 CLEAN 在通量估计上仍更优，DL 方法缺乏显式通量校准机制
5. **从鲁棒性到适应性**：不仅验证了 PSF 不匹配下的鲁棒性，还展示了快速微调的适应性

## 局限与展望

1. 通量估计精度不如 CLEAN，缺乏显式通量校准机制
2. 仅在图像平面操作（非可见度域），可能损失相位信息
3. 训练数据基于 T-RECS 模拟，与真实天空仍有差距
4. 分块拼接可能在块边界产生不连续性（文中未详细讨论）
5. 未来方向：通量后处理校准、端到端可见度域方法、更复杂的天体形态模拟

## 相关工作与启发

- **POLISH (Connor et al. 2022)**：本文的基线方法，仅在 2,048 像素和 $\sim 10^2$ 动态范围上测试
- **R2D2**：展开网络方法，支持 512 像素和 $5 \times 10^5$ 动态范围
- **CLEAN**：射电天文标准方法，受限于 PSF 分辨率
- 启发：深度学习去卷积方法在天文成像中的"杀手级应用"是超分辨率带来的科学发现能力

## 评分

- 新颖性: ⭐⭐⭐ 技术上是现有方法的实用改进（分块训练+非线性变换），核心贡献在于工程规模和应用
- 实验充分度: ⭐⭐⭐⭐⭐ 涵盖源检测、形状估计、通量估计、强透镜发现、PSF 鲁棒性/适应性
- 写作质量: ⭐⭐⭐⭐ 天文背景和 DL 方法结合流畅，问题定义清楚
- 价值: ⭐⭐⭐⭐ 面向 DSA 部署的实际价值高，强透镜产量 10× 提升是重要科学贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery](../../CVPR2025/image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting](bhcast_unlocking_black_hole_plasma_dynamics_from_a_single_blurry_image_with_long.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)

</div>

<!-- RELATED:END -->
