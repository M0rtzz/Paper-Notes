---
title: >-
  [论文解读] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction
description: >-
  [CVPR 2026][图像恢复][射电干涉成像] 在 POLISH 框架基础上提出 POLISH+ 和 POLISH++，通过分块训练-拼接策略和基于 arcsinh 的非线性变换，实现宽视场（12,960×12,960 像素）和高动态范围（$\sim 10^6$）条件下的射电干涉图像重建与超分辨率，并首次展示深度学习方法可超分辨强引力透镜系统。
tags:
  - CVPR 2026
  - 图像恢复
  - 射电干涉成像
  - 深度学习去卷积
  - 超分辨率
  - 高动态范围
  - 强引力透镜
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

POLISH 系列采用端到端 CNN（基于 WDSR 架构），直接学习从低分辨率脏图到高分辨率干净天空图的映射。本文在此基础上：
- **POLISH+**：引入分块训练/推理
- **POLISH++**：在 POLISH+ 基础上增加 arcsinh 非线性变换

正向模型：$I_{\text{dirty}} = [k * (I_{\text{true}} + n)]{\downarrow_s}$，其中 $k$ 是 PSF，$s=2$ 为下采样因子。

### 关键设计

1. **分块训练与拼接策略 (Patch-wise Processing)**：将 12,960×12,960 的全视场图像分割为 $J$ 个不重叠的 324×324 小块，分别构成训练对。关键洞察：

    - 从全视场脏图中裁剪的 patch 包含**跨块污染**（相邻亮源的 PSF 旁瓣伪影）
    - 这与直接在小图上训练完全不同——网络必须学会处理这些非局部伪影
    - 18 张全视场图 → 28,800 个训练对（含 600 万可检测星系）
    - 推理时各 patch 独立预测后拼接回全视场

   设计动机：GPU 显存无法容纳 $10^8$ 像素的完整图像训练；分块策略使大规模训练成为可能。

2. **Arcsinh 非线性变换 (AsinhStretch)**：解决 $10^4 \sim 10^6$ 的极高动态范围问题：
    $\text{AsinhStretch}(x; a) = \frac{\operatorname{arcsinh}(x/a)}{\operatorname{arcsinh}(1/a)}$
    - 对数形式：将跨多个数量级的像素值压缩到相同数量级
    - 可处理正负值（脏图可能包含负值），适合干涉成像
    - 训练损失在变换空间中计算：
    $\theta^* = \arg\min_\theta \frac{1}{NJ}\sum_{i,j} \|\text{G}_\theta(\text{AsinhStretch}(I_{\text{dirty}}^{[j]}; a_d)) - \text{AsinhStretch}(I_{\text{true}}^{[j]}; a_t)\|_1$
    - 推理时通过逆变换 $\text{AsinhStretch}^{-1}$ 恢复原始强度尺度

   设计动机：直接在原始强度空间训练时，$\ell_1$ 损失被极少数亮源主导，暗源恢复不佳。

3. **模型鲁棒性与适应性**：

    - 鲁棒性：仅在理想 PSF 上训练的模型，面对随机扰动的 PSF（$\gamma \in [0, 30]$）仍保持视觉一致性
    - 适应性：微调比从头训练快 5 倍以上（11 vs 57 epochs），可快速适配不同观测条件

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

## 相关论文

- [\[CVPR 2025\] POLISH'ing the Sky: Wide-Field and High-Dynamic Range Interferometric Image Reconstruction with Application to Strong Lens Discovery](../../CVPR2025/image_restoration/polishing_the_sky_wide-field_and_high-dynamic_range_interferometric_image_recons.md)
- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](toward_real-world_infrared_image_super-resolution_a_unified_autoregressive_frame.md)
- [\[CVPR 2026\] BHCast: Unlocking Black Hole Plasma Dynamics from a Single Blurry Image with Long-Term Forecasting](bhcast_unlocking_black_hole_plasma_dynamics_from_a_single_blurry_image_with_long.md)

<!-- RELATED:END -->
