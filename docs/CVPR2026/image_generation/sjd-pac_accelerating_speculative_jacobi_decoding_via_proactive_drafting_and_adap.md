---
title: >-
  [论文解读] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation
description: >-
  [CVPR 2026][图像生成][自回归图像生成] 本文分析了 Speculative Jacobi Decoding (SJD) 在文本到图像生成中接受长度分布严重偏斜的瓶颈，提出 SJD-PAC 框架…
tags:
  - "CVPR 2026"
  - "图像生成"
  - "自回归图像生成"
  - "推理加速"
  - "推测解码"
  - "Jacobi解码"
  - "无损加速"
---

# SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation

**会议**: CVPR 2026  
**arXiv**: [2603.18599](https://arxiv.org/abs/2603.18599)  
**作者**: Jialiang Kang (北京大学), Han Shu, Wenshuo Li, Yingjie Zhai, Xinghao Chen (华为)
**代码**: 无  
**领域**: 图像生成  
**关键词**: 自回归图像生成, 推理加速, 推测解码, Jacobi解码, 无损加速

## 一句话总结
本文分析了 Speculative Jacobi Decoding (SJD) 在文本到图像生成中接受长度分布严重偏斜的瓶颈，提出 SJD-PAC 框架，通过 Proactive Drafting (PD) 和 Adaptive Continuation (AC) 两项技术，在严格无损的前提下实现 3.8× 推理加速，显著超越原始 SJD 的约 2× 加速。

## 研究背景与动机

**领域现状**：自回归（AR）文本到图像模型（如Lumina-mGPT、Emu3）生成质量已经可以与扩散模型竞争，但推理延迟严重——需要序列化生成数千个token。Speculative Decoding（SD）是LLM推理加速的主流方法，但在T2I场景下效果不佳。

**现有痛点**：标准SD方法（如EAGLE）在T2I模型上几乎无加速效果，因为图像生成的高熵特性导致draft token的接受率极低——即使在标准采样温度下，图像token也有很多候选几乎等概率可行。现有SJD方法虽然是training-free和lossless的，但只能提供约2×的温和加速。

**核心矛盾**：作者深入分析发现SJD的接受长度分布高度偏斜——约50%的前向传播只接受1个token（即完全没有加速），平均2×的加速主要由少数成功接受大量token的步骤贡献。这种"长尾分布"是性能瓶颈的根源。

**本文目标**
   - 如何减少单token接受（低效步骤）的频率？
   - 如何增加每步成功验证的token数？

**切入角度**：单token接受的根因是"上下文失配的级联效应"——当位置i被拒绝后，后续所有proposal的上下文都无效了。对此需要两个方向的优化：（1）在拒绝点提供多样化候选减少后续级联拒绝（PD），（2）拒绝后不立即终止而是继续验证后续token（AC）。

**核心 idea**：拒绝后不停下来，而是继续验证+主动多路起草，双管齐下最大化每步接受长度。

## 方法详解

### 整体框架
SJD-PAC在标准SJD框架上修改验证循环和起草策略。每次迭代：（1）对整个draft序列做一次并行前向传播得到target分布 $P^t$；（2）AC循环——遍历所有L个位置做accept/reject，即使遇到拒绝也不中断；（3）如果有拒绝发生，在第一个拒绝点启动PD构建多路树形proposal供下一轮迭代使用。

### 关键设计

1. **Adaptive Continuation (AC)**:

    - 功能：消除标准SJD验证循环中的"first-reject break"机制
    - 核心思路：标准SJD在位置i遭遇拒绝后立即终止，丢弃所有后续token $X_{i+1:L}^{t-1}$。AC移除了这个break——拒绝后对位置i重采样 $x_i^t \sim \max(p_i - q_i, 0)$，然后继续对 $j > i$ 的位置做标准rejection sampling验证。若后续token通过验证则保留（accept），否则也重采样
    - 设计动机：图像token具有强局部性——通过测量不同距离处上下文扰动对输出分布的Total Variation影响，发现图像token的 $d_{TV}$ 随距离j快速趋近于0（而文本token保持高敏感）。这意味着即使上下文在位置i被修改，远处token的target分布变化极小。因此用stale分布继续验证是有效的
    - 与标准SJD的区别：标准SJD拒绝后重采样整个后续序列，AC只替换被拒绝的个别token，保留被接受的后续token。每个被保留的token概率为 $1 - d_{TV}(p_i^{t-1}, p_i^t) > 0.7$，远高于重新采样能命中同一token的概率（$< 0.01$，因为图像token高熵）

2. **Proactive Drafting (PD)**:

    - 功能：在拒绝点构建多路树形proposal，减少后续迭代的级联拒绝
    - 核心思路：当位置i发生拒绝后，不只是简单采样一条后续序列，而是构建一棵"浅而宽"的树：
        - 树部分（depth D=3, width K=4）：对位置 $i+1$ 到 $i+D$，每个位置从target分布 $p(\cdot|X_{<j}^{t-1})$ 中无放回采样K个候选token
        - 链部分：从K条路径中选一条的末端，自回归延伸到完整长度L
    - 设计动机：拒绝后新采样的 $x_i^t$ 与后续token的上下文不匹配是级联拒绝的根源。通过在关键的post-rejection边界提供K个多样选择，增加至少一条路径在下一轮验证中有效的概率。树只在拒绝点局部构建，不需要额外的模型前向传播
    - vs 标准树形推测解码：标准tree-based SD需要多次前向传播来构建树，而PD的树是基于当前（可能stale的）分布构建的，只增加采样开销不增加计算开销

3. **PD + AC协同工作**:

    - AC让序列在拒绝后保持稳定（更多token被保留为下一轮的draft），PD在拒绝点提供多样化候选增加接受概率
    - 两者都严格保持无损（lossless）：AC的后续验证使用standard rejection sampling保证分布正确性，PD的tree仅作为draft不影响最终输出分布

### 算法流程
完整算法（Algorithm 1）：并行前向传播 → AC循环（遍历所有L个位置，记录first_rej_idx但不break）→ 若有拒绝则对first_rej_idx启动PD → 返回新序列和概率。

## 实验关键数据

### 主实验（Lumina-mGPT，MS-COCO 2017）

| 方法 | 训练免？ | 无损？ | 步压缩↑ | 时延加速↑ | FID↓ | CLIP↑ |
|------|---------|--------|---------|----------|------|-------|
| 原始AR | ✓ | ✓ | 1.00× | 1.00× | 30.79 | 31.31 |
| EAGLE | ✗ | ✓ | 2.94× | 2.10× | 30.68 | 31.73 |
| SJD | ✓ | ✓ | 2.22× | 2.05× | 31.13 | 31.33 |
| GSD (有损) | ✓ | ✗ | 3.39× | 3.62× | 33.12 | 31.25 |
| SJD2 (有损) | ✗ | ✗ | 4.02× | 2.81× | 31.40 | 31.80 |
| **SJD-PAC** | **✓** | **✓** | **4.51×** | **3.80×** | **30.69** | **31.21** |

SJD-PAC以无损方式超越所有有损方法的加速比。

### 跨模型验证（Emu3，MS-COCO 2017）

| 方法 | 无损？ | 步压缩↑ | 时延加速↑ | FID↓ |
|------|--------|---------|----------|------|
| SJD | ✓ | 2.32× | 2.01× | 30.74 |
| SJD2 | ✗ | 5.62× | 2.54× | 31.50 |
| **SJD-PAC** | **✓** | **4.31×** | **3.25×** | **31.10** |

SJD2虽然步压缩更高但窗口长度翻倍导致额外开销，实际wall-clock加速远低于SJD-PAC。

### 消融实验

| 配置 | 步压缩↑ | 说明 |
|------|---------|------|
| SJD baseline (L=32) | 2.31× | 原始方法 |
| + PD | 2.71× | 主动起草减少级联拒绝 |
| + PD + AC | 3.52× | 自适应续行大幅提升 |
| + PD + AC (L=64) | **4.51×** | 更大窗口充分利用AC |

### 关键发现
- AC是贡献最大的组件（+0.81×压缩比 vs PD的+0.40×），因为它直接保留了更多有效token
- AC启用后L=32成为瓶颈，因为token更快稳定，需要更大窗口L=64来充分发挥
- 图像token的总变差距离 $d_{TV}$ 随距离快速衰减的特性是AC有效的关键理论支撑——与文本生成形成鲜明对比
- 修改单个token（0.04%的总量）即可引入严重视觉伪影，证明了无损保证对T2I的必要性
- PD的树参数D=3, K=4是甜点——太深浪费采样，太浅不够多样

## 亮点与洞察
- **对SJD接受长度的细粒度分析**揭示了"50%的步骤贡献0%加速"的洞察，这个分析本身就很有价值，为后续加速方法提供了clear的optimization target
- **AC利用图像token局部性**的思路很巧妙——文本token的长距离依赖使得类似方法在LLM上不可行，但图像token的强局部性使得用stale分布验证是有效的
- **PD + AC的正交性设计**使两者可以独立分析和组合，这种模块化设计值得学习

## 局限与展望
- 仅在Lumina-mGPT和Emu3两个模型上测试，对更新的AR T2I模型的泛化性未知
- 窗口大小L>64后收益递减但计算开销增加——硬件特定的最优L值不通用
- PD的树构建基于stale分布，理论上不如full forward pass构建的准确，可能在更高质量要求下有优化空间
- 可以探索自适应的D和K参数——根据当前区域的熵动态调整树的深度和宽度

## 相关工作与启发
- **vs SJD原版**: SJD-PAC在其基础上修改验证循环和起草策略，将加速比从2×提升到3.8×，且保持training-free和lossless
- **vs EAGLE**: EAGLE需要训练draft model且在T2I上效果差（2.10×），SJD-PAC无需训练且加速更强（3.80×）
- **vs GSD/LANTERN++**: 这些有损方法通过放松接受标准加速，但可能引入视觉伪影。SJD-PAC以无损方式甚至超越它们的加速比
- **vs SJD2**: SJD2需要训练且有损，步压缩虽高但实际加速低（因为窗口大），SJD-PAC更实用

## 评分
- 新颖性: ⭐⭐⭐⭐ AC和PD个别看不算特别新，但结合起来针对T2I的高熵特性设计合理
- 实验充分度: ⭐⭐⭐⭐ 两个模型、两个benchmark、详细消融和分析，但缺少更大规模模型测试
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析深入，从分布偏斜观察到两个解决方案的推导逻辑链非常清晰
- 价值: ⭐⭐⭐⭐ 对AR T2I推理加速有直接实用价值，training-free + lossless是强卖点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICML 2026\] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation](../../ICML2026/image_generation/speculative_coupled_decoding_for_training-free_lossless_acceleration_of_autoregr.md)
- [\[CVPR 2026\] Depth Adaptive Efficient Visual Autoregressive Modeling](depthvar_depth_adaptive_var.md)
- [\[CVPR 2026\] D2C: Accelerating Diffusion Model Training under Minimal Budgets via Condensation](d2c_diffusion_dataset_condensation.md)

</div>

<!-- RELATED:END -->
