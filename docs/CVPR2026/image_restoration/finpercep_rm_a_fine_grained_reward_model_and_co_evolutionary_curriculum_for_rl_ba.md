---
title: >-
  [论文解读] FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution
description: >-
  [CVPR 2026][图像恢复][图像超分辨率] 提出细粒度感知奖励模型 FinPercep-RM 和协同进化课程学习（CCL）策略，解决 RLHF 应用于真实世界超分辨率时的奖励黑客和训练不稳定问题，通过同时输出全局质量分数和空间退化热力图实现局部缺陷感知。
tags:
  - CVPR 2026
  - 图像恢复
  - 图像超分辨率
  - 奖励模型
  - 图像复原
  - 细粒度质量评估
  - 课程学习
---

# FinPercep-RM: A Fine-grained Reward Model and Co-evolutionary Curriculum for RL-based Real-world Super-Resolution

**会议**: CVPR 2026  
**arXiv**: [2512.22647](https://arxiv.org/abs/2512.22647)  
**代码**: [https://github.com/lyd-2022/FinPercep-RM](https://github.com/lyd-2022/FinPercep-RM)  
**领域**: 图像修复 / 超分辨率  
**关键词**: 图像超分辨率, 奖励模型, RLHF, 细粒度质量评估, 课程学习

## 一句话总结

提出细粒度感知奖励模型 FinPercep-RM 和协同进化课程学习（CCL）策略，解决 RLHF 应用于真实世界超分辨率时的奖励黑客和训练不稳定问题，通过同时输出全局质量分数和空间退化热力图实现局部缺陷感知。

## 研究背景与动机

基于扩散模型的真实世界图像超分辨率（Real-ISR）已取得显著进展，利用大规模 T2I 模型的生成先验合成丰富纹理。RLHF 作为 T2I 领域的成功优化范式，自然被迁移到 ISR 任务中——用图像质量评估（IQA）模型作为奖励信号引导 SR 模型。

然而，现有 IQA 模型（如 CLIP-IQA、MANIQA）仅输出单一全局分数，**对局部细粒度失真不敏感**——一张有明显局部伪影的图像可能获得与原图相近的高分。这导致严重的**奖励黑客（Reward Hacking）**：生成器学会"讨好"不完善的奖励信号，收敛到全局分数高但充满局部伪影和"油画感"外观的结果。

另一方面，直接使用更复杂的细粒度奖励模型会引发**训练不稳定**——高方差的空间惩罚信号导致策略梯度振荡和收敛失败。这形成了**稳定性-鲁棒性两难**：简单全局奖励稳定但被黑客，复杂细粒度奖励鲁棒但不稳定。

核心思想：**一个好的奖励模型不仅要评估"质量是什么（What）"，还要诊断"缺陷在哪里（Where）"。** 同时用课程学习从简单到复杂逐步引入细粒度奖励。

## 方法详解

### 整体框架

框架包含三个核心组件：(1) 基于 T2I 先验的 SR 生成器；(2) FinPercep-RM 诊断奖励模型——编码器产生全局质量分数，解码器输出细粒度感知退化图（fg-PDM）；(3) CCL 协同进化课程——奖励模型从简单全局 IQA 逐步扩展为完整的细粒度版本，生成器训练同步从简单奖励过渡到复杂奖励。

### 关键设计

1. **FinPercep-RM 编码器-解码器架构**:

    - 功能：同时提供全局质量评分和空间缺陷定位
    - 核心思路：编码器（基于 CLIP-IQA 骨干）提取多尺度特征 $\{f_i\}_{i=1}^N$，最深层特征 $f_N$ 用于全局评分。解码器通过上采样和跨层融合重建与输入同分辨率的感知退化图 $M_{fg-pdm} \in [0,1]$（Sigmoid 归一化）。关键创新：全局分数 $S_{fgc-global}$ 不是直接从 $f_N$ 回归，而是先用退化图调制全局特征 $f_N \odot \text{interpolate}(M_{fg-pdm})$，再经 MLP 回归——使全局分数内在地依赖于局部缺陷
    - 设计动机：传统 IQA 的全局分数与局部质量脱钩是奖励黑客的根源。通过架构设计强制全局分数"看到"空间缺陷

2. **FGR-30k 细粒度奖励数据集**:

    - 功能：提供有空间缺陷标注的训练数据
    - 核心思路：收集高质量图像 $I_{GT}$→退化得到 $I_{LR}$→多个 SR 模型生成 $I_{SR}$→**区域交换**策略将 $I_{SR}$ 的伪影区域"植入"$I_{GT}$（用随机掩码和 SAM 语义掩码）得到合成样本 $I_{syn}$。真值退化图 $M_{gt}$ 融合像素级 L1 距离和 DINOv3 特征级余弦距离
    - 设计动机：现有 IQA 和偏好数据集缺乏空间缺陷标注。区域交换可控地"植入"局部缺陷，生成精确的训练监督

3. **协同进化课程学习（CCL）**:

    - 功能：在稳定性和鲁棒性之间取得平衡
    - 核心思路：双路径协同进化——(a) 奖励模型渐进扩展：从简单全局 IQA（$RM_0$）开始，分阶段在 FGR-30k 上训练并引入解码器参数，逐步从全局分数进化为带热力图的完整模型；(b) 生成器课程共进化：先用稳定的 $RM_0$ 收敛，再逐步切换到更严格的 $RM_k$。这种从易到难的设计让生成器在早期获得稳定初始化，晚期获得鲁棒的细粒度监督
    - 设计动机：直接使用完整 FinPercep-RM 导致策略梯度振荡和收敛失败（实验曲线显示剧烈振荡）。课程学习让训练过程平滑过渡

### 损失函数 / 训练策略

FinPercep-RM 训练损失：$\mathcal{L}_{total} = \lambda_{map} \mathcal{L}_{map} + \lambda_{rank} \mathcal{L}_{rank} + \lambda_{align} \mathcal{L}_{align}$。$\mathcal{L}_{map}$ 为热力图 L1 损失；$\mathcal{L}_{rank}$ 为三元组排序损失（$S_{SR} < S_{syn} < S_{GT}$）；$\mathcal{L}_{align}$ 为锚定对齐损失（防止跨阶段分数漂移）。RL 优化使用标准 RLHF pipeline。

## 实验关键数据

### 主实验

**DRealSR 数据集，应用到不同 SR 基线**

| 基线模型 | 配置 | LPIPS↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|---------|------|--------|--------|---------|---------|
| SUPIR | Baseline | 0.452 | 65.665 | 0.629 | 0.572 |
| SUPIR | + 标准 IQA 奖励 | 0.465 | 64.892 | 0.612 | 0.589 |
| SUPIR | **+ FinPercep-RM** | **0.428** | **67.234** | **0.648** | **0.586** |
| DreamClear | Baseline | 0.317 | 65.077 | 0.605 | 0.543 |
| DreamClear | + 标准 IQA 奖励 | 0.332 | 64.123 | 0.591 | 0.567 |
| DreamClear | **+ FinPercep-RM** | **0.295** | **67.891** | **0.632** | **0.561** |

### 消融实验

| 配置 | 训练稳定性 | 奖励黑客 | 最终质量 | 说明 |
|------|----------|---------|---------|------|
| 标准 IQA 奖励 | ✅ 稳定 | ❌ 严重 | 中等 | 局部伪影明显 |
| FinPercep-RM (无 CCL) | ❌ 振荡 | ✅ 缓解 | 差（不收敛） | 直接使用不稳定 |
| **FinPercep-RM + CCL** | **✅ 稳定** | **✅ 缓解** | **最优** | 完整方法 |

### 关键发现

- 标准 IQA 奖励会导致 LPIPS 上升（失真增加）但 MUSIQ 分数下降——典型的奖励黑客：模型学会了取悦 IQA 但实际质量下降
- FinPercep-RM 在所有 SR 基线上一致提升（包括 ResShift、SUPIR、DreamClear、DiffBIR、SeeSR、DIT4SR）
- CCL 的训练曲线平滑且最终收敛到更高奖励值，直接使用 FinPercep-RM 的曲线剧烈振荡
- 退化图可视化显示 FinPercep-RM 能准确定位纹理伪影区域

## 亮点与洞察

- **将奖励黑客归因于感知粒度不足**是精准的诊断：全局分数确实无法区分"整体好但局部差"和"整体好且局部好"
- **架构耦合全局-局部**的设计巧妙：用退化图调制全局特征再回归分数，使两者在架构层面不可分离
- **CCL 课程学习**解决了一个普遍的 RLHF 问题——复杂奖励信号导致训练不稳定——可迁移到其他 RLHF 应用

## 局限与展望

- FGR-30k 的退化图真值依赖于像素+特征差异的启发式融合，可能不完美反映人类感知
- CCL 的阶段划分和切换时机需要手动调整，自适应课程更理想
- 仅在 ISR 任务上验证，是否适用于其他图像生成 RLHF（如 T2I）待探索
- 编码器-解码器结构增加了计算开销，实时性有限

## 相关工作与启发

- **vs CLIP-IQA/MANIQA**: 仅全局分数，缺乏空间感知，易被奖励黑客
- **vs 大规模 IQA（如 Q-Align）**: 语义感知好但计算成本过高，不适合迭代训练
- **vs NPN（反问题空间方法）**: 不同方向——NPN 在零空间层面工作，FinPercep-RM 在奖励信号层面

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统诊断 ISR-RLHF 的奖励黑客问题并提出架构+课程双重解决方案
- 实验充分度: ⭐⭐⭐⭐⭐ 6 个 SR 基线、多数据集、训练曲线可视化、用户研究
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 对 RLHF 在视觉生成中的应用有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)
- [\[CVPR 2026\] UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution](ucan_unified_convolutional_attention_lightweight_sr.md)
- [\[CVPR 2026\] Towards Universal Computational Aberration Correction in Photographic Cameras: A Comprehensive Benchmark Analysis](unicac_universal_computational_aberration_correction_benchmark.md)

</div>

<!-- RELATED:END -->
