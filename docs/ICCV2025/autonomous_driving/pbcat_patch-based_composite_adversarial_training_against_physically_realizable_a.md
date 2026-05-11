---
title: >-
  [论文解读] PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection
description: >-
  [ICCV 2025][自动驾驶][对抗训练] 提出 PBCAT（Patch-Based Composite Adversarial Training），通过结合小面积梯度引导对抗补丁和全局不可感知扰动进行对抗训练，统一防御多种物理可实现攻击（对抗补丁+对抗纹理），在行人检测任务上比之前 SOTA 防御方法提升 29.7% AP。
tags:
  - ICCV 2025
  - 自动驾驶
  - 对抗训练
  - 物理可实现攻击
  - 目标检测
  - 对抗纹理
  - 对抗补丁
---

# PBCAT: Patch-Based Composite Adversarial Training against Physically Realizable Attacks on Object Detection

**会议**: ICCV 2025  
**arXiv**: [2506.23581](https://arxiv.org/abs/2506.23581)  
**代码**: [GitHub](https://github.com/LixiaoTHU/oddefense-PatchAT)  
**领域**: 自动驾驶  
**关键词**: 对抗训练, 物理可实现攻击, 目标检测, 对抗纹理, 对抗补丁

## 一句话总结

提出 PBCAT（Patch-Based Composite Adversarial Training），通过结合小面积梯度引导对抗补丁和全局不可感知扰动进行对抗训练，统一防御多种物理可实现攻击（对抗补丁+对抗纹理），在行人检测任务上比之前 SOTA 防御方法提升 29.7% AP。

## 研究背景与动机

目标检测在自动驾驶、视频监控等安全关键应用中发挥核心作用，但近期研究表明检测器极易被物理可实现攻击欺骗。这些攻击分为两大类：

**对抗补丁（Adversarial Patches）**：在固定区域放置对抗模式，面积小但扰动强度大

**对抗纹理（Adversarial Textures）**：如 AdvTexture、AdvCaT，将对抗扰动覆盖在衣物表面，面积大且从多角度有效，可直接在物理世界中穿着欺骗检测器

现有防御方法存在三大关键限制：

- **针对性过窄**：大多数防御仅针对补丁攻击，对纹理攻击几乎无效（SAC 等非 AT 方法在 AdvTexture 下 AP 接近 0）
- **缺乏自适应鲁棒性**：输入预处理和特征过滤方法可被自适应攻击绕过
- **$\ell_\infty$-AT 不匹配物理威胁**：全局不可感知扰动的威胁模型与物理攻击截然不同，虽然有一定迁移效果但仍不够

将 patch-based AT 从分类扩展到检测面临额外挑战：检测场景需处理多个目标的多个补丁位置，计算开销随候选位置数急剧增加。

## 方法详解

### 整体框架

PBCAT 的核心思路是将两种类型的对抗扰动组合用于训练：小面积高强度的梯度引导对抗补丁 + 全局低强度的 $\ell_\infty$ 约束扰动。最终对抗样本为：

$$\delta = \text{Apply}(\delta_p \odot \mathbf{M}, \mathbf{x}) + \delta_g$$

其中 $\delta_p$ 是补丁扰动（$\|\delta_p\|_\infty \leq \beta$），$\delta_g$ 是全局扰动（$\|\delta_g\|_\infty \leq \epsilon$），$\mathbf{M}$ 是梯度引导选择的二值 mask。

### 关键设计

1. **从 $\ell_\infty$-AT 到 Patch-based AT 的扩展**：将攻击预算形式从全局约束改为局部 mask 约束。对检测任务，每个 bounding box 内可能包含一个对抗补丁。关键公式：

$$\theta = \arg\min_\theta \mathbb{E}_\mathbf{x}\left\{\max_{\|\delta_p \odot \mathbf{M}\|_\infty \leq \beta} \mathcal{L}_d(f_\theta(\mathbf{x} + \delta_p \odot \mathbf{M}), y)\right\}$$

补丁在 bounding box 内随机放置（以 bbox 中心为均值的高斯分布采样），避免模型利用补丁位置做回归（信息泄露），补丁边长 $s = \lambda \cdot \sqrt{w_{bbox}^2 + h_{bbox}^2}$。每个 bbox 有 50% 概率附加补丁以维持干净对象的检测能力。

2. **梯度引导的补丁分割与选择**：将采样的补丁分割为 $n \times n$ 个子补丁（默认 $N=64$，即 $8\times8$），一次前向+反向传播后计算每个子补丁的平均梯度范数，选择梯度最大的 top 50% 区域构建二值 mask $\mathbf{M}$。这一设计的核心优势：

    - 仅需单次前向/反向传播，计算开销几乎为零
    - 梯度大的区域是模型的脆弱区域，在此处添加扰动最有效
    - 生成不规则形状的扰动区域，更接近真实物理攻击的多样性（而非简单方形补丁）

3. **局部补丁 + 全局噪声的组合策略**：这是 PBCAT 最关键的创新。仅用小补丁训练无法防御大面积纹理攻击，但直接增大补丁面积又会导致训练崩溃（物体信息被严重破坏）。解决方案：

    - 引入 $\ell_\infty$ 约束的全局扰动（$\epsilon = 4/255$），强度低不会破坏物体信息
    - 全局噪声覆盖整张图像，补偿小补丁无法覆盖的区域
    - 两种扰动互补：补丁提供局部高强度对抗信号，全局噪声提供空间覆盖

### 损失函数 / 训练策略

- 基于 FreeAT 加速：复用梯度扰动，避免完整 PGD 内循环的高计算成本
- 使用对抗预训练的 backbone（来自 AdvOD）
- 补丁扰动步长 $\alpha = 8/255$，强度 $\beta = 64/255$
- FreeAT replay 参数 $r = 8$
- 在 MS-COCO 上训练通用检测器，直接迁移到安全关键的行人检测任务
- 补丁和全局扰动在同一次反向传播中同时计算梯度

## 实验关键数据

### 主实验

**行人检测 AP50（自适应白盒攻击）**

| 方法 | Clean(Inria) | AdvPatch | Clean(Synth) | AdvTexture | AdvCaT |
|------|-------------|----------|-------------|-----------|--------|
| Vanilla | 96.2 | 37.3 | 86.4 | 0.2 | 0.3 |
| SAC | 96.2 | 57.1 | 85.4 | 0.3 | 0.6 |
| Jedi | 92.3 | 64.4 | 88.1 | 2.3 | 0.7 |
| $\ell_\infty$-AT (AdvOD) | 95.9 | 56.1 | 92.5 | 30.5 | 39.6 |
| **PBCAT** | 95.4 | **77.6** | 92.5 | **60.2** | **56.4** |

### 消融实验

**各组件贡献（行人检测 AP50）**

| Patch | Global | Gradient | AdvPatch | AdvTexture | AdvCaT |
|-------|--------|----------|----------|-----------|--------|
| ✓ | | | 35.4 | 1.6 | 0.8 |
| ✓ | ✓ | | 72.8 | 24.9 | 19.5 |
| ✓ | ✓ | ✓ | **77.6** | **63.3** | **56.4** |

**子补丁数量影响**

| 子补丁数 | AdvPatch | AdvTexture | AdvCaT | 说明 |
|---------|----------|-----------|--------|------|
| 16 | **78.3** | 50.8 | 46.2 | 过粗，纹理防御弱 |
| 64 | 77.6 | **60.2** | 56.4 | 最佳平衡 |
| Pixel-level | 67.4 | 20.4 | **59.4** | 过细，补丁防御退化 |

### 关键发现

- 全局噪声是防御纹理攻击的关键——仅用补丁训练，AdvTexture AP 仅 1.6%；加入全局噪声跃升至 24.9%
- 梯度引导选择进一步提升所有攻击的鲁棒性，比随机选择高约 10-20%
- 所有非 AT 防御方法（LGS, SAC, Jedi 等）在自适应攻击下对纹理攻击基本无效（AP < 6%）
- PBCAT 在 FCOS 和 DN-DETR 上同样有效，证明方法的检测器无关性
- 在物理世界视频评估中，PBCAT 成功检测穿着对抗纹理服装的行人

## 亮点与洞察

- **统一防御框架**：首次用单一 AT 方法同时有效防御补丁和纹理两类物理攻击，填补了一个重要空白
- **组合扰动的巧妙设计**：局部强扰动+全局弱扰动的组合既避免训练崩溃，又提供足够的空间覆盖
- **高效的梯度引导**：与之前需要多次前向推理搜索最优位置的方法相比，仅需一次前向/反向传播
- **实际安全价值**：直接应对穿着对抗服装隐身的真实威胁，对视频监控和自动驾驶安全意义重大

## 局限与展望

- 与大多数 AT 工作类似，PBCAT 会轻微降低干净数据上的准确率
- 物理可实现攻击的鲁棒性与干净准确率之间是否存在内在权衡仍是开放问题
- 当前补丁的形状仍为规则的方形区域，物理攻击可能有更不规则的形状
- 可探索引入风格迁移或 GAN 生成更多样化的训练扰动

## 相关工作与启发

- 弥合了 $\ell_\infty$-AT（非物理攻击）和 patch-based AT（仅限补丁）之间的鸿沟
- FreeAT 加速策略使对抗训练成本接近标准训练，推动 AT 在检测任务的实用化
- 梯度引导分割的思路可推广到其他需要空间选择的对抗训练场景
- 对自动驾驶安全领域的攻防研究具有直接参考价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 组合扰动策略和梯度引导选择设计精巧，统一了补丁和纹理防御
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖多种攻击/检测器/数据集，消融极为详尽，物理世界验证
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法描述循序渐进，实验设计严谨
- **价值**: ⭐⭐⭐⭐⭐ — 直接提升安全关键检测系统的鲁棒性，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] MOBA: A Material-Oriented Backdoor Attack against LiDAR-based 3D Object Detection](../../AAAI2026/autonomous_driving/moba_a_material-oriented_backdoor_attack_against_lidar-based_3d_object_detection.md)
- [\[ICCV 2025\] 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation](3d_gaussian_splatting_driven_multiview_robust_physical_adver.md)
- [\[ICCV 2025\] EVT: Efficient View Transformation for Multi-Modal 3D Object Detection](evt_efficient_view_transformation_for_multi-modal_3d_object_detection.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[AAAI 2026\] Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning](../../AAAI2026/autonomous_driving/backdoor_attacks_on_open_vocabulary_object_detectors_via_multi-modal_prompt_tuni.md)

</div>

<!-- RELATED:END -->
