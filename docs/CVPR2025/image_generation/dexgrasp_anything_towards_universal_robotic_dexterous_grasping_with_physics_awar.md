---
title: >-
  [论文解读] DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness
description: >-
  [CVPR 2025][图像生成][灵巧抓取] 提出 DexGrasp Anything，将三种物理约束（表面拉力、外穿透排斥、自穿透排斥）集成到扩散模型的训练和采样中，配合 LLM 增强的点云+语义物体表示，在 5 个基准上达到 SOTA 灵巧抓取性能（Suc.6 57.5%，比 SceneDiffuser 提升 2.2×）。
tags:
  - CVPR 2025
  - 图像生成
  - 灵巧抓取
  - 物理约束扩散
  - 表面拉力
  - 穿透排斥
  - LLM增强
---

# DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness

**会议**: CVPR 2025  
**arXiv**: [2503.08257](https://arxiv.org/abs/2503.08257)  
**代码**: https://github.com/4DVLab/DexGrasp-Anything  
**领域**: 机器人 / 扩散模型  
**关键词**: 灵巧抓取, 物理约束扩散, 表面拉力, 穿透排斥, LLM增强

## 一句话总结
提出 DexGrasp Anything，将三种物理约束（表面拉力、外穿透排斥、自穿透排斥）集成到扩散模型的训练和采样中，配合 LLM 增强的点云+语义物体表示，在 5 个基准上达到 SOTA 灵巧抓取性能（Suc.6 57.5%，比 SceneDiffuser 提升 2.2×）。

## 研究背景与动机

**领域现状**：灵巧手（如 ShadowHand）抓取需要为任意物体生成合理的多指抓取姿态。基于扩散的抓取生成可以产生多样化姿态，但经常出现手-物穿透、手指自穿透、非接触悬空等物理不合理的结果。

**现有痛点**：(1) 现有扩散方法不感知物理——生成后才检查穿透，此时已太迟；(2) 物体表示仅用几何（点云），缺乏语义信息（如"杯子的手柄适合抓握"）；(3) 数据集规模不足（现有最大的 DexGraspNet 130 万抓取 / 5355 物体）。

**核心矛盾**：扩散模型的自由度使其容易生成物理不合理的抓取，需要在训练和采样过程中注入物理约束而不破坏生成多样性。

**切入角度**：将三种物理力直接集成到扩散过程：训练时用估计的干净样本收物理约束梯度，采样时用 classifier guidance 将物理约束作为引导信号。

**核心 idea**：表面拉力（拉手掌向物体）+ 外穿透排斥力 + 自穿透排斥力，集成到扩散训练和采样的每一步 + LLM 语义描述增强物体感知。

## 方法详解

### 关键设计

1. **三重物理约束**：(a) 表面拉力——将手内表面点拉向物体最近表面，促进接触；(b) 外穿透排斥力——检测手-物穿透并推离；(c) 自穿透排斥力——检测手指间碰撞并推离。三种力通过球面高斯约束建模

2. **物理感知训练+采样**：训练时从带噪预测中估计干净样本，计算物理损失并加到扩散训练损失中。采样时用 classifier guidance 将物理约束作为引导信号

3. **LLM 增强物体表示**：Point Transformer 编码几何 + BERT 编码 LLM 生成的语义描述（如"圆柱形杯子，有一个小把手"），融合几何和语义信息

4. **大规模数据集**：340 万抓取姿态，15,698 个物体（现有最大的 2.6×）

## 实验关键数据（DexGraspNet 测试集）

| 方法 | Suc.6↑ | Suc.1↑ | 穿透(mm)↓ | 多样性↑ |
|------|--------|--------|----------|--------|
| SceneDiffuser | 26.6 | 66.9 | 31.0 | 0.15 |
| UGG | 46.9 | 79.0 | 25.2 | 0.14 |
| **DexGrasp Anything** | **57.5** | **90.6** | **17.8** | **0.23** |

在 5 个基准上均 SOTA。MultiDex: Suc.6 **79.1%**, Suc.1 **98.1%**。

## 评分
- 新颖性: ⭐⭐⭐⭐ 物理约束集成到扩散训练+采样双重机制有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个基准+最大数据集+消融
- 写作质量: ⭐⭐⭐⭐ 清晰
- 价值: ⭐⭐⭐⭐⭐ 对机器人灵巧操作有重大推动

<!-- RELATED:START -->

## 相关论文

- [InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [CustAny: Customizing Anything from A Single Example](custany_customizing_anything_from_a_single_example.md)
- [UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models](uibdiffusion_universal_imperceptible_backdoor_attack_for_diffusion_models.md)
- [UniReal: Universal Image Generation and Editing via Learning Real-world Dynamics](unireal_universal_image_generation_and_editing_via_learning_real-world_dynamics.md)
- [OminiControl: Minimal and Universal Control for Diffusion Transformer](../../ICCV2025/image_generation/ominicontrol_minimal_and_universal_control_for_diffusion_transformer.md)

<!-- RELATED:END -->
