---
title: >-
  [论文解读] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models
description: >-
  [CVPR 2026][图像生成][机器遗忘] RAZOR通过比率感知的梯度评分联合衡量遗忘压力与保留对齐来选择最关键的层/注意力头，配合三部分约束损失和迭代扩展机制，在CLIP、Stable Diffusion和VLM上实现了精准高效的目标遗忘且量化后性能不退化。
tags:
  - CVPR 2026
  - 图像生成
  - 机器遗忘
  - 比率感知编辑
  - 多层选择
  - 模型不可学习
  - 量化鲁棒性
---

# RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2603.14819](https://arxiv.org/abs/2603.14819)  
**代码**: https://github.com/raviranjan-ai/RAZOR-cvpr2026 (有)  
**领域**: 模型安全 / 机器遗忘  
**关键词**: 机器遗忘, 比率感知编辑, 多层选择, 模型不可学习, 量化鲁棒性

## 一句话总结

RAZOR通过比率感知的梯度评分联合衡量遗忘压力与保留对齐来选择最关键的层/注意力头，配合三部分约束损失和迭代扩展机制，在CLIP、Stable Diffusion和VLM上实现了精准高效的目标遗忘且量化后性能不退化。

## 研究背景与动机

**领域现状**：大规模视觉-语言模型和扩散模型在海量数据上训练，不可避免地嵌入了敏感或不期望的信息。GDPR等法规要求模型能"忘记"特定数据，但从头重训代价极高，因此机器遗忘（machine unlearning）作为替代方案受到关注。

**现有痛点**：现有方法各有局限——(1) 梯度上升类方法（如SalUn）仅依据遗忘集梯度选参数，忽视保留集冲突，导致遗忘不彻底或保留性能严重下降；(2) 单层编辑方法（如SLUG）虽高效但过于脆弱，当知识分布在多层时一个层不够；(3) 全模型更新方法在量化后遗忘效果严重退化。

**核心矛盾**：参数选择只看遗忘集显著性，保留冲突事后才弥补——这种"先忘后补"的顺序策略无法避免遗忘与保留动态的耦合。

**本文目标** (1) 如何同时考虑遗忘压力和保留对齐来选择编辑位置；(2) 如何在多层编辑中显式控制遗忘-保留权衡；(3) 如何保证量化后遗忘效果不崩溃。

**切入角度**：将遗忘集和保留集的梯度同时计算，用遗忘梯度强度与遗忘-保留梯度正交性的乘积来衡量每层的"编辑价值"，高评分意味着强遗忘影响且低保留损害。

**核心 idea**：用遗忘/保留梯度的比率感知评分联合定位关键层，通过约束多目标损失和迭代扩展实现精准多层遗忘。

## 方法详解

### 整体框架

给定预训练模型和遗忘/保留数据集，RAZOR分三阶段工作：(1) 对每层计算遗忘和保留梯度，用比率感知评分选出高影响低损害的层子集 $\mathcal{K}$；(2) 在选中层上通过三部分约束损失进行一步或少步梯度更新；(3) 若遗忘指标未达阈值，迭代扩展 $\mathcal{K}$ 直到收敛。整个流程是模型无关的，统一适用于CLIP、Stable Diffusion和VLM。

### 关键设计

1. **比率感知梯度评分（Ratio-Aware Gradient Scoring）**:

    - 功能：为每层/注意力头计算一个综合评分，决定是否需要编辑
    - 核心思路：对每层 $l$ 一次性计算遗忘梯度 $g_l^f = \nabla_{\theta_l}\mathcal{L}_{\text{forget}}$ 和保留梯度 $g_l^r = \nabla_{\theta_l}\mathcal{L}_{\text{retain}}$，评分为 $\phi(l) = \frac{\|g_l^f\|_2}{\|\theta_l\|_2+\varepsilon} \cdot (1-\cos(g_l^f, g_l^r))^\alpha$。第一项衡量遗忘梯度相对参数的显著性，第二项衡量遗忘与保留梯度的方向差异——越正交代表编辑对保留影响越小
    - 设计动机：之前方法仅用遗忘梯度排序，忽视保留冲突。RAZOR将两者在选择阶段就联合考虑，从根本上避免"先忘后补"的耦合问题

2. **三部分约束损失（Three-Loss Objective）**:

    - 功能：在选中层上执行约束更新，显式平衡遗忘、保留和稳定性
    - 核心思路：$\mathcal{L}_{\text{RAZOR}} = \mathcal{L}_{\text{retain}} + \lambda_f \rho \mathcal{L}_{\text{forget}} + \lambda_m \mathcal{L}_{\text{mismatch}}$。保留损失维持任务性能（如CLIP的InfoNCE），遗忘损失通过梯度上升拉开遗忘对齐（余弦嵌入损失），错配损失正则化嵌入相似度相对冻结模型的漂移。比率超参 $\rho$ 显式控制遗忘强度
    - 设计动机：单一遗忘目标要么遗忘不彻底要么破坏保留，三部分设计解耦三个目标，各自有梯度方向指导

3. **迭代扩展机制（Iterative Growing of $\mathcal{K}$）**:

    - 功能：如果初始选中层的更新未达遗忘阈值，动态添加新层
    - 核心思路：每轮重新计算更新后参数的评分 $\phi_t(l)$，将最高评分的未编辑层加入 $\mathcal{K}$ 并更新，最多迭代6轮
    - 设计动机：避免一次性选中过多层导致过度编辑，渐进策略确保精准遗忘同时控制附带损害

### 损失函数 / 训练策略

模块化设计——三个损失的具体形式随底层模型变化：CLIP用InfoNCE+余弦嵌入+相似度漂移正则；Stable Diffusion用去噪损失+文本编码器余弦损失+生成引导漂移正则；VLM用InfoNCE+视觉编码器余弦损失+中性QA漂移正则。每层学习率 $\eta_l$ 通过轻量二分搜索确定最大稳定步长。

## 实验关键数据

### 主实验

| 方法 | CIFAR-10 M1↓ | CIFAR-10 M4↑ | CIFAR-10 M5↑ | ImageNet M1↓ | ImageNet M4↑ | LAION M1↓ | LAION M4↑ |
|------|-------------|-------------|-------------|-------------|-------------|----------|----------|
| SSD | 52.00 | 25.00 | 97.50 | 52.50 | 30.00 | 42.00 | 48.00 |
| SalUn | 97.00 | 83.00 | 84.50 | 88.00 | 84.00 | 48.00 | 88.00 |
| SLUG | 67.50 | 87.50 | 96.50 | 68.00 | 88.00 | 48.00 | 88.00 |
| **RAZOR** | **52.50** | **89.00** | **100.00** | **53.50** | **92.00** | **40.00** | **94.00** |

SD-V3 UnlearnCanvas 风格/物体遗忘：

| 方法 | 风格UA↑ | 风格IRA↑ | 风格CRA↑ | 物体UA↑ | 物体IRA↑ | 物体CRA↑ |
|------|---------|---------|---------|---------|---------|---------|
| ESD | 99.62 | 89.97 | 98.86 | 97.44 | 68.47 | 82.37 |
| SalUn | 90.36 | 92.33 | 97.02 | 91.06 | 98.35 | 99.59 |
| SLUG | 88.20 | 85.59 | 91.00 | 85.44 | 79.50 | 91.00 |
| **RAZOR** | **99.40** | **98.97** | **100.00** | **98.80** | **98.35** | **100.00** |

### 消融实验

| 配置 | 效率权衡分↑ | 时间↓ | 内存↓ | 存储↓ |
|------|-----------|-------|-------|-------|
| ESD | 11.97 | 6163s | 17.8GB | 4.30GB |
| SLUG | 59.42 | 39s | 3.6GB | 0.04GB |
| **RAZOR** | **66.86** | 78s | 4.2GB | 0.06GB |

VLM（LLaVA-1.6-8B）身份遗忘：遗忘准确率降至2.2%，同时MME认知/感知保持301/1362+，GQA保持60+。

### 关键发现

- RAZOR在5个CLIP指标上全面领先，M3（隐私泄露）和M5（保留稳定性）均达满分级别
- 量化鲁棒性突出：4-bit量化后RAZOR各指标仅降约0.5%，而全模型更新方法(SSD/SalUn)降5-10%
- 效率权衡分最高（66.86），比SLUG的59.42高12%，兼顾了遗忘质量和计算效率
- SD-V3上6个指标中5个最优，证明方法对新架构的泛化性

## 亮点与洞察

- **比率感知评分的核心洞察**：将遗忘与保留的梯度方向差异纳入选层标准，解决了"先忘后补"的根本性设计缺陷。这种联合评分思路可迁移到任何需要选择性参数更新的场景（如持续学习中避免灾难性遗忘）。
- **编辑子集极小**：$|\mathcal{K}| \ll |\mathcal{L}|$，仅存储变更权重，存储开销0.06GB远低于全模型方法的4GB+，真正实现"外科手术式"遗忘。
- **模型无关的模块化损失设计**：同一框架跨CLIP/SD/VLM三大模型族，只需替换损失形式表格中的具体损失函数即可适配。

## 局限与展望

- 超参数（$\alpha$, $\rho$, $\tau$, $\lambda_f$, $\lambda_m$）较多，不同模型/任务可能需要调参
- 迭代扩展最多6轮的硬限制可能在极端分布式知识的模型上不够
- 仅在身份/风格/物体遗忘上验证，概念级或更细粒度的遗忘效果未知
- 对抗性恢复攻击（beyond量化）的鲁棒性未测试

## 相关工作与启发

- **vs SLUG**: SLUG限制单层编辑，知识分布时失败；RAZOR多层编辑+迭代扩展，灵活性和覆盖率远超
- **vs SalUn**: SalUn仅用遗忘梯度选参数，保留冲突事后弥补；RAZOR在选择阶段就联合考虑遗忘和保留
- **vs ESD**: ESD在采样时用负引导抑制概念但非真正删除；RAZOR在权重层面删除，更彻底

## 评分

- 新颖性: ⭐⭐⭐⭐ 比率感知评分和模型无关框架设计有新意，但整体仍是梯度选择+约束优化范式
- 实验充分度: ⭐⭐⭐⭐⭐ 跨三大模型族（CLIP/SD/VLM）+量化鲁棒性+效率对比，覆盖全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式完整，但符号较多需要反复对照
- 价值: ⭐⭐⭐⭐ 提供了实用的遗忘框架，量化鲁棒性是重要加分项

<!-- RELATED:START -->

## 相关论文

- [Pluggable Pruning with Contiguous Layer Distillation for Diffusion Transformers](pluggable_pruning_with_contiguous_layer_distillation_for_diffusion_transformers.md)
- [From Inpainting to Layer Decomposition: Repurposing Generative Inpainting Models for Image Layer Decomposition](from_inpainting_to_layer_decomposition_repurposing_generative_inpainting_models_.md)
- [EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](segquant_diffusion_model_quantization.md)
- [Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)

<!-- RELATED:END -->
