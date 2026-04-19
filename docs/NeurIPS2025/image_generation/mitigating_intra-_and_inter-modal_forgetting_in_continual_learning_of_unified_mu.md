---
title: >-
  [论文解读] Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models
description: >-
  [NeurIPS 2025][图像生成][统一多模态生成模型] 提出Modality-Decoupled Experts (MoDE)，通过将文本和图像的适配器解耦为独立的T-MoE和V-Adapter子空间，配合知识蒸馏，在统一多模态生成模型的持续指令微调中同时缓解模态内遗忘和模态间遗忘。
tags:
  - NeurIPS 2025
  - 图像生成
  - 统一多模态生成模型
  - 模态间遗忘
  - 模态内遗忘
  - LoRA专家混合
  - 知识蒸馏
---

# Mitigating Intra- and Inter-modal Forgetting in Continual Learning of Unified Multimodal Models

**会议**: NeurIPS 2025  
**arXiv**: [2512.03125](https://arxiv.org/abs/2512.03125)  
**代码**: [GitHub](https://github.com/Christina200/MoDE-official)  
**领域**: 多模态生成 / 持续学习  
**关键词**: 统一多模态生成模型, 模态间遗忘, 模态内遗忘, LoRA专家混合, 知识蒸馏

## 一句话总结

提出Modality-Decoupled Experts (MoDE)，通过将文本和图像的适配器解耦为独立的T-MoE和V-Adapter子空间，配合知识蒸馏，在统一多模态生成模型的持续指令微调中同时缓解模态内遗忘和模态间遗忘。

## 研究背景与动机

统一多模态生成模型（UMGMs，如Chameleon、Janus-Pro）将视觉理解和图像生成统一在单个自回归框架中。然而，在持续学习新任务时面临严重的灾难性遗忘。

现有持续学习研究主要关注的是**模态内遗忘**（intra-modal）：即在同一输出模态内学新任务忘旧任务（如连续学习多个VQA任务）。但UMGMs引入了一个全新且未被充分探索的挑战——**模态间遗忘**（inter-modal）：在文本理解任务上微调时，模型的图像生成能力也会退化。

作者通过实验验证了这一现象：在Chameleon模型上顺序微调3个VQA数据集后，图像生成质量显著下降，且文本-图像对齐性恶化。理论上，这源于**模态梯度冲突**：

**定义**：文本生成损失的梯度 $g_t = \nabla_\theta \mathcal{L}_t$ 和图像生成损失的梯度 $g_v = \nabla_\theta \mathcal{L}_v$ 在内积 $\langle g_v, g_t \rangle < 0$ 时发生冲突。此时对文本任务的SGD更新 $\theta \leftarrow \theta - \eta g_t$ 会导致视觉损失增加：$\Delta \mathcal{L}_v = -\eta \langle g_t, g_v \rangle + \frac{\eta^2}{2} g_t^\top H_v g_t$，当梯度冲突时首项为正，即导致视觉性能退化。

现有方法（如CL-MoE、Model Tailor）无法同时有效解决两种遗忘。

## 方法详解

### 整体框架

MoDE在冻结的UMGM线性层上集成两类轻量级适配器：处理文本token的T-MoE（Mixture-of-Experts LoRA）和处理图像token的V-Adapter（单个LoRA），实现模态解耦。在持续指令微调中，只训练MoDE组件，UMGM原始参数保持冻结。

### 关键设计

1. **V-Adapter（视觉LoRA适配器）**

   专门处理图像token的LoRA模块，用于视觉理解和图像生成。采用标准LoRA形式：
   $$\Delta W = \frac{\alpha}{r} BA$$
   
   输入token表示 $h$ 经过修改的线性变换为 $f(h) = hW^\top + \frac{\alpha}{r}hA^\top B^\top$。设计动机：将图像相关的参数更新隔离在独立子空间中，避免与文本更新互相干扰。

2. **T-MoE（文本LoRA专家混合）**

   针对文本token的MoE-LoRA，通过路由器将输入分配给多个专家：
   $$g_j(x) = \text{softmax}(xW_g)_j$$
   $$f(h) = hW^\top + \frac{\alpha}{r}\sum_{j=1}^{n} g_j(x) hA_j^\top B_j^\top$$
   
   其中 $n$ 为专家数量。路由机制使不同任务自动激活不同专家组合，从而缓解模态内遗忘。动机：利用专家的多样性实现任务特异性适配，避免参数覆盖。

3. **模态解耦的理论保证**

   MoDE可证明地将模态间干扰从 $\mathcal{O}(\eta)$ 降低到 $\mathcal{O}(\eta^2)$：
   
   当T-MoE参数 $\phi$ 更新时，对视觉损失的影响为：
   $$\Delta \mathcal{L}_v = \frac{\eta^2}{2} \lambda_{\max}(\nabla^2_{\phi\phi}\mathcal{L}_v) \|\nabla_\phi \mathcal{L}_t\|^2$$
   
   由于 $\nabla_\phi \mathcal{L}_v = 0$（T-MoE不直接处理视觉token），一阶冲突项消失，仅剩二阶项。

### 损失函数 / 训练策略

- **T-MoE训练**：使用标准交叉熵损失 $\mathcal{L}_{\text{CE}} = -\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(X_i^{ans} | X^{img}, X^{ins}, X_{<i}^{ans})$

- **V-Adapter训练**：交叉熵损失 + 知识蒸馏损失的加权组合：
  $$\mathcal{L}_{\text{V-Adapter}} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{KD}}$$
  
  其中KD损失对齐教师（冻结预训练模型）和学生（V-Adapter增强模型）的softened logits：
  $$\mathcal{L}_{\text{KD}} = \beta^2 \sum_{i=1}^{L} D_{\text{KL}}(\text{Softmax}(z_i^T/\beta) \| \text{Softmax}(z_i^S/\beta))$$
  
  参考数据从LAION-5B采样，$\lambda=0.3$。

## 实验关键数据

### 主实验

在Chameleon上对5个数据集顺序微调（ScienceQA→TextVQA→ImageNet→GQA→VizWiz）：

| 方法 | 文本对齐↑ | 图像对齐↑ | FID↓ | 理解准确率↑ | 遗忘↓ | 综合Δ↓ |
|------|----------|----------|------|-----------|-------|--------|
| Zero-shot | 0.2592 | 0.5205 | 52.13 | 22.48 | - | 34.84 |
| Seq LoRA | 0.2162 | 0.5150 | 56.12 | 28.43 | 35.33 | 28.57 |
| MoELoRA | 0.2248 | 0.5095 | 65.16 | 33.01 | 30.77 | 24.31 |
| CL-MoE | 0.2081 | 0.5150 | 65.87 | 32.86 | 30.95 | 24.46 |
| **MoDE** | **0.2458** | **0.5170** | **53.74** | **33.47** | **25.99** | **22.78** |

### 消融实验

| 配置 | FID↓ | 理解准确率↑ | 遗忘↓ | 说明 |
|------|------|-----------|-------|------|
| Chameleon原始 | 52.13 | 22.48 | - | 基线 |
| + T-MoE LoRA | 51.28 | 33.03 | 28.65 | 仅文本专家，图像完全保留 |
| + MoDE w/o KD | 54.61 | 33.07 | 26.49 | 无蒸馏，图像质量下降 |
| + MoDE (完整) | 53.74 | 33.47 | 25.99 | KD有效平衡两种能力 |

### 关键发现

- MoDE是唯一能同时缓解模态内和模态间遗忘的方法：FID 53.74接近预训练的52.13，同时理解准确率33.47最优
- MoELoRA和CL-MoE虽提升理解能力，但共享参数导致图像生成严重退化（FID > 65 vs 预训练52.13）
- DualPrompt虽保留图像质量，但理解能力学习不充分
- 知识蒸馏对保持图像生成能力至关重要（w/o KD的FID 54.61 vs 完整MoDE的53.74）

## 亮点与洞察

- 首次系统性地识别和研究UMGMs中的模态间遗忘问题，填补了持续学习研究的空白
- 梯度冲突的理论分析简洁有力，直接导出了模态解耦的设计动机
- MoDE的解耦设计可证明地将干扰降低一个量级（$\mathcal{O}(\eta) \to \mathcal{O}(\eta^2)$）
- 方法轻量且即插即用，适用于各种基于Transformer的UMGM架构

## 局限与展望

- 目前验证的任务序列相对较短（5个任务），更长序列下的表现待验证
- V-Adapter使用单个LoRA，在大量图像相关任务持续增加时可能也会遗忘
- 知识蒸馏需要额外的存储开销和参考数据
- 未探索任务间的正向迁移（forward transfer）

## 相关工作与启发

- 模态梯度冲突的分析思路可扩展到其他多任务/多模态学习场景
- 解耦设计原则（不同模态使用独立参数空间）对统一模型的持续演化有指导意义
- KD作为"记忆锚"的策略可与其他持续学习方法结合

## 评分

- **新颖性**: ⭐⭐⭐⭐ 模态间遗忘的问题定义新颖且重要，但解决方案（MoE + KD）是较成熟的技术组合
- **实验充分度**: ⭐⭐⭐⭐ 多个基线、消融实验、定性结果完整，但数据集和模型多样性可以更丰富
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义→理论分析→方法设计的逻辑链非常清晰
- **价值**: ⭐⭐⭐⭐ 对UMGMs的持续学习提供了重要的基准和解决方案

<!-- RELATED:START -->

## 相关论文

- [Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [Joint Diffusion Models in Continual Learning](../../ICCV2025/image_generation/joint_diffusion_models_in_continual_learning.md)
- [Show-o2: Improved Native Unified Multimodal Models](show-o2_improved_native_unified_multimodal_models.md)
- [ConceptGuard: Continual Personalized Text-to-Image Generation with Forgetting and Confusion Mitigation](../../CVPR2025/image_generation/conceptguard_continual_personalized_text-to-image_generation_with_forgetting_and.md)
- [Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning](../../ECCV2024/image_generation/diffusion-driven_data_replay_a_novel_approach_to_combat_forgetting_in_federated_.md)

<!-- RELATED:END -->
