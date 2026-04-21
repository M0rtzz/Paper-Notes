---
title: >-
  [论文解读] L2D: Large Language Models to Diffusion Finetuning
description: >-
  [ICML 2025][自监督学习][LLM微调] 提出L2D微调方法，将预训练LLM视为单步扩散模型，引入并行扩散路径实现多步推理缩放，不修改原始权重即可随推理步数增加获得单调递增的准确率，在4个LLM上的数学/编码/推理任务上取得一致提升。
tags:
  - ICML 2025
  - 自监督学习
  - LLM微调
  - 扩散框架
  - 测试时缩放
  - LoRA
  - Classifier-free Guidance
---

# L2D: Large Language Models to Diffusion Finetuning

**会议**: ICML 2025  
**arXiv**: [2501.15781](https://arxiv.org/abs/2501.15781)  
**代码**: [github.com/SakanaAI/L2D](https://github.com/SakanaAI/L2D)  
**领域**: LLM / 测试时计算缩放  
**关键词**: LLM微调, 扩散框架, 测试时缩放, LoRA, Classifier-free Guidance

## 一句话总结
提出L2D微调方法，将预训练LLM视为单步扩散模型，引入并行扩散路径实现多步推理缩放，不修改原始权重即可随推理步数增加获得单调递增的准确率，在4个LLM上的数学/编码/推理任务上取得一致提升。

## 研究背景与动机
**领域现状**：自回归LLM在语言领域取得巨大成功，但本质上缺乏按需缩放推理计算的能力——每个token的计算量固定，无法为关键决策投入更多计算。

**现有痛点**：(1) 现有测试时缩放方法（prompting、token-level search）受限于生成token空间，扩展性有限；(2) 语言扩散模型从头训练远远落后于自回归对应物，质疑其在语言领域的适用性；(3) LoRA等参数高效微调虽然轻量但无法提供推理时缩放能力。

**核心矛盾**：如何在保留LLM已有"系统1"理解能力的同时，赋予其扩散框架的推理时缩放特性？

**切入角度**：将LLM的下一token预测（无先验信息 $t=0$）视为单步扩散的特例，通过微调引入多步扩散能力作为自然扩展。

**核心 idea**：不从头训练语言扩散模型，而是在预训练LLM上加一条并行扩散路径，复用其知识实现多步推理。

## 方法详解

### 整体框架
L2D在冻结的LLM主路径旁引入一条并行"扩散路径"。训练时，对每个目标token $y^k$，采样timestep $t$ 和噪声token $x_t = t \cdot V_y + (1-t) \cdot x_0$（$x_0 \sim \mathcal{N}(0, \sigma^2 I)$），扩散路径通过交叉注意力访问主路径的KV缓存来预测 $y$。推理时，从纯噪声出发，通过Euler积分逐步去噪，每步采样token embedding后更新 $x_t$，最终输出最终预测。

### 关键设计
1. **并行扩散路径架构**:
    - 功能：在冻结LLM旁构建一条完全并行的轻量级Transformer路径
    - 核心思路：扩散路径 $f_{\theta_d}$ 与主路径 $f_{\theta_l}$ 同层数，每层包含MLP（复用主路径权重+LoRA）和交叉注意力（query来自扩散token，key/value来自主路径自注意力的KV缓存）。仅在最终层通过加权和 $f_{\theta_l} + w_d(t) f_{\theta_d}$ 融合，其中 $w_d(t) = w_{\theta_d}(t) - w_{\theta_d}(0)$ 确保 $t=0$ 时不影响原始LLM输出
    - 设计动机：(1) 冻结主路径保护原始能力；(2) 共享KV缓存使推理时主路径只需计算一次；(3) 独立timestep采样使训练可跨序列并行化

2. **交叉熵扩散训练**:
    - 功能：用标准CE损失（而非MSE）训练语言扩散模型
    - 核心思路：损失 $L^{CE}(\theta) = -\mathbb{E}_{x_0, x_1, t}[\log(f_\theta(x_t, t, c)_y)]$，其中 $x_t = t \cdot V_y + (1-t) \cdot x_0$。扩散路径仍输出vocabulary logits，但额外接收含有目标token部分信息的 $x_t$（$t=0$为纯噪声，$t=1$为完美信息）。采用rectified flow调度 $\alpha_t = t, \beta_t = 1-t$
    - 设计动机：CE损失与标准LM训练直接对接——$t=0$时等价于标准next-token prediction，使L2D成为LM的自然扩展

3. **Classifier-Free Guidance + 自适应ODE求解**:
    - 功能：引入扩散领域的强力引导技术和自适应计算分配
    - 核心思路：训练时以概率dropout类别embedding $g_j$，推理时构造引导预测 $\hat{x}_g = w_g f_\theta(x_t,t,g_j,c) - (1-w_g) f_\theta(x_t,t,g_0,c)$。自适应ODE求解器（二阶Runge-Kutta）根据扩散误差自动调节每个token的推理步数
    - 设计动机：guidance使LLM获得面向特定任务的专家级生成能力；自适应求解器让模型自主决定难题多花计算

### 损失函数 / 训练策略
使用交叉熵扩散损失 $L^{CE}$ 训练1个epoch，AdamW优化器，100步warmup+线性衰减，$\sigma=64$（高噪声标准差使扩散步集中在有意义的区间），扩散维度 $\bar{d}=256$，LoRA秩8。推理默认采用midpoint求解器+8个离散步（15次 $f_{\theta_d}$ 评估）。

## 实验关键数据

### 主实验（跨4个LLM）

| 模型 | 方法 | 数学 | 编码 | 通用知识 | 平均 | 参数量 |
|------|------|------|------|---------|------|--------|
| Llama 1B | 基线 | 11.93 | 47.63 | 28.54 | 28.54 | - |
| | +LoRA ft. | 18.68 | 44.82 | - | 29.97 | 3M |
| | +Full ft. | 22.94 | 31.04 | - | 27.04 | 1235M |
| | **+L2D** | **28.02** | **49.80** | - | **35.50** | 73M |
| Qwen 2.5 7B | 基线 | 11.98 | 73.01 | - | 46.65 | - |
| | +LoRA ft. | 51.95 | 83.83 | - | 63.34 | 10M |
| | **+L2D** | **63.21** | **84.00** | - | **67.58** | 233M |

### 扩展实验（Llama 1B）

| 方法 | 数学 | 编码 | 全部任务 |
|------|------|------|---------|
| L2D (15步) | 28.02 | 49.80 | 35.50 |
| L2D (127步) | 28.39 | 51.90 | 36.24 |
| L2D (自适应solver) | 30.26 | 49.53 | 36.34 |
| L2D + token search | **35.95** | 49.79 | **38.57** |
| LoRA ft. → L2D | 29.19 | 48.45 | 35.51 |

### 关键发现
- L2D的推理步数增加→准确率单调递增，复现了扩散模型的缩放特性
- 自适应求解器在MATH/MMLU等难题上自动分配更多步数（平均118步 vs 固定15步）
- L2D与传统微调和搜索正交——三者可叠加（L2D+token search达38.57）
- 编码任务上Full ft.严重下降（31.04 vs 47.63基线），L2D反而提升（49.80）

## 亮点与洞察
- "LLM是单步扩散模型"这一观察建立了自回归和扩散框架之间的统一视角
- $w_d(0)=0$的设计确保L2D永远不损害原始LLM的单步能力——真正的"只增不减"
- 自适应ODE求解器让LLM实现逐token的计算自主分配，类比于"System 2 thinking"但不依赖CoT
- L2D与LoRA/Full ft./token search均正交兼容，开辟了一个新的缩放维度

## 局限与展望
- 推理开销线性增长（15次 $f_{\theta_d}$ 评估），对实时应用仍有挑战
- 73M-281M新参数虽远小于全量微调但仍显著高于LoRA的3-13M
- 仅在instruction-following数据上微调，对需要新世界知识的任务提升有限
- Classifier-Free Guidance需要预定义任务类别，限制了通用性

## 相关工作与启发
- **vs MDLM/Plaid (NeurIPS24)**: 从头训练语言扩散模型，远落后自回归LM。L2D通过微调预训练LM绕过此痛点
- **vs LoRA (ICLR22)**: 3M参数轻量但无推理缩放能力。L2D用73M参数换取了质的飞跃（28.54→35.50）
- **vs Chain-of-Thought**: CoT通过生成更多token实现"思考"，但计算分配不灵活。L2D的自适应solver可逐token分配
- 启示：扩散框架的"iterative refinement"范式可能是LLM推理缩放的一个被忽视的正交方向

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将扩散框架的缩放特性引入自回归LLM的思路极具创新性
- 实验充分度: ⭐⭐⭐⭐ 4个模型、6个任务、多种缩放方式、消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 从单步扩散到多步扩散的叙事链条清晰优雅
- 价值: ⭐⭐⭐⭐⭐ 开辟LLM推理缩放的新维度，与传统方法正交兼容

<!-- RELATED:START -->

## 相关论文

- [PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)
- [From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models without Finetuning](../../AAAI2026/self_supervised/from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho.md)
- [Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [Towards Benchmarking Foundation Models for Tabular Data With Text](towards_benchmarking_foundation_models_for_tabular_data_with_text.md)
- [AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)

<!-- RELATED:END -->
