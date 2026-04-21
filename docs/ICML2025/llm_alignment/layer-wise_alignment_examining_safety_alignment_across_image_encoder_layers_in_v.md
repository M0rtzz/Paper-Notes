---
title: >-
  [论文解读] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models
description: >-
  [ICML 2025][LLM对齐][VLM安全对齐] 本文发现了 VLM 中图像编码器的"早退出"漏洞（ICET）——跳过图像编码器的部分层会大幅增加有害输出概率，提出 Layer-wise PPO (L-PPO) 修改 Clipped-PPO 算法在不同层级做多模态 RLHF，在 ASR 上降低高达 48%、毒性分数降低 33.64%。
tags:
  - ICML 2025
  - LLM对齐
  - VLM安全对齐
  - 图像编码器
  - 层级安全性
  - 早退出漏洞
  - RLHF
---

# Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models

**会议**: ICML 2025  
**arXiv**: [2411.04291](https://arxiv.org/abs/2411.04291)  
**代码**: 无  
**领域**: LLM对齐/RLHF  
**关键词**: VLM安全对齐, 图像编码器, 层级安全性, 早退出漏洞, RLHF

## 一句话总结
本文发现了 VLM 中图像编码器的"早退出"漏洞（ICET）——跳过图像编码器的部分层会大幅增加有害输出概率，提出 Layer-wise PPO (L-PPO) 修改 Clipped-PPO 算法在不同层级做多模态 RLHF，在 ASR 上降低高达 48%、毒性分数降低 33.64%。

## 研究背景与动机
**领域现状**：VLM（如 LLaVA-1.5、LLaVA-NeXT、Llama 3.2 Vision）在多模态理解上取得了巨大进步，但安全对齐仍是挑战。现有的安全训练方法（SFT、RLHF、unlearning）主要在默认层嵌入下训练。

**现有痛点**：已有研究发现 LLM 的特定层保留了不同类型的信息，跳过特定层会影响有害内容生成。VLM 的多模态架构使得这一风险更加复杂——图像编码器的中间层嵌入在安全训练中从未被覆盖。

**核心矛盾**：安全对齐训练仅在图像编码器的默认层（通常是倒数第二层）进行，但攻击者可以使用中间层嵌入绕过安全防线，因为这些中间嵌入构成了分布外（OOD）场景。

**本文目标**：（1）系统化揭示 VLM 图像编码器的层级安全分布不均问题；（2）提出有效的防御方法使安全对齐覆盖不同层级。

**切入角度**：从神经网络早退出（early exit）这一效率优化技术切入，发现它对 VLM 安全性有灾难性影响。

**核心 idea**：使用来自不同图像编码器层的嵌入进行多层级 RLHF 训练（L-PPO），使安全对齐不局限于单一层。

## 方法详解

### 整体框架
1. **漏洞发现（ICET）**：系统测试不同图像编码器层的早退出对 VLM 安全输出的影响
2. **防御方法（L-PPO）**：修改 Clipped-PPO 算法，在训练时使用特定中间层的嵌入而非默认层嵌入，使安全对齐覆盖到潜在的漏洞层

### 关键设计

1. **ICET 漏洞（Image enCoder Early-exiT）**:

    - 功能：发现并系统性地量化图像编码器早退出对 VLM 安全性的影响
    - 核心发现：当使用 LLaVA-1.5 的第 18 层（而非默认的倒数第二层）嵌入时，即使输入图像安全、仅文本有害，VLM 也会生成有害响应
    - 原理分析：中间层嵌入构成 OOD 输入，语言骨干对这些嵌入的理解方式不同，安全对齐在该区域失效
    - 关键区分：中间层嵌入虽然产生**连贯**的输出（语义相关、逻辑一致），但安全机制被突破
    - 设计动机：早退出是神经网络的常见优化手段，在实际部署中可能被无意或故意触发

2. **Layer-wise PPO (L-PPO)**:

    - 功能：修改 Clipped-PPO 算法使其在特定的图像编码器中间层嵌入上执行 RLHF
    - 核心思路：既然漏洞来自特定中间层的嵌入分布不在安全训练覆盖范围内，就直接用这些层的嵌入做安全对齐训练
    - 具体修改：在标准 PPO 训练时，将输入给 VLM 的视觉嵌入从默认层替换为目标漏洞层的嵌入
    - PPO 目标函数保持 Clipped-PPO 的标准形式：$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)]$
    - 与标准 PPO 的区别：标准多模态 RLHF 使用默认层嵌入训练，L-PPO 针对性地使用中间层嵌入
    - 理论基础：提供了 L-PPO 有效性的理论证明

3. **评估指标体系**:

    - ASR（Attack Success Rate）：使用 Llama Guard 判断响应是否有害
    - TR（Total Rewards）：使用专门的安全奖励模型评估安全性
    - TS（Toxicity Score）：使用 Perspective API 评估毒性
    - 三种指标从不同角度评估 VLM 的安全性

### 损失函数 / 训练策略
- 使用 Clipped-PPO 目标函数，核心修改在于输入嵌入层的选择
- 奖励模型针对安全性训练（安全响应高分，有害响应低分）
- 训练数据：Redteam 2K、minijailbreak-V28K 等包含安全图像+有害文本的数据集
- 在三个 VLM（LLaVA-1.5、LLaVA-NeXT、Llama 3.2）上验证

## 实验关键数据

### ICET 漏洞严重程度（LLaVA-1.5 早退出不同层）

| 编码器层 | ASR ↓ | Toxicity Score ↓ | 说明 |
|---------|-------|-----------------|------|
| 默认层（倒数第二层）| 低 | 低 | 正常安全推理 |
| 第 18 层 | 显著升高 | 显著升高 | 安全对齐失效 |
| 部分中间层 | 不等 | 不等 | 有害信息在层间不均匀分布 |

### L-PPO 防御效果

| 模型 | 指标 | 无 L-PPO | 有 L-PPO | 改善 |
|------|------|---------|---------|------|
| LLaVA-1.5 | ASR | 高 | 大幅降低 | 最高 48% |
| LLaVA-NeXT | ASR | 高 | 大幅降低 | 显著 |
| Llama 3.2 | ASR | 高 | 大幅降低 | 显著 |
| 跨数据集 | Toxicity | 高 | 大幅降低 | 最高 33.64% |

### 关键发现
- ICET 漏洞在 LLaVA-1.5、LLaVA-NeXT 和 Llama 3.2 三个 VLM 上均存在，说明这是架构性问题
- 有害信息在图像编码器的不同层**不均匀分布**，某些层特别脆弱
- 安全训练方法的有限泛化性是漏洞根源：仅覆盖默认层的安全训练无法推广到中间层
- L-PPO 可以有效缓解 ICET，但需要针对每个漏洞层单独训练
- 中间层嵌入产生的输出仍然语义连贯，仅是安全性被破坏——这使得漏洞更加危险

## 亮点与洞察
- **揭示了一个此前未知的安全漏洞**：图像编码器早退出可破坏 VLM 安全对齐，对 VLM 部署安全性具有直接的实践警示意义
- 从"效率优化技术的安全隐患"视角切入非常新颖——早退出本是为了加速推理，却成为安全漏洞
- L-PPO 方法虽然简单，但直指问题本质：安全对齐的覆盖范围不足
- 跨三个 VLM 的验证增强了发现的普适性
- 提出了一个重要的安全原则：安全训练应当覆盖模型可能被使用的各种配置，而非仅限于默认配置

## 局限与展望
- L-PPO 需要预先识别哪些层是漏洞层，对新的 VLM 架构可能需要重新分析
- 修复一个层的漏洞是否会引入新的漏洞（层间 safety alignment 的 trade-off）尚未充分讨论
- 未来可以探索一次性覆盖所有层的训练方法（如多层嵌入混合训练或随机层采样训练）
- 对实际攻击场景的威胁模型定义可以更精确（攻击者如何访问中间层嵌入？）
- 缓存文件较短（64行），部分实验细节（如具体 ASR 数值）未完全获取

## 相关工作与启发
- 与 LLM 层级安全性研究（如 Zhao et al. 2023 发现跳层影响有害内容生成）形成呼应
- 早退出研究通常关注效率-精度权衡，本文新增了"效率-安全"的权衡维度
- 为 VLM 安全评估提供了新的 red-teaming 思路：不仅测试输入端，还应测试架构变体
- L-PPO 的思想可推广：在任何可能的 OOD 嵌入空间上做安全对齐
- 与多模态对抗攻击研究互补——本文的攻击不需要对抗 token，仅需改变嵌入层选择

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning](../../NeurIPS2025/llm_alignment/safevla_towards_safety_alignment_of_vision-language-action_model_via_constrained.md)
- [Safety Alignment Can Be Not Superficial With Explicit Safety Signals](safety_alignment_can_be_not_superficial_with_explicit_safety_signals.md)
- [MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)

<!-- RELATED:END -->
