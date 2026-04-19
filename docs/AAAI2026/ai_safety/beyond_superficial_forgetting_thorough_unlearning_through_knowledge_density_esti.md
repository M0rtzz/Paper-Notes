---
title: >-
  [论文解读] Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion
description: >-
  [AAAI 2026][AI安全][machine unlearning] 提出 KUnBR 框架，通过梯度引导的知识密度估计定位有害知识富集层，并采用块重插入策略绕过 cover layer 的梯度遮蔽效应，实现对 LLM 有害知识的深度遗忘而非表面抑制。
tags:
  - AAAI 2026
  - AI安全
  - machine unlearning
  - Knowledge Density
  - Block Re-insertion
  - LLM safety
  - RTT Attack
---

# Beyond Superficial Forgetting: Thorough Unlearning through Knowledge Density Estimation and Block Re-insertion

**会议**: AAAI 2026  
**arXiv**: [2511.11667](https://arxiv.org/abs/2511.11667)  
**代码**: [github.com/llmgfffffff/Beyond-Superficial-Forgetting-KUnBR](https://github.com/llmgfffffff/Beyond-Superficial-Forgetting-KUnBR)  
**领域**: ai_safety  
**关键词**: machine unlearning, Knowledge Density, Block Re-insertion, LLM safety, RTT Attack

## 一句话总结

提出 KUnBR 框架，通过梯度引导的知识密度估计定位有害知识富集层，并采用块重插入策略绕过 cover layer 的梯度遮蔽效应，实现对 LLM 有害知识的深度遗忘而非表面抑制。

## 研究背景与动机

- **机器遗忘的核心需求**：LLM 在预训练过程中可能吸收隐私敏感、有害或受版权保护的内容，需要在不从头重训的前提下选择性移除这些知识，以满足 GDPR 等"被遗忘权"法规要求。
- **现有方法的表面遗忘问题**：梯度上升（GA）、梯度差分（GD）、RMU 等方法虽能在输出层面抑制有害内容，但实际上只修改了少量"cover layer"的参数，有害知识仍驻留在模型深层参数中。
- **RTT 攻击暴露脆弱性**：Retraining on T（RTT）攻击表明，对遗忘集的一小部分数据进行微调即可恢复大量被"遗忘"的知识，说明现有方法并未真正从参数中消除目标知识。
- **Cover Layer 的梯度遮蔽**：遗忘训练时梯度主要集中在少数输出端层，形成 cover layer 屏蔽效应，导致深层知识富集块无法得到有效更新。
- **精准定位的缺失**：此前缺乏量化各层有害知识密度的系统性方法，无法精确识别哪些层最需要被深度遗忘。
- **通用能力保持的挑战**：RIA、NPO 等方法在追求更低遗忘精度的同时往往严重损害模型的推理、事实回答等通用能力，缺乏遗忘-保留的平衡。

## 方法详解

### 总体框架

KUnBR 分三阶段：(1) 全参数预遗忘（warm-up），用标准梯度差分对整个模型做初步遗忘训练；(2) 知识密度估计 + 块选择，定位有害知识富集块；(3) 块重插入 + 二次遗忘，将选出的块嫁接回原始模型进行深度遗忘。

### 关键设计 1：知识密度估计（Knowledge Density Estimation）

- **功能**：为模型每一层计算一个知识密度分数 $K_l$，量化该层包含的待遗忘知识量。
- **核心思路**：在遗忘集上做前向-反向传播，取每层参数梯度的 L1 范数期望作为知识密度指标。梯度绝对值越大，说明该层对遗忘集信息越敏感，含有更多待消除的知识。归一化后得到 $K_l^{norm} = K_l / \sum_{i=1}^H K_i$。
- **设计动机**：基于"MLP 层是 LLM 的神经记忆单元"这一洞察，梯度大意味着参数与目标知识的关联度高。该步骤仅计算不更新参数，为后续精准定位提供依据。

### 关键设计 2：块选择策略（Block Selection Strategy）

- **功能**：将模型 $H$ 层划分为 $M$ 个块，每块 $N = \lfloor H/M \rfloor$ 层，累加各层知识密度得到块级密度 $K_{block,m}$，选取 Top-K 高密度块。
- **核心思路**：采用两条规则——(a) Top-K 选择：选知识密度最高的 K 个块；(b) 排除头部层：忽略含最后两层的块，因为末尾层的高梯度是输出生成的产物而非知识存储。
- **设计动机**：逐层操作粒度太细、效率低，块级分组兼顾定位精度与实操效率，同时排除干扰层避免误判。

### 关键设计 3：块重插入策略（Re-insertion Strategy）

- **功能**：从已经过预遗忘的 $\text{LLM}_{unlearning}$ 中提取选中的高密度块，将其"嫁接"到未经遗忘的原始 $\text{LLM}_{original}$ 的对应位置，冻结原始模型其余层，仅对插入块施加梯度差分遗忘训练。
- **核心思路**：原始模型中不存在 cover layer（因为未做过遗忘训练），所以被插入的块能直接暴露给遗忘梯度，不再被其他已修改层的遮蔽效应干扰，实现更深度的知识消除。训练完成后，这些块回归 $\text{LLM}_{unlearning}$，残留知识量远低于标准方法。
- **设计动机**：直接在已遗忘模型上继续训练会受 cover layer 梯度阻断的影响，重插入策略通过绕过该阻断，让遗忘梯度直达目标块。

### 损失函数与训练

预遗忘与重插入阶段均使用 **Gradient Difference** 损失：对遗忘集做梯度上升（增大损失）以消除知识，同时对保留集做梯度下降（减小损失）以维持通用能力。预遗忘阶段为全参数训练，重插入阶段仅训练选中块、其余层冻结。

## 实验关键数据

### 表 1：RTT 攻击下遗忘性能（LLaMA3-8B-Instruct，↓ 越低越好）

| 方法 | Random Birthdays Forget. | RTT. | Rec. | WMDP Forget. | RTT. | Rec. | Years Forget. | RTT. | Rec. |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GA | 23.5 | 87.2 | 63.7 | 29.2 | 66.8 | 37.6 | 25.9 | 50.6 | 24.7 |
| GD | 64.9 | 80.2 | 15.3 | 30.5 | 62.4 | 31.9 | 25.9 | 68.3 | 42.4 |
| RMU | 36.3 | 88.5 | 52.2 | 29.9 | 64.9 | 35.0 | 24.2 | 68.3 | 44.1 |
| NPO | 71.3 | 78.3 | 7.0 | 35.6 | 58.4 | 22.8 | 26.5 | 67.7 | 41.2 |
| **KUnBR** | **36.9** | **43.9** | **7.0** | **29.2** | **38.8** | **9.6** | **25.9** | **36.0** | **10.1** |

KUnBR 在所有数据集上 RTT 攻击后的恢复率（Rec.）均为最低或并列最低，表明有害知识被更彻底地移除。

### 表 2：通用能力保持（LLaMA3-8B，RKWU 指标，↑ 越高越好）

| 方法 | Rea. | Fac. | Tru. | Flu. |
|------|:---:|:---:|:---:|:---:|
| GA | 40.2 | 56.3 | 36.8 | 706.2 |
| RIA | 39.5 | 56.1 | 36.8 | 705.9 |
| NPO | 39.8 | 54.3 | 36.8 | 703.7 |
| **KUnBR** | **41.2** | **56.1** | **36.6** | **706.7** |

KUnBR 在推理能力和流畅度上取得最优，事实性和真实性也与最佳方法持平，证明块级局部遗忘有效保护了通用能力。

### 消融实验（表 3，Years 数据集）

| 变体 | Forget.↓ | RTT.↓ |
|------|:---:|:---:|
| KUnBR（完整） | 25.9 | 36.0 |
| 去掉重插入（退化为 GD） | 25.9 | 68.3 |
| 去掉预遗忘 | 25.9 | 36.7 |

去掉重插入后 RTT 准确率从 36.0% 飙升至 68.3%，证明重插入策略是抵抗 RTT 攻击的关键。

## 亮点

- **问题洞察深刻**：首次系统分析 cover layer 对遗忘训练的梯度遮蔽效应，揭示现有方法"表面遗忘"的本质原因。
- **知识密度估计简洁有效**：利用梯度 L1 范数量化各层知识密度，无需额外探针或复杂分析工具。
- **重插入策略设计巧妙**：通过将目标块嫁接回原始模型绕过 cover layer，思路新颖且实现简单。
- **RTT 攻击鲁棒性显著提升**：在多个数据集上恢复率远低于所有基线方法。
- **通用能力损失极小**：块级局部遗忘 + 冻结其余层，有效避免了全局能力退化。

## 局限性

- 仅在 7B-8B 规模模型上验证，未测试更大模型（如 70B）的可扩展性。
- 知识密度估计需要在整个遗忘集上做全模型反向传播，对大规模遗忘集的计算开销较高。
- 块数 $M$ 和 Top-K 的选择需要预实验调优，虽然作者声称跨架构稳定，但最优配置可能因任务不同而异。
- 仅评估了多选题形式的遗忘，未考虑开放式生成场景下的遗忘效果。
- 排除最后两层的规则基于经验观察，缺乏严格的理论证明。

## 相关工作

- **GA / GD / NPO**：基于梯度的遗忘方法，通过梯度上升或偏好优化抑制输出，但知识仍留存于参数中。
- **RMU**：修改中间层表示实现遗忘，但单纯的表示扰动易被 RTT 攻击恢复。
- **RIA**：引导模型学习错误答案，遗忘效果有限且损害通用能力。
- **RTT 攻击**：参数级攻击方法，揭示了遗忘方法的脆弱性，是本文的核心对比攻击手段。
- **层级知识分析**：Geva et al. 发现 MLP 是 LLM 的键值记忆单元，Hong et al. 发现遗忘训练主要修改少量层，为本文的知识密度估计提供了理论基础。

## 评分

- 新颖性: ⭐⭐⭐⭐ — cover layer 分析和重插入策略是新颖的视角和解决方案
- 实验充分度: ⭐⭐⭐⭐ — 4 个数据集、2 个骨干模型、完整消融和块选择分析
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述到位，图示直观
- 价值: ⭐⭐⭐⭐ — 解决了机器遗忘中的根本性问题，对 LLM 安全部署有实际意义

<!-- RELATED:START -->

## 相关论文

- [An Information Theoretic Evaluation Metric for Strong Unlearning](an_information_theoretic_evaluation_metric_for_strong_unlear.md)
- [Matrix-Free Two-to-Infinity and One-to-Two Norms Estimation](matrix-free_two-to-infinity_and_one-to-two_norms_estimation.md)
- [The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](../../NeurIPS2025/ai_safety/the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [Privacy-protected Retrieval-Augmented Generation for Knowledge Graph Question Answering](privacy-protected_retrieval-augmented_generation_for_knowledge_graph_question_an.md)
- [Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity](breaking_the_dyadic_barrier_rethinking_fairness_in_link_prediction_beyond_demogr.md)

<!-- RELATED:END -->
