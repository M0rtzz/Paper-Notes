---
title: >-
  [论文解读] Certified Circuits: Stability Guarantees for Mechanistic Circuits
description: >-
  [ICML 2026][可解释性][机械可解释性] 提出 Certified Circuits 框架，通过基于删除的随机平滑为机械可解释性中的电路发现提供可证明的数据集级稳定性保证，使得发现的电路在概念数据集的有界编辑距离扰动下保持不变，从而产生更紧凑、更准确且 OOD 泛化更好的电路。
tags:
  - "ICML 2026"
  - "可解释性"
  - "机械可解释性"
  - "电路发现"
  - "随机平滑"
  - "鲁棒性认证"
  - "数据集稳定性"
---

# Certified Circuits: Stability Guarantees for Mechanistic Circuits

**会议**: ICML 2026  
**arXiv**: [2602.22968](https://arxiv.org/abs/2602.22968)  
**代码**: https://github.com/AlaaAnani/certified-circuits  
**领域**: AI安全/可解释性  
**关键词**: 机械可解释性, 电路发现, 随机平滑, 鲁棒性认证, 数据集稳定性  

## 一句话总结
提出 Certified Circuits 框架，通过基于删除的随机平滑为机械可解释性中的电路发现提供可证明的数据集级稳定性保证，使得发现的电路在概念数据集的有界编辑距离扰动下保持不变，从而产生更紧凑、更准确且 OOD 泛化更好的电路。

## 研究背景与动机

**领域现状**：机械可解释性（Mechanistic Interpretability）通过识别电路（circuits）——负责特定行为的最小子网络——来理解神经网络的内部计算。在视觉模型中，识别"鳄鱼"的电路可能由检测鳞片、尖牙等特征的卷积滤波器组成；在语言模型中，通过激活补丁和因果追踪可以隔离负责特定任务的注意力头和 MLP 神经元。

**现有痛点**：当前的电路发现方法极其脆弱——添加、删除或替换概念数据集中的少量语义等价样本就可能导致发现的电路发生不可预测的变化。例如，用陆地上的鳄鱼照片发现的电路可能在水中鳄鱼、卡通鳄鱼上表现很差。更严重的是，发现的电路无法可靠地迁移到分布外（OOD）数据。

**核心矛盾**：这种不稳定性的根本原因在于现有方法过拟合到特定的概念数据集，而非恢复真正的概念表示。电路中包含的不稳定组件往往对应虚假特征（如识别鳄鱼时关注背景中的鸟），而非类别定义特征（如牙齿、鳞片）。

**本文目标**：为电路发现提供可证明的数据集级鲁棒性保证，确保发现的电路在概念数据集的有界编辑扰动下保持不变。

**切入角度**：作者将分类任务中的随机平滑（Randomized Smoothing）思想迁移到电路发现场景——通过在多个随机子采样数据集上运行电路发现算法并聚合结果，将经验稳定性转化为可证明的鲁棒性保证。这个角度有希望是因为编辑距离下的 RS-Del 删除平滑已被证明在离散对象上有效。

**核心 idea**：用随机删除平滑包裹任意黑盒电路发现算法，对每个电路组件的纳入/排除决策进行逐组件投票认证，不确定的组件弃权，从而产生可证明稳定且更紧凑的电路。

## 方法详解

### 整体框架
给定模型计算图 $G=(V,E)$、概念数据集 $D$ 和任意黑盒电路发现算法 $A$，Certified Circuits 通过五步流程产生认证电路：①接收输入（数据集、模型图、算法）；②对数据集进行多次随机删除采样；③在每个子采样上运行 $A$ 获得候选电路；④聚合每个组件的投票频率并认证；⑤输出认证电路，保证在编辑距离 $r$ 内的所有数据集扰动下不变。

### 关键设计

1. **数据集删除平滑（Dataset Deletion Smoothing）**:

    - 功能：将数据集级编辑扰动转化为可处理的随机删除分布
    - 核心思路：对概念数据集 $D = (x_1, \ldots, x_{|D|})$ 中的每个样本独立采样伯努利掩码 $\varepsilon_i \sim \text{Bernoulli}(1 - p_{\text{del}})$，保留 $\varepsilon_i=1$ 的样本得到子数据集 $D \odot \varepsilon$。通过在 $n$ 个这样的随机子数据集上运行基础算法 $A$，计算每个组件 $u$ 的平滑包含概率 $p_u(D) = P[A_u(D \odot \varepsilon) = 1]$，衡量组件在随机删除下被选入电路的频率
    - 设计动机：直接枚举编辑距离内的所有数据集不可行（指数级），而基于 RS-Del 的删除平滑可以将随机删除下的经验一致性转化为对完整编辑距离（插入、删除、替换）的认证保证

2. **逐组件三值认证（Per-Component Certification with Abstention）**:

    - 功能：将投票概率转化为可证明的纳入/排除/弃权决策
    - 核心思路：设置置信阈值 $\tau \in [0.5, 1)$，对每个组件 $u$ 做三值判定：若 $p_u > \tau$ 则认证纳入（1），若 $1 - p_u > \tau$ 则认证排除（0），否则弃权（$\oslash$）。弃权机制是关键——它自适应地剪除那些在随机删除下不稳定的组件，这些组件往往对应虚假特征
    - 设计动机：借鉴分割任务中的逐像素认证思想，将二值决策扩展为三值，使得不确定的组件被排除而非错误地纳入，从而产生更紧凑且准确的电路

3. **认证半径保证（Certified Radius Guarantee）**:

    - 功能：提供可证明的稳定性半径，保证电路在有界数据集编辑下不变
    - 核心思路：核心定理（Theorem 3.1）证明，对任意非弃权组件 $u$，在置信度 $\geq 1-\alpha$ 下，对满足 $\text{dist}_{\text{edit}}(D, D') \leq r$ 的任何扰动数据集 $D'$，认证决策不变。认证半径为 $r = \lfloor \log(1.5 - \tau) / \log p_{\text{del}} \rfloor$。例如 $p_{\text{del}}=0.6, \tau=0.95$ 时 $r=1$，保证在任意 1 次数据集编辑下电路不变
    - 设计动机：提供形式化的最坏情况保证而非经验性启发式，使电路发现从"可能稳定"提升到"可证明稳定"，覆盖指数级大小的数据集族

## 实验关键数据

### 主实验

实验覆盖三种架构（ResNet-101/50, ViT-B/16, GPT-2 Small）和两种模态（视觉、语言）。

| 数据集/任务 | 模型 | 完整模型 cACC | Baseline cACC | Certified cACC | 提升 | 电路大小 K 减少 |
|-------------|------|---------------|---------------|----------------|------|-----------------|
| ImageNet (ID) | ResNet-101 | 0.78 | 0.83 | 0.95 | ↑14% | ↓52% |
| ImageNet-A (OOD) | ResNet-101 | 0.07 | 0.60 | 0.94 | ↑56% | ↓15% |
| OOD-CV (OOD) | ResNet-101 | 0.20 | 0.73 | 0.93 | ↑28% | ↓33% |
| ImageNet-C (OOD) | ResNet-101 | 0.57 | 0.72 | 0.92 | ↑28% | ↓41% |
| IOI (ID) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓58% |
| Greater-Than (ID) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓75% |
| Greater-Than (OOD) | GPT-2 | 1.00 | 1.00 | 1.00 | 0% | ↓80% |

### 消融/分析实验

| 分析维度 | 关键结果 | 说明 |
|----------|----------|------|
| 特征可视化 | 认证纳入的神经元响应类别定义特征 | 弃权神经元响应共现但非类别特异的虚假线索 |
| 结构稳定性 (IoU) | 认证电路中位 IoU 更高且分布更紧 | 在 OOD-CV 上 ∆cACC=100% 对应最大稳定性增益 |
| 跨随机种子稳定性 | 认证电路跨种子收敛 | 认证过程本身可靠 |
| 计算开销 | ~2.4× 基线算法开销 (n=1000) | 通过缓存前向/反向传播保持低开销 |

### 关键发现
- 认证电路在所有视觉 OOD 数据集上显著优于基线，最大提升出现在最困难的分布偏移上（ImageNet-A ↑56%）
- 弃权机制不仅移除了无信息组件，还移除了"有害"组件——这些组件将基线偏向虚假类别关联
- 语言任务中准确率已饱和，但认证电路仍能以最多 80% 更少的边达到相同性能
- 认证电路在 OOD 数据上重新发现时收敛到相似的不变核心，验证了其捕获真正概念表示的能力

## 亮点与洞察
- **从分类到电路的随机平滑迁移**：将 RS-Del 从分类输入扰动迁移到电路发现的数据集扰动，结合分割中的逐组件弃权，是一个优雅的"旧工具新应用"。这种迁移思路可推广到任何需要认证结构化输出稳定性的场景
- **弃权即剪枝**：三值认证的弃权机制本质上实现了自适应稀疏化——不需要手动选择 top-K，认证过程自动识别并排除不稳定组件，产生更紧凑的电路。这暗示传统 top-K 选择中有大量组件是虚假的
- **算法无关性**：框架包裹任意黑盒算法而不需要访问其内部，使其可以即插即用地增强已有的电路发现方法

## 局限与展望
- 更大的认证半径需要更高的删除率，在概念数据集很小时可能降低基础算法性能（但在 $|D|$ 较大时可支持 $r$ 达 59）
- 目前使用统一的稀疏度 $K$ 跨所有层，分层预算分配可能产生更精细的稀疏化
- 尚未在更大的语言模型（如 LLaMA）、多模态模型和稀疏激活架构上验证
- 认证保证相对于特定威胁模型和概念数据集，不应被误解为对模型行为的完整解释

## 相关工作与启发
- **ACDC** (Conmy et al., 2023)：通过迭代边剪枝自动发现电路，是本文视觉实验的基线之一
- **EAP-IG** (Hanna et al., 2024)：通过积分梯度对计算图边评分的语言模型电路发现方法，是本文语言实验的基础算法
- **RS-Del** (Huang et al., 2023)：基于随机删除的编辑距离认证方法，是本文认证框架的理论基础
- **Fischer et al. (2021)**：分割任务中的逐像素认证方法，启发了本文的逐组件弃权设计

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](../../ACL2025/interpretability/reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)

</div>

<!-- RELATED:END -->
