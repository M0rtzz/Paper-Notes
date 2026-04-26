---
title: >-
  [论文解读] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions
description: >-
  [NeurIPS 2025][AI安全][机器遗忘] 本文提出R2D（Rewind-to-Delete），首个适用于一般非凸损失函数的一阶、黑盒认证机器遗忘算法，通过"回溯"到训练过程中的较早检查点再对保留数据执行梯度下降来实现数据删除，同时提供(ε,δ)认证遗忘保证和隐私-效用-效率的理论权衡。
tags:
  - NeurIPS 2025
  - AI安全
  - 机器遗忘
  - 差分隐私
  - 非凸优化
  - 认证遗忘
  - 隐私保护
---

# Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions

**会议**: NeurIPS 2025  
**arXiv**: [2409.09778](https://arxiv.org/abs/2409.09778)  
**代码**: 无  
**领域**: AI安全 / 机器遗忘  
**关键词**: 机器遗忘, 差分隐私, 非凸优化, 认证遗忘, 隐私保护

## 一句话总结
本文提出R2D（Rewind-to-Delete），首个适用于一般非凸损失函数的一阶、黑盒认证机器遗忘算法，通过"回溯"到训练过程中的较早检查点再对保留数据执行梯度下降来实现数据删除，同时提供(ε,δ)认证遗忘保证和隐私-效用-效率的理论权衡。

## 研究背景与动机
机器遗忘旨在高效地从已训练模型中移除特定数据的影响，而无需从头重新训练。这在GDPR等隐私法规的"被遗忘权"背景下尤为重要，同时对移除错误、过时或有害数据也有实际意义。

**现有痛点**：
- 大多数认证遗忘算法（如Newton Step、Descent-to-Delete）仅适用于凸或强凸函数，无法处理深度学习中的非凸损失
- 已有的非凸遗忘方法存在明显缺陷：二阶方法（CNS）计算代价高、需要O(d²)存储；准二阶方法（HF）需要训练时记录每个样本的统计向量，是"白盒"方法；一阶方法（Langevin Unlearning）需要每步注入噪声，也是白盒

**核心矛盾**：如何在非凸设置下同时满足一阶（计算高效）、黑盒（无需修改训练过程）和认证遗忘保证？

**切入角度**：利用"回溯"到训练轨迹的较早检查点，再在保留数据上继续梯度下降。通过近端点算法可事后计算检查点，实现完全黑盒。

## 方法详解

### 整体框架
R2D分为学习和遗忘两个阶段。学习阶段：对全数据集执行T步标准梯度下降，保存第T-K步的检查点，最终模型加高斯噪声后用于推理。遗忘阶段：加载第T-K步检查点，对保留数据集执行K步梯度下降，同样加高斯噪声输出。整个过程不需要在训练期间做任何特殊操作。

### 关键设计

1. **回溯机制（Rewinding）**:

    - 核心洞察：Descent-to-Delete从训练终点θ_T直接"下降"的策略在非凸设置下失效，因为非凸函数没有唯一全局最小值来吸引轨迹收敛
    - R2D通过回溯到更早的检查点θ_{T-K}，使遗忘轨迹更接近从头重训的轨迹，从而减小"遗忘偏差"导致的轨迹分歧
    - K越大->回溯越多->需要的噪声σ越小->模型效用越好，但计算代价越高
    - 当K=T时退化为无噪声的完整重训

2. **近端点算法事后计算检查点**:

    - 对于训练时没有保存检查点的模型，可以通过近端点方法（Proximal Point Method）从θ_T反向计算θ_{T-K}
    - 关键引理：当步长η < 1/L时，前一步的梯度下降迭代等价于对-f的近端算子，由于强凸性可以通过标准梯度下降或Newton法求解近端子问题
    - 实验表明对K < T/2的检查点可以较忠实重建，但更早的检查点会因累积近似误差而不稳定

3. **高斯噪声机制**:

    - 基于差分隐私的高斯机制，在学习和遗忘的最终输出上各加一次独立的高斯噪声
    - 噪声标准差σ由遗忘保证参数(ε,δ)、数据集大小n、删除数据大小m、回溯步数K等共同决定
    - σ与n成反比——更大的数据集需要更少的噪声
    - σ随K增大通过h(K)函数单调递减——更多回溯允许更小的噪声

### 损失函数 / 训练策略
- 学习阶段使用标准梯度下降最小化全数据经验风险$f_\mathcal{D}(\theta)$
- 遗忘阶段在保留数据上最小化$f_{\mathcal{D}'}(\theta)$
- 对满足Polyak-Łojasiewicz (PL)不等式的非凸函数，可获得线性收敛率和泛化保证
- PL函数与过参数化神经网络在初始化附近局部满足PL条件密切相关
- 实际中建议：先选择保持模型效用的σ（如0.01），再选择计算预算内的K（如10%×T），最后计算达到的隐私水平

## 实验关键数据

### 主实验

| 数据集 | 算法 | D_retain AUC | D_unlearn AUC (Δ) | D_test AUC | MIA AUC↓ |
|--------|------|-------------|-------------------|-----------|---------|
| eICU | R2D 74% | 0.7333 | 0.7439 (-0.0012) | 0.7326 | **0.481** |
| eICU | Hessian-Free | 0.7476 | 0.7587 (-0.0027) | 0.7465 | 0.524 |
| eICU | CNS | 0.7622 | 0.7848 (-0.0012) | 0.7601 | 0.543 |
| Lacuna-100 | R2D 74% | 0.9712 | 0.8998 (-0.0995) | 0.9534 | **最优** |
| Lacuna-100 | Hessian-Free | 0.9757 | 0.9555 (-0.0191) | 0.9652 | 较差 |
| Lacuna-100 | Finetune | 0.9997 | 0.9975 (-0.0018) | 0.9854 | 较差 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| R2D 7-10% | unlearn AUC下降适中 | 少量回溯即有效果 |
| R2D 36-37% | unlearn AUC持续下降 | 回溯比例与遗忘效果正相关 |
| R2D 74% | unlearn AUC下降最大 | 更多回溯=更强遗忘，但同时保留数据的效用也略降 |
| σ=0 vs σ=0.01 | 少量噪声即显著改善MIA防御 | 认证遗忘方法天然比非认证方法更好防御MIA |
| 训练速度 | R2D比HF快高达88倍 | 黑盒方法的巨大优势 |

### 关键发现
- R2D在D_unlearn上展示显著更大的性能下降，表明遗忘更彻底
- R2D在两种成员推断攻击（经典MIA和MIA-U）中均表现最佳
- 即使少量的噪声扰动，认证遗忘方法也比非认证方法（Finetune、SCRUB、Fisher Forgetting）更能防御MIA
- R2D作为黑盒方法，训练阶段比HF快高达88倍

## 亮点与洞察
- **黑盒+一阶+非凸**：首次在非凸设置下同时满足这三个实用性要求，可直接应用于任何以标准梯度下降训练的预训练模型
- **回溯优于下降**：核心洞察"rewinding > descending"提供了比Descent-to-Delete更强大的非凸遗忘基线
- **近端点反向计算**：即使训练时没有保存检查点，也可通过近端点算法事后反向计算。这使R2D完全黑盒化
- **理论-实践统一**：显式的隐私-效用-效率三方权衡公式（σ、K、ε的关系），允许从业者根据具体需求灵活配置
- 新颖的实验框架：按用户维度删除数据，更贴近真实遗忘场景（而非随机删除或删整类）

## 局限与展望
- 理论保证依赖于均匀有界梯度和Lipschitz平滑性假设，对所有实际模型可能过强
- 近端点算法反向计算检查点在K > T/2时不稳定，限制了黑盒模式下的回溯深度
- 使用全批梯度下降分析，实际中用小批量SGD时理论保证会松弛
- PL函数的泛化保证需要较强的假设，实际深度网络是否局部满足PL条件存在争议
- 未在大语言模型等超大规模模型上验证

## 相关工作与启发
- **vs Descent-to-Delete (D2D)**: D2D从θ_T直接微调，仅适用于强凸函数；R2D通过回溯绕过了非凸函数没有唯一全局最小值的根本困难
- **vs CNS**: 二阶方法需O(d²)存储且不可扩展；R2D仅需O(d)存储
- **vs Hessian-Free**: 白盒方法需训练时每个样本每步记录统计向量，开销巨大；R2D完全不需修改训练过程
- **启发**：回溯+微调的思路非常通用，可能启发其他需要"撤销"模型行为的场景（如偏见消除、版权数据删除）

## 评分
- 新颖性: ⭐⭐⭐⭐ "回溯到检查点"的思路直觉简洁，近端点反向计算是亮点，但与已有的rewinding思想有一定重叠
- 实验充分度: ⭐⭐⭐⭐ 新颖的用户级遗忘实验框架、多种基线对比、MIA评估全面，但数据集规模偏小
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，问题定义和动机阐述到位，与相关工作的对比客观公允
- 价值: ⭐⭐⭐⭐ 填补了非凸认证遗忘的重要空白，但实际大模型场景的适用性有待验证

<!-- RELATED:START -->

## 相关论文

- [\[ICML 2025\] A Certified Unlearning Approach without Access to Source Data](../../ICML2025/ai_safety/a_certified_unlearning_approach_without_access_to_source_data.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[CVPR 2025\] Towards Source-Free Machine Unlearning](../../CVPR2025/ai_safety/towards_source-free_machine_unlearning.md)

<!-- RELATED:END -->
