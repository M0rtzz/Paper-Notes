---
title: >-
  [论文解读] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning
description: >-
  [ICCV 2025][AI安全][成员推断] 本文提出 Active MINT（aMINT），一种多任务学习框架，在训练审核模型的同时联合训练 MINT 模型，使模型能够以超过 80% 的准确率检测特定数据是否被用于训练，显著优于現有的被动 MINT 和成员推断攻击方法。 随着 AI 技术的快速发展…
tags:
  - "ICCV 2025"
  - "AI安全"
  - "成员推断"
  - "数据审计"
  - "多任务学习"
  - "AI可信赖性"
  - "模型透明度"
---

# Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning

**会议**: ICCV 2025  
**arXiv**: [2509.07879](https://arxiv.org/abs/2509.07879)  
**代码**: [GitHub](https://github.com/DanieldeAlcala/Membership-Inference-Test.git)  
**领域**: AI 安全与可审计性  
**关键词**: 成员推断, 数据审计, 多任务学习, AI可信赖性, 模型透明度

## 一句话总结
本文提出 Active MINT（aMINT），一种多任务学习框架，在训练审核模型的同时联合训练 MINT 模型，使模型能够以超过 80% 的准确率检测特定数据是否被用于训练，显著优于現有的被动 MINT 和成员推断攻击方法。

## 研究背景与动机
随着 AI 技术的快速发展，欧盟 AI 法案（2024年6月）和美国白宫备忘录（2024年10月）等法规都要求对 AI 模型进行审计，确保模型训练数据的合法合规使用。数据拥有者有权知道其数据是否被用于模型训练，而模型开发者可能在未获许可的情况下使用受保护的数据（如生物特征数据、版权内容）。

成员推断测试（MINT）于 2024 年提出，旨在检测特定数据是否被用于训练 AI 模型。与成员推断攻击（MIA）不同，MINT 是作为审计工具而非攻击手段，允许与模型开发者有一定程度的合作（如有限的模型访问权限），这在现有法规框架下是合理的。

然而，现有的被动 MINT（Passive MINT）方法是在模型训练完成后才训练 MINT 模型，其检测准确率受限。核心矛盾在于：MINT 模型需要分析审核模型的内部激活模式来判断数据是否参与训练，但如果审核模型本身没有为这一任务优化，其激活模式中的训练/非训练数据可区分性很弱。

本文的切入角度是：如果模型开发者主动参与审计过程（Active MINT），在训练审核模型的同时联合优化 MINT 模型，那么共享的底层特征可以同时服务于主任务和审计任务，从而大幅提升 MINT 检测准确率。

核心 idea：将"数据审计"作为培训优化目标之一纳入多任务学习，同时训练审核模型和 MINT 模型。

## 方法详解

### 整体框架
Active MINT 构建一个增强审核模型 $M^+$，包含两部分：(1) 审核模型 $M$，执行原始任务（如图像分类）；(2) MINT 模型 $T$，判断输入数据是否属于训练集。两个模型共享底层网络层，在某个提取点（extraction point）分叉为两条路径。训练集 $\mathcal{D}$（50%）和外部数据 $\mathcal{E}$（50%）混合组成 batch 进行多任务联合训练。

### 关键设计
1. **多任务学习架构（Multi-task Learning）**:

    - 功能：将 MINT 任务与主任务联合训练，使网络底层特征同时编码数据成员信息
    - 核心思路：网络前几层为共享层（橙色），之后分叉为审核模型路径（灰色）和 MINT 模型路径（蓝色）。$\mathcal{D}$ 中的样本同时通过两条路径，$\mathcal{E}$ 中的样本仅通过 MINT 路径。多任务损失函数：$$\mathcal{L}_{\text{Multi-task}} = \lambda_1 \frac{\mathcal{L}_{\text{Audited}}}{\|\mathcal{L}_{\text{Audited}}\|} + \lambda_2 \frac{\mathcal{L}_{\text{MINT}}}{\|\mathcal{L}_{\text{MINT}}\|} + R(\mathbf{w}^+)$$
    - 设计动机：被动 MINT 中审核模型已训练完成，其激活模式固定不可优化。主动 MINT 让共享层在反向传播中同时接收两个任务的梯度，使底层特征更具"成员可区分性"。

2. **辅助可审计数据（Auxiliary Auditable Data, AAD）提取**:

    - 功能：从网络的两个中间层提取激活图（activation maps）作为 MINT 模型的输入
    - 核心思路：选择两个中间层的激活图 $\text{AAD} = N(d|\mathbf{w}')$，通过 CNN 或全连接网络分析激活模式，预测二分类结果（训练数据 vs 外部数据）
    - 设计动机：两级激活图提供不同粒度的信息——底层捕获纹理级差异，高层捕据语义级差异。本文扩展了前人只用单层激活图的做法，发现两层组合效果更好。

3. **三种 Setup 的对比分析**:

    - 功能：系统评估激活图提取位置的影响——Entry（靠近输入层）、Middle（中间层）、Output（靠近输出层）
    - 核心发现：Entry 和 Middle Setup 效果相当且远优于 Output Setup
    - 设计动机/原因分析：审核任务追求泛化（训练和测试表现一致），而 MINT 任务追求区分（训练数据和非训练数据表现不同），两个目标本质矛盾。Output Setup 中两个任务共享大量网络层，矛盾最尖锐。Entry Setup 中共享层最少，两个任务的干扰最小，因此在审核准确率上有微弱优势。

### 损失函数 / 训练策略
- $\mathcal{L}_{\text{Audited}}$：取决于主任务（如分类任务用交叉熵）
- $\mathcal{L}_{\text{MINT}}$：二元交叉熵，判断数据是否为训练集成员
- 两个损失函数通过各自的范数归一化以保持相同量级
- $\lambda_2 / \lambda_1$ 的比值随任务难度调整（MNIST 为 10，Tiny ImageNet 为 10000）
- L2 正则化 $R(\mathbf{w}^+)$ 防止过拟合
- 采用早停策略，约 50-100 个 epoch

## 实验关键数据

### 主实验

| 数据集 | 架构 | Passive MINT | Active MINT | MINT 提升 | 审核任务影响 |
|--------|------|-------------|------------|----------|------------|
| MNIST | ResNet50 | 0.51 | **0.83** | +32% | 0.97→0.97 (无损) |
| CIFAR-10 | DenseNet121 | 0.60 | **0.86** | +26% | 0.80→0.80 (无损) |
| GTSRB | Xception | 0.59 | **0.86** | +27% | 0.99→0.99 (无损) |
| Tiny ImageNet | Xception | 0.65 | **0.88** | +23% | 0.28→0.28 (无损) |
| CASIA WebFace | MobileNet | 0.60 | **0.86** | +26% | 0.17→0.15 (微降) |
| CIFAR-10 | ResNet50 | 0.66 | **0.86** | +20% | 0.55→0.53 (微降) |

### 消融实验

| Setup | MNIST MINT↑ | MNIST Aud | CIFAR MINT↑ | CIFAR Aud |
|-------|------------|-----------|-------------|-----------|
| Entry | 0.83-0.86 | 0.94-0.99 | 0.86-0.91 | 0.19-0.80 |
| Middle | 0.81-0.88 | 0.92-0.99 | 0.86-0.91 | 0.19-0.80 |
| Output | 0.77-0.82 | 0.80-0.98 | 0.82-0.88 | 0.19-0.76 |

Output Setup 在所有场景下均为最差，验证了共享层越多、任务矛盾越剧烈的理论分析。

### 与 MIA 方法的对比

| 方法 | CIFAR-10 | GTSRB |
|------|----------|-------|
| Salem et al. MIA | 0.61 | 0.67 |
| Yeom et al. MIA | 0.64 | 0.79 |
| Watson et al. MIA | 0.63 | 0.79 |
| Passive MINT | 0.66 | 0.61 |
| **Active MINT (Ours)** | **0.86** | **0.86** |

Active MINT 大幅超越所有 MIA 方法和被动 MINT，尽管 MIA 和 MINT 的实验条件不完全相同。

### 关键发现
- Active MINT 在所有模型架构和数据集上一致地超越 Passive MINT，MINT 准确率提升 20-32 个百分点
- 在多数情况下，审核任务性能几乎不受影响（<1%）或完全保持不变
- 从轻量级 MobileNet 到复杂的 Vision Transformer 均有效，方法泛化性强
- Entry Setup 是最佳选择：MINT 和审核任务均表现最好

## 亮点与洞察
- **多任务学习的巧妙应用**：将"可审计性"作为训练目标纳入网络优化，而非事后分析
- **实际部署考量周全**：论文详细讨论了 Docker 容器、数字签名日志、多方计算等确保训练可信的部署策略
- **与法规接轨**：紧密呼应 EU AI Act 和 US 白宫备忘录的监管需求
- **矛盾分析深刻**：清晰地阐述了审核任务（追求泛化）与 MINT 任务（检测过拟合）之间的本质矛盾，并用 Setup 实验验证

## 局限与展望
- 需要模型开发者主动参与训练 MINT 模型，在对抗性场景下开发者可能不配合
- 训练数据量被对半分割（50% 训练数据 + 50% 外部数据），限制了主任务性能
- 目前仅在图像分类任务上实验，未扩展到 LLM、生成模型等更复杂场景
- 高阶运动检测（如 gradient-based MINT, gMINT）是否能与 Active MINT 结合尚未探索
- 在类别数极多（如 Tiny ImageNet 200 类）的场景下，审核任务准确率本身就偏低，限制了方法评估

## 相关工作与启发
- **vs Passive MINT**: 核心区别在于训练时机——被动 MINT 是事后审计，主动 MINT 是训练时审计。主动 MINT 不需要开发者暴露训练数据或提供模型访问权限，但需要开发者参与训练
- **vs MIA (Shokri et al.)**: MIA 是攻击，需要训练影子模型来模拟目标模型行为；MINT 是审计工具，允许开发者合作。Active MINT 进一步通过联合训练消除了对影子模型的需求
- **vs Nasr et al. (对抗正则化)**: Nasr 等人训练模型对抗 MIA（使 MIA 更难成功），而本文训练模型配合 MINT（使审计更易成功），两者目标恰好相反

## 评分
- 新颖性: ⭐⭐⭐⭐ 将可审计性纳入训练目标的主动 MINT 思路新颖，且理论动机清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 6种架构 × 5个数据集 × 3种Setup，实验极为充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，概念定义明确，部署讨论有实际价值
- 价值: ⭐⭐⭐⭐ 回应当前 AI 监管的迫切需求，有明确的社会价值和应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICLR 2026\] Toward Enhancing Representation Learning in Federated Multi-Task Settings](../../ICLR2026/ai_safety/toward_enhancing_representation_learning_in_federated_multi-task_settings.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ACL 2025\] Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries](../../ACL2025/ai_safety/multi-task_adversarial_attacks_against_black-box_model_with_few-shot_queries.md)
- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)

</div>

<!-- RELATED:END -->
