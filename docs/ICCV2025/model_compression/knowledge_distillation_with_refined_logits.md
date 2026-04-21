---
title: >-
  [论文解读] Knowledge Distillation with Refined Logits
description: >-
  [ICCV 2025][模型压缩][知识蒸馏] RLD 通过 Sample Confidence（样本置信度）和 Masked Correlation（掩码相关性）两种精炼知识，在不破坏类别相关性的前提下修正教师错误预测的负面影响，在 CIFAR-100 和 ImageNet 上全面超越现有 logit 蒸馏方法。
tags:
  - ICCV 2025
  - 模型压缩
  - 知识蒸馏
  - logit蒸馏
  - 类别相关性
  - 教师纠错
---

# Knowledge Distillation with Refined Logits

**会议**: ICCV 2025  
**arXiv**: [2408.07703](https://arxiv.org/abs/2408.07703)  
**代码**: https://github.com/zju-SWJ/RLD  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, logit蒸馏, 类别相关性, 教师纠错, 模型压缩

## 一句话总结

RLD 通过 Sample Confidence（样本置信度）和 Masked Correlation（掩码相关性）两种精炼知识，在不破坏类别相关性的前提下修正教师错误预测的负面影响，在 CIFAR-100 和 ImageNet 上全面超越现有 logit 蒸馏方法。

## 研究背景与动机

知识蒸馏利用高性能教师模型指导轻量学生模型训练，其中 **logit 蒸馏** 因其简单性、有效性和通用性而重新受到关注。Zhao et al. (DKD) 通过解耦经典 KL 损失，证明 logit 蒸馏可以匹敌甚至超越 feature 蒸馏。然而，**现有 logit 蒸馏方法几乎都忽略了一个关键问题：教师模型也会犯错**。

教师预测错误时，标准蒸馏损失（KL 散度，要求学生对齐教师输出）和交叉熵损失（要求学生预测正确类别）之间产生严重分歧。这种分歧会阻碍学生性能提升——学生被迫在"模仿教师" 和"预测正确"之间左右为难。

现有的纠正方法（LA、RC、LR）试图用标签信息修正教师 logits：
- **LA (Label Augment)**：交换预测最大类和真实类的值
- **RC (Review & Correction)**：放大真实类的概率
- **LR (Label Revision)**：混合 one-hot 标签和教师 soft label

但本文通过一个玩具示例（图 1）清楚展示了这些方法的致命问题：**交换/增强操作会破坏类别间的相关性**。例如图像是"狮子"但教师预测为"森林"，交换操作后"狮子"和"老虎"（高度相关的类）的排名被严重打乱，阻碍了"dark knowledge"（类别间语义关系）的传递。

核心矛盾：**如何纠正教师的错误预测，同时保留类别间有价值的相关性信息**。

RLD 的核心 idea：将教师知识精炼为两种互补形式——(1) **Sample Confidence**：只传递教师对样本的置信水平，不强制学生复制错误的类别排名；(2) **Masked Correlation**：动态掩蔽教师预测中可能错误的部分（排名高于真实类的类别），保留剩余类别间的相关性。教师犯的错越多，掩蔽的类越多，传递的知识越少（但不会传递错误信息）。

## 方法详解

### 整体框架

RLD 的总损失由三部分组成：

$$L_{RLD} = L_{CE} + \alpha \cdot L_{SCD} + \beta \cdot L_{MCD}$$

- $L_{CE}$：标准交叉熵损失（学生预测 vs 真实标签）
- $L_{SCD}$：样本置信度蒸馏损失
- $L_{MCD}$：掩码相关性蒸馏损失

三者协同：$L_{CE}$ 确保正确性，$L_{SCD}$ 传递教师的置信水平，$L_{MCD}$ 传递类别间的语义关系。

### 关键设计

1. **Sample Confidence Distillation (SCD)**：

    - 功能：将教师和学生的 logit 分布压缩为二元分布，传递"教师对当前样本有多确信"这一信息
    - 核心思路：
        - 教师二元分布：$b^T = \{\hat{p}_{max}^T, 1 - \hat{p}_{max}^T\}$（最大预测概率 vs 其余）
        - 学生二元分布：$b^S = \{\hat{p}_{true}^S, 1 - \hat{p}_{true}^S\}$（真实类概率 vs 其余）
        - 对齐：$L_{SCD} = \tau^2 \text{KL}(b^T, b^S)$
    - **关键巧妙之处**：教师端用预测最大类的概率，学生端用真实类的概率。这意味着当教师预测正确时，两端对齐的是同一类；当教师预测错误时，学生被引导"像教师一样自信地预测真实类"，而不是像教师一样预测错误的类
    - 梯度分析：$\frac{\partial L_{SCD}}{\partial z_i} = p_i^S - p_{max}^T$（$i$ = true）与 $\frac{\partial L_{CE}}{\partial z_i} = p_i^S - y_i$ 不同——SCD 提供了更柔和的监督信号，避免过拟合
    - 设计动机：纯交叉熵将目标硬编码为 1（过拟合风险），而 SCD 将目标设为教师的最大预测概率（通常 <1），提供了正则化效果

2. **Masked Correlation Distillation (MCD)**：

    - 功能：动态掩蔽教师 logit 中排名高于真实类的所有类别，仅对剩余类别做 KL 对齐
    - 核心思路：
        - 掩码集合：$M_{ge} = \{i | z_i^T \geq z_{true}^T, 1 \leq i \leq C\}$
        - 掩蔽后的分布：$\tilde{p}_i = \frac{\exp(z_i/\tau)}{\sum_{c \notin M_{ge}} \exp(z_c/\tau)}$，仅对 $i \notin M_{ge}$ 的类计算
        - 对齐：$L_{MCD} = \tau^2 \text{KL}(\tilde{p}^T, \tilde{p}^S)$
    - **自适应特性**：教师预测越准确（真实类排名越高），被掩蔽的类越少，传递的类别相关性越多；教师预测越差，掩蔽越多（极端情况下只掩蔽真实类本身），减少错误信息的传递
    - 为什么用 $M_{ge}$（≥）而非 $M_g$（>）：消融实验和玩具示例（图 6）表明，$M_g$ 会将真实类的知识同时放入 SCD 和 MCD 中，导致两个损失间的冲突。$M_{ge}$ 确保真实类始终被掩蔽，避免冲突
    - 设计动机（图 3b）：被掩蔽的类给予学生充分的自由度，其预测可以与教师完全不同而不会产生大的损失。这让学生可以"纠正"教师的排名错误

3. **与 DKD 的理论联系**：

    - 当教师始终预测正确时，RLD 退化为 DKD
    - DKD 的 NCKD 部分（掩蔽真实类后的分布对齐）之所以有效，RLD 提供了新解释：给学生调整真实类排名的自由度，间接缓解教师错误的影响
    - RLD 相比 DKD 更适合较大的 $\alpha$ 值，因为精炼后的知识消除了错误信息

### 损失函数 / 训练策略

- 默认超参数：$\alpha = 1, \beta = 4$
- 温度 $\tau$ 遵循标准设置
- 训练框架与 LSKD、DKD、CRD 保持一致以确保公平比较
- 所有结果为三次实验的平均值

## 实验关键数据

### 主实验

**CIFAR-100 同构蒸馏（教师学生同架构族）**：

| 教师→学生 | KD | DKD | LA | RC | LR | **RLD** |
|-----------|-----|-----|-----|-----|-----|---------|
| ResNet32×4→ResNet8×4 | 73.33 | 76.32 | 73.46 | 74.68 | 76.06 | **76.64** |
| VGG13→VGG8 | 72.98 | 74.68 | 73.51 | 73.37 | 74.66 | **74.93** |
| WRN-40-2→WRN-40-1 | 73.54 | 74.81 | 73.75 | 74.07 | 74.42 | **74.88** |
| ResNet56→ResNet20 | 70.66 | 71.97 | 71.24 | 71.63 | 70.74 | **72.00** |
| ResNet110→ResNet32 | 73.08 | 74.11 | 73.39 | 73.44 | 73.52 | 74.02 |
| ResNet110→ResNet20 | 70.67 | 71.06 | 70.86 | 71.41 | 70.61 | **71.67** |

**ImageNet（大规模验证）**：

| 教师→学生 | KD | DKD | RC | LR | **RLD** |
|-----------|-----|-----|-----|-----|---------|
| Res34→Res18 Top-1 | 71.03 | 71.70 | 71.59 | 70.29 | **71.91** |
| Res50→MN-V1 Top-1 | 70.50 | 72.05 | 71.86 | 71.76 | **72.75** |

RLD 在 ImageNet 上超越所有 feature 和 logit 蒸馏方法，且提升幅度大于 CIFAR-100。

### 消融实验

| $L_{CE}$ | $L_{SCD}$ | $L_{MCD}$ (掩码) | 准确率 | 说明 |
|----------|-----------|------------------|--------|------|
| ✓ | | | 72.50 | 仅交叉熵 |
| ✓ | ✓ | | 73.55 | +SCD 提供正则化 |
| ✓ | | ✓ ($M_g$) | 75.50 | MCD 用 $M_g$ 掩码 |
| ✓ | | ✓ ($M_{ge}$) | 75.64 | MCD 用 $M_{ge}$ 掩码（更好） |
| ✓ | ✓ | ✓ ($M_g$) | 75.53 | SCD + MCD($M_g$) 冲突 |
| ✓ | ✓ | ✓ ($M_{ge}$) | **76.64** | 完整 RLD（最优） |

### 关键发现

- **教师训练准确率越低，RLD vs DKD 的优势越大**（图 4）。ImageNet 上教师训练准确率低于 CIFAR-100，因此 RLD 在 ImageNet 上提升更显著
- **逆向蒸馏**（弱教师指导强学生）：RLD 仍大幅超越 DKD（Δ 最大 +0.69%），说明精炼知识在教师能力有限时更加重要
- $M_{ge}$ vs $M_g$ 的微妙差异被消融实验清楚解释：$M_g$ 导致 SCD 和 MCD 在真实类上的损失冲突
- RLD 训练的学生与教师的 logit 差异反而比 DKD 更大（图 5），但性能更好——证明"无脑对齐教师"不是最优策略

## 亮点与洞察

1. **"纠正错误但保留关系"**这一问题被首次清楚地提出和解决。以往的纠正方法（LA、RC、LR）关注纠正预测值，却忽视了类别相关性这一关键 trade-off
2. SCD 中"教师用最大预测概率、学生用真实类概率"的巧妙不对称设计，一举实现了纠错和正则化两个目标
3. MCD 的自适应掩蔽机制优雅地实现了"错得多则学得少"的理念，无需额外超参数控制掩蔽比例
4. 与 DKD 的理论联系清晰，为 DKD 中 NCKD 为何有效提供了新解释
5. 可与 LSKD 等 logit 标准化技术正交组合（表 5）

## 局限与展望

- $\alpha$ 和 $\beta$ 的最优值在不同蒸馏对间差异显著（图 7），自动调参是重要的改进方向
- 仅在分类任务上验证，未扩展到检测、分割等下游任务
- 掩蔽策略基于 logit 排名，可能在 logit 值差异极小时不够稳定
- 未探索与 feature 蒸馏的组合（论文在 Future Work 中提及，可通过 CAM 对齐 feature）

## 相关工作与启发

- DKD（解耦知识蒸馏）是最直接的对比方法，RLD 在其基础上通过精炼解决了教师错误问题
- CTKD（课程温度）的动态温度思想可与 RLD 组合
- 对于其他需要从有噪声信号学习的场景（如 noisy label learning），"自适应掩蔽不可靠信号"的思路有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

## 相关论文

- [A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [Local Dense Logit Relations for Enhanced Knowledge Distillation](local_dense_logit_relations_for_enhanced_knowledge_distillation.md)
- [Perspective-Aware Teaching: Adapting Knowledge for Heterogeneous Distillation](perspective-aware_teaching_adapting_knowledge_for_heterogeneous_distillation.md)
- [Fuse Before Transfer: Knowledge Fusion for Heterogeneous Distillation](fuse_before_transfer_knowledge_fusion_for_heterogeneous_distillation.md)
- [Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

<!-- RELATED:END -->
