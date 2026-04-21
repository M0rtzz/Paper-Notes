---
title: >-
  [论文解读] HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning
description: >-
  [AAAI 2026][AI安全][分割联邦学习] 提出 HealSplit，首个针对分割联邦学习（SFL）的统一防御框架，通过拓扑感知检测（TAS）识别中毒样本、GAN 生成语义一致的替代表示、对抗多教师蒸馏训练一致性验证学生模型，实现端到端检测与恢复，在五类投毒攻击下均大幅超越十种 SOTA 防御方法。
tags:
  - AAAI 2026
  - AI安全
  - 分割联邦学习
  - 数据投毒攻击
  - 拓扑异常检测
  - 对抗蒸馏
  - 自愈防御
---

# HealSplit: Towards Self-Healing through Adversarial Distillation in Split Federated Learning

**会议**: AAAI 2026  
**arXiv**: [2511.11240](https://arxiv.org/abs/2511.11240)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 分割联邦学习, 数据投毒攻击, 拓扑异常检测, 对抗蒸馏, 自愈防御

## 一句话总结

提出 HealSplit，首个针对分割联邦学习（SFL）的统一防御框架，通过拓扑感知检测（TAS）识别中毒样本、GAN 生成语义一致的替代表示、对抗多教师蒸馏训练一致性验证学生模型，实现端到端检测与恢复，在五类投毒攻击下均大幅超越十种 SOTA 防御方法。

## 研究背景与动机

**分割联邦学习（SFL）** 结合了联邦学习（FL）和分割学习（SL）的优势，每个客户端执行本地前向传播并将中间表示（smashed data）传输到服务器。然而 SFL 存在多个攻击面：

**多种投毒攻击类型**：
   - 标签投毒（LP）：修改标签 $y_j' = (y_j + \delta_y) \mod C$
   - 数据投毒（DP）：修改输入 $x_j' = x_j + \delta_x$
   - Smashed data 投毒（SP）：修改中间表示 $z_j' = g_\phi(x_j) + \delta_z$
   - 权重投毒（WP）：修改模型参数 $\theta' = \theta + \Delta_\theta$
   - 多向量组合攻击（DP+SP、WP+SP 等）

**现有防御的不足**：
   - Krum、Trimmed Mean、Bulyan 等统计聚合方法 → 假设能访问完整模型更新或原始梯度，在 SFL 的分割架构下假设不成立
   - FLTrust、DnC 等高级防御 → 主要针对单一攻击向量，泛化性差
   - ShieldFL 使用加密余弦相似度 → 在组合攻击下仍然脆弱

**核心动机**：SFL 中 smashed data 是投毒攻击的主要通道 → 保障 smashed data 安全即可防御广泛的攻击类型。但检测后还需要恢复机制，否则丢弃样本会降低模型效用。

## 方法详解

### 整体框架

HealSplit 包含三个核心组件，形成检测→替换→验证的端到端流程：

1. **拓扑感知检测** → 在 smashed data 上建图，通过 TAS 识别投毒样本
2. **语义一致替换** → GAN 生成替代表示
3. **对抗多教师蒸馏** → 训练一致性验证学生模型确保替代质量

### 关键设计

1. **拓扑感知恶意数据检测（Topology-Aware Detection）**

   **核心观察**：投毒样本在特征空间中倾向于形成局部密集但全局孤立的簇——即彼此高度相似但与良性数据连接很弱。

   **图构建**：对 smashed data 构建 KNN 加权图，邻接矩阵：
   $$W_{kj} = \begin{cases} \exp(-\gamma \|z_k - z_j\|^2), & \text{if } z_j \in \mathcal{N}_k \text{ and } z_k \in \mathcal{N}_j \\ 0, & \text{otherwise} \end{cases}$$

   **拓扑异常评分（TAS）**：使用 Personalized PageRank（PPR）进行迭代传播，捕获局部和全局结构异常：
   $$r_k^{(t+1)} = \mathbb{I}_{[t=0]} \cdot \frac{1}{d_k + \epsilon} + \mathbb{I}_{[t \geq 1]} \cdot \left(\alpha \sum_{w \in \mathcal{N}(k)} \frac{r_w^{(t)}}{d_w} + (1-\alpha) v_k \right)$$

   **自适应阈值**：通过核密度估计（KDE）自动确定检测阈值：
   $$T = \min\left(\underset{r}{\operatorname{argmin}} \hat{f}(r), Q_\rho(\{r_k\})\right)$$

   TAS 低于阈值的样本被标记为投毒样本。

   **设计动机**：社交网络中的图传播理论启发——异常节点的传播模式与正常节点在拓扑维度上存在可检测差异。PPR 兼顾局部邻域和全局图结构，比单纯的特征距离更鲁棒。

2. **语义一致替换（Semantically Consistent Substitution）**

   使用 vanilla GAN 在已清洁的 smashed data 上训练，生成替代表示：
   $$\mathcal{L}_D = -\mathbb{E}_z[\log D(z)] - \mathbb{E}_{\tilde{z}}[\log(1 - D(\tilde{z}))]$$
   $$\mathcal{L}_G = -\mathbb{E}_{\tilde{z}}[\log D(\tilde{z})]$$

   GAN 仅使用当前轮次的 smashed data 训练。由于训练数据有限可能导致生成语义不一致的样本，需要一致性验证学生模型进行过滤。

3. **对抗多教师蒸馏（Adversarial Multi-Teacher Distillation）**

   两个互补教师：

   **(a) AD（Anomaly-Influence Debiasing）教师**：
   - 定义三个任务：投毒识别(a)、客户端识别(b)、类别分类(c)
   - 计算梯度交互评分（GIS）矩阵 $\mathbf{G}_p$：衡量任务间梯度对齐程度
   - 构建任务间影响评分矩阵 $\mathbf{M}_p$：结合 TAS 矩阵 $\mathbf{R}$ 和 GIS 矩阵
   - 损失函数通过影响评分矩阵动态调整标签影响权重

   **(b) Vanilla 教师**：仅在已清洁数据上训练，保持语义完整性

   学生模型的蒸馏损失使用 KL 散度：
   $$\mathcal{L}_{VS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{van}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$$
   $$\mathcal{L}_{IS} = \tau^2 \cdot KL(LogSoftmax(h_{T_{AD}}(z_i)/\tau), Softmax(h_S(z_i)/\tau))$$

### 损失函数 / 训练策略

**学生总损失**：$\mathcal{L}_{Stu} = \sum_k (\mathcal{L}_a + \lambda_b \mathcal{L}_b + \mu \mathcal{L}_{VS} + \eta \mathcal{L}_{IS})$

**动量自适应优化**：动态平衡两个教师的贡献，防止任一方主导：
$$\mu_t = m \cdot \mu_{t-1} + (1-m) \cdot \sigma\left(\kappa \cdot \frac{\mathcal{L}_{VS} - \mathcal{L}_{IS}}{\mathcal{L}_{VS} + \mathcal{L}_{IS} + \epsilon}\right)$$

**理论保证**：证明 HealSplit 通过提高梯度相似性降低服务器端梯度方差（SGV），从而改善收敛性。

## 实验关键数据

### 主实验：多攻击类型下的鲁棒性

在 MNIST 上，10 个客户端（20% 恶意），ResNet-18 backbone：

| 防御方法 | 无攻击 | DP | WP | SP | LP | DP+SP | WP+SP | LP+SP |
|---------|--------|----|----|----|----|-------|-------|-------|
| FedAvg | 96.90 | 10.12 | 44.74 | 96.90 | 79.23 | 9.19 | 68.22 | 64.82 |
| Krum | 96.66 | 76.77 | 15.91 | 71.62 | 82.95 | 70.48 | 76.20 | 70.68 |
| ShieldFL | 97.58 | 83.73 | 84.24 | 96.35 | 78.18 | 75.54 | 75.16 | 12.97 |
| DnC | 97.27 | 80.58 | 82.18 | 95.33 | 80.43 | 76.34 | 78.82 | 75.33 |
| FLTrust | 96.52 | 76.48 | 48.70 | 94.42 | 55.56 | 73.39 | **11.33** | 32.41 |
| **HealSplit** | 97.17 | **96.86** | **95.99** | **96.75** | **96.72** | **93.88** | **92.44** | **93.88** |

关键发现：
- HealSplit 在所有攻击场景下保持 92%+ 准确率，波动极小
- SOTA 方法 FLTrust 在 WP+SP 组合攻击下暴跌到 11.33%
- ShieldFL 在 LP+SP 下也暴跌到 12.97%
- HealSplit 是攻击无关的——无需预知攻击类型

### 消融实验

| 组件 | MNIST | F-MNIST | CIFAR-10 | HAM10k |
|------|-------|---------|----------|--------|
| **HealSplit (完整)** | **93.88** | **84.11** | **53.87** | **72.27** |
| w/o Vanilla Teacher | 90.99 | 80.64 | 51.27 | 69.64 |
| w/o AD Teacher | 87.34 | 75.17 | 46.40 | 63.20 |
| w/o Distillation | 74.38 | 69.65 | 42.75 | 59.61 |
| w/o Adversarial | 92.74 | 82.59 | 51.55 | 70.40 |

- AD 教师和蒸馏机制贡献最大（去掉后分别下降 ~6.5% 和 ~19.5%）
- Vanilla 教师提供训练稳定性
- 对抗机制增强抗强攻击能力

### 关键发现

1. **跨数据集泛化**：在 MNIST、F-MNIST、CIFAR-10、HAM10000（非 IID）上均优于所有基线
2. **跨模型泛化**：在 ResNet-18、ResNet-152、VGG16 上均一致领先
3. **客户端数量鲁棒**：增加客户端数量时 HealSplit 保持稳定高精度，DnC 显著退化
4. **恶意比例鲁棒**：恶意客户端比例从 10% 增加到 50% 时，HealSplit 仅轻微下降
5. **自适应攻击鲁棒**：即使攻击者尝试最小化 TAS 差异来逃避检测，HealSplit 仍超越最强基线

## 亮点与洞察

1. **首个 SFL 统一防御框架**：覆盖五类攻击，而非针对单一攻击设计
2. **检测+恢复的端到端设计**：不是简单丢弃可疑样本，而是生成替代并验证一致性
3. **拓扑视角的创新**：用图传播理论分析 smashed data 异常模式，比特征空间距离更鲁棒
4. **理论支撑**：证明 HealSplit 降低 SGV，为鲁棒性提供收敛性理论保证
5. **无需攻击先验**：自动实时检测异常并自适应调整阈值，无需手动调参

## 局限与展望

- 仍然依赖图像分类任务评估，对 NLP 或其他模态的适用性未验证
- GAN 训练延迟可能影响 SFL 系统效率（需与训练轮次同步）
- 拓扑检测假设投毒样本形成"局部密集、全局孤立"的簇 → 若攻击者刻意分散投毒样本，可能绕过检测
- 自适应攻击实验虽然展示了性能下降但仍领先 → 更强的自适应攻击（如基于梯度的逃逸）值得进一步探索
- 仅在 classification 任务上验证，生成任务或回归任务的适用性待考察

## 相关工作与启发

- **与 FL 防御的区别**：传统 FL 防御（Krum、Bulyan、FLTrust）假设能访问完整模型更新，SFL 中不成立
- **对抗蒸馏的应用**：DTDBD 和 B-MTARD 的双教师框架启发了本文的 AD Teacher + Vanilla Teacher 设计
- **实践意义**：SFL 正成为隐私保护分布式学习的热门范式，HealSplit 为其提供了第一个针对性的安全保障

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（SFL 统一防御框架的首创性 + 拓扑检测+对抗蒸馏的有机结合）
- 实验充分度: ⭐⭐⭐⭐⭐（4 数据集 × 5 攻击类型 × 10 基线 × 多维度消融+泛化实验）
- 写作质量: ⭐⭐⭐⭐（框架复杂但阐述有条理，公式推导清晰）
- 价值: ⭐⭐⭐⭐⭐（填补 SFL 安全防御的重要空白）

<!-- RELATED:START -->

## 相关论文

- [FedAFD: Multimodal Federated Learning via Adversarial Fusion and Distillation](../../CVPR2026/ai_safety/fedafd_multimodal_federated_learning_via_adversarial_fusion_and_distillation.md)
- [Self-Refining Language Model Anonymizers via Adversarial Distillation](../../NeurIPS2025/ai_safety/self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [InfoDecom: Decomposing Information for Defending Against Privacy Leakage in Split Inference](infodecom_decomposing_information_for_defending_against_privacy_leakage_in_split.md)

<!-- RELATED:END -->
