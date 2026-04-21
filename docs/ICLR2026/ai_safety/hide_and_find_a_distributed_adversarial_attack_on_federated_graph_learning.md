---
title: >-
  [论文解读] Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning
description: >-
  [ICLR2026][AI安全][Federated Graph Learning] 提出 FedShift，一种两阶段"隐藏-发现"分布式对抗攻击框架：第一阶段通过温和的分布偏移（distributional shift）向训练图中植入隐蔽的 shifter，第二阶段以 shifter 生成器为起点高效搜索对抗扰动，多恶意客户端聚合扰动形成最终对抗样本，在六个大规模数据集上实现最高攻击成功率，同时逃逸三种主流防御算法且收敛速度提升 90% 以上。
tags:
  - ICLR2026
  - AI安全
  - Federated Graph Learning
  - adversarial attack
  - Backdoor Attack
  - 图神经网络
  - Data Poisoning
---

# Hide and Find: A Distributed Adversarial Attack on Federated Graph Learning

**会议**: ICLR2026  
**arXiv**: [2603.07743](https://arxiv.org/abs/2603.07743)  
**代码**: 待公开  
**领域**: ai_safety  
**关键词**: Federated Graph Learning, adversarial attack, Backdoor Attack, Graph Neural Networks, Data Poisoning

## 一句话总结
提出 FedShift，一种两阶段"隐藏-发现"分布式对抗攻击框架：第一阶段通过温和的分布偏移（distributional shift）向训练图中植入隐蔽的 shifter，第二阶段以 shifter 生成器为起点高效搜索对抗扰动，多恶意客户端聚合扰动形成最终对抗样本，在六个大规模数据集上实现最高攻击成功率，同时逃逸三种主流防御算法且收敛速度提升 90% 以上。

## 背景与动机
联邦图学习（Federated Graph Learning, FedGL）允许多客户端在不共享图数据的前提下协作训练 GNN，广泛应用于疾病预测、推荐系统等场景。然而 FedGL 面临严重安全威胁：恶意客户端可通过后门攻击或对抗攻击使全局模型产生错误预测。

现有攻击方法存在三个核心困境：

1. **后门信号被平滑**：恶意客户端产生的后门信号在联邦聚合过程中被正常客户端的信号稀释，攻击效果大幅下降
2. **隐蔽性与有效性矛盾**：增大攻击预算可抵抗信号平滑，但大规模数据投毒容易被防御算法检测
3. **对抗攻击收敛困难**：图结构的离散性和优化目标的非凸性导致对抗扰动搜索收敛缓慢、计算开销大

## 核心问题
如何设计一种分布式攻击方法，同时解决后门攻击的信号平滑问题和对抗攻击的收敛效率问题，在保持高隐蔽性的同时实现高攻击成功率？

## 方法详解

### 整体框架：两阶段 "Hide and Find"

FedShift 将后门攻击的思路融入对抗攻击，通过两阶段设计互补解决各自的局限。

### Stage 1: Gentle Data Poisoning（温和数据投毒）

**目标**：在联邦训练开始前，为每个恶意客户端训练一个自适应 shifter 生成器，将隐蔽的 shifter 注入训练数据。

**步骤 1 — 训练本地 GNN 模型**：每个恶意客户端用本地数据预训练一个 GAT 模型 $\theta_i^*$，用于提取高质量图嵌入并指导后续 shifter 生成器训练。

**步骤 2 — 自适应 Shifter 生成器**，包含两个子模块：

- **Shifter 位置学习**：基于图的拓扑结构计算节点聚类系数，选择影响力最高的节点子集 $\mathcal{V}_p$ 作为注入位置
- **Shifter 形状学习**：生成器 $G_{\text{gen}}$ 为选定节点产生特征扰动 $\Delta\mathbf{X}_p$，仅修改节点特征而不改变边连接，增强隐蔽性

**关键损失函数设计**：

1. **分布接近损失 $L_{\text{dist}}$**：对目标类嵌入做 K-means 聚类得到质心，最小化投毒图嵌入与最近质心的余弦距离。这使投毒图的表示温和地靠近目标类决策边界但**不越过**，不修改标签、不构建强制映射，因此恶意客户端行为几乎与良性客户端不可区分
2. **同质性损失 $L_{\text{homo}}$**：基于图同质性假设，惩罚相连节点特征差异过大的情况，确保扰动后图在结构上保持自然
3. **边界平衡交叉熵损失 $L_{\text{ce}}$**：要求投毒图仍被本地模型分类为原始标签，防止分布偏移越过决策边界

总损失：$L_{\text{stage1}} = \lambda_{\text{dist}} L_{\text{dist}} + \lambda_{\text{homo}} L_{\text{homo}} + \lambda_{\text{ce}} L_{\text{ce}}$

**在线微调**：在联邦训练过程中，可利用从服务器接收的全局模型动态微调 shifter 生成器。

### Stage 2: Adversarial Perturbation Finding（对抗扰动搜索）

联邦训练完成后，冻结全局模型 $\theta^*$，以 Stage 1 训练好的 shifter 生成器作为**高质量初始化**继续优化。此时优化目标转为最大化攻击成功率：

$$L_{\text{stage2}} = L_{\text{attack}} + \lambda_{\text{homo}} L_{\text{homo}}$$

其中 $L_{\text{attack}} = \text{CrossEntropy}(f(G_p; \theta^*), y_t)$ 驱动扰动图跨越决策边界。

**最终攻击**：聚合多个恶意客户端产生的不同对抗扰动，形成最终对抗样本，实现 "1+1>2" 的效果。

## 实验关键数据

在 6 个大规模图数据集（DD、NCI109、Mutagenicity、FRANKENSTEIN、Eth-Phish&Hack、Gossipcop）上与 4 种 SOTA 方法对比。

### Q1：抵抗联邦信号平滑
- 恶意客户端比例从 0.2 降到 0.05 时，传统方法 ASR 平均下降 25.6%–53.3%
- FedShift 的 ASR 下降不到 5%，始终保持最优攻击效果
- 在 |C_M|/N=0.2 设定下，FedShift 在 6 个数据集上均取得最高 AAS

### Q2：逃逸防御算法
- 面对 FoolsGold、FedKrum、FedBulyan 三种主流鲁棒联邦学习防御
- FedShift 在所有场景下保持最高 AAS，平均领先最强 baseline 4.9%

### Q3：收敛效率
- 相比从零优化的 NI-GDBA，FedShift（无联邦微调）所需 epoch 减少 **90.3%**
- 完整 FedShift（含联邦微调）所需 epoch 减少 **98.3%**

### 消融实验
- 仅 Stage 1 → +FL-Tune：平均 AAS 提升 17.0%
- 仅 Stage 1 → +Stage 2：平均 AAS 提升 32.2%
- Stage 1 + Stage 2 的效果已接近完整模型，说明 Stage 2 是提升攻击有效性的关键

## 亮点
1. **新颖的两阶段攻击范式**：首次在统一框架中结合后门攻击（植入）和对抗攻击（搜索），利用整个联邦学习过程的信息
2. **温和分布偏移策略**：不修改标签、仅推向决策边界而不越过，从根本上解决隐蔽性问题
3. **优化初始化优势**：Stage 1 的 shifter 生成器为 Stage 2 提供高质量起点，收敛效率提升一个数量级
4. **多客户端扰动聚合**：不同恶意客户端产生的互补扰动聚合后效果远超单一扰动
5. **实验全面扎实**：6 个跨领域大规模数据集、3 种防御算法、详尽的消融和超参分析

## 局限与展望
1. **威胁模型假设较强**：攻击者需在联邦训练全程参与，实际场景中恶意客户端可能被动态踢出
2. **仅针对图分类任务**：未验证在节点分类、链接预测等其他图任务上的表现
3. **防御方法有限**：仅测试了 3 种防御算法，未包含更新的防御方法如 FLTrust、FLAME 等
4. **GNN 架构单一**：所有实验仅用 GAT 作为骨干网络，未验证对 GIN、GraphSAGE 等其他架构的泛化性
5. **缺少投毒检测分析**：未评估数据层面的异常检测方法是否能发现 shifter 注入的异常

## 与相关工作的对比

| 方法 | 类型 | 隐蔽性 | 抗平滑 | 收敛速度 | 综合效果 |
|------|------|--------|--------|----------|----------|
| Rand-GDBA | 后门（静态触发器） | 低 | 差 | - | 低 |
| GTA | 后门（自适应触发器） | 中 | 中 | - | 中 |
| Opt-GDBA | 后门（优化触发器） | 中 | 中 | - | 中 |
| NI-GDBA | 对抗（从零优化） | 高 | 不适用 | 慢 | 中高 |
| **FedShift** | **后门+对抗** | **高** | **强** | **快** | **最高** |

FedShift 的核心创新在于打破了后门攻击和对抗攻击的范式壁垒：后门攻击的植入阶段提供隐蔽性和优化起点，对抗攻击的搜索阶段提供最终攻击有效性。

## 启发与关联
- **防御视角**：该工作揭示了在特征空间中进行温和分布偏移的攻击方式极难检测，未来防御应关注训练过程中嵌入分布的微小异常漂移
- **联邦学习安全**：说明仅依赖聚合层面的鲁棒防御（如 Krum、Bulyan）不足以抵御此类攻击，需结合数据层面的检测手段
- **攻击方法融合**：两阶段"植入-搜索"范式可推广到其他联邦学习任务（文本、图像），是一种通用的攻击设计模板

## 评分
- 新颖性: ⭐⭐⭐⭐ — 两阶段"Hide and Find"范式是首创，将后门与对抗自然统一
- 实验充分度: ⭐⭐⭐⭐ — 6 个大规模数据集、3 种防御、丰富消融，但 GNN 架构和防御方法可更丰富
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、三个挑战-三个解决对应明确，逻辑性强
- 价值: ⭐⭐⭐⭐ — 对联邦图学习安全研究有重要参考价值，新范式可启发后续工作

<!-- RELATED:START -->

## 相关论文

- [Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](../../ICCV2025/ai_safety/find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [ATEX-CF: Attack-Informed Counterfactual Explanations for Graph Neural Networks](atex-cf_attack-informed_counterfactual_explanations_for_graph_neural_networks.md)
- [Less is More: Towards Simple Graph Contrastive Learning](less_is_more_towards_simple_graph_contrastive_learning.md)
- [Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions](faithful_bi-directional_model_steering_via_distribution_matching_and_distributed.md)
- [Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)

<!-- RELATED:END -->
