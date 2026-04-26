---
title: >-
  [论文解读] Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning
description: >-
  [ECCV 2024][AI安全][联邦学习] 本文提出Self-Driven Fisher Calibration（SDFC），利用Fisher信息度量参数对不同分布的重要程度差异，在异质联邦学习场景中有效区分恶意后门客户端并进行参数校准，突破了现有防御方法依赖数据同质性和恶意节点少数假设的局限。
tags:
  - ECCV 2024
  - AI安全
  - 联邦学习
  - 后门攻击
  - Fisher信息
  - 异质性
  - 参数校准
---

# Fisher Calibration for Backdoor-Robust Heterogeneous Federated Learning

**会议**: ECCV 2024  
**arXiv**: 无  
**代码**: [GitHub](https://github.com/WenkeHuang/SDFC)  
**领域**: AI安全 / 联邦学习  
**关键词**: 联邦学习, 后门攻击, Fisher信息, 异质性, 参数校准

## 一句话总结

本文提出Self-Driven Fisher Calibration（SDFC），利用Fisher信息度量参数对不同分布的重要程度差异，在异质联邦学习场景中有效区分恶意后门客户端并进行参数校准，突破了现有防御方法依赖数据同质性和恶意节点少数假设的局限。

## 研究背景与动机

1. **领域现状**: 联邦学习（Federated Learning）作为一种隐私保护的分布式学习范式，在视觉任务协作中展现了巨大潜力。然而，联邦学习系统极易受到后门攻击（backdoor attack）的威胁——恶意客户端通过在训练数据中注入触发器（trigger），诱导全局模型在推理时对含触发器的输入产生定向误分类。

2. **现有痛点**: 现有的后门防御方案通常基于两个核心假设：（a）各客户端数据分布同质（IID），（b）恶意客户端占少数。基于这两个假设，现有方法通过客户端级别的异常检测规则（如模型更新的范数裁剪、中值聚合等）来识别和过滤恶意客户端。然而，这些假设在现实场景中往往不成立——不同客户端自然拥有不同的数据分布（Non-IID/异质性），且恶意客户端比例可能较高。

3. **核心矛盾**: 在异质联邦学习中，数据异质的正常客户端和后门攻击者都会产生偏离全局优化方向的梯度更新。这使得基于梯度/参数偏差来区分它们变得极其困难——正常的数据异质性与恶意的后门注入在参数更新层面产生了类似的"异常"信号，传统方法无法精确区分。

4. **本文目标**: 在不依赖数据同质性和恶意节点少数假设的条件下，如何在异质联邦学习中有效防御后门攻击。

5. **切入角度**: 虽然异质客户端和后门攻击者都导致参数偏离，但它们的偏离在参数重要性维度上存在差异。利用Fisher Information可以度量每个参数对特定分布的重要程度——正常异质客户端的参数偏差集中在对其本地分布重要的参数上（合理偏差），而后门攻击者的偏差则出现在对触发器模式重要但对正常分布不重要的参数上（恶意偏差）。

6. **核心 idea**: 用Fisher信息衡量参数对"有意义分布"（本地数据和全局验证集）的重要性差异，据此校准异质客户端的参数更新，自然过滤掉后门信号。

## 方法详解

### 整体框架

SDFC在标准的联邦学习聚合框架中引入了基于Fisher信息的参数校准和聚合权重分配两个核心机制。每轮通信中：（1）服务器将全局模型分发给各客户端；（2）客户端在本地数据上训练后，不仅上传模型更新，还上传本地Fisher信息矩阵的估计；（3）服务器利用Fisher信息计算每个参数的重要性差异，对那些在本地分布和全局分布之间重要性差异大的参数进行校准；（4）为各客户端分配差异化的聚合权重，使得与全局分布更接近的客户端贡献更大。

### 关键设计

1. **Fisher信息度量参数重要性**: SDFC的核心创新在于利用Fisher Information Matrix（FIM）来刻画模型参数对特定数据分布的重要程度。对于每个客户端k，计算两组Fisher信息：（a）关于本地数据分布的Fisher信息矩阵 $F_k^{local}$，反映参数对本地任务的重要性；（b）关于全局验证分布的Fisher信息矩阵 $F^{global}$，反映参数对全局目标分布的重要性。通过比较这两组Fisher信息的差异 $|F_k^{local} - F^{global}|$，可以识别出哪些参数的本地重要性与全局重要性存在显著不一致。正常的异质客户端虽然数据分布不同，但其参数重要性模式与全局分布仍有较高一致性（因为都是同一类任务的不同子分布）；而后门攻击者由于注入了触发器模式，会在一些对正常分布不重要但对触发器响应至关重要的参数上表现出异常高的本地重要性。

2. **参数校准（Parameter Calibration）**: 基于Fisher信息差异，SDFC对重要性差异大的参数进行校准。具体操作是：对于本地重要性远高于全局重要性的参数（可能包含后门信号），降低其在聚合时的影响权重；对于重要性差异小的参数（通常是对任务本身有用的特征），保持其正常贡献。这种校准是参数级别的精细操作，不是简单地丢弃或保留整个客户端模型。这使得即使恶意客户端的大部分参数更新是正常的（仅少部分参数被后门污染），SDFC也能精准切除后门信号同时保留有用的知识贡献。

3. **自适应聚合权重分配**: 除了参数级别的校准外，SDFC还在客户端级别分配差异化的聚合权重。权重分配的依据是客户端整体参数差异的大小——本地Fisher信息与全局Fisher信息整体差异越小，说明其数据分布越接近全局分布，应给予更高的聚合权重。这个策略的效果是鼓励那些拥有更具代表性数据的客户端对全局模型贡献更多。后门攻击者由于引入了异常的触发器分布，其整体Fisher信息差异通常较大，自然获得较低的聚合权重。

### 损失函数 / 训练策略

- 客户端本地训练使用标准的交叉熵损失
- Fisher信息通过损失函数对参数的二阶导数的对角近似计算：$F_{ii} = \mathbb{E}[(\partial \mathcal{L}/\partial w_i)^2]$
- 全局Fisher信息通过在服务器端维护的小型验证集计算（不含敏感用户数据）
- 参数校准在服务器端聚合时执行，不增加客户端的计算负担
- 兼容FedAvg、FedProx等主流联邦学习聚合算法

## 实验关键数据

### 主实验

| 数据集/场景 | 指标 | SDFC | 之前SOTA防御 | 提升 |
|------------|------|------|------------|------|
| CIFAR-10 (Non-IID + Backdoor) | 主任务准确率 | 高 | 基线方法 | 维持正常性能 |
| CIFAR-10 (Non-IID + Backdoor) | 后门攻击成功率↓ | **显著降低** | 防御效果有限 | 明显提升 |
| CIFAR-100 (Non-IID + Backdoor) | 主任务准确率 | 高 | 基线方法 | 稳定 |
| CIFAR-100 (Non-IID + Backdoor) | 后门攻击成功率↓ | **显著降低** | 较高 | 大幅下降 |

### 消融实验

| 配置 | 后门成功率 | 主任务准确率 | 说明 |
|------|-----------|-------------|------|
| FedAvg (无防御) | 高 | 基线 | 后门攻击几乎100%成功 |
| + Norm Clipping | 仍较高 | 轻微下降 | 简单裁剪不足以防御 |
| + Krum/Multi-Krum | 中等 | 显著下降 | 剔除正常异质客户端 |
| + SDFC (仅参数校准) | 较低 | 保持 | 参数级校准有效切除后门 |
| + SDFC (校准+权重) | **最低** | 保持最高 | 完整方案效果最优 |

### 关键发现

- 在非IID + 后门攻击的联合挑战下，传统防御方法（Krum、中值聚合等）会错误地丢弃正常异质客户端的有用更新，导致主任务性能严重下降
- Fisher信息确实能有效区分正常异质性引起的参数偏差和后门攻击引起的参数偏差
- 参数级别的精细校准比客户端级别的粗粒度过滤更加有效，能在去除后门的同时保留有用知识
- 即使恶意客户端比例较高（如40-50%），SDFC仍能保持有效防御

## 亮点与洞察

1. **问题定义精准**: 明确指出了异质联邦学习中后门防御的核心困难——异质性和恶意性在参数层面的混淆——并提出了有效的解决思路
2. **Fisher信息的新应用**: 将Fisher信息从传统的模型压缩/持续学习领域引入联邦学习安全，巧妙利用其"参数重要性度量"特性解决新问题
3. **参数级精细操作**: 不是简单地丢弃或接受整个客户端模型，而是对每个参数分别校准，既能切除后门又能保留知识
4. **理论与实践的结合**: Fisher信息提供了坚实的理论基础，同时方法实现简洁高效

## 局限与展望

1. Fisher信息的计算需要服务器端维护一个验证集来估计全局分布，这在某些隐私严格的场景下可能不现实
2. Fisher信息的对角近似是常用但粗糙的近似，可能丢失参数间的相关性信息
3. 对于更隐蔽的自适应后门攻击（攻击者根据防御策略调整攻击方式），SDFC的鲁棒性需进一步验证
4. 实验主要在图像分类任务上进行，更复杂的视觉任务（目标检测、分割）上的效果值得探索
5. 通信开销可能增加——客户端需要额外上传Fisher信息矩阵

## 相关工作与启发

- **FedAvg (McMahan et al., 2017)**: 联邦学习的基础算法，对后门攻击没有防御能力
- **Krum/Multi-Krum (Blanchard et al., 2017)**: 通过选择最接近多数的客户端更新进行聚合，但在异质场景下可能错误过滤正常客户端
- **EWC (Kirkpatrick et al., 2017)**: 最早在持续学习中使用Fisher信息度量参数重要性，SDFC的思路与之有理论联系
- **FLTrust (Cao et al., 2022)**: 使用服务器端验证集构建信任分数，与SDFC在服务器端验证集的使用上有相似之处
- 启发：Fisher信息在安全领域的应用值得更多探索，例如检测联邦学习中的数据投毒、模型逆向等攻击

## 评分

- **新颖性**: ⭐⭐⭐⭐ Fisher信息用于后门防御是新颖的切入点，参数级校准的思路有创造性
- **实验充分度**: ⭐⭐⭐⭐ 多种攻击方式、多种异质度设置、多种基线方法的全面对比
- **写作质量**: ⭐⭐⭐⭐ 问题动机阐述清晰，核心矛盾分析透彻
- **价值**: ⭐⭐⭐⭐ 解决了异质联邦学习中后门防御的实际痛点，对联邦学习安全部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](../../CVPR2026/ai_safety/domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)
- [\[ECCV 2024\] SkyMask: Attack-Agnostic Robust Federated Learning with Fine-Grained Learnable Masks](skymask_attack-agnostic_robust_federated_learning_with_fine-grained_learnable_ma.md)
- [\[ECCV 2024\] Towards Multi-modal Transformers in Federated Learning](towards_multimodal_transformers_in_federated_learning.md)
- [\[NeurIPS 2025\] FedFACT: A Provable Framework for Controllable Group-Fairness Calibration in Federated Learning](../../NeurIPS2025/ai_safety/fedfact_a_provable_framework_for_controllable_group-fairness_calibration_in_fede.md)
- [\[CVPR 2025\] Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)

<!-- RELATED:END -->
