---
title: >-
  [论文解读] Split Adaptation for Pre-trained Vision Transformers
description: >-
  [CVPR 2025][AI安全][隐私保护] 本文提出 Split Adaptation (SA)，将预训练 ViT 分割为前端（量化后发送给客户端）和后端（留在服务器），通过双层噪声注入保护数据隐私，配合OOD增强和patch检索增强缓解噪声影响和过拟合，在保护模型和数据的前提下实现高效少样本下游适配。
tags:
  - "CVPR 2025"
  - "AI安全"
  - "隐私保护"
  - "Transformer"
  - "分割学习"
  - "少样本适配"
  - "模型知识产权"
---

# Split Adaptation for Pre-trained Vision Transformers

**会议**: CVPR 2025  
**arXiv**: [2503.00441](https://arxiv.org/abs/2503.00441)  
**代码**: [https://github.com/conditionWang/Split_Adaptation](https://github.com/conditionWang/Split_Adaptation)  
**领域**: AI安全  
**关键词**: 隐私保护, Vision Transformer, 分割学习, 少样本适配, 模型知识产权

## 一句话总结
本文提出 Split Adaptation (SA)，将预训练 ViT 分割为前端（量化后发送给客户端）和后端（留在服务器），通过双层噪声注入保护数据隐私，配合OOD增强和patch检索增强缓解噪声影响和过拟合，在保护模型和数据的前提下实现高效少样本下游适配。

## 研究背景与动机

**领域现状**：预训练 ViT 已成为基础模型，广泛用于各领域的下游任务适配。在医疗、金融等隐私敏感领域，客户端持有数据不愿共享，服务端持有模型作为知识产权不愿泄露，形成双重保护需求。

**现有痛点**：(1) 标准适配方法需要直接访问数据，隐私域不可行；(2) 将模型发送到客户端做本地适配（如联邦学习）会泄露模型IP且计算负担大；(3) 分割学习(SL)传输中间表示可被数据重建攻击(DRA)恢复原始数据；(4) Offsite Tuning 发送模型仿真器，仿真器参数完全暴露，攻击者可据此创建高性能模型。

**核心矛盾**：需要同时保护数据（不离开客户端）和模型（不暴露完整参数），且客户端计算开销要小。

**本文目标**：设计一个保护数据和模型双重安全的 ViT 适配方法，在少样本场景下实现高效适配。

**切入角度**：将 ViT 分割为前端（浅层）和后端（深层），仅将量化后的前端发给客户端，客户端提取特征表示后注入噪声再传回服务器。量化保护模型参数，噪声保护数据隐私。

**核心 idea**：模型分割+量化保护IP，双层噪声（前端参数噪声+表示噪声）保护隐私，OOD增强+patch检索增强弥补噪声带来的性能损失。

## 方法详解

### 整体框架
SA 在客户端-服务器框架下工作：(1) 服务器将 ViT 的前 $k$ 层量化为低位值发送给客户端；(2) 客户端在量化前端注入随机噪声后提取数据表示，再对表示注入噪声后发回服务器；(3) 服务器在后端训练任务模块。

### 关键设计

1. **模型分割与量化保护**:

    - 功能：保护预训练 ViT 的模型知识产权
    - 核心思路：将 ViT 的前 $k$ 层参数用低位量化（如INT4）替代原始浮点值，使攻击者无法从量化前端恢复出原始模型参数。客户端只接收到量化版本的浅层网络，无法据此重建完整模型
    - 设计动机：直接发送原始参数等同泄露模型，量化引入的精度损失恰好成为模型保护手段

2. **双层噪声注入**:

    - 功能：保护客户端数据隐私，抵御数据重建攻击
    - 核心思路：客户端向接收到的量化前端参数注入随机噪声（模型级），再对提取出的中间表示注入额外噪声（数据级），双重噪声使得服务器即使尝试 DRA 也无法恢复原始图像
    - 设计动机：单层噪声可能被高级攻击方法破解，双层噪声提供更强的隐私保障

3. **OOD增强与Patch检索增强**:

    - 功能：缓解噪声注入和少样本带来的性能下降
    - 核心思路：数据级OOD增强在表示空间施加增强变换提升泛化性；模型级OOD增强通过在后端训练时适应噪声分布；Patch检索增强利用存储的patch特征进行检索式数据增强，生成多样化表示缓解少样本过拟合
    - 设计动机：噪声注入不可避免地降低表示质量，需要配套的增强策略来补偿

### 损失函数 / 训练策略
标准交叉熵损失用于下游分类任务。训练在服务器端进行（仅训练后端+任务模块），客户端计算量极小（仅需前向传播前端）。

## 实验关键数据

### 主实验

| 方法 | 数据保护 | 模型保护 | 客户端开销 | 准确率(平均) |
|------|---------|---------|-----------|-------------|
| SA (本文) | ✓ | ✓ | 小 | **最优** |
| Offsite Tuning | ✓ | ✗ | 大 | 次优 |
| 标准分割学习 | 部分 | 部分 | 大 | 中等 |
| Linear Probing | ✓ | ✓ | 无 | 较低 |

### 消融实验

| 配置 | 关键效果 |
|------|---------|
| 无噪声注入 | 性能最高但无隐私保护 |
| 仅模型级噪声 | 部分隐私保护 |
| 双层噪声（无OOD增强） | 隐私好但性能下降明显 |
| 双层噪声 + OOD增强 + Patch检索 | 最佳平衡点 |

### 关键发现
- SA 成功抵御了 FSHA、PCAT、Ginver、FORA 等先进数据重建攻击
- 量化到INT4即可有效防止模型参数恢复
- Patch检索增强在少样本场景下对缓解过拟合贡献最大
- 客户端计算开销仅为分割学习的一小部分

## 亮点与洞察
- 将量化的精度损失"一石二鸟"地转化为模型保护手段，而非单纯的压缩工具
- 双层噪声+OOD增强的"先破坏再修复"策略构思巧妙，在隐私和性能间取得平衡
- Patch检索增强可迁移到其他少样本学习场景

## 局限与展望
- 量化位数和噪声强度的选择需要权衡隐私-性能tradeoff
- 仅针对分类任务验证，密集预测任务（分割、检测）适用性待验证
- 对于更强的攻击（如结合辅助信息的攻击）鲁棒性未知

## 相关工作与启发
- **vs Offsite Tuning**: 仿真器参数完全暴露，模型不安全。SA通过量化保护模型IP
- **vs 标准分割学习**: 中间表示可被DRA恢复。SA的双层噪声提供更强隐私保护
- **vs 联邦学习**: 需将完整模型发送给客户端且计算重。SA仅发送量化前端

## 评分
- 新颖性: ⭐⭐⭐⭐ 量化作为模型保护的创新思路，双层噪声设计完善
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+攻击抵御评测+消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法描述细致
- 价值: ⭐⭐⭐⭐ 对隐私敏感领域的模型适配有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Semantic Alignment and Reinforcement for Data-Free Quantization of Vision Transformers](../../ICCV2025/ai_safety/semantic_alignment_and_reinforcement_for_data-free_quantization_of_vision_transf.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](../../ICCV2025/ai_safety/a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ECCV 2024\] Towards Multi-modal Transformers in Federated Learning](../../ECCV2024/ai_safety/towards_multi-modal_transformers_in_federated_learning.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](../../ICCV2025/ai_safety/fret_feature_redundancy_elimination_for_test_time_adaptation.md)

</div>

<!-- RELATED:END -->
