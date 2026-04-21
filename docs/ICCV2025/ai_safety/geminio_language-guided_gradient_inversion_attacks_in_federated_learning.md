---
title: >-
  [论文解读] Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning
description: >-
  [ICCV 2025][AI安全][联邦学习] 本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。
tags:
  - ICCV 2025
  - AI安全
  - 联邦学习
  - 梯度反转攻击
  - 视觉语言模型
  - 隐私攻击
  - 自然语言引导
---

# Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning

**会议**: ICCV 2025  
**arXiv**: [2411.14937](https://arxiv.org/abs/2411.14937)  
**代码**: [GitHub](https://github.com/HKU-TASR/Geminio)  
**领域**: AI安全  
**关键词**: 联邦学习, 梯度反转攻击, 视觉语言模型, 隐私攻击, 自然语言引导

## 一句话总结
本文提出Geminio，首个利用视觉语言模型（VLM）实现自然语言引导的梯度反转攻击（GIA），使联邦学习中的恶意服务器可以用自然语言描述想要窃取的数据类型，并从大batch梯度中精准定位和重建匹配的隐私样本，同时不影响正常的FL模型训练。

## 研究背景与动机
联邦学习（FL）允许客户端共享梯度而非原始数据来协作训练模型，但梯度反转攻击（GIA）可以从共享梯度中重建私有训练数据，构成严重隐私威胁。

**现有GIA的核心瓶颈**：当受害者使用大batch训练时，梯度反转的搜索空间随batch大小指数增长，重建质量急剧下降。现有方法在batch > 8时基本无法恢复可辨识的图像。

**缩小重建范围的尝试**：
- **Fishing/GradFilt**：将某些类别参数设为极大值，使特定类的梯度主导。但只能按类别筛选，粒度太粗且参数异常易被检测
- **SEER/LOKI/Abandon**：引入特殊神经架构保留特定样本梯度。但只能针对亮度等语义无关条件，控制力极弱
- 所有现有方法都无法进行**语义层面的、跨类别的、实例级的**定向重建

**核心矛盾**：攻击者真正关心的是特定语义内容的数据（如"任何武器"、"人脸"），而非随机样本或特定类别，但现有方法缺乏灵活的语义指定能力。

**核心idea**：利用预训练VLM（如CLIP）的文本-图像关联能力，将攻击者的自然语言查询转化为对恶意全局模型的loss surface重塑，使匹配查询的样本产生高梯度，非匹配样本的梯度被抑制。

## 方法详解

### 整体框架
Geminio分为两个阶段：
1. **准备阶段**：接收自然语言查询 → 利用VLM和辅助数据集 → 优化恶意全局模型参数 $\boldsymbol{\Theta}_{\mathcal{Q}}$，重塑其loss surface
2. **攻击阶段**：将恶意模型发送给受害者 → 受害者用私有数据计算梯度并上传 → 服务器用任意现有重建方法从梯度中恢复匹配查询的样本

### 关键设计
1. **VLM引导的Loss Surface重塑**:

    - 功能：训练恶意全局模型使其loss surface与VLM的文本-图像相似度surface匹配
    - 核心思路：对辅助数据集中每个样本 $\boldsymbol{x}$，用VLM计算其与查询 $\mathcal{Q}$ 的相似度 $s(\boldsymbol{x}; \mathcal{Q}) = \mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(\mathcal{Q})$，然后batch内softmax归一化：
    $\alpha(\boldsymbol{x}; \mathcal{Q}, \boldsymbol{\mathcal{B}}_{\text{aux}}) = \frac{\exp(s(\boldsymbol{x}; \mathcal{Q}))}{\sum_{\boldsymbol{x}'} \exp(s(\boldsymbol{x}'; \mathcal{Q}))}$
   训练目标为：
    $\mathcal{L}_{\text{Geminio}} = \frac{\sum_{\boldsymbol{x}} \mathcal{L}(F_{\boldsymbol{\Theta}_{\mathcal{Q}}}(\boldsymbol{x}); y)(1 - \alpha(\boldsymbol{x}))}{|\boldsymbol{\mathcal{B}}_{\text{aux}}| \sum_{\boldsymbol{x}'} \mathcal{L}(F_{\boldsymbol{\Theta}_{\mathcal{Q}}}(\boldsymbol{x}'); y')(1 - \alpha(\boldsymbol{x}'))}$
    - 设计动机：高匹配度样本的系数 $(1-\alpha)$ 接近0，模型不会学到降低其loss；低匹配度样本的系数大，模型会降低其loss。最终，匹配样本保持高loss → 高梯度，非匹配样本loss→0 → 梯度可忽略

2. **VLM引导的辅助标签生成**:

    - 功能：消除辅助数据集需要标注的限制
    - 核心思路：用VLM为每个辅助样本生成软标签：
    $y_i = \frac{\mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(c_i)}{\sum_{j=1}^{K} \mathcal{V}_{\text{image}}(\boldsymbol{x})^{\intercal} \mathcal{V}_{\text{text}}(c_j)}$
   其中 $c_1, ..., c_K$ 是FL任务的类别名称
    - 设计动机：辅助数据集可以是公开的无标注图像（如从互联网爬取），大大降低了攻击门槛

3. **梯度主导机制**:

    - 功能：确保受害者提交的梯度被匹配查询的样本主导
    - 核心思路：通过loss surface重塑，对于匹配样本 $\boldsymbol{x}_{\text{target}}$：
    $\|\nabla_{\boldsymbol{\Theta}_{\mathcal{Q}}} \mathcal{L}(F(\boldsymbol{x}); y)\| \ll \|\nabla_{\boldsymbol{\Theta}_{\mathcal{Q}}} \mathcal{L}(F(\boldsymbol{x}_{\text{target}}); y_{\text{target}})\|$
   for all $\boldsymbol{x} \neq \boldsymbol{x}_{\text{target}}$
    - 设计动机：per-sample梯度大小与per-sample loss成正比，所以控制loss surface等价于控制梯度分布

### 损失函数 / 训练策略
- 使用 $\mathcal{L}_{\text{Geminio}}$ 训练恶意模型参数，标准SGD优化
- 梯度重建阶段可直接使用任何现有的重建优化方法（DLG、InvertingGrad、HFGradInv等）
- 攻击与正常训练并行：非受害客户端的梯度正常聚合，不影响全局模型训练
- 恶意模型仅在攻击轮次发送给目标受害者

## 实验关键数据

### 主实验（定向重建 - ImageNet/CIFAR-20/FER）

| 方法 | 攻击类型 | 粒度 | 语义控制 | ImageNet成功率 |
|------|---------|------|---------|--------------|
| Vanilla GIA | 全batch重建 | 全部 | 无 | 极低(batch≥8) |
| Fishing | 类别级 | 单样本 | 仅按类 | 中等 |
| GradFilt | 类别级 | 全类 | 仅按类 | 中等 |
| SEER/LOKI | 条件级 | 随机 | 亮度等 | 低 |
| **Geminio** | **实例级** | **语义匹配** | **自然语言** | **高** |

### 大batch定向重建（CIFAR-20, HFGradInv重建）

| Batch大小 | Geminio Recall(%) | Geminio Precision(%) | Baseline Recall(%) |
|-----------|-------------------|---------------------|-------------------|
| 2 | 90.12 | 85.34 | 45.23 |
| 8 | 82.45 | 80.12 | 12.56 |
| 32 | 74.38 | 73.21 | ~0 |
| 64 | 70.25 | 69.83 | ~0 |
| 128 | 67.51 | 67.12 | ~0 |
| 256 | 64.96 | 65.67 | ~0 |

### 消融/鲁棒性实验

| 配置 | F-1(%) | 说明 |
|------|--------|------|
| 辅助数据20%同分布 | 68.13 | 默认设置 |
| 辅助数据5%同分布 | 60.37 | 少量数据仍可用 |
| 辅助数据=ImageNet(跨域) | 64.37 | 跨域辅助数据有效 |
| 辅助数据=Caltech256(跨域) | 50.48 | 域差距越大效果略降 |
| 梯度剪枝防御(95%) | ≈不受影响 | 剪枝无法缓解 |
| 梯度剪枝防御(99%) | 显著下降 | 但也摧毁正常训练 |
| 拉普拉斯噪声(0.10) | ≈不受影响 | 低噪声无效 |
| 拉普拉斯噪声(0.50) | 显著下降 | 但也摧毁正常训练 |
| 参数最大值: Clean=0.35, Geminio=1.64 | 隐蔽 | Fishing=2772, GradFilt=1000 |

### 关键发现
- **Geminio是唯一支持跨类别、实例级、语义驱动定向重建的方法**
- 即使batch高达256，仍可保持约65%的召回率和精度
- 作为插件增强现有方法（DLG、InvertingGrad、HFGradInv），所有方法均显著提升
- 模型参数隐蔽性远优于Fishing和GradFilt（最大参数值仅为1.64 vs 2772/1000）
- 支持FedAvg协议（通过控制学习率）和多种架构（ResNet、MobileNet、EfficientNet、ViT）
- ViT和EfficientNet比ResNet更容易被攻击，暗示更强的模型可能更脆弱

## 亮点与洞察
- **VLM滥用的新威胁**：揭示了预训练VLM可以被武器化，为攻击者提供自然语言"通信接口"来表达攻击意图
- **Loss Surface重塑的思路优雅**：避免了二阶导数的不稳定性，通过控制每样本loss间接控制梯度分布
- **极强的实用性**：辅助数据不需要标注、不需要与FL任务相关、查询不需要与FL任务相关
- **隐蔽性设计**：模型参数自然、不修改架构、可在任意轮次发起、不影响正常训练
- **防御困难**：现有的梯度剪枝、噪声注入、参数检查均无法有效防御，揭示了严重的安全隐患

## 局限与展望
- 依赖VLM（如CLIP）的文本-图像对齐质量，对特定领域的查询可能不够精确
- 攻击成功率随batch增大而逐渐下降，对超大batch（>512）的效果未知
- FedAvg下需要控制客户端学习率，实际FL系统中客户端可能有自己的学习率设置
- 未考虑安全聚合（Secure Aggregation）等更强的防御方案
- 查询的语义粒度受VLM能力限制（如无法查询"10月3日拍的照片"）
- 重建质量仍依赖底层重建优化方法的能力

## 相关工作与启发
- Imperio（语言引导后门攻击）到Geminio（语言引导梯度反转）的延伸表明，自然语言正在成为攻击者的通用接口
- 对FL隐私防御研究有重要启示：需要设计能从根本上阻止loss surface重塑的机制
- 模型能力与隐私脆弱性的正相关性暗示"能力-隐私"权衡可能是FL的基本困境
- 为FL安全审计提供了新的威胁模型和测试工具

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个自然语言引导的GIA，VLM武器化是全新方向，问题定义和解决方案均出彩
- 实验充分度: ⭐⭐⭐⭐⭐ 3数据集×5攻击方法×4防御×多架构×FedSGD+FedAvg，极为全面
- 写作质量: ⭐⭐⭐⭐⭐ 图表精美，攻击场景可视化直观，方法描述清晰，安全分析深入
- 价值: ⭐⭐⭐⭐⭐ 揭示了FL面临的全新威胁范式，对隐私保护和AI安全研究有深远影响

<!-- RELATED:START -->

## 相关论文

- [Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning](../../CVPR2025/ai_safety/gradient_inversion_attacks_on_parameter-efficient_fine-tuning.md)
- [Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)
- [FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)

<!-- RELATED:END -->
