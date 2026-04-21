---
title: >-
  [论文解读] A Framework for Double-Blind Federated Adaptation of Foundation Models
description: >-
  [ICCV 2025][AI安全][联邦学习] 本文提出BlindFed框架，通过全同态加密（FHE）友好的架构改造、两阶段分割学习和隐私增强策略，实现了基础模型的"双盲"联邦适配——数据方看不到模型，服务方看不到数据，在CIFAR-10上达到94.28%准确率，接近LoRA的95.92%。
tags:
  - ICCV 2025
  - AI安全
  - 联邦学习
  - 全同态加密
  - 基础模型适配
  - 隐私保护
  - 分割学习
---

# A Framework for Double-Blind Federated Adaptation of Foundation Models

**会议**: ICCV 2025  
**arXiv**: [2502.01289](https://arxiv.org/abs/2502.01289)  
**代码**: [https://github.com/tnurbek/blindfed](https://github.com/tnurbek/blindfed)  
**领域**: AI安全与隐私  
**关键词**: 联邦学习, 全同态加密, 基础模型适配, 隐私保护, 分割学习

## 一句话总结
本文提出BlindFed框架，通过全同态加密（FHE）友好的架构改造、两阶段分割学习和隐私增强策略，实现了基础模型的"双盲"联邦适配——数据方看不到模型，服务方看不到数据，在CIFAR-10上达到94.28%准确率，接近LoRA的95.92%。

## 研究背景与动机
基础模型（如ViT、CLIP）在零样本任务上表现出色，但针对特定下游任务（如医学影像）的适配仍需任务特定数据训练。然而在实际场景中，存在两个核心矛盾：

(1) 数据隐私：多个数据持有方（如多家医院）的数据不能相互共享，也不能发送给模型服务方，隐私法规严格限制数据流通。

(2) 模型隐私：学习服务提供者（LSP）投入巨大成本训练的基础模型是其核心资产，不愿将模型参数分享给数据方。

传统联邦学习解决了数据隐私问题，但要求将模型发送给客户端进行本地训练，违反了模型隐私约束。LoRA等参数高效微调方法仍需通过模型反向传播。而全同态加密虽能保护数据，但基础模型的复杂非线性操作使得加密推理极其困难。

本文提出"双盲"联邦适配概念：数据方看不到基础模型（也看不到彼此的数据），服务方看不到任务数据。核心idea是将FHE友好的架构改造、知识蒸馏、并行适配器和安全聚合巧妙组合，在不通过基础模型反向传播的前提下实现模型适配。

## 方法详解

### 整体框架
BlindFed分为三个阶段：(1) FHE友好架构改造：用多项式近似替换Transformer中的非线性操作；(2) 离线蒸馏：服务端用辅助数据集将原始模型的知识蒸馏到近似模型；(3) 在线适配：客户端加密数据发送给服务端进行加密推理，基于中间表示在本地训练并行适配器和分类头，最终通过安全聚合获得全局模型。

### 关键设计
1. **FHE友好架构改造**:

    - 功能：将Transformer中不兼容FHE的非线性操作替换为多项式近似
    - 核心思路：Softmax中的指数函数用Taylor展开近似 $e^x \approx \sum_{i=0}^d \frac{x^i}{i!}$（$d=6$）；GELU激活用二次函数近似 $\text{GELU}(x) \approx 0.125x^2 + 0.25x + 0.5$；LayerNorm和Softmax中的除法用Goldschmidt算法近似 $\frac{1}{x} \approx \prod_{i=0}^d (1+(1-x)^{2^i})$（$d=7$）
    - 设计动机：FHE（如CKKS方案）仅支持多项式运算，所有非线性操作必须用多项式替代

2. **两阶段分割学习**:

    - 功能：在不泄露模型和数据的前提下完成基础模型的推理和适配
    - 核心思路：
        - **阶段一（离线蒸馏）**：服务端用辅助数据集将原始FM（teacher）蒸馏到近似FM（student），前半epochs蒸馏embedding/注意力矩阵/隐藏状态，后半epochs蒸馏预测层
        - **阶段二（在线适配）**：客户端加密本地数据发送给服务端；服务端逐块执行加密推理 $\mathcal{E}(\mathbf{b}_\ell) = \hat{\mathcal{B}}_{\hat{\psi}_\ell}(\mathcal{E}(\mathbf{b}_{\ell-1}))$；每块输出发回客户端解密-重加密后传入下一块。客户端获得明文中间表示后，在本地训练并行适配器。适配器输出：$\mathbf{h}_\ell = g_\ell(\mathbf{b}_\ell + \mathbf{h}_{\ell-1}) + \mathbf{h}_{\ell-1}$，其中 $g_\ell(\mathbf{z}) = \alpha \mathbf{W}_\ell^u \text{GELU}(\mathbf{W}_\ell^d \mathbf{z})$
    - 设计动机：逐块加密推理避免了FHE在深层网络上的乘法深度限制和bootstrapping问题；并行适配器（而非LoRA）的选择是因为它不需要通过FM反向传播

3. **模型隐私增强策略**:

    - 功能：防止恶意客户端利用中间表示发起模型提取攻击
    - 核心思路：
        - **样本级置换**：服务端对每个batch内的样本应用随机置换矩阵 $\Pi_\ell$，客户端仅收到置换后的batch。服务端额外发送相邻块的逆置换乘积 $\Pi_{\ell-1}^{-1} \cdot \Pi_\ell$，使客户端可以正确计算适配器但无法恢复对应关系
        - **随机块采样（SBS）**：每次前向传播仅返回部分块的输出，且避免返回相邻块（相邻块特征相似度高，可被利用）。采样策略：若块ℓ被采样则ℓ+1必不采样，若ℓ未采样则ℓ+1以50%概率采样
    - 设计动机：中间表示以明文形式暴露给客户端，$(b_{\ell-1}, b_\ell)$ 配对可用于模型提取；置换和采样打破了这种对应关系

### 损失函数 / 训练策略
- 蒸馏阶段：前半epochs用MSE蒸馏中间表示，后半epochs用交叉熵+KL散度蒸馏预测
- 适配阶段：使用交叉熵损失，SGD优化器（lr=0.001），50轮通信，学习率在第25/40轮衰减0.1
- 安全聚合使用MPC实现FedAvg

## 实验关键数据

### 主实验

| 数据集 | 方法 | 双盲? | 集中式 | 联邦(α=100) | 联邦(α=1) | 联邦(α=0.01) |
|--------|------|-------|--------|-------------|-----------|-------------|
| CIFAR-10 | Full fine-tuning | ✗ | 0.9635 | 0.9759 | 0.9725 | 0.8857 |
| CIFAR-10 | LoRA | ✗ | 0.9592 | 0.9736 | 0.9718 | 0.8979 |
| CIFAR-10 | Linear probing | ✓ | 0.9226 | 0.9203 | 0.9191 | 0.7447 |
| CIFAR-10 | **BlindFed** | **✓** | **0.9428** | **0.9471** | **0.9413** | **0.8540** |
| CIFAR-100 | LoRA | ✗ | 0.8349 | 0.8593 | 0.8568 | 0.7647 |
| CIFAR-100 | **BlindFed** | **✓** | **0.7930** | **0.7929** | **0.7808** | **0.6620** |

### 消融实验（可扩展性和隐私增强）

| 配置 | K=10 | K=20 | K=50 | 说明 |
|------|------|------|------|------|
| Full fine-tuning | 0.9739 | 0.9513 | N/A | GPU不够 |
| LoRA | 0.9661 | 0.9584 | 0.9482 | - |
| Linear probing | 0.9167 | 0.9142 | 0.9007 | - |
| BlindFed | 0.9446 | 0.9422 | 0.9287 | 随客户端数增加下降最少 |
| BlindFed + SBS | 0.9425 | 0.9411 | **0.9388** | SBS几乎不影响精度 |

### 关键发现
- BlindFed在双盲约束下大幅超越linear probing（CIFAR-10: 94.28% vs 92.26%），接近非双盲的LoRA（95.92%）
- SBS对精度影响极小（有时甚至略好），但显著增强模型隐私
- 在极端数据异质性（α=0.01）下，BlindFed优势更明显：CIFAR-10上85.40% vs linear probing的74.47%
- 辅助数据集可以是域外数据（Tiny-ImageNet），OOD蒸馏对所有方法均有效
- 通信开销：加密中间表示约17.33MB/块，对大模型适配场景可接受

## 亮点与洞察
- "双盲"隐私约束的提出非常务实——现实中模型方和数据方确实互不信任
- 并行适配器（而非LoRA）的选择看似受限实则巧妙——完美适配了FHE+分割学习的约束
- 样本置换+随机块采样的隐私增强方案在不损失精度的前提下有效缓解模型提取攻击，设计优雅

## 局限与展望
- 通信成本高：每轮需传输 $N_k \times L \times C$ 的加密中间表示
- 服务端计算负担大：需要在加密域上执行大量矩阵运算
- 仅考虑半诚实（semi-honest）威胁模型，对恶意攻击者的鲁棒性需进一步分析
- 多项式近似引入的误差在更深/更大模型上可能积累

## 相关工作与启发
- **vs 标准联邦学习(FedAvg)**: FedAvg需要将完整模型发给客户端，违反模型隐私；BlindFed通过加密推理避免模型泄露
- **vs 私有推理(PI)方法**: SAL-ViT、Iron等关注推理阶段的隐私，BlindFed关注训练/适配阶段的双向隐私
- **vs LoRA/PEFT**: 这些方法需要通过模型反向传播，不适用于模型不可共享的场景；并行适配器是更合适的选择

## 评分
- 新颖性: ⭐⭐⭐⭐ "双盲"联邦适配概念新颖，FHE+分割学习+并行适配器的组合方案设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 4个数据集，多种数据分割策略，可扩展性分析，但缺少大模型实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题形式化清晰，框架组件层次分明，安全性分析严谨
- 价值: ⭐⭐⭐⭐ 对隐私敏感领域（医疗、金融）的基础模型适配有重要实际意义

<!-- RELATED:START -->

## 相关论文

- [FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [Geminio: Language-Guided Gradient Inversion Attacks in Federated Learning](geminio_language-guided_gradient_inversion_attacks_in_federated_learning.md)
- [Client2Vec: Improving Federated Learning by Distribution Shifts Aware Client Indexing](client2vec_improving_federated_learning_by_distribution_shifts_aware_client_inde.md)
- [LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)

<!-- RELATED:END -->
