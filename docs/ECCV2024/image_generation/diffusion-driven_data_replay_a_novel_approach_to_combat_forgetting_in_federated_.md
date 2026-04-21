---
title: >-
  [论文解读] Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning
description: >-
  [ECCV 2024][图像生成][联邦类别持续学习] 提出 DDDR 框架，首次将预训练扩散模型引入联邦类别持续学习（FCCL），通过 Federated Class Inversion 技术为每个类别学习一个紧凑的 class embedding，利用扩散模型高质量回放历史数据以对抗灾难性遗忘，并通过对比学习弥合生成数据与真实数据的域差距。
tags:
  - ECCV 2024
  - 图像生成
  - 联邦类别持续学习
  - 灾难性遗忘
  - 扩散模型
  - 数据回放
  - 对比学习
---

# Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning

**会议**: ECCV 2024  
**arXiv**: [2409.01128](https://arxiv.org/abs/2409.01128)  
**代码**: [有 (GitHub)](https://github.com/jinglin-liang/DDDR)  
**领域**: 扩散模型 / 联邦学习  
**关键词**: 联邦类别持续学习, 灾难性遗忘, 扩散模型, 数据回放, 对比学习

## 一句话总结

提出 DDDR 框架，首次将预训练扩散模型引入联邦类别持续学习（FCCL），通过 Federated Class Inversion 技术为每个类别学习一个紧凑的 class embedding，利用扩散模型高质量回放历史数据以对抗灾难性遗忘，并通过对比学习弥合生成数据与真实数据的域差距。

## 研究背景与动机

**领域现状**：联邦学习（FL）在医疗、金融等隐私敏感领域至关重要，但真实应用中客户端会不断引入新类别数据，形成了联邦类别持续学习（FCCL）这一新兴问题。FCCL 的核心挑战是灾难性遗忘——模型在学习新任务时丧失对旧任务的记忆。

**现有方法的痛点**：

**经验回放不可用**：持续学习中最有效的方法是存储旧任务数据进行回放，但在联邦场景下，隐私保护要求禁止客户端长期保留历史数据，且参与者的退出会导致存储数据丢失

**GAN 训练不稳定**：FedCIL 使用联邦训练 ACGAN 来生成历史数据，但 GAN 训练本身就不稳定，在联邦设置中问题更加严重

**无数据蒸馏质量差**：Target 和 MFCL 使用数据无关的知识蒸馏训练生成器，但生成的样本往往更接近对抗样本（adversarial examples），与真实数据分布差距大，对模型的指导能力有限

**核心矛盾**：在不存储任何真实历史数据的前提下，如何高质量地重建历史类别数据来对抗遗忘？

**本文切入角度**：利用预训练扩散模型的强大生成能力，不训练生成模型本身，而是在其输入空间中为每个类别搜索一个条件嵌入（class embedding），大幅降低计算和通信成本，同时保证生成质量。

## 方法详解

### 整体框架

DDDR 分为两个阶段：(1) **Federated Class Inversion Phase** — 利用冻结的预训练 LDM 为每个新类别反向工程出一个 class embedding；(2) **Replay-Augmented Training Phase** — 使用历史任务的 class embedding 生成回放数据，结合真实新任务数据训练分类器。

### 关键设计

1. **Federated Class Inversion（联邦类别反演）**

   核心思想：不训练扩散模型，而是利用冻结的预训练 Latent Diffusion Model (LDM) 进行反向工程。将冻结的文本提示 "a photo of" 经文本编码器得到 $c_\theta(p)$，再拼接一个可学习的 class embedding $v$，形成引导条件 $[c_\theta(p); v]$。优化目标为：

   $$v_i^* = \arg\min_v \mathbb{E}_{z \sim \mathcal{E}(X_i), p, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\alpha_t}z + \sqrt{1-\alpha_t}\epsilon, t, [c_\theta(p); v])\|_2^2 \right]$$

   其中 $X_i$ 是第 $i$ 类的图像集合，$\mathcal{E}$ 是 LDM 的编码器，$\epsilon_\theta$ 是冻结的去噪模型。

   **设计动机**：Class embedding 可视为类别的压缩表示，只需优化和通信这个小向量（而非整个生成模型），大幅减少计算和通信资源。同时由于不修改 LDM 参数，生成与原始数据完全相同图像的概率更低，增强了隐私保护。

2. **Global Class Embedding Aggregation**

   在联邦设置中，各客户端本地优化 class embedding 后，服务器使用 FedAvg 聚合：$v_i = \frac{1}{k}\sum_{j=1}^k v_i^{(j)}$。迭代进行本地训练和全局聚合直到收敛，保存 class embedding 用于后续数据回放。

   **设计动机**：仅传输 class embedding 而非数据本身，既保护隐私又能聚合分布在不同客户端的类别知识。

3. **对比学习约束**

   生成数据与真实数据存在分布差异（domain gap）。通过有监督对比学习损失缩小同类别内生成数据和真实数据在特征空间中的距离：

   $$\mathcal{L}_{SCL} = \mathbb{E}_{e_i, e_p \sim P(e_i)} \left[ \log \frac{\exp(sim(e_i, e_p)/\tau)}{\sum_{i \neq j} \exp(sim(e_i, e_j)/\tau)} \right]$$

   其中 $P(e_i)$ 是与 $e_i$ 同类的正样本集合，$sim$ 通过 MLP 映射到 $l_2$ 归一化空间后计算相似度。

   **设计动机**：增强分类器在生成域和真实域上的泛化能力，间接提升生成数据对真实数据的代表能力。

### 损失函数 / 训练策略

分类器训练的最终目标函数：

$$\mathcal{F}^* = \arg\min_\mathcal{F} \underbrace{\mathcal{L}_{CE}}_{\text{当前任务}} + w_1 \underbrace{\mathcal{L}_{SCL}}_{\text{对比学习}} + w_2 \underbrace{\mathcal{L}_{PCE}}_{\text{历史任务CE}} + w_3 \underbrace{\mathcal{L}_{KD}}_{\text{知识蒸馏}}$$

其中 $w_1=1, w_2=0.5, w_3=10$。$\mathcal{L}_{CE}$ 对当前任务的真实和生成数据计算交叉熵；$\mathcal{L}_{PCE}$ 对历史任务的生成数据计算交叉熵；$\mathcal{L}_{KD}$ 使用 KL 散度将前一轮模型的知识蒸馏到当前模型。

训练细节：使用 ResNet-18 作为分类器，5个客户端，LDM 预训练于 LAION-400M。Class Inversion 阶段 10 轮通信 × 50 步本地训练；Replay-Augmented 阶段 100 轮 × 5 epoch 本地训练。同时为当前任务也生成数据以缓解 non-IID 问题。

## 实验关键数据

### 主实验

**CIFAR-100 对比实验（Table 1）：**

| 方法 | IID T=5 Acc↑ | IID T=5 FM↓ | IID T=10 Acc↑ | IID T=10 FM↓ | non-IID T=5 Acc↑ | non-IID T=5 FM↓ |
|---|---|---|---|---|---|---|
| Finetune | 17.33 | 0.83 | 9.03 | 0.88 | 16.48 | 0.81 |
| FedEWC | 21.35 | 0.69 | 11.76 | 0.73 | 20.96 | 0.70 |
| Target | 34.40 | 0.48 | 22.95 | 0.49 | 34.35 | 0.48 |
| MFCL | 42.67 | 0.37 | 31.35 | 0.46 | 41.19 | 0.34 |
| **Ours** | **51.04** | **0.29** | **43.45** | **0.32** | **48.45** | **0.26** |

**Tiny-ImageNet 对比实验（Table 2）：**

| 方法 | IID T=5 Acc↑ | IID T=5 FM↓ | non-IID T=10 Acc↑ | non-IID T=10 FM↓ |
|---|---|---|---|---|
| Finetune | 12.29 | 0.60 | 6.58 | 0.64 |
| Target | 17.56 | 0.45 | 11.28 | 0.42 |
| MFCL | 15.11 | 0.52 | 8.54 | 0.51 |
| **Ours** | **25.47** | **0.36** | **16.65** | **0.27** |

在 CIFAR-100 IID T=5 上超越 SOTA (MFCL) **+8.37%** 准确率，遗忘度降低 **0.08**；CIFAR-100 IID T=10 上更是提升 **+12.10%**。Tiny-ImageNet 上同样全面领先。

### 消融实验

**CIFAR-100, T=5, non-IID（Table 3）：**

| 编号 | 历史生成数据 | 当前生成数据 | 对比学习 | Acc↑ | FM↓ | 说明 |
|---|---|---|---|---|---|---|
| 1 | ✓ | ✓ | ✓ | **48.45** | **0.26** | 完整模型 |
| 2 | ✗ | ✓ | ✓ | 17.63 | 0.84 | 无历史回放→几乎全忘 |
| 3 | ✓ | ✗ | ✓ | 44.29 | 0.36 | 无当前生成数据→non-IID影响 |
| 4 | ✓ | ✓ | ✗ | 45.34 | 0.28 | 无对比学习→域差距untouched |
| 8 | ✗ | ✗ | ✗ | 16.48 | 0.81 | 等同于 Finetune |

### 关键发现

- **历史回放数据是核心**：去掉后模型几乎全忘（行2 vs 行1：Acc 48.45→17.63），FM从0.26飙升到0.84
- **当前任务生成数据缓解 non-IID**：提升 +4.16% Acc，降低 0.10 FM，通过共享同分布生成数据减轻 non-IID
- **对比学习需要与当前生成数据配合**：单独使用有效（行1 vs 4：+3.11% Acc），但没有当前生成数据时反而有害（行3 vs 6：-0.84%），说明其效果来自弥合生成和真实数据的域差距
- 生成质量可视化显示 DDDR 的生成图像远接近真实分布，而 Target/MFCL 生成的更像对抗样本

## 亮点与洞察

1. **预训练模型的巧妙复用**：不训练扩散模型，仅在输入空间搜索条件嵌入——计算量从"训练整个模型"降到"优化一个向量"，资源效率极高
2. **Class Embedding 作为类别压缩表示**：一个小向量就能代表一个类别的全部视觉信息，概念简洁有力
3. **隐私保护的自然性**：只传输 class embedding，不修改 LDM 参数，生成完全相同图像的概率极低
4. **对比学习的条件有效性**：实验发现对比学习仅在有当前任务生成数据时有效，这一洞察对后续工作有参考价值

## 局限与展望

- 依赖预训练的 LDM（LAION-400M），当目标域与 LDM 训练域差距大时效果可能退化
- 仅在 CIFAR-100 和 Tiny-ImageNet 上验证，缺乏更大规模或更多样化数据集的实验
- Class Inversion 需要额外的通信轮次（10轮），在通信受限场景下可能是瓶颈
- 未探讨异质任务（如从分类到检测）的持续学习场景
- Diffusion 模型的推理成本较高，每次训练新任务都需要生成大量回放样本

## 相关工作与启发

- **Textual Inversion / DreamBooth**：个性化生成模型的思路启发了 Class Inversion——在扩散模型的输入空间搜索而非微调模型
- **FedCIL / Target / MFCL**：FCCL 的前沿工作，分别使用 GAN 和数据无关蒸馏进行回放，但生成质量有限
- **启发**：预训练大模型（如扩散模型、大语言模型）可以作为持续学习中的"知识银行"，通过轻量级反演即可提取和回放知识

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次将预训练扩散模型引入 FCCL，Class Inversion 的思路巧妙且实用
- **实验充分度**: ⭐⭐⭐⭐ — 两个数据集 × 4种设置（IID/non-IID × T=5/10）全面覆盖，消融实验深入细致
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法描述充分，消融分析有洞察力
- **价值**: ⭐⭐⭐⭐ — CIFAR-100 上绝对提升 8-12%，Tiny-ImageNet 上近 10%，显著推进了 FCCL 的 SOTA

<!-- RELATED:START -->

## 相关论文

- [Mutual Learning for Acoustic Matching and Dereverberation via Visual Scene-driven Diffusion](mutual_learning_for_acoustic_matching_and_dereverberation_via_visual_scene-drive.md)
- [Autoencoder-Based Hybrid Replay for Class-Incremental Learning](../../ICML2025/image_generation/autoencoder-based_hybrid_replay_for_class-incremental_learning.md)
- [Toward Tiny and High-quality Facial Makeup with Data Amplify Learning](toward_tiny_and_high-quality_facial_makeup_with_data_amplify_learning.md)
- [Idempotent Unsupervised Representation Learning for Skeleton-Based Action Recognition](idempotent_unsupervised_representation_learning_for_skeleton-based_action_recogn.md)
- [Prompting Future Driven Diffusion Model for Hand Motion Prediction](prompting_future_driven_diffusion_model_for_hand_motion_prediction.md)

<!-- RELATED:END -->
