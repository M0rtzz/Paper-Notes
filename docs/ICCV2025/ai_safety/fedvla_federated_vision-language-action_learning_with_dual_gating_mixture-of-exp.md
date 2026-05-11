---
title: >-
  [论文解读] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation
description: >-
  [ICCV 2025][AI安全][联邦学习] 本文提出 FedVLA——首个面向视觉-语言-动作（VLA）模型的联邦学习框架，通过指令导向场景解析（IOSP）增强任务感知特征提取、双门控混合专家（DGMoE）实现自适应知识路由、以及专家驱动聚合（EDA）策略确保跨客户端有效知识整合，在保护数据隐私的同时达到与集中式训练相当的任务成功率。
tags:
  - ICCV 2025
  - AI安全
  - 联邦学习
  - 视觉-语言-动作模型
  - 混合专家
  - 机器人操作
  - 隐私保护
---

# FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation

**会议**: ICCV 2025  
**arXiv**: [2508.02190](https://arxiv.org/abs/2508.02190)  
**代码**: 无  
**领域**: AI安全 / 隐私保护 / 机器人操作  
**关键词**: 联邦学习, 视觉-语言-动作模型, 混合专家, 机器人操作, 隐私保护

## 一句话总结

本文提出 FedVLA——首个面向视觉-语言-动作（VLA）模型的联邦学习框架，通过指令导向场景解析（IOSP）增强任务感知特征提取、双门控混合专家（DGMoE）实现自适应知识路由、以及专家驱动聚合（EDA）策略确保跨客户端有效知识整合，在保护数据隐私的同时达到与集中式训练相当的任务成功率。

## 研究背景与动机

**领域现状**：视觉-语言-动作（VLA）模型是机器人操作领域的重要进展，使机器人能够根据自然语言指令理解场景并执行任务。代表性工作包括 RT-2、OpenVLA 等，它们在大规模数据集上训练后展现出对新指令和未见物体的泛化能力。

**现有痛点**：训练 VLA 模型需要大量用户特定的室内场景数据（如家庭环境中机器人操作的视频），这些数据涉及用户隐私——包含个人生活空间、物品摆放、作息习惯等敏感信息。集中式训练需要上传所有数据到云端，存在隐私泄露风险，限制了 VLA 模型的广泛应用。

**核心矛盾**：隐私保护（数据不能共享）与模型性能（需要大量多样化数据）之间存在根本张力。现有联邦学习方法（如 FedAvg）主要针对单模态设计，简单平均聚合忽略了不同客户端任务的异质性——在 VLA 场景中，不同家庭的机器人执行完全不同的任务（开抽屉 vs 扫地），特征分布差异极大。

**本文目标**：设计首个面向 VLA 模型的联邦学习框架，在保护用户数据隐私的前提下，有效处理多客户端任务异质性，达到与集中式训练相当的机器人操作性能。

**切入角度**：作者观察到 VLA 任务的异质性可以通过混合专家（MoE）来处理——不同任务激活不同专家。进一步，专家的激活模式可以作为客户端任务相似性的信号，指导联邦聚合。

**核心 idea**：提出三个协同组件——IOSP 分解场景为对象级表征增强任务理解，DGMoE 让专家自主决定是否响应 token 实现高效路由，EDA 基于专家激活相似性做智能聚合。

## 方法详解

### 整体框架

FedVLA 采用客户端-服务器架构。每个客户端有本地 VLA 模型（Stem + Trunk + Head），处理本地任务数据。模型分为三部分：Stem（视觉编码 + IOSP）、Trunk（包含 DGMoE 的 Transformer 层）、Head（动作预测）。每轮训练中，客户端本地训练 5 个 epoch，然后将 Trunk 参数和专家激活统计发送到服务器；服务器执行 EDA 聚合后将全局 Trunk 分发回客户端。Stem 和 Head 保持个性化，不参与聚合。

### 关键设计

1. **Instruction-Oriented Scene-Parsing（IOSP，指令导向场景解析）**:

    - 功能：根据语言指令将场景图像分解为结构化的对象级表征，增强任务感知特征提取
    - 核心思路：给定指令和图像，首先用命名实体识别提取指令中的目标物体名称，同时用 YOLOv8 检测图像中的所有物体。然后用 CLIP 模型计算物体名称与指令的余弦相似度，将物体分为三类：目标物体（TOs，指令直接涉及的）、周围物体（SOs，前景但非目标的）、背景物体（BOs）。用 CLIP 视觉编码器将图像 token 分配给各组（每组选 top-8 最相关 token），通过 MoE 模块精化各组特征后拼接回主序列
    - 设计动机：在联邦场景中，不同客户端的图像背景和物体分布差异很大。IOSP 帮助模型聚焦于任务相关物体，过滤掉客户端特有的背景干扰，这对跨客户端泛化至关重要

2. **Dual Gating Mixture-of-Experts（DGMoE，双门控混合专家）**:

    - 功能：自适应地为不同 token 选择合适数量的专家，实现任务感知的知识路由
    - 核心思路：每个 DGMoE 层包含 $K$ 个专家和两个门控机制。Token 侧门控 $G_t$：由软路由器计算 token 对各专家的偏好分数，并通过残差连接融合前一层的分数（Eq.3），允许 token 继承前层的专家选择先验。专家侧门控 $G_e$：每个专家有可学习的阈值参数 $W_e$，只有当 token 分数超过 $\lambda W_e$（$\lambda=0.5$）时专家才被激活（Eq.4-5）。最终输出 $y = \sum_{i=1}^K g_i(x) E_i(x)$，其中 $g_i(x)$ 在专家被拒绝时为 0
    - 设计动机：传统 MoE 固定选择 top-k 个专家，不适应任务复杂度变化——简单任务可能只需 1 个专家，复杂任务需要多个。DGMoE 的双向选择机制（token 选专家 + 专家选 token）实现了动态稀疏激活，实验中平均每 token 仅激活约 1.22 个专家，大幅节省计算

3. **Expert-Driven Aggregation（EDA，专家驱动聚合）**:

    - 功能：在联邦服务器上根据专家激活相似性智能聚合客户端模型
    - 核心思路：每个客户端记录专家激活矩阵 $\mathbf{V}_i \in \mathbb{R}^{L \times K}$（$L$ 层 $K$ 专家，记录每个专家被激活次数）。对于第 $l$ 层，计算客户端 $C_i$ 和 $C_j$ 的专家选择向量余弦相似度 $s_{i,j}^{(l)}$（Eq.8），然后归一化得到聚合权重 $w_{l,i}$（Eq.9）。相似度高的客户端对彼此贡献更大权重
    - 设计动机：FedAvg 简单平均聚合在任务异质场景下会相互抵消不同任务的知识。EDA 让执行相似任务（激活相似专家模式）的客户端更多地互相学习，保持了任务特异性

### 损失函数 / 训练策略

客户端使用 Huber 损失优化动作预测。基于 HPT（Heterogeneous Pre-trained Transformers）骨干，仿真中学习率 $5 \times 10^{-6}$，真实世界 $2 \times 10^{-5}$。Batch size 256，Adam 优化器。共 1000 轮联邦通信，每轮本地训练 5 个 epoch。

## 实验关键数据

### 主实验

仿真实验（MuJoCo + Meta-World，Sawyer 机器人，4 个任务）：

| 方法 | 平均成功率 | Door Lock | Close Drawer | Sweep Into | Open Window |
|------|----------|-----------|-------------|-----------|-------------|
| Centralized | 65.0% | 86.7% | 73.3% | 53.3% | 46.7% |
| FedAvg | 51.7% | 66.7% | 73.3% | 40.0% | 26.7% |
| FedVLA | **63.3%** | 80.0% | **80.0%** | 53.3% | 40.0% |

真实世界实验（UR3 机器人，4 个任务）：

| 方法 | 平均成功率 | Clean Up | Trash Collect | Open Drawer | Sorting Pills |
|------|----------|----------|-------------|------------|--------------|
| Centralized | 63.4% | 46.7% | 46.7% | 86.7% | 73.3% |
| FedAvg | 53.3% | 46.7% | 40.0% | 60.0% | 66.7% |
| FedVLA | **63.3%** | 53.3% | 46.7% | 80.0% | **73.3%** |

### 消融实验

| 配置 | 平均成功率 | Clean Up | Trash Collect | Open Drawer | Sorting Pills |
|------|----------|----------|-------------|------------|--------------|
| w/o IOSP | 41.1% | 40.0% | 13.3% | 66.7% | 46.7% |
| w/o DGMoE | 31.7% | 20.0% | 20.0% | 46.7% | 40.0% |
| w/o EDA | 26.7% | 26.7% | 20.0% | 33.3% | 26.7% |
| FedVLA (完整) | **63.3%** | 53.3% | 46.7% | 80.0% | 73.3% |

### 关键发现

- **FedVLA 几乎追平集中式训练**：平均成功率 63.3% vs 63.4%（仿真也是 63.3% vs 65.0%），说明联邦框架确实能在不共享数据的情况下达到集中式水平
- **EDA 是最关键组件**：去掉 EDA 后成功率从 63.3% 暴降至 26.7%（下降 57.8%），因为简单的 FedAvg 聚合会抵消 DGMoE 中不同专家的知识
- **IOSP 在多物体场景中至关重要**：Trash Collection 任务去掉 IOSP 后从 46.7% 降至 13.3%，因为该任务物体数量最多
- **DGMoE 实现高效稀疏激活**：平均每 token 仅激活 ~1.22 个专家（vs vanilla top-k 的固定 k 个），大幅节省计算。且不同物体类型（目标 vs 背景）激活不同专家，证明了任务感知路由的有效性

## 亮点与洞察

- **首个联邦 VLA 框架**：开创了"隐私保护机器人学习"这一交叉方向。随着家用机器人普及，VLA 模型的隐私问题将越来越重要，FedVLA 提供了可行的解决方案
- **双向专家选择机制的巧妙设计**：传统 MoE 只有 token→expert 的单向选择，DGMoE 加入 expert→token 的反向选择（自感知专家），实现"双向匹配"，既提升性能又提高效率。这种设计可以迁移到其他需要高效 MoE 的场景
- **专家激活作为客户端相似度信号**：利用 DGMoE 的专家激活模式来衡量客户端任务相似性，比直接比较模型参数更有语义意义。这种"行为相似性"指导聚合的思路很巧妙

## 局限与展望

- 实验规模有限：只有 4 个客户端/任务，每个任务仅 30-80 条轨迹。大规模异质场景（数十个客户端、更复杂任务）下的表现未知
- 仅聚合 Trunk 模块——Stem 和 Head 完全个性化，可能限制了跨客户端的知识共享
- 未考虑通信效率和安全攻击（如模型反推数据的攻击），这些是联邦学习中的重要问题
- DGMoE 的自感知阈值 $\lambda=0.5$ 是手动设置的，可能不是最优值
- 未来可以扩展到更多客户端、更复杂的操作任务，以及探索差分隐私等更强的隐私保护

## 相关工作与启发

- **vs FedAvg**: FedAvg 简单平均聚合，对任务异质性无感。FedVLA 通过 EDA 实现任务感知聚合，真实世界成功率 63.3% vs 53.3%
- **vs OpenVLA/RT-2**: 这些集中式 VLA 模型需要汇聚所有数据训练，存在隐私风险。FedVLA 证明了分布式训练不牺牲性能的可行性
- **vs FedHCA2**: FedHCA2 是联邦多任务学习方法，但针对单模态。FedVLA 面向多模态（视觉+语言+动作）的机器人场景，挑战更大
- **vs DeepSeekMoE/LLaVA-MoE**: 这些工作将 MoE 用于大模型，但使用固定 top-k 选择。DGMoE 的自感知专家实现了动态稀疏激活，更灵活

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个联邦 VLA 框架，DGMoE 和 EDA 设计新颖，但各组件（联邦学习、MoE、场景解析）单独看并不新
- 实验充分度: ⭐⭐⭐ 仿真 + 真实世界验证，消融完整，但规模较小（4 任务/客户端），对比方法少（只有 FedAvg）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，公式和算法伪代码完整
- 价值: ⭐⭐⭐⭐ 开创联邦 VLA 方向，隐私保护+机器人学习的交叉前沿，但需要更大规模验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[CVPR 2026\] When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/ai_safety/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICML 2025\] Privacy-Shielded Image Compression: Defending Against Exploitation from Vision-Language Pretrained Models](../../ICML2025/ai_safety/privacy-shielded_image_compression_defending_against_exploitation_from_vision-la.md)

</div>

<!-- RELATED:END -->
