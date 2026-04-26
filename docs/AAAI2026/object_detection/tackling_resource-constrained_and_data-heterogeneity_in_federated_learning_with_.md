---
title: >-
  [论文解读] Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack
description: >-
  [AAAI 2026][目标检测][联邦学习] 提出FedCSPACK，一种基于余弦稀疏化参数打包和双权重聚合的个性化联邦学习方法，通过在包级别进行参数选择和共享，同时平衡了数据异质性和客户端资源约束，训练速度提升2-5倍、通信量压缩高达96%，同时模型精度提升3.34%。
tags:
  - AAAI 2026
  - 目标检测
  - 联邦学习
  - 数据异质性
  - 资源受限
  - 稀疏通信
  - 个性化联邦学习
---

# Tackling Resource-Constrained and Data-Heterogeneity in Federated Learning with Double-Weight Sparse Pack

**会议**: AAAI 2026  
**arXiv**: [2601.01840](https://arxiv.org/abs/2601.01840)  
**代码**: [https://github.com/NigeloYang/FedCSPACK](https://github.com/NigeloYang/FedCSPACK)  
**领域**: 目标检测  
**关键词**: 联邦学习, 数据异质性, 资源受限, 稀疏通信, 个性化联邦学习

## 一句话总结

提出FedCSPACK，一种基于余弦稀疏化参数打包和双权重聚合的个性化联邦学习方法，通过在包级别进行参数选择和共享，同时平衡了数据异质性和客户端资源约束，训练速度提升2-5倍、通信量压缩高达96%，同时模型精度提升3.34%。

## 研究背景与动机

### 联邦学习面临的双重挑战

联邦学习（FL）需要在不交换原始数据的情况下协同训练模型，但面临两个交织的核心挑战：

**挑战1：数据异质性（Non-IID）**
- 不同地理位置的边缘设备生成的数据分布差异显著
- 单一全局模型难以适应所有客户端的本地数据
- 导致收敛缓慢、推理性能差

**挑战2：系统资源异质性**
- 客户端设备的处理器、内存、带宽各不相同
- 资源受限的客户端难以跟上复杂全局模型的协同训练
- 造成通信瓶颈、计算延迟和参与不平衡

### 现有方法的不足

现有方法通常只解决其中一个问题：

- **应对数据异质性**：FedProx（L2正则化）、MOON（对比学习）、FedNTD（知识蒸馏）等——忽略了资源约束
- **应对资源受限**：FedSPU（参数稀疏化）、模型分割方法——缺乏对数据异质性的考虑

**关键洞察**：在现实世界中，数据异质性和资源约束不是孤立的，而是协同作用的核心挑战。需要同时解决两者。

## 方法详解

### 整体框架

FedCSPACK的工作流程分为4个步骤：

1. **服务器**：聚合全局模型 $W^t$ 和全局掩码 $M^t$，广播给所有客户端
2. **客户端训练**：在本地数据上更新模型
3. **参数打包与选择**：将模型参数展平后打包，用余弦相似度选择Top-K个最有贡献的包共享，同时生成带双权重的掩码
4. **服务器聚合**：利用掩码的双权重完成加权聚合

### 关键设计

#### 1. **基于Top-K余弦的参数打包（Cosine Parameter Packing）**：解决通信瓶颈

**问题**：频繁的全模型传输对资源受限客户端造成巨大通信负担。传统Top-K稀疏化方法难以在动态训练中选择合适的参数子集。

**解决方案**：在"包"级别进行参数选择，而非参数级别。

**具体流程**：
1. 将本地模型和全局模型展平为一维向量 $FW_i^t, FW^t$
2. 计算整体余弦相似度阈值：

$$\theta_a^t = \text{CosSim}(FW_i^t, FW^t) = \frac{FW_i^t \cdot FW^t}{\|FW_i^t\| \|FW^t\|}$$

3. 按 $PACK$ 大小将展平向量分割为参数包 $PW_{i,j}^t$
4. 计算每个包的相似度 $\theta_{i,j}^t$
5. 使用Top-K选择满足 $\theta_{i,j}^t < \theta_a^t$ 的K个包作为共享参数包

**设计动机**：
- 余弦相似度低意味着推理损失高，改善这些参数包可能带来更好的模型性能
- 包级别操作比参数级别更高效
- 未被选中的参数包作为客户端的独特本地知识保留

#### 2. **掩码矩阵与双权重聚合（Mask Double Weight Aggregation）**：缓解非对齐聚合问题

**问题**：稀疏参数包可能导致服务器聚合错位或错误，降低全局模型性能。同时，仅靠余弦相似度无法衡量参数值的距离差异和幅度偏移。

**解决方案**：双权重掩码 = 方向权重（余弦相似度）+ 分布距离权重（KL散度）

**方向权重**：由余弦相似度 $\theta_{i,k}^t$ 提供，反映参数更新方向的一致性。

**分布距离权重**：使用KL散度计算参数包之间的分布距离：

$$\beta_{i,j}^t = \sum PW_{i,j}^t \log \frac{PW_{i,j}^t}{PW_j^t}$$

**最终掩码**：

$$M_{i,j}^t = \begin{cases} \theta_{i,k}^t + \beta_{i,k}^t & \text{共享包位置} \\ 0 & \text{非共享位置} \end{cases}$$

**聚合公式**：

$$PW_{i,j}^t = \begin{cases} \sum_{i=1}^{S_t} \frac{M_{i,j}^t}{M_j^{t+1}} PW_{i,k}^t & M_{i,j}^t \neq 0 \\ 0 & \text{否则} \end{cases}$$

**设计动机**：
- KL散度补充了余弦相似度不能反映的幅度信息
- 双权重使服务器能更全面地评估每个参数包的贡献
- 保留方向对齐优势的同时有效缓解分布差异的影响

#### 3. **个性化知识保留**：未共享参数包中的本地特色

客户端只共享K个参数包，其余 $PW_{i,j \setminus k}^t$ 保留为客户端独有特征，避免其他客户端异质性的干扰。当客户端接收新全局模型时，只有被掩码标记为有效的位置才会被全局模型更新，保留了本地个性化知识。

### 损失函数 / 训练策略

- 本地训练使用标准交叉熵损失
- SGD优化器用于本地模型更新：$W_i^t \leftarrow W_i^t - \eta \nabla f_i(W_i^t)$
- 每轮从N个客户端中随机采样子集 $S^t$ 参与训练
- 全局epoch T, 本地epoch E

## 实验关键数据

### 主实验

**四个数据集上的Top-1准确率（Table 1，Dirichlet采样）**：

| 方法 | FMNIST Dir(0.3) | CIFAR-10 Dir(0.3) | CIFAR-100 Dir(0.3) | EMNIST Dir(0.6) |
|------|----------------|-------------------|-------------------|----------------|
| FedAvg | 84.39 | 69.71 | 39.15 | 84.03 |
| FedProx | 84.39 | 69.58 | 38.48 | 84.12 |
| MOON | 85.44 | 70.03 | 38.43 | 84.08 |
| FedNTD | 84.35 | 70.32 | 39.44 | 84.49 |
| FedSPU | 85.29 | 67.38 | 37.81 | 84.22 |
| **FedCSPACK** | **88.13** | **73.23** | **41.60** | **86.26** |

在CIFAR-10 Dir(1.0)上达到78.71%，CIFAR-100 Dir(1.0)上达到43.20%，均为最优。

**资源消耗（Table 2，T=100, Dir(0.3)）**：

| 数据集/模型 | 指标 | FedAvg | FedSPU | FedCSPACK |
|------------|------|--------|--------|-----------|
| EMNIST/CNN | 通信量(GB) | 18.18 | 4.29 | **0.73** |
| EMNIST/CNN | 时间(h) | 11.16 | 13.92 | **11.32** |
| CIFAR-100/ResNet18 | 通信量(GB) | 251.00 | 49.22 | **9.24** |
| CIFAR-100/ResNet18 | 时间(h) | 0.81 | 1.01 | **0.88** |

通信量压缩比：EMNIST上96.0%（18.18GB → 0.73GB），CIFAR-100上27倍压缩。

### 消融实验

**双权重影响（Table 3，CIFAR-10 Dir(0.5)）**：

| 权重类型 | Round 10 | Round 50 | Round 100 |
|---------|---------|---------|----------|
| 仅CS（余弦） | 0.33 | 0.65 | 0.74 |
| 仅KL | 0.30 | 0.68 | 0.69 |
| **双权重** | **0.33** | **0.71** | **0.79** |

**PACK大小影响**：PACK增大时模型性能基本不变，但训练时间逐渐减少，尤其在大数据量异质场景下改善更明显。

### 关键发现

1. **通信效率极高**：在EMNIST上实现96%的通信压缩比，CIFAR-100上实现27倍压缩
2. **精度与效率的最佳平衡**：在大幅减少通信量的同时，精度反而比其他方法更高
3. **双权重优于单权重**：方向权重+距离权重的组合比任何单一权重更有效
4. **对低参与率场景鲁棒**：即使客户端参与率很低，FedCSPACK仍保持稳定性能
5. **全局模型泛化性优异**：在最差客户端（Client 3）上泛化性能比最佳SOTA高20%

## 亮点与洞察

1. **首个包级别的个性化联邦学习**：将参数打包提升为一等公民，而非简单的参数级别稀疏化
2. **同时解决双重异质性**：在单一框架内平衡数据异质性和资源约束，填补了现有方法的空白
3. **PACK大小的鲁棒性**：模型性能对PACK大小不敏感，给工程部署提供了灵活性
4. **KL散度作为分布距离权重的引入**：补充了余弦相似度无法捕获的幅度信息
5. **训练时间几乎无增加**：尽管引入了额外的打包和权重计算，训练时间基本不变

## 局限与展望

1. **仅在图像分类上验证**：未涉及目标检测、语义分割等更复杂的视觉任务
2. **模型架构限制**：仅使用CNN和ResNet-18，对大型模型（如ViT）的表现未知
3. **PACK大小的自适应选择**：当前PACK大小是固定的，动态调整可能进一步优化
4. **参数包的KL散度假设**：将参数值视为分布计算KL散度的合理性需要更多理论分析
5. **安全性考虑不足**：在恶意客户端存在时的鲁棒性未讨论

## 相关工作与启发

- **FedAvg**（McMahan et al. 2017）：标准联邦平均基线
- **FedProx**（Li et al. 2020）：通过L2正则化约束本地模型与全局模型的偏移
- **FedSPU**（Niu et al. 2025）：基于Top-K的模型稀疏化方法，但只考虑通信效率
- **FedNTD**（Lee et al. 2022）：通过知识蒸馏缓解灾难性遗忘

**启发**：在分布式系统中，"选择性共享"往往比"全量共享"更高效。包级别的粒度控制比参数级别更具工程可行性，且通过双权重聚合可以有效弥补信息损失。

## 评分

- 新颖性: ⭐⭐⭐⭐（包级别操作和双权重是有意义的创新，但核心技术组件相对常规）
- 实验充分度: ⭐⭐⭐⭐（4数据集、10个对比方法、多种异质性设置、消融全面）
- 写作质量: ⭐⭐⭐（整体结构清晰，但公式符号有时前后不一致）
- 价值: ⭐⭐⭐⭐（对联邦学习在资源受限场景下的部署有实际意义）

<!-- RELATED:START -->

## 相关论文

- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](../../ICLR2026/object_detection/sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[AAAI 2026\] TRACE: A Generalizable Drift Detector for Streaming Data-Driven Optimization](trace_a_generalizable_drift_detector_for_streaming_data-driven_optimization.md)
- [\[ICLR 2026\] Long-Context Generalization with Sparse Attention](../../ICLR2026/object_detection/long-context_generalization_with_sparse_attention.md)

<!-- RELATED:END -->
