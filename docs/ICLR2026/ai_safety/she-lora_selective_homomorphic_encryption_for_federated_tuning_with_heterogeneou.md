---
title: >-
  [论文解读] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA
description: >-
  [ICLR 2026][AI安全][联邦学习] 提出SHE-LoRA——将选择性同态加密(SHE)与LoRA结合用于跨设备联邦LLM微调：基于参数敏感度的列级加密子集协商 + 列交换参数混淆 + 列感知自适应聚合，在保持与非隐私基线可比的模型性能同时，通信开销减少99.71%、加密时间减少99.87%，完全抵御SOTA梯度反演攻击DAGER。
tags:
  - ICLR 2026
  - AI安全
  - 联邦学习
  - 同态加密
  - LoRA
  - 隐私保护
  - 异构设备
---

# SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA

**会议**: ICLR 2026  
**arXiv**: [2505.21051](https://arxiv.org/abs/2505.21051)  
**代码**: [GitHub](https://github.com/liyan2015/SHE-LoRA)  
**领域**: AI安全/隐私保护  
**关键词**: 联邦学习, 同态加密, LoRA, 隐私保护, 异构设备

## 一句话总结
提出SHE-LoRA——将选择性同态加密(SHE)与LoRA结合用于跨设备联邦LLM微调：基于参数敏感度的列级加密子集协商 + 列交换参数混淆 + 列感知自适应聚合，在保持与非隐私基线可比的模型性能同时，通信开销减少99.71%、加密时间减少99.87%，完全抵御SOTA梯度反演攻击DAGER。

## 研究背景与动机

**领域现状**：联邦微调LLM需要在保持数据隐私的同时提升领域特定性能。LoRA因高效性成为联邦PEFT的主流选择，但研究表明传输的参数/梯度可被梯度反演攻击(DAGER)重建私有数据。

**现有痛点**：(1) DP在LoRA矩阵乘法中噪声被放大，损害模型性能；(2) MPC需复杂同步协议，不适合异构设备；(3) 现有SHE方法两个问题——LoRA矩阵乘法导致加密位置扩展，异构客户端加密子集合并导致密文膨胀。

**核心矛盾**：跨设备场景下客户端硬件能力、数据分布、加密预算各不相同。naive的FedAvg分别聚合A和B矩阵在数学上不等价于聚合BA乘积，且异构设备的不同加密位置在聚合时union扩大导致密文膨胀。

**切入角度**：(a) 只加密A矩阵(直接作用于用户数据更易泄露)；(b) 按列评估参数重要性(整列加密避免矩阵乘法导致的扩展)；(c) 服务器协商全局加密子集控制密文膨胀；(d) 列交换聚簇加密/非加密参数提高效率。

**核心 idea**：通过按列的参数敏感度评估+全局子集协商+列交换混淆+列感知聚合，在异构联邦LoRA中实现极低开销的强隐私保护。

## 方法详解

### 整体框架
四步工作流：(1) HE子集协商：客户端评估参数重要性与服务器协商全局加密子集 (2) 选择性加密：列交换+CKKS批加密 (3) 自适应聚合：分别聚合明文和密文 (4) 重参数化：解密+SVD+合并为匹配本地rank的新LoRA参数。

### 关键设计

1. **HE子集协商机制**：

    - 功能：让异构客户端协调出全局最优的加密列子集
    - 核心思路：每个客户端用Wanda方法评估各列敏感度 $S_j = \sum_k |W_{kj}| \cdot \|x_j\|_2$。用OPE加密敏感度排序后发送服务器。服务器维护Common列表(按频率)和Sensitivity列表(按敏感度)，协商每个客户端可承受的全局子集
    - 设计动机：按列加密而非按元素——矩阵乘法会将单元素加密扩展到整列。OPE保护敏感参数位置不被窥探。协商避免密文膨胀

2. **列交换参数混淆与选择性加密**：

    - 功能：对A矩阵列交换使加密列连续聚簇到右侧
    - 核心思路：加密部分用CKKS分块批加密。三重好处：(1) 批量加密减少开销；(2) 未加密列直接矩阵运算；(3) 列交换作为混淆增加攻击难度
    - 设计动机：散乱加密位置增加分块和加密开销，聚簇后利用CKKS向量化大幅提效

3. **列感知自适应聚合与重参数化**：

    - 功能：处理异构客户端不同加密列数的聚合
    - 核心思路：明文部分：$\Delta W_i^{plain} = B_i A_i^{plain}$，按列加权平均，SVD分解后按各客户端rank切片。密文部分类似。客户端解密后拼接 $B_g = [B_p, B_c]$, $A_g = [A_p; A_c]$，再SVD调整到本地rank
    - 设计动机：先乘得到全秩更新再SVD分解保证了聚合LoRA乘积的数学正确性。证明不丢失有意义的模型更新

### 损失函数 / 训练策略
- 50个客户端，200轮联邦训练，Dirichlet alpha=0.3的Non-IID划分
- 4种设备类型：rank 8-32，加密预算0.125%-1.6%
- HE实现：TenSEAL CKKS，多项式度8192

## 实验关键数据

### 主实验：隐私攻击防御 (DAGER攻击, SST2数据集)

| 方法 | B=4 R-1 | B=8 R-1 | B=16 R-1 |
|------|---------|---------|----------|
| Flex-LoRA(无保护) | 95.18 | 61.14 | 10.27 |
| Flex-LoRA-DP | 86.25 | 80.28 | 68.62 |
| MaskCrypt(同等HE开销) | 89.16 | 61.49 | 10.91 |
| **SHE-LoRA** | **0.72** | **0.98** | **0.0** |

### 消融：效率对比 (OpenLLaMA-3B)

| 指标 | 全加密Baseline | MaskCrypt | SHE-LoRA |
|------|---------------|----------|---------|
| 加密时间 | ~480s | ~50s | ~0.6s |
| 通信开销 | 最高 | 中等 | 最低(-99.71%) |
| 时间波动 | [311s,653s] | [1.6s,105s] | 几乎无波动 |

### 关键发现
- **极低加密即可完全防御**: 仅加密0.125%参数即可使DAGER完全失败(R-1=0)
- **列交换是安全关键**: 扰动梯度正交补在LoRA低秩空间的结构，导致DAGER的span check失败
- **模型性能无损**: 在GLUE/MMLU上与非隐私SOTA性能可比
- **互信息验证**: Max策略(优先加密最重要参数)的互信息下降速率远快于Min/Random
- **MaskCrypt需100x开销**: 才能匹配SHE-LoRA的安全水平

## 亮点与洞察
- **按列加密**精准命中了LoRA矩阵乘法导致加密扩展的根源——整列加密既高效又避免扩展
- **协商机制**巧妙平衡异构设备的隐私需求和加密能力，避免密文膨胀
- **列交换双重作用**：工程优化(批加密效率) + 安全增强(参数混淆)，一举两得
- **安全保证有理论支撑**(Bayesian CRLB)——选择性加密最敏感参数最大化重建误差下界

## 局限与展望
- 假设semi-honest adversary，未处理恶意行为(投毒/后门)
- 多方HE场景需multi-key HE扩展
- OPE暴露顺序信息，可以用更强的ORE/MPC替代
- 更复杂的生成任务待探索

## 相关工作与启发
- **vs Flex-LoRA**: 性能最好的异构联邦LoRA但无隐私保护。SHE-LoRA在保持可比性能时提供强隐私
- **vs MaskCrypt**: SOTA SHE联邦方法但不考虑LoRA矩阵乘法扩展和异构膨胀，需100x开销才能匹配SHE-LoRA安全水平
- **vs DP**: DP在LoRA中噪声被矩阵乘法放大，且DAGER攻击下防御效果远不如SHE

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将SHE与异构LoRA联邦微调结合，列级加密+协商设计创新
- 实验充分度: ⭐⭐⭐⭐⭐ 多模型多任务，安全性/效率/性能三维度全面
- 写作质量: ⭐⭐⭐⭐ 问题阐述清晰，motivation chain严密
- 价值: ⭐⭐⭐⭐ 对联邦LLM微调的实际部署有重要意义

<!-- RELATED:START -->

## 相关论文

- [Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement](../../ICCV2025/ai_safety/lora-fair_federated_lora_fine-tuning_with_aggregation_and_initialization_refinem.md)
- [Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/ai_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [FedALT: Federated Fine-Tuning through Adaptive Local Training with Rest-of-World LoRA](../../AAAI2026/ai_safety/fedalt_federated_fine-tuning_through_adaptive_local_training_with_rest-of-world_.md)
- [FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/ai_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)

<!-- RELATED:END -->
