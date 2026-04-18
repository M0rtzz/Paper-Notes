---
title: >-
  [论文解读] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge
description: >-
  [CVPR 2026][模型压缩][持续学习] 提出 CPS-Prompt 框架，通过任务感知的关键 patch 采样（CPS）和解耦 prompt-分类器训练（DPCT）两个模块，在边缘设备上实现 Prompt-based 持续学习的训练时内存和计算效率提升约 1.6 倍，同时准确率仅下降约 2%。
tags:
  - CVPR 2026
  - 模型压缩
  - 持续学习
  - 边缘设备
  - 提示学习
  - Token Reduction
  - 训练效率
---

# Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge

**会议**: CVPR 2026  
**arXiv**: [2604.07399](https://arxiv.org/abs/2604.07399)  
**代码**: [https://github.com/laymond1/cps-prompt](https://github.com/laymond1/cps-prompt) (有)  
**领域**: 模型压缩 / 持续学习  
**关键词**: 持续学习, 边缘设备, Prompt-based CL, Token Reduction, 训练效率

## 一句话总结
提出 CPS-Prompt 框架，通过任务感知的关键 patch 采样（CPS）和解耦 prompt-分类器训练（DPCT）两个模块，在边缘设备上实现 Prompt-based 持续学习的训练时内存和计算效率提升约 1.6 倍，同时准确率仅下降约 2%。

## 研究背景与动机

**领域现状**：持续学习（CL）在边缘设备（家用机器人、无人机、手机）上需要在有限内存和算力下不断适应新任务。Prompt-based 持续学习（PCL）通过冻结 ViT 骨干+轻量可学习 prompt 实现参数高效学习，但既有工作主要关注精度和推理效率。

**现有痛点**：PCL 方法如 C-Prompt 虽然精度高，但训练时内存开销巨大（4.3× 于本文方法），不适合部署在内存受限的边缘设备上。OS-Prompt 虽然简化了两阶段流水线，但反向传播时峰值内存仍然很高。

**核心矛盾**：现有 token reduction 方法（ToMe、PatchDropout）在与 PCL 结合时会丢弃任务关键 patch，导致精度严重下降——因为它们是"任务无关"的。

**本文要解决**：如何在 PCL 的两阶段架构中实现训练时内存和计算的显著节省，同时保持竞争力的精度？

**切入角度**：利用冻结 query encoder 最后一层的注意力和 value 信号来估计 patch 重要性，做任务感知的稀疏化；再通过解耦训练消除稀疏训练与全 patch 推理之间的表征错位。

**核心 idea**：任务感知的 patch 采样 + 解耦的 prompt/分类器训练 = 训练高效 + 精度保持。

## 方法详解

### 整体框架
CPS-Prompt 沿用 PCL 标准两阶段架构：(1) 冻结 query encoder $f_q$ 前向生成任务线索；(2) prompt-injected backbone $f_p$ 做分类。在两阶段之间插入 CPS 模块选择关键 patch，并用 DPCT 策略分两阶段训练 prompt 和分类器。

### 关键设计

1. **Critical Patch Sampling (CPS)**：

    - **功能**：从 query encoder 最后一层提取 class token 到 patch token 的注意力权重 $A^L_{\text{cls},j}$ 和 value 向量的 L2 范数 $\|V^L_j\|_2$，计算临界分数：
    $s_j = A^L_{\text{cls},j} \cdot \|V^L_j\|_2$
    - **核心思路**：注意力权重反映 patch 对类别表征的贡献强度，value 范数反映特征的显著性。两者乘积构成综合重要性分数。经温度缩放 softmax 转为采样概率：
    $p_j = \frac{\exp(s_j/\tau)}{\sum_i \exp(s_i/\tau)}$
      然后以无替换多项式采样选 $k = \lfloor(1-r) \cdot N\rfloor$ 个 patch。
    - **设计动机**：利用冻结 backbone 的先验知识做"免训练"的任务感知稀疏化；随机采样引入多样性避免过拟合；温度 $\tau$ 控制确定性 vs 探索性的 trade-off。

2. **Decoupled Prompt and Classifier Training (DPCT)**：

    - **功能**：将 $E$ 个 epoch 分为两个阶段——前 $\lfloor \lambda \cdot E \rfloor$ 个 epoch 联合优化 prompt $\phi$ 和分类器 $\theta$（用稀疏 patch 输入）；后 $E - E_p$ 个 epoch 冻结 prompt，仅用全 patch 输入微调分类器。
    - **核心思路**：
    $\mathcal{L}_p = \mathcal{L}(f_p(\mathbf{X}_{\text{sampled}}; \theta, \phi), y)$
    $\mathcal{L}_{\text{cls}} = \mathcal{L}(f_p(\mathbf{X}_{\text{full}}; \theta, \phi), y), \quad \text{(}\phi \text{ frozen)}$
    - **设计动机**：稀疏 patch 训练导致 prompt 学到的表征与全 patch 推理时不匹配；分类器单独用全 patch 对齐可消除此错位。同时冻结 prompt 后，梯度不再回传到 prompt，减少计算开销。

3. **温度控制的随机采样 vs 确定性 Top-k**：

    - 实验证明随机采样优于确定性 Top-k，因为受控的随机性促进了训练中的探索多样性，有助于泛化到新任务。

### 损失函数 / 训练策略
- 使用标准交叉熵损失
- Prompt 阶段和分类器阶段各用 Adam 优化器
- 学习率 cosine decay，起始 0.001
- 最优超参：patch 削减率 $r=0.4$，温度 $\tau=0.1$

## 实验关键数据

### 主实验

| 数据集 | 指标 | CPS-Prompt | C-Prompt (SOTA) | CODA-Prompt | 差异说明 |
|--------|------|-----------|-----------------|-------------|---------|
| CIFAR-100 | ACC↑ | 66.89 | **68.34** | 67.06 | 仅差 1.45% |
| ImageNet-R | ACC↑ | 49.96 | **53.32** | 50.24 | 差 3.36% |
| CUB-200 | ACC↑ | 52.85 | 52.64 | **53.96** | 与 CODA 持平 |

效率比较（Jetson Orin Nano 上测量）：

| 方法 | 峰值内存倍率 | 训练时间倍率 | 能耗倍率 |
|------|------------|------------|---------|
| CPS-Prompt | **1×** | **1×** | **1×** |
| CODA-Prompt | ~1.6× | ~1.5× | ~1.6× |
| C-Prompt | ~4.3× | ~3.1× | ~3.3× |

### 消融实验

| 配置 | ACC↑(ImageNet-R) | 内存 | 训练时间 | 说明 |
|------|-----------------|------|---------|------|
| CODA-Prompt 基线 | 50.24 | 440MB | 1788s | 基线 |
| + PD (随机丢 patch) | 45.32 | 253MB | 1388s | 精度大幅下降 |
| + CPS (任务感知) | 47.16 | 253MB | 1389s | 比 PD 好 1.8% |
| + PD + DPCT | 47.96 | 253MB | 1126s | DPCT 恢复精度 |
| + CPS + DPCT (完整) | **49.28** | **253MB** | **1126s** | 最优配置 |

### 关键发现
- CPS 和 DPCT 提供互补收益：CPS 提升 patch 质量，DPCT 消除表征错位
- 即使内存削减超过 60%，CPS-Prompt 仍保持基线 90% 以上精度
- 随机采样在低 phase ratio 下表现尤其优于确定性 Top-k
- 温度 $\tau=0.1$（较尖锐分布）在所有数据集上表现最佳

## 亮点与洞察
- **真正的边缘部署视角**：在 Jetson Orin Nano 上做了完整的实测（内存、时间、能耗），而非仅仅报告理论 FLOPs
- **任务感知 token reduction**：巧妙利用 PCL 两阶段架构中本已存在的 query forward pass 信号，零额外训练开销
- **解耦训练的简洁性**：分两阶段分别用稀疏/全 patch 训练 prompt/分类器，设计简单但有效

## 局限与展望
- 仅在 ViT-Tiny/16 上验证，更大模型（ViT-Base/Large）上的表现未知
- 固定的 patch 削减比 $r=0.4$，未探索动态自适应策略
- 仅考虑 class-incremental 设定，未涉及 task-incremental 或 domain-incremental
- 未与更新的 VLM-based CL 方法对比

## 相关工作与启发
- 与 ToMe（token merge）和 PatchDropout 的对比表明，任务无关的 token reduction 在 PCL 中表现糟糕
- DPCT 的思路与知识蒸馏中的"训练-推理不一致"问题异曲同工
- 可以启发将 CPS 思路推广到其他需要 token reduction 的 ViT 下游任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 任务感知 patch 采样+解耦训练的组合新颖，但单独看每个模块技术贡献有限
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集+真实边缘硬件+完整消融+效率分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法流程图和伪代码完备
- 价值: ⭐⭐⭐⭐ 对边缘持续学习有实际意义，但整体 scope 偏小众
