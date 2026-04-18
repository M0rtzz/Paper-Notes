---
title: >-
  [论文解读] DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning
description: >-
  [NeurIPS 2025][AI安全][同态加密] 提出 DESIGN 框架，在全同态加密(FHE)下通过服务器端输入图剪枝和自适应多项式激活度分配两阶段优化，相比 SEAL 基线加速 FHE GNN 推理约 2× 并维持有竞争力的准确率。
tags:
  - NeurIPS 2025
  - AI安全
  - 同态加密
  - 图神经网络
  - 图剪枝
  - 自适应多项式激活
  - 隐私保护
---

# DESIGN: Encrypted GNN Inference via Server-Side Input Graph Pruning

**会议**: NeurIPS 2025  
**arXiv**: [2507.05649](https://arxiv.org/abs/2507.05649)  
**代码**: https://github.com/LabRAI/DESIGN  
**领域**: ai_safety  
**关键词**: 同态加密, GNN推理, 图剪枝, 自适应多项式激活, 隐私保护

## 一句话总结
提出 DESIGN 框架，在全同态加密(FHE)下通过服务器端输入图剪枝和自适应多项式激活度分配两阶段优化，相比 SEAL 基线加速 FHE GNN 推理约 2× 并维持有竞争力的准确率。

## 研究背景与动机

**领域现状**：GNN 在推荐系统、生物信息、社交网络等领域表现出色，但处理敏感图数据（用户画像、金融交易、社交关系）时面临严重隐私风险。FHE 允许在加密数据上直接计算，是理想的隐私保护方案。

**现有痛点**：FHE 下的基本运算（尤其是同态乘法）比明文慢数个数量级。现有 FHE GNN 方法（CryptoGCN、LinGCN、Penguin）主要优化模型内部结构（如稀疏邻接矩阵、简化激活函数），但忽视了输入图数据本身的冗余性——不是所有节点和边都同等重要。

**核心矛盾**：三个基本挑战——(a) 在加密域中无法直接访问明文图指标做在线剪枝；(b) 现有方法对所有节点统一使用相同阶多项式激活，高阶节点拉高整体成本而低阶节点浪费精度；(c) 优化方案需架构无关可通用。

**本文目标** 在 FHE 约束下实现服务器端的输入图自适应处理：(a) 对不重要节点/边进行逻辑剪枝；(b) 对不同重要性节点使用不同阶多项式激活。

**切入角度**：利用节点度数作为 FHE 友好的重要性指标（仅需加法运算），通过近似同态比较生成多级重要性掩码，进而指导剪枝和自适应激活。

**核心 idea**：在加密图上用节点度数计算重要性分数 → 同态比较生成多级掩码 → 掩码驱动图剪枝 + 自适应多项式激活度分配。

## 方法详解

### 整体框架
客户端加密图 $\tilde{\mathcal{G}} = (\text{Enc}(\mathcal{V}), \text{Enc}(\mathcal{E}), \text{Enc}(\mathbf{X}))$ 上传给服务器。服务器执行两阶段处理：**Stage 1** FHE 兼容的重要性评分与分区 → **Stage 2** 带剪枝和自适应激活的 FHE GNN 推理。最终返回加密结果 $\tilde{\mathbf{Y}}$ 给客户端解密。

### 关键设计

1. **加密节点度数计算**:

    - 功能：在加密邻接矩阵上计算每个节点的度数作为重要性分数
    - 核心思路：$\tilde{s}_v = \bigoplus \tilde{A}_{vu}$，仅依赖同态加法 HE.Add（和可能的 HE.Rotate），完全避免昂贵的同态乘法
    - 设计动机：特征范数等更丰富的指标需要 HE.Mult，其开销可能抵消后续剪枝收益。节点度数是 FHE 效率与信息量的最优平衡点

2. **同态分区与掩码生成**:

    - 功能：将节点分为 $m+1$ 级——Level 0 待剪枝，Level 1~$m$ 保留并分配不同精度
    - 核心思路：使用 $m$ 个预定义阈值 $\tau_1 > \tau_2 > \cdots > \tau_m$，通过近似同态比较算子 HE.AprxCmp 生成二值掩码
    - 剪枝掩码：$\tilde{M}_{0,v} \approx \tilde{\mathbf{1}}_v \ominus \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_m)$
    - 层级掩码：$\tilde{M}_{i,v} \approx \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_i) \odot (\tilde{\mathbf{1}}_v \ominus \texttt{HE.AprxCmp}(\tilde{s}_v, \tilde{\tau}_{i-1}))$
    - 代价：HE.AprxCmp 需要评估高阶比较多项式，是计算最密集的阶段，但其一次性开销可在后续推理中摊销

3. **同态图剪枝**:

    - 功能：用 keep mask $(1 - M_0)$ 逻辑清零不重要节点的特征和关联边
    - 核心思路：$\tilde{\mathbf{X}}'_v = (\tilde{\mathbf{1}}_v \ominus \tilde{M}_{0,v}) \odot \tilde{\mathbf{X}}_v$，$\tilde{A}'_{uv} = (\tilde{\mathbf{1}}_u \ominus \tilde{M}_{0,u}) \odot (\tilde{\mathbf{1}}_v \ominus \tilde{M}_{0,v}) \odot \tilde{A}_{uv}$
    - 仅在推理开始时增加常数乘法深度，后续所有层级在缩减后的图上计算

4. **自适应多项式激活**:

    - 功能：根据节点重要性级别分配不同阶数的多项式近似激活函数
    - 核心思路：第 $l$ 层激活输出为 $\tilde{H}_v^{(l+1)} = \bigoplus_{i=1}^{m}(\tilde{M}_{i,v} \odot \texttt{HE.PolyEval}(P_{d_i}, \tilde{Z}_v^{(l+1)}))$
    - 高重要性节点用高阶多项式（如 7 阶）保持精度，低重要性节点用低阶（如 2 阶）降低开销
    - 效率来源：$P_{d_i}$ 的评估需 $O(d_i)$ 次同态乘法，低阶多项式大幅减少平均计算量

### 层级 GNN 计算
标准 GNN 层的同态计算：$\tilde{\mathbf{Z}}^{(l+1)} = (\tilde{\mathbf{A}}' \otimes \tilde{\mathbf{H}}^{(l)} \otimes \mathbf{W}_1^{(l)}) \oplus (\tilde{\mathbf{H}}^{(l)} \otimes \mathbf{W}_2^{(l)}) \oplus \mathbf{b}^{(l)}$，其中权重为明文。

## 实验关键数据

### 主实验（FHE GNN 推理延迟，单位：秒）

| 方法类别 | 方法 | Cora | Citeseer | PubMed | Yelp | ogbn-proteins |
|---|---|---|---|---|---|---|
| 基础 FHE | SEAL | 1656.6 | 4207.6 | 528.9 | 321.1 | 14.2 |
| 基础 FHE | OpenFHE | 813.7 | 3659.4 | 244.5 | 158.4 | 7.7 |
| 优化 FHE | CryptoGCN | 1035.8 | 2017.1 | 284.3 | 169.9 | 10.4 |
| 优化 FHE | LinGCN | 951.1 | 1850.4 | 260.8 | 155.9 | 9.5 |
| 优化 FHE | Penguin | 892.7 | 1928.1 | 249.7 | 162.3 | 9.1 |
| **本文** | **DESIGN** | **806.1** | **1760.0** | **239.3** | **146.1** | **8.5** |

### 准确率（节点分类，%）

| 方法 | Cora | Citeseer | PubMed | Yelp | ogbn-proteins |
|---|---|---|---|---|---|
| GNN (明文) | 84.0 | 60.0 | 80.0 | 56.8 | 59.2 |
| LinGCN | 72.5 | 52.1 | 58.3 | 48.2 | 58.2 |
| Penguin | 76.5 | 44.6 | 61.5 | 49.5 | 58.1 |
| **DESIGN** | 74.0 | 45.0 | 55.0 | **51.3** | 56.1 |

### 消融实验（Cora 数据集）

| 配置 | 延迟(s) | 准确率(%) |
|---|---|---|
| Baseline FHE GNN (BFG) | ~1650 | ~68 |
| Pruning Only (PO) | ~1000 | ~72 |
| Adaptive Activation Only (AAO) | ~1200 | ~73 |
| **Full Framework (FF)** | **~806** | **74** |

### 关键发现
- 剪枝和自适应激活各自独立贡献显著，组合后效果叠加
- 剪枝比例 40% 时在多数据集上取得延迟-精度最优平衡
- 多项式阶数集合 PSet2(5,3,2) 在中等保真度下最优：比 PSet1(7,5,3) 快 30%+ 而精度仅降 2-3%
- DESIGN 在 Yelp 和 ogbn-proteins 上准确率甚至优于部分 FHE baseline，说明低度节点使用低阶多项式反而避免了高阶近似带来的噪声累积

## 亮点与洞察
- **节点度数作为 FHE 友好重要性指标**的选择非常实用——只需加法运算，完全避免了乘法的噪声积累和深度消耗
- **双重优化策略**（剪枝减数据量 + 自适应激活减计算复杂度）相互正交，提供了灵活的效率-精度旋钮
- 将图剪枝思想从明文域迁移到加密域，是一个有前景的研究方向
- 框架设计与 GNN 架构无关，可即插即用到 GCN、GraphSAGE 等不同变体

## 局限与展望
- 同态比较阶段（HE.AprxCmp）本身计算密集，对大图可能抵消部分剪枝收益
- 仅用节点度数衡量重要性，可能不能充分捕捉任务相关的节点重要性
- 部分数据集上准确率低于 Penguin（如 Citeseer 45% vs 44.6%），说明激进剪枝对某些图结构有损
- PubMed 上准确率方差较大（±12.5%），稳定性需改善
- 未探索与模型压缩技术（如知识蒸馏到 FHE 友好架构）的结合

## 相关工作与启发
- **vs CryptoGCN**: 仅优化模型内部（稀疏邻接 + 简化激活），不处理输入冗余；DESIGN 同时优化数据和计算
- **vs LinGCN/Penguin**: 更好的多项式近似策略，但仍为统一阶数；DESIGN 根据节点重要性差异化分配
- **vs 明文图剪枝**: 明文方法可直接访问特征做精细剪枝，加密域受限于 FHE 可计算性，需使用度数等"廉价"指标

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将输入图剪枝和自适应激活组合应用于 FHE GNN，但核心组件较直觉
- 实验充分度: ⭐⭐⭐⭐ 五个数据集、多种 baseline、完整消融和灵敏度分析
- 写作质量: ⭐⭐⭐ 符号体系完整但偏重，主文论述有些冗长
- 价值: ⭐⭐⭐⭐ 实用性强，为 FHE GNN 推理效率优化开辟了输入端优化方向
