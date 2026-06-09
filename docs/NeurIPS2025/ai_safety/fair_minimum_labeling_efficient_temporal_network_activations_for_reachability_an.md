---
title: >-
  [论文解读] Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity
description: >-
  [NeurIPS 2025][AI安全][公平性] 本文提出公平最小标注（FML）问题，旨在设计最小代价的时序边激活方案，使网络中各节点组均有足够的时序路径可达性以满足公平覆盖要求；证明该问题是 NP-hard 且难以近似，并基于概率树嵌入给出匹配下界的近似算法。
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "公平性"
  - "时序图"
  - "最小标注"
  - "可达性"
  - "近似算法"
---

# Fair Minimum Labeling: Efficient Temporal Network Activations for Reachability and Equity

**会议**: NeurIPS 2025  
**arXiv**: [2510.03899](https://arxiv.org/abs/2510.03899)  
**代码**: [有](https://gitlab.com/tgdesign/fml)  
**领域**: AI安全  
**关键词**: 公平性, 时序图, 最小标注, 可达性, 近似算法

## 一句话总结

本文提出公平最小标注（FML）问题，旨在设计最小代价的时序边激活方案，使网络中各节点组均有足够的时序路径可达性以满足公平覆盖要求；证明该问题是 NP-hard 且难以近似，并基于概率树嵌入给出匹配下界的近似算法。

## 研究背景与动机

**领域现状**：现代联网系统（联邦学习、推荐系统、分布式传感器网络）中，资源有限且需要保证参与者或数据源之间的公平性。图结构描述潜在交互，而时序维度即连接何时激活决定了信息、更新和影响的实际流向。

**现有方法不足**：
   - 已有的公平性图挖掘方法（公平聚类、公平节点嵌入、公平影响力最大化）主要针对静态图，不涉及时序交互
   - 时序图连通性研究（Minimum Labeling 问题）关注全局连通性，但未考虑分组公平约束
   - 简单启发式可能过度消耗瓶颈资源，或系统性忽略难到达的节点和少数群体

**核心动机**：需要一种形式化框架，能同时优化时序边激活的资源效率和分组公平的可达性保障。FML 填补了公平感知图优化与时序连通性模型之间的空白。

## 方法详解

### 整体框架

**问题定义（FML）**：给定带颜色的静态图 G=(V,E,c)，颜色集 C，终端节点集 T，每种颜色 c 的需求函数 fc，目标是找到最小化总时间标签数的时序标注，使得每种颜色 c 中有足够多的节点能通过时序路径到达终端节点。

**近似框架（Algorithm 1）**：
1. 计算图 G 上的最短路径度量
2. 使用 FRT 算法生成概率树嵌入，采样加权树 T
3. 以终端 t 为根的树上求解 FML（精确或近似 DP）
4. 将树上解投影回原图 G

### 关键设计

**1. 概率树嵌入**

采用 Fakcharoenphol-Rao-Talwar (FRT) 算法：任意 n 点度量空间可嵌入到树度量分布上，期望拉伸因子 O(log n)，此界是紧的。核心性质：对所有 x,y 有 d(x,y) <= d_T(x,y)（支配性）和 E[d_T(x,y)] <= O(log n) * d(x,y)（期望拉伸）。将 FML 在一般图上的求解转化为在树上的 DP 求解，利用了树结构的简洁性。

**2. 精确树 DP 算法（Section 5.1）**

自底向上计算标签 (b, r, c)：b 和 r 分别为子树中蓝色和红色节点数，c 为连接这些节点的权重和。

- 叶子节点：蓝色节点标签 (1,0,0)，红色节点 (0,1,0)
- 内部节点 v 有 l 个子节点时，每个子节点最多选一个标签
- 每个 (b,r) 对只保留最小代价 c 的标签，上界 O(n^2) 标签每节点
- 总复杂度：O(n^5)

**3. 双准则近似算法（Section 5.2）**

通过几何分桶（geometric bucketing）降低标签数：

- 将 [0,n] 分为几何间隔，每个桶对只保留最小代价标签
- 标签数降至 O(eps^{-2} * log^2 n)
- 覆盖违约因子：xi = (1+eps)^{H+1}，H 为树高
- 近似保证：代价 c' <= c_opt，覆盖 b' >= b_opt/xi
- 复杂度：O(n^2 + n * eps^{-4} * log^4 n)

**4. 树解到图解的投影**

树上每条时序边投影为原图 G 中最短路径上的一系列时序边激活。严格递增时间戳保证时序路径有效性。投影保持覆盖约束（精确或近似因子 xi）。

### 损失函数 / 训练策略

FML 本身是组合优化问题，不涉及神经网络训练。在实验的多源学习任务中，使用 MLP 分类器（两个 32 神经元隐藏层，ReLU，sigmoid 输出），Adam 优化器（lr=0.01, batch=64, weight decay=0.01），二元交叉熵损失，100 epoch + early stopping。

**理论结果**：
- NP-hard 且 Omega(log n) 近似困难（即使只有单终端，通过从 Set Cover 归约证明）
- 精确算法给出 O(log n) 期望代价近似（匹配下界）
- 双准则算法在代价和公平性上均给出有界保证

## 实验关键数据

### 主实验（多源数据收集）

在随机几何图（|V|=1024, 连接半径 r=0.2）上的学习任务：

| 算法 | 时间(s) | 代价 | 覆盖B | 覆盖R | AccB | AccR |
|------|---------|------|-------|-------|------|------|
| Greedy | 0.49 | 88.3 | 64.0 | 0.0 | 1.00 | 0.27 |
| Closest | 0.73 | 224.0 | 44.3 | 32.0 | 0.97 | 0.94 |
| Alternating | 0.71 | 212.8 | 34.2 | 32.0 | 0.96 | 0.94 |
| FMLapprox | 69.20 | 74.3 | 33.5 | 32.3 | 0.98 | 0.93 |
| FMLbiApprox | 8.54 | 73.9 | 33.1 | 31.3 | 0.97 | 0.94 |

- Greedy 完全忽视 R 组（覆盖=0），导致严重不公平
- FML 方法以仅约 74 的代价实现两组近乎等比覆盖，远低于 Closest/Alternating 的 212-224

### 实际网络可扩展性（Pokec 数据集）

20,000 节点子图（201,900 边）上的测试：

| epsilon | 运行时间(s) | 准确率 |
|---------|-------------|--------|
| 0.01 | 3625 +/- 421 | 93.1% |
| 0.001 | 4662 +/- 991 | 99.3% |

理论保证：树高 H 约 6.68 时，eps=0.01 保证约 92.6% 覆盖，实测 93.1%，验证了理论界。

### 消融实验

- FMLbiApprox vs FMLapprox：在 Barabasi-Albert 图上，biApprox 速度快一个数量级，eps=0.001 时覆盖比率接近 1.0
- 终端位置影响：无论终端在中心（高度数）或边缘（低度数），biApprox 均达到 98%+ 覆盖
- 99.9% 以上的运行时间花在 DP 求解树上，树嵌入时间可忽略

### 关键发现

- 公平无关的 Greedy 虽代价低但完全系统性忽视一组，可能导致有偏模型
- FML 方法在代价、公平性和下游模型精度之间取得了最佳平衡
- 双准则近似显著提升可扩展性（约8倍加速），代价仅有微小增加

## 亮点与洞察

1. **形式化新问题 FML**：首次将时序边激活的代价优化与分组公平可达性约束统一建模，泛化了经典的 ML 和 MSL 问题
2. **紧近似界**：Omega(log n) 下界与 O(log n) 上界匹配，理论上最优
3. **概率树嵌入的创造性应用**：将一般图上的 NP-hard 问题转化为树上可精确求解的 DP，兼顾理论保证和实际效率
4. **实验设计巧妙**：两组数据源编码正交分类规则，不训练两组数据就无法两组准确，自然展示了公平数据收集的必要性

## 局限与展望

1. 当前算法仅支持单终端、两组的情况，多终端和多组需要更复杂的 DP 状态空间
2. 假设中心化控制边激活，不适用于去中心化场景（如经典联邦学习）
3. FML 定义的公平是组级时序可达性，不涉及个体公平或随时间变化的公平
4. 精确 DP 的 O(n^5) 复杂度在大规模网络上仍有挑战
5. 实验的数据生成较为合成，真实网络场景的验证仅限 Pokec 一个数据集

## 相关工作与启发

- **Minimum Labeling**（Mertzios et al. / Klobas et al.）：FML 泛化了这些经典时序图设计问题
- **FRT 概率树嵌入**（Fakcharoenphol et al.）：O(log n) 紧近似的理论基础
- **公平影响力最大化**：静态图上的公平优化，FML 将其扩展到时序设定
- 可启发分布式联邦学习中的公平通信调度设计

## 评分

4/5 优秀。理论扎实（NP-hard 证明加匹配的近似界），问题形式化新颖且实际相关。不足在于当前仅解决了受限情形（单终端两组），距离通用解决方案仍有距离。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[ICCV 2025\] Vulnerability-Aware Spatio-Temporal Learning for Generalizable Deepfake Video Detection](../../ICCV2025/ai_safety/vulnerability-aware_spatio-temporal_learning_for_generalizable_deepfake_video_de.md)
- [\[NeurIPS 2025\] Efficient Fairness-Performance Pareto Front Computation](efficient_fairness-performance_pareto_front_computation.md)
- [\[NeurIPS 2025\] Cost Efficient Fairness Audit Under Partial Feedback](cost_efficient_fairness_audit_under_partial_feedback.md)
- [\[NeurIPS 2025\] DictPFL: Efficient and Private Federated Learning on Encrypted Gradients](dictpfl_efficient_and_private_federated_learning_on_encrypted_gradients.md)

</div>

<!-- RELATED:END -->
