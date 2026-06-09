---
title: >-
  [论文解读] Laplacian Representations for Decision-Time Planning
description: >-
  [ICML 2026][强化学习][Laplacian表示] 本文提出 ALPS，将图 Laplacian 的特征向量空间（缩放后近似 commute-time distance）作为分层决策时规划的潜空间，先用 k-means 在该空间发现子目标并跑 Dijkstra 生成高层路径…
tags:
  - "ICML 2026"
  - "强化学习"
  - "Laplacian表示"
  - "分层规划"
  - "决策时规划"
  - "CEM"
  - "离线目标条件RL"
---

# Laplacian Representations for Decision-Time Planning

**会议**: ICML 2026  
**arXiv**: [2602.05031](https://arxiv.org/abs/2602.05031)  
**代码**: https://github.com/machado-research/ALPS  
**领域**: 强化学习 / 表示学习 / 基于模型的RL  
**关键词**: Laplacian表示、分层规划、决策时规划、CEM、离线目标条件RL  

## 一句话总结
本文提出 ALPS，将图 Laplacian 的特征向量空间（缩放后近似 commute-time distance）作为分层决策时规划的潜空间，先用 k-means 在该空间发现子目标并跑 Dijkstra 生成高层路径，再用带行为先验的 CEM 在原始状态空间做短程低层规划，在 OGBench 的离线目标条件 RL 上首次让基于模型的规划方法系统性击败 model-free SOTA。

## 研究背景与动机

**领域现状**：基于模型的强化学习相比 model-free RL 在样本效率、泛化性和适应速度上都有理论优势，决策时规划（如 MPC、MCTS）是把"学到的模型"转成行为的常见方式。OGBench 这种困难的离线目标条件 RL 长期由 HIQL、CRL、QRL 等 model-free 方法垄断，model-based 规划方法在长程任务上几乎不见踪影。

**现有痛点**：决策时规划的核心难题是**复合误差**——学到的一步模型在长 horizon 反复 rollout 后预测轨迹会迅速偏离真实动力学，导致 CEM 等优化器在"幻觉轨迹"上做出错误决策。分层规划是公认的解药（高层只挑子目标、低层只跑短程），但需要一个潜空间同时满足两个矛盾的需求：邻近状态在潜空间里要近（支持局部代价计算），长程可达性也要保留（支持高层路径搜索）。

**核心矛盾**：常用对比学习潜空间（如 PcLast 用的随机游走对比目标）在低层距离估计上不错，但**没有显式编码全局连通性**，导致 k-means 出来的子目标常常跨越障碍、生成不可达高层路径；用原始欧氏距离则完全无视环境动力学（迷宫里两点欧氏近但要绕一圈才能到）。

**本文目标**：找到一个潜空间，使得（1）距离近似 commute-time distance（CTD），同时支持高低两层规划；（2）天然适合谱聚类做子目标发现；（3）能从样本可学，不依赖 $O(|\mathcal{S}|^3)$ 的精确特征分解。

**切入角度**：作者注意到图 Laplacian 的特征向量正是为表达"多时间尺度的图结构"而设计的——前几个特征向量编码全局结构（房间、区域），后续特征向量编码局部细节；同时谱聚类的理论保证它会沿"瓶颈"切分图，恰好对应导航任务里走廊连接的房间。更关键的是，缩放后的 Laplacian 表示 $\psi_i(s) = \phi_i(s)/\sqrt{\lambda_i}$ 在欧氏距离下等价于 CTD，可同时当低层代价和高层距离用。

**核心 idea**：用学到的缩放 Laplacian 表示 $\psi$ 作为统一潜空间，在 $\psi$ 空间做 k-means 聚类得到子目标、Dijkstra 出高层路径，低层用带行为先验的 CEM 在原始状态空间做短程优化，把 model-based 规划重新带回 OGBench 的领奖台。

## 方法详解

### 整体框架
ALPS 是一套预训练 + 决策时规划的两阶段算法。**预训练**阶段从离线数据集 $\mathcal{D}$ 里学三个组件：（1）Laplacian 表示 $\phi$（再缩放成 $\psi$），（2）原始状态空间上的一步前向模型 $f$，（3）目标条件行为先验 $\pi_{\text{prior}}$；然后在 $\psi$ 空间跑 k-means 得到 $C$ 个簇，把簇心当顶点、簇间观察到的转移当边，构建簇图 $G_c$。**决策时**给定 $(s_{\text{start}}, s_{\text{goal}})$，先把两端映到 $\psi$ 空间找所在簇 $(c_s, c_g)$，用 Dijkstra 在 $G_c$ 上算最短簇路径 $\mathcal{P}_G$ 作为高层计划；每一步用 CEM 朝当前目标簇心的 $\psi$ 表示做短程优化，agent 进入下一个簇后高层指针前进，如果偏离 $\mathcal{P}_G$ 则重规划。

### 关键设计

**1. 缩放 Laplacian 表示 $\psi$ 作为统一潜空间：让同一个空间既能当低层代价、又能当高层距离**

分层规划需要一个潜空间同时满足两个矛盾需求——邻近状态要近（支持局部代价），长程可达性也要保留（支持高层路径搜索）；对比学习空间擅长前者却没显式编码全局连通性，原始欧氏距离则无视环境动力学。作者的关键观察是：把图 Laplacian 前 $D$ 个非零特征向量 $\phi$ 按特征值缩放成 $\psi(s)=\phi(s)\oslash\sqrt{\lambda}$ 后，欧氏距离恰好近似 commute-time distance（CTD），$c(u,v)\approx\|\psi(u)-\psi(v)\|^2$，而 CTD 同时编码了"一步可达"和"绕路距离"。为避开精确特征分解 $O(|\mathcal{S}|^3)$ 的代价，$\phi$ 用 ALLO（Augmented Lagrangian Laplacian Objective）从样本学：目标 $\max_\beta \min_u \sum_i \langle u_i, L u_i \rangle + \sum_{j,k} \beta_{jk}(\langle u_j, [[u_k]]\rangle - \delta_{jk}) + B\cdot(\cdot)^2$ 用 stop-gradient $[[\cdot]]$ 和拉格朗日乘子 $\beta$ 强制正交归一约束，特征值直接从对偶变量读出 $\lambda_i=-\beta_{ii}/2$、缩放得 $\psi_i=\sqrt{2}\phi_i/\sqrt{-\beta_{ii}}$，训练对 $(S_t,S_{t+\Delta})$ 的 $\Delta\sim\text{Geom}(1-\gamma_s)$ 从几何分布抽。正因为 CTD 一体两用，ALPS 不必像 PcLast 那样维护两个不同潜空间：低层 CEM 拿它当代价，高层 k-means 在同一空间分簇也会自动沿环境瓶颈切分。

**2. 行为先验加速的 CEM 低层规划：把通用黑盒优化器升级成数据感知规划器**

CEM 通常从无信息高斯采样动作序列，在高维动作空间收敛极慢，而且一步模型反复 rollout 会累积复合误差。ALPS 在每个高层子目标 $z_{\text{sub}}$ 下优化代价 $J^m=\sum_{t=1}^H(\|\psi(\hat{S}_t^m)-z_{\text{sub}}\|_2^2+\lambda\|A_t^m\|_2^2)$，但不再从零搜索：先用目标条件行为克隆学一个确定性先验 $\pi_{\text{prior}}(S_t,\psi(S_t),\psi(S_{t+k}))$（$k\sim U(1,K_{\max})$ 回归 $A_t$），规划时让它配合多步自回归前向模型 $f$（训练用 $\frac{1}{H_f}\sum_{\tau=1}^{H_f}\|\hat{S}_{t+\tau}-S_{t+\tau}\|_2^2$ 把误差反传通时间）先滚出一条均值动作序列 $\mathbf{a}_{t:t+H-1}$，再围绕它加时间相关高斯噪声生成 $N_s$ 个候选，按代价排序取 top-$N_e$ 更新分布、迭代 $N_{\text{iter}}$ 次。行为先验直接把初始搜索分布偏向"看起来像数据集里目标导向轨迹"的动作，多步前向模型又把 rollout 控制在模型可信窗口内，二者组合让 CEM 在高维动作上少跑几轮就能收敛。

**3. 基于簇图的高层 Dijkstra 规划与漂移重规划：把长程问题切成"过几个房间"的离散搜索**

复合误差的根治办法是把长 horizon 切成短 horizon 子任务。ALPS 在 $\psi$ 空间跑 k-means 聚成 $C$ 簇（在 CTD 空间跑 k-means 等价于谱聚类，会沿环境瓶颈切分——迷宫的房间被走廊分开、CTD 大的状态对必落到不同簇），簇心当顶点、数据集里观察到的"簇 $i\to$ 簇 $j$"转移当边，并用 nucleus sampling 只保留每个簇 top-$p\%$ 频繁邻居以剔除不可达边。决策时 Dijkstra 在这张 $|C|$ 顶点的簇图上算最短路径 $\mathcal{P}_G$，把"怎么从起点走到目标"降成"过哪几个房间"，每段任务长度落在前向模型可信窗口内；agent 每步检查所在簇 $c_{\text{curr}}$，一旦低层 CEM 走歪进了 off-plan 簇（$c_{\text{curr}}\notin\mathcal{P}_G$），就从当前簇重新 Dijkstra 补救。图搜索可秒级完成，复杂度从原始连续状态空间降到离散图。

### 一个完整示例：在 pointmaze-giant 里从起点导航到目标

给定 $(s_{\text{start}}, s_{\text{goal}})$，ALPS 先把两端映到 $\psi$ 空间，分别落在簇 $c_s$（左下角房间）和 $c_g$（右上角房间）。Dijkstra 在簇图 $G_c$ 上算出最短簇路径，比如 $\mathcal{P}_G = c_s \to c_3 \to c_7 \to c_g$（穿过三道走廊连接的四个房间）——注意因为 $\psi$ 是 CTD 几何，这条路径绝不会"抄近道"穿墙，而是老老实实绕走廊。执行时高层指针指向第一个子目标 $c_3$ 的簇心，低层 CEM 朝它的 $\psi$ 表示做 $H$ 步短程优化、执行首个动作；agent 一旦进入 $c_3$，指针前进到 $c_7$。若某步 CEM 被噪声推到了不在路径上的簇 $c_5$，高层立刻从 $c_5$ 重新 Dijkstra，得到新路径 $c_5 \to c_7 \to c_g$ 接着走。整个长程导航就这样被拆成三四段"走到隔壁房间"的短任务，每段都在前向模型可信范围内，避免了一次性 rollout 几十步带来的幻觉。



### 损失函数 / 训练策略
ALLO 用 $\gamma_s$ 控制时间尺度（几何分布参数），$B$ 是 barrier 系数（论文报告对其不敏感）。前向模型用 $H_f$ 步自回归 MSE 训练。行为先验是 MSE 行为克隆。CEM 关键超参：planner horizon $H$、采样数 $N_s$、elite 数 $N_e$、迭代数 $N_{\text{iter}}$、动作惩罚 $\lambda$、子目标到达阈值 $\epsilon$。

## 实验关键数据

### 主实验

| 数据集 | 指标 | ALPS | 之前最强 model-free | 提升 |
|--------|------|------|---------|------|
| pointmaze-large-stitch-v0 | 成功率 % | 96 ±2 | QRL 84 ±15 | +12 |
| pointmaze-giant-stitch-v0 | 成功率 % | 98 ±1 | QRL 50 ±8 | +48 |
| antmaze-large-navigate-v0 | 成功率 % | 93 ±5 | HIQL 91 ±2 | +2 |
| antmaze-giant-navigate-v0 | 成功率 % | 69 ±9 | HIQL 65 ±5 | +4 |
| pointmaze-giant-navigate-v0 | 成功率 % | 67 ±11 | QRL 68 ±7 | -1 (打平) |

OGBench 整体 ALPS 用 Holm-Bonferroni 校正的 Wilcoxon 检验在 $p<0.001$ 下显著优于所有 model-free baseline（GCBC/GCIVL/GCIQL/QRL/CRL/HIQL）。最戏剧的提升在 stitch 类数据集——这类数据集只有短轨迹要求方法学会拼接，model-free 几乎全部失败（HIQL 在 pointmaze-giant-stitch 只有 0），而 ALPS 拿到 98%。

### 消融实验

| 配置 | Hallway | Rooms | Spiral | 说明 |
|------|---------|-------|--------|------|
| PcLast (1 cluster, 仅低层) | 51 ±4 | 30 ±3 | 35 ±4 | 对比学习潜空间 |
| PcLast (16 clusters) | 62 ±4 | 57 ±10 | 60 ±6 | 加高层规划 |
| ALPS† (1 cluster, 仅低层) | 94 ±3 | 92 ±3 | 91 ±4 | 换成 $\psi$ 空间，无行为先验 |
| ALPS† (16 clusters) | 97 ±2 | 96 ±2 | 94 ±2 | 加高层规划 |

### 关键发现
- $\psi$ 空间本身就是关键贡献：仅替换潜空间（ALPS† 1 cluster vs PcLast 1 cluster），Hallway 从 51% 涨到 94%、Rooms 从 30% 涨到 92%，说明 Laplacian/CTD 几何对低层代价估计远比对比学习目标更可靠。
- 高层规划在 PcLast 上是刚需（去掉掉 11–27 个百分点），在 ALPS 上只是锦上添花（1 cluster vs 16 cluster 差 2–5 点）——这是因为 $\psi$ 空间的距离本身已经隐含了全局拓扑，低层 CEM 直接朝 $\psi(g)$ 走就能避开多数障碍。
- Stitch 类数据集是 model-based 规划的天然优势区——只要前向模型能学好局部转移，规划器就能在数据里没出现过的子轨迹拼接里发现新路径，model-free 价值函数则受困于数据分布。
- Teleport 类任务（带瞬移机制破坏 CTD 假设）是 ALPS 弱项：pointmaze-teleport-stitch 只有 13%，因为缩放 Laplacian 距离假设了局部平滑动力学，瞬移破坏了"距离近 = 步数少"。

## 亮点与洞察
- 用一个潜空间同时承担"高层子目标发现 + 高层距离 + 低层代价"三重身份，避免了 PcLast 之类方法维护多个表示带来的不一致。这个统一性来自 CTD 与谱聚类的数学等价关系——前者给距离、后者给聚类，刚好对应低层、高层两个需求。
- 把 commute-time distance 这种来自图论的经典几何引回深度 RL，并用 ALLO 这种可微采样目标使其在连续状态空间可学，是表示学习与规划的优雅结合；这种"显式编码动力学的潜空间"思路可迁移到任何需要长程推理的领域（具身导航、分子设计、组合优化）。
- 用行为先验把 CEM 从"通用黑盒优化器"升级成"数据感知规划器"，是把离线数据的隐含知识喂回规划循环的便宜手段；行为先验的成本只是一个 BC 网络，但能让 CEM 在高维动作空间里少跑几轮迭代。

## 局限与展望
- 作者承认：teleport-stitch 是显著短板，因为缩放 Laplacian 假设了局部平滑动力学；任何带瞬移、传送门、状态跳跃的环境都会让 CTD 失真。
- 自己发现：ALLO 学的是基于行为策略 $\pi$ 的 Laplacian，所以表示质量强依赖数据集覆盖率——explore 类数据集（高噪声但高覆盖）和 navigate/stitch 类的最优表示可能不同，论文没有系统消融 dataset type 对 $\psi$ 质量的影响。
- 改进思路：可以把多个数据集策略合并训 $\psi$、或在线 fine-tune $\psi$；高层 Dijkstra 是确定性的、不考虑不确定性，可换成 belief-MDP 规划；CEM 的代价函数只用了 $\psi$ 距离，可以加上不确定性惩罚（用前向模型的 ensemble variance）。

## 相关工作与启发
- **vs PcLast**: PcLast 用对比学习目标学潜空间、k-means 找子目标、CEM 跑低层，整体框架几乎相同；ALPS 把潜空间换成缩放 Laplacian 并加上行为先验。直接对比（Table 1）显示 $\psi$ 空间相比对比空间在所有 Maze2D 任务上平均提升 30+ 个百分点，证明潜空间几何是关键而非框架结构。
- **vs HIQL/QRL**: HIQL 是分层 model-free（高层预测子目标表示、低层做 IQL），QRL 学满足三角不等式的 quasimetric；ALPS 优势在于 stitch 类任务上能拼接出训练分布外的轨迹（这是 model-based 规划的本质优势），劣势在 teleport 类。
- **vs MuZero/Dreamer**: 后两者也是 model-based RL 但用学到的隐空间一步预测做 MCTS/想象 rollout，没有显式建模 commute-time 几何；ALPS 强调"规划用的潜空间应该匹配规划的代价语义"，给 model-based RL 提供了表示设计的新视角。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把图论里 commute-time distance + 谱聚类的经典工具用 ALLO 端到端学到深度 RL 设置里，并用统一潜空间承担三重角色，组合很优雅。
- 实验充分度: ⭐⭐⭐⭐ OGBench 全套（locomotion + manipulation、state-based + pixel-based、navigate/stitch/explore），8 seed，Wilcoxon 校正显著性检验；PcLast 直接对比 + 多个消融。
- 写作质量: ⭐⭐⭐⭐ Section 3 把 CTD/谱聚类/Laplacian 三件事的关系讲得清晰；动机和方法分离得当；图 1 的 $\psi$ 空间可视化直击要点。
- 价值: ⭐⭐⭐⭐ OGBench 长期被 model-free 垄断的局面被打破，给 model-based 规划重新进入主流离线 RL benchmark 提供了具体路径；统一潜空间设计思想对其他长程规划任务有迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning to Search and Searching to Learn for Generalization in Planning](learning_to_search_and_searching_to_learn_for_generalization_in_planning.md)
- [\[ICML 2026\] Quantifying and Optimizing Simplicity via Polynomial Representations](quantifying_and_optimizing_simplicity_via_polynomial_representations.md)
- [\[ICML 2026\] DR.Q: Debiased Model-based Representations for Sample-efficient Continuous Control](debiased_model-based_representations_for_sample-efficient_continuous_control.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)

</div>

<!-- RELATED:END -->
