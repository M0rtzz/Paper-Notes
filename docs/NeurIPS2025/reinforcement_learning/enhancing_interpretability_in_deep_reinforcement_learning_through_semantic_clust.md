---
title: >-
  [论文解读] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering
description: >-
  [NeurIPS 2025][interpretability] 提出语义聚类模块(SCM)，将特征降维网络(FDR)与改进的 VQ-VAE 在线聚类相结合，无缝集成到 DRL 训练流程中，解决了 t-SNE 可视化不稳定的问题，揭示 DRL 内在具有基于语义的动态聚类特性。
tags:
  - NeurIPS 2025
  - interpretability
  - semantic clustering
  - VQ-VAE
  - dimensionality reduction
  - Procgen
---

# Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering

**会议**: NeurIPS 2025  
**arXiv**: [2409.17411](https://arxiv.org/abs/2409.17411)  
**代码**: [github.com/ualiangzhang/semantic_rl](https://github.com/ualiangzhang/semantic_rl)  
**领域**: reinforcement_learning / 可解释性  
**关键词**: interpretability, semantic clustering, VQ-VAE, dimensionality reduction, Procgen

## 一句话总结
提出语义聚类模块(SCM)，将特征降维网络(FDR)与改进的 VQ-VAE 在线聚类相结合，无缝集成到 DRL 训练流程中，解决了 t-SNE 可视化不稳定的问题，揭示 DRL 内在具有基于语义的动态聚类特性。

## 研究背景与动机

**领域现状**：深度强化学习(DRL)在机器人、游戏等领域广泛应用，但其"黑盒"决策过程缺乏可解释性。语义聚类在 NLP（Word2Vec、GloVe）和 CV（图像特征空间）中已被充分研究，但在 DRL 中仍欠探索。

**现有痛点**：(a) 先前工作（Mnih et al. 2015, Zahavy et al. 2016）仅在 Atari 固定场景游戏上用 t-SNE 分析 DRL 特征，无法区分聚类是来自像素相似性还是语义理解；(b) t-SNE 结果不稳定——对初始化、随机种子、样本数量高度敏感；(c) t-SNE 没有自动聚类机制，需要大量人工标注。

**核心矛盾**：要理解 DRL 模型的内部语义组织，需要稳定、自动化的聚类方法，但现有可视化工具不满足这些要求。

**本文目标**：(a) 验证 DRL 是否内在具有语义聚类能力（排除像素相似性混淆因素）；(b) 提供稳定的替代 t-SNE 的分析工具；(c) 基于聚类结果开发策略分析方法。

**切入角度**：使用 Procgen 而非 Atari——Procgen 的程序生成关卡确保场景多样性，如果聚类仍然成立则说明是真正的语义理解而非像素记忆。将 VQ-VAE 的离散编码机制改造为在线聚类工具。

**核心 idea**：用改进 VQ-VAE 的 codebook 作为在线聚类中心，与 DRL 端到端训练，既增强聚类质量又保持策略性能。

## 方法详解

### 整体框架
在标准 DRL（如 PPO）的特征提取器 $f$ 之后接入语义聚类模块(SCM)。SCM 包含两部分：(1) FDR 网络 $g$ 将高维特征降到 2D；(2) VQ 量化器将 2D 特征分配到最近的 codebook 嵌入（相当于聚类）。VQ 编号 $k$ 通过维度扩展后加回原始特征，实现条件化策略训练 $\pi(a|s,k)$。

### 关键设计

1. **特征降维网络 (FDR Network)**:

    - 功能：学习一个从高维特征到 2D 的稳定映射，替代 t-SNE
    - 核心思路：训练 FDR 网络 $g$ 使低维空间保持高维空间的距离关系。使用 Student's t-分布计算成对相似度：
    $p_{ij} = \frac{d(i,j)}{\sum_{k\neq l}d(k,l)}, \quad d(m,n) = \left(1+\frac{\|f(\mathbf{s}_m)-f(\mathbf{s}_n)\|^2}{\alpha}\right)^{-\frac{\alpha+1}{2}}$
   FDR 特征的相似度 $q_{ij}$ 同公式但用 $g \circ f$ 替代 $f$。FDR 损失为交叉熵：$\mathcal{L}_{\text{FDR}} = -\sum_i\sum_j p_{ij}\log(q_{ij})$
    - 设计动机：t-SNE 不稳定因为其非凸目标函数对初始化敏感。FDR 网络训练后产生确定性映射，不随随机种子或样本数变化
    - 与 t-SNE 的关键区别：高维和低维使用**相同的**自由度参数 $\alpha$，确保原始距离关系被严格保持（不只保留近邻关系）

2. **改进的 VQ-VAE 在线聚类**:

    - 功能：自动将 FDR 特征分配到离散聚类
    - 核心思路：仅保留 VQ-VAE 损失的第二项（嵌入更新项），使 codebook 嵌入作为在线 k-means 聚类中心：$\mathcal{L}'_{\text{VQ-VAE}} = \|sg[g(f(\mathbf{s}))] - \mathbf{e}_k\|_2^2$，其中 $\mathbf{e}_k$ 是最近的 codebook 嵌入
    - 设计动机：(a) 丢弃 VQ-VAE 的重建项（不需要重建观测）和 commitment loss（由 FDR 损失替代）；(b) stop-gradient 防止聚类目标直接拉动 FDR 特征，而是通过 $\mathcal{L}_{\text{FDR}}$ 联合训练间接增强聚类紧致性

3. **条件化策略训练**:

    - 功能：将聚类信息注入策略
    - 核心思路：VQ 编号 $k$ 先扩展到特征维度 $\mathbf{k}^{\text{expand}} = \text{expand}(k, \dim(\mathbf{f}))$，然后与原始特征逐元素相加 $\mathbf{f}^{\text{fused}} = \mathbf{f} + \mathbf{k}^{\text{expand}}$，策略和值函数基于融合特征输出
    - 设计动机：使策略能利用聚类信息，为下游分层学习奠定基础（如将聚类作为 macro-action 选择的基础）

4. **自适应控制因子 $\lambda_{\text{ctrl}}$**:

    - 功能：在训练早期降低 SCM 损失权重
    - 核心思路：根据训练性能动态调整 $\lambda_{\text{ctrl}}$，早期语义分布不清晰时减小权重避免干扰策略学习
    - 设计动机：有效的语义聚类依赖清晰的语义分布，训练早期尚未形成

### 损失函数 / 训练策略
总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DRL}} + \lambda_{\text{ctrl}}(w_{\text{FDR}}\mathcal{L}_{\text{FDR}} + w_{\text{VQ-VAE}}\mathcal{L}'_{\text{VQ-VAE}})$

联合训练 PPO 参数 $\theta$、FDR 参数 $\phi$、codebook 嵌入 $\{\mathbf{e}_k\}_{k=1}^K$。使用 $K=8$ 个聚类。

## 实验关键数据

### t-SNE vs FDR 稳定性对比

| 对比维度 | t-SNE | FDR (本文) |
|---------|-------|-----------|
| 改变随机种子 | 可视化结构剧变 | 映射完全不变 |
| 减少 50% 样本 | 分布结构改变 | 仅密度降低,空间分布不变 |
| 聚类边界 | 无清晰边界,需人工检查 | 清晰分离的聚类 |
| 自动聚类 | 不支持 | VQ 编码自动分配 |
| 分析方式 | 静态可视化 | 支持动态策略分割 |

### 人类评估结果 (5点 Likert 量表)

| 评估声明 | Jumper | FruitBot | Ninja |
|---------|--------|----------|-------|
| 每个聚类一致展示相同技能 | 4.24 (±0.15) | 4.10 (±0.11) | 4.30 (±0.15) |
| 聚类匹配给定的技能描述 | 4.36 (±0.16) | 4.16 (±0.11) | 4.20 (±0.17) |
| 识别的技能助于理解决策过程 | 4.50 (±0.22) | 4.10 (±0.18) | 4.20 (±0.20) |

所有评分均 >4.0（15名评估者），说明人类普遍认同 DRL 具有有意义的语义聚类属性。

### 聚类语义分析示例 (Ninja 游戏, 8 个聚类)

| 聚类 | 语义描述 |
|------|---------|
| 0 | 在第一个平台行走,然后高跳到更高平台 |
| 1 | 在场景中部做小跳 |
| 2 | 从最左端走到起始位置/在无更高平台时行走准备 |
| 3 | 在平台上走动准备跳到更高平台 |
| 4 | 高跳后失去下方平台视线 |
| 5 | 在平台上走动准备跳到同高或更低平台 |
| 6 | 高跳中但保持下方平台可见 |
| 7 | 向场景右边移动并触碰蘑菇 |

### 关键发现
- **语义聚类是 DRL 的内在属性**：不依赖外部约束（如 bisimulation 或对比学习），DRL 模型内在地按语义组织特征空间
- **动态而非静态的聚类**：与 NLP/CV 中基于单个输入的聚类不同，DRL 的语义聚类是时序的——连续的状态序列被归为同一聚类，可用自然语言描述（类似"技能"）
- **超越像素距离**：在 Procgen 的程序生成关卡中，视觉完全不同的状态因语义相同被聚到一起，证明聚类基于语义理解而非像素相似
- **策略层次发现**：Episode 的聚类分割揭示了策略的层次结构（cluster 5→7 的切换由右侧墙壁检测驱动而非蘑菇出现）
- **SCM 对策略性能影响可忽略**：集成 SCM 后 PPO 的游戏性能基本不变

## 亮点与洞察
- **VQ-VAE 的巧妙改造**：将生成模型的量化机制重新解释为聚类工具——去掉解码器和重建损失，只保留嵌入更新项。这个"减法创新"简洁有效
- **聚类 = 技能的自然发现**：每个聚类对应一个人类可理解的行为阶段（"走到平台边缘"、"高跳"、"触碰蘑菇"），这暗示可以将聚类直接用作分层 RL 中的选项/宏动作
- **可视化工具的实用性**：交互式 hover 工具和动态 episode 分割为 DRL 调试提供了新的分析范式——观察聚类切换点可以发现策略错误（如 Figure 6c 中误检测右侧墙壁的案例）

## 局限与展望
- **聚类数量的选择**：使用固定 $K=8$，未探索自适应聚类数量方法（如 elbow method、silhouette score）
- **依赖清晰的语义分布**：当策略远离最优时语义分布模糊，聚类不稳定
- **人工语义描述**：聚类的语义标签仍需人工提供（每个环境约 15 分钟），作者计划用 GPT-4V 自动化
- **仅在 Procgen 验证**：未在连续控制或更复杂任务（如 StarCraft）中验证
- **聚类使用的 affinity**：当前使用 Student's t-分布成对相似度，可探索余弦相似度或 bisimulation metric

## 相关工作与启发
- **vs Mnih et al. (2015, DQN t-SNE)**：他们在 Atari 固定场景中发现特征聚类，但无法排除像素相似性假设。本文用 Procgen 程序生成排除此混淆
- **vs Zahavy et al. (2016, Graying DQN)**：他们为特定游戏手动定义特征进行分析，人力成本高。本文的端到端方法自动完成
- **vs PW-Net/DIGR 等可解释性方法**：它们聚焦单帧决策解释（原型/显著图），本文揭示时序语义结构——更接近人类对"策略"的整体理解

## 评分
- 新颖性: ⭐⭐⭐⭐ VQ-VAE 改造为聚类工具的 idea 简洁优雅,DRL 语义聚类的系统性研究填补空白
- 实验充分度: ⭐⭐⭐ Procgen 上三个游戏的分析+人类评估=基本充分，但缺乏连续控制和复杂多主体场景
- 写作质量: ⭐⭐⭐⭐ 图表丰富，可视化出色，方法说明清晰
- 价值: ⭐⭐⭐⭐ 揭示 DRL 内在语义组织的分析工具，对可解释 RL 和分层 RL 都有启发
