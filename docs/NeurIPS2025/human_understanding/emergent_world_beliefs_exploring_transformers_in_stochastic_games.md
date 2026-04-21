---
title: >-
  [论文解读] Emergent World Beliefs: Exploring Transformers in Stochastic Games
description: >-
  [NeurIPS 2025][人体理解][涌现世界模型] 将LLM涌现世界模型的研究从完全信息游戏（Othello、国际象棋）扩展到不完全信息领域（德州扑克），通过在PHH格式扑克数据上预训练GPT-2并探测其内部激活，证明模型不仅学习了确定性特征（牌型识别~98%准确率），还自发发展了对随机性特征（胜率/equity，相关系数0.59）的内部表示。
tags:
  - NeurIPS 2025
  - 人体理解
  - 涌现世界模型
  - Transformer
  - 扑克
  - POMDP
  - 信念状态
  - 激活探针
  - GPT-2
---

# Emergent World Beliefs: Exploring Transformers in Stochastic Games

**会议**: NeurIPS 2025  
**arXiv**: [2512.23722](https://arxiv.org/abs/2512.23722)  
**代码**: [GitHub](https://anonymous.4open.science/r/poker-interp-4653/)  
**领域**: 机械可解释性 / 世界模型 / 不完全信息博弈  
**关键词**: 涌现世界模型, Transformer, 扑克, POMDP, 信念状态, 激活探针, GPT-2  

## 一句话总结
将LLM涌现世界模型的研究从完全信息游戏（Othello、国际象棋）扩展到不完全信息领域（德州扑克），通过在PHH格式扑克数据上预训练GPT-2并探测其内部激活，证明模型不仅学习了确定性特征（牌型识别~98%准确率），还自发发展了对随机性特征（胜率/equity，相关系数0.59）的内部表示。

## 背景与动机
Transformer LLM的内部世界表示研究已在完全信息游戏中取得重要进展：

- **OthelloGPT**（Li et al.）：仅在合法走子序列上训练的模型自发发展了对棋盘状态的内部表示，非线性探针可解码方格占据状态，因果干预可翻转方格并影响下游走子
- **ChessGPT**（多个工作）：线性解码器可恢复棋子/方格特征，编辑这些特征可预测性地改变走子概率
- **Nanda et al.**：证明OthelloGPT的世界模型甚至可以通过线性探针提取

然而，以上工作均限于**完全信息**（perfect information）游戏——所有玩家可观测完整游戏状态。在现实世界中，大量决策场景涉及**不完全信息**（incomplete information），如扑克、商业谈判、军事博弈。在这些POMDP环境中，模型需要维护一个关于隐藏状态的**信念分布**（belief state），而非确定性状态。

核心问题：LLM能否在不完全信息游戏的序列数据中，自发发展出对应于POMDP信念状态的内部表示？

## 核心问题
1. 在部分可观测的随机游戏（如扑克）中，Transformer是否能涌现出确定性世界模型（如牌型识别）？
2. 更重要的是，Transformer是否能涌现出**随机性世界模型**——即对隐藏信息（如对手底牌、胜率）的概率估计？

## 方法详解

### 数据集生成
由于缺乏大规模完整的扑克手牌数据集，作者通过模拟生成：
- 6人No-Limit Texas Hold'em对局
- 使用模拟equity估算驱动决策的AI代理
- 每个代理随机初始化不同的游戏风格参数（加注倾向、紧密度、诈唬频率、跟注意愿、下注尺度等）
- 生成超过200万局PHH（Poker Hand History）格式的合成对局数据

### 模型训练
- **架构**：GPT-2 base配置（12个注意力头，隐藏维度768，约87M参数）
- **训练**：AdamW优化器（$\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$），学习率$5 \times 10^{-5}$
- **特殊设计**：在输入序列中插入`<GAP>`特殊token替代部分token，原始token移到`<ANS>`标记后出现，损失仅计算`<ANS>`后的token
- **训练规模**：13个epoch（验证loss不再下降后停止），有效batch size 128，95-5训练测试划分
- **硬件**：NVIDIA H200 GPU，约7小时

### 探针设计

#### 线性分类探针
$$p_\theta(x_t^l) = \text{argmax}(W x_t^l), \quad \theta = \{W \in \mathbb{R}^{C \times F}\}$$
其中$F$为激活向量维度，$C$为类别数。

#### 两层MLP探针
$$p_\theta(x_t^l) = \text{argmax}(W_1 \text{ReLU}(W_2 x_t^l)), \quad \theta = \{W_1 \in \mathbb{R}^{C \times H}, W_2 \in \mathbb{R}^{H \times F}\}$$
其中$H$为隐藏维度。对随机性特征（连续值），去掉argmax直接输出回归值。

### 确定性世界模型探测
探测**牌型**（hand rank）如高牌、一对、两对、三条等。为避免高频牌型（如高牌、一对）的过度表示：
- 对每类牌型按第40百分位数截断样本数量
- 在每层分别训练独立的探针

### 随机性世界模型探测
探测**手牌胜率**（equity）：
- 使用模拟方法的equity估算值作为标签
- 故意遮蔽所有除Player 1外的底牌
- 训练两层MLP回归探针预测equity值

### 理论支撑：信念状态的线性可解码性

作者在附录中给出了正式的理论分析。在POMDP中，信念状态$b_t(s) = \Pr(S_t = s | H_t)$是决策的充分统计量。对任意有限时域事件$F$：

$$\Pr_\pi(F | H_t) = \sum_{s \in \mathcal{S}} b_t(s) v_F(s) = \langle b_t, v_F \rangle$$

即未来事件的预测是信念状态的**线性函数**。如果Transformer在残差流中存储了信念状态的仿射变换，则存在线性探针可以恢复相关预测——这为线性可探测性提供了理论基础。

## 实验关键数据

### 确定性特征：牌型识别

| 探针类型 | 准确率 | 备注 |
|---------|--------|------|
| 线性探针 | ~80% | 牌型分类 |
| MLP探针 | **~98%** | 牌型分类，40th百分位平衡 |

- 强对角混淆矩阵证实模型内部激活可靠编码了牌型信息
- 低方差跨种子表明表示的高一致性
- 早期层（Layer 0-3）编码最强

### 随机性特征：手牌胜率(Equity)

| 指标 | 数值 | 备注 |
|------|------|------|
| 相关系数 $r$ | **0.59** | 跨种子平均，Layer 0 |
| $R^2$最强层 | Layer 0-4 | 随深度递减 |
| $R^2$衰减模式 | 信息瓶颈式 | 深层保留更与token预测相关的信息 |

### 动作识别（附加实验）
遮蔽动作token（fold, check/call等）后探测：
- 线性探针和MLP探针均达到~80%准确率
- 模型学会了将动作与上下文关联

### 激活空间可视化
- **t-SNE**：清晰的牌型聚类，pair和three-of-a-kind形成紧凑区域
- **PCA**：三角形激活结构，类似于Shai et al.描述的信念状态几何
- **UMAP**：部分聚类行为
- 单个牌型（如pair）出现**多个子聚类**，暗示模型学习了更精细的子类别（不同面值的pair）

## 亮点
- **领域扩展的重要性**：首次将LLM涌现世界模型的研究推进到不完全信息游戏，这是概念上的重要一步
- **随机性信念表示的证据**：0.59的equity相关系数虽不算极高，但证明了模型确实在无监督训练中自发发展了概率性估计能力
- **理论-实验的良好配合**：POMDP信念状态线性可解码性的理论证明为探针实验提供了坚实基础
- **信息瓶颈模式**：$R^2$随层深度递减的模式与信息瓶颈理论一致，深层压缩掉了与直接token预测无关的输入变量信息
- **多聚类现象**：pair的多个子聚类暗示模型学习了超越简单分类的丰富内部结构

## 局限与展望
- **数据集规模有限**：合成数据生成计算成本高，数据量影响模型能力和探针结果的上界
- **合成数据的简化性**：AI代理使用简单的equity+启发式策略，不具备人类或复杂博弈agent的适应性学习能力，可能导致数据多样性不足
- **稀有牌型不足**：数据集中顺子、同花等稀有牌型样本不够，无法准确探测这些类别
- **模型规模小**：GPT-2 base (87M)，在更大模型上信念表示可能更强或表现出不同的编码模式
- **equity相关系数有提升空间**：0.59虽有意义但距离高保真信念表示仍有距离
- **缺乏因果干预实验**：仅有探针（被动观测），未像OthelloGPT那样进行激活编辑的因果验证，无法确认发现的表示是否确实被模型决策使用
- **无法在其他POMDP域验证**：结果是否能推广到非扑克的不完全信息域尚不清楚

## 评分
- 新颖性: ⭐⭐⭐⭐ 从完全信息到不完全信息的领域扩展具有重要意义
- 实验充分度: ⭐⭐⭐ 确定性探测充分，随机性探测的相关系数说服力中等，缺乏因果验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰，理论附录扎实，实验呈现系统
- 价值: ⭐⭐⭐⭐ 为理解LLM如何内部建模不确定性打开了新窗口

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

## 相关论文

- [BubbleFormer: Forecasting Boiling with Transformers](bubbleformer_forecasting_boiling_with_transformers.md)
- [Vision Transformers for Cosmological Fields: Application to Weak Lensing Mass Maps](vision_transformers_for_cosmological_fields_application_to_weak_lensing_mass_map.md)
- [nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers](nnterp_a_standardized_interface_for_mechanistic_interpretability_of_transformers.md)
- [LLM Unlearning with LLM Beliefs](../../ICLR2026/human_understanding/llm_unlearning_with_llm_beliefs.md)
- [Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](../../ICCV2025/human_understanding/lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)

<!-- RELATED:END -->
