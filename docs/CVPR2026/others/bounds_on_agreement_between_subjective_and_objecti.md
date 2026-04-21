---
title: >-
  [论文解读] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR 2026][quality assessment] 从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。
tags:
  - CVPR 2026
  - quality assessment
  - MOS
  - subjective test
  - PCC bound
  - MSE bound
  - BinoVotes
---

# Bounds on Agreement between Subjective and Objective Measurements

**会议**: CVPR 2026  
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)  
**代码**: 无  
**领域**: 多媒体质量评估 / 信号处理  
**关键词**: quality assessment, MOS, subjective test, PCC bound, MSE bound, BinoVotes

## 一句话总结
从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。

## 研究背景与动机

**客观估计器评估的困境**：多媒体质量的客观估计器通常通过与主观"真值"（MOS）的PCC和MSE来评判。但追求PCC=1.0或MSE=0.0既不现实也不可重复——MOS本身因评分尺度离散性、投票人数有限、个人偏差等因素包含固有噪声。

**现有应对方式的不足**：(1) 有些工作提出新的评价指标（如分类错误率、分辨力、ε-insensitive RMSE），但这些指标缺乏统一理论基础；(2) MOS不确定性模型（如高斯分布建模）可能违反MOS的离散性和有限范围约束，clip等修补引入新偏差。

**本文方法的独特性**：不提出新指标，而是为两个最常用的既有指标（PCC、MSE）推导理论界。仅基于"投票期望等于真实质量"这一最基本假设，不需要MOS不确定性模型。界自然反映了MOS的离散性质和对投票数的依赖。

## 方法详解

### 整体框架
三步推进：(1) 推导PCC和MSE的理论界（基于投票方差）；(2) 提出BinoVotes投票模型提供投票方差估计（当实际方差不可用时）；(3) 在18项主观测试数据上验证界的有效性。

### 关键设计

1. **PCC上界和MSE下界的推导**：
    - 功能：推导任何客观估计器在给定主观测试条件下能达到的最优PCC和最低MSE
    - 核心思路：考虑最佳可能的客观估计器——即oracle估计器能直接访问真实质量 $Y$。则 $Y$ 与 MOS $X$ 的PCC/MSE就是任何估计器能达到的极限。关键结果：$\mathbb{E}(D^2) = \frac{\mathbb{E}(v_r(Y))}{n_v}$，MSE下界仅取决于投票方差的期望 $\mathbb{E}(v_r(Y))$ 和投票人数 $n_v$。PCC上界类似，利用 $\text{Var}(X) = \frac{\mathbb{E}(v_r(Y))}{n_v} + \text{Var}(Y)$
    - 设计动机：基于概率论基础（全期望定律、全方差定律、i.i.d.votes假设），推导严格且假设最少。"well-behaved"条件 $\mathbb{E}(R_j|Y)=Y$ 是所有主观测试的基石假设

2. **BinoVotes投票模型**：
    - 功能：当实际投票方差不可用时，提供投票分布和方差的理论估计
    - 核心思路：将投票建模为变换后的二项分布——真实质量 $Y \in [s_L, s_H]$ 映射到 $p = (Y-s_L)/(s_H-s_L) \in [0,1]$，投票分布为 $R = \frac{s_H-s_L}{n_s-1}\text{Binom}(n_s-1, p) + s_L$。BinoVotes的方差 $v_r(Y) = \frac{(s_H-s_L)^2}{(n_s-1)} p(1-p)$——在质量范围两端为0，中间最大
    - 设计动机：二项分布天然满足评分尺度的离散性和有限范围约束。BinoVotes的方差是真实质量的抛物线函数——在极端质量（如"明显好"或"明显差"）时投票一致性高（方差小），在中间质量时投票分歧大（方差大），符合直觉

3. **BinoMOS模型**：
    - 功能：对BinoVotes取平均得到MOS的理论分布模型
    - 核心思路：$n_v$ 个BinoVotes的平均，自然继承了MOS的离散值域——$|M| = n_v(n_s-1)+1$ 个可能取值——和对投票数的依赖。随 $n_v$ 增大，BinoMOS趋近真实质量（中心极限定理）
    - 设计动机：直接对MOS建模的方法（如高斯分布）往往违反MOS的离散性。BinoMOS从投票模型自然推导，保持了所有数学性质

### 界的三种计算方法
1. **完全数据驱动**：主观测试直接提供投票方差 $v_r(Y_i)$ → 直接代入界公式
2. **借用方差信息**：从其他提供方差的主观测试中借用经验方差函数
3. **BinoVotes模型**：用BinoVotes的理论方差公式估计 → 仅需知道评分尺度和投票人数

## 实验关键数据

### 主实验（18项主观测试的PCC/MSE界验证）

| 验证方式 | PCC上界吻合度 | MSE下界吻合度 |
|---------|-------------|-------------|
| 数据驱动 vs BinoVotes | 非常接近 | 非常接近 |
| 投票多的测试 (n_v大) | 界接近1.0/0.0 | 界小 |
| 投票少的测试 (n_v小) | 界明显<1.0 | 界较大 |

跨18项测试（涵盖语音、音频、视频质量评估），BinoVotes模型产生的界与完全数据驱动的界高度一致——验证了BinoVotes作为投票方差近似的有效性。

### 消融实验

| 影响因素 | 对MSE下界的影响 | 对PCC上界的影响 |
|---------|---------------|---------------|
| 增加投票数 $n_v$ | 下界降低 → 更紧 | 上界升高 → 更接近1 |
| 质量分布集中vs均匀 | 均匀时界更紧 | 取决于 $\text{Var}(Y)$ |
| 评分尺度 $n_s$ | $n_s$越大投票方差越大 | 界更松 |

### 关键发现
- 当投票数 $n_v$ 较少时（如4-10票），MSE下界可达0.1-0.3（5分制），意味着客观估计器达到MSE=0.1已接近理论极限
- BinoVotes假设投票方差是质量的抛物线函数，这与真实投票方差的经验分布高度吻合
- 4项无投票方差信息的主观测试也可通过BinoVotes模型给出合理的界
- 界的计算仅需知道投票数和评分尺度——无需其他复杂假设

## 亮点与洞察
- **理论贡献**：首次为PCC和MSE推导了仅基于MOS内在数学性质的严格界，完全避免了模型化MOS不确定性的需要
- BinoVotes模型的简洁性和有效性令人印象深刻——二项分布天然匹配评分尺度的离散性
- 实用价值：任何客观质量估计器的开发者都可用这些界判断"还有多少提升空间"或"已经达到理论极限"
- 显示了per-subject bias在数学上等价于增加投票方差——不需要显式建模偏差

## 局限与展望
- i.i.d.投票假设在某些场景下可能过强（如疲劳效应、顺序效应）
- BinoVotes模型假设方差是质量的对称抛物线，真实分布可能不对称
- 仅验证了"overall quality"评分，其他属性（如noisiness、sharpness）的界可能不同
- 未讨论非线性映射后的界变化——实际中PLCC常在映射后计算

## 相关工作与启发
- **MOS不确定性研究**：大量前期工作建模MOS噪声（alpha-stable、高斯、混合高斯），本文提供了更基础的替代方案
- **ε-insensitive RMSE**：试图容忍主观噪声的评价指标，本文的界提供了更精确的容忍基准
- **客观质量评估领域**：POLQA、VISQOL、VMAF等估计器的开发者可直接使用这些界
- 启发：在任何依赖"有噪声真值"评估模型的场景中（如LLM评分、crowd-sourced标注），类似的界推导可提供有价值的性能预期

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次从MOS数学性质推导PCC/MSE界，BinoVotes模型简洁有效
- 实验充分度: ⭐⭐⭐⭐ 18项跨域主观测试的大规模验证，覆盖语音/音频/视频
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，符号一致，逻辑链清晰
- 价值: ⭐⭐⭐⭐ 为多媒体质量评估研究社区提供了基础性理论工具

<!-- RELATED:START -->

## 相关论文

- [What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [ELogitNorm: Enhancing OOD Detection with Extended Logit Normalization](enhancing_outofdistribution_detection_with_extende.md)
- [Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [Deconstructing the Failure of Ideal Noise Correction: A Three-Pillar Diagnosis](deconstructing_the_failure_of_ideal_noise_correcti.md)

<!-- RELATED:END -->
