---
title: >-
  [论文解读] Bounds on Agreement between Subjective and Objective Measurements
description: >-
  [CVPR 2026][quality assessment] 从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。 现有痛点：多媒体质量的客观估计器通常通过与主观"真值"（MOS…
tags:
  - "CVPR 2026"
  - "quality assessment"
  - "MOS"
  - "subjective test"
  - "PCC bound"
  - "MSE bound"
  - "BinoVotes"
---

# Bounds on Agreement between Subjective and Objective Measurements

**会议**: CVPR 2026  
**arXiv**: [2603.13204](https://arxiv.org/abs/2603.13204)  
**代码**: 无  
**领域**: 其他  
**关键词**: quality assessment, MOS, subjective test, PCC bound, MSE bound, BinoVotes

## 一句话总结
从MOS的数学性质出发推导出主观测试结果与任何客观估计器之间PCC上界和MSE下界的理论公式，并提出BinoVotes/BinoMOS投票模型，在18项主观测试数据上验证了界的有效性和模型的准确性。

## 研究背景与动机

**现有痛点**：多媒体质量的客观估计器通常通过与主观"真值"（MOS）的PCC和MSE来评判。但追求PCC=1.0或MSE=0.0既不现实也不可重复——MOS本身因评分尺度离散性、投票人数有限、个人偏差等因素包含固有噪声。

**现有应对方式的不足**：(1) 有些工作提出新的评价指标（如分类错误率、分辨力、ε-insensitive RMSE），但这些指标缺乏统一理论基础；(2) MOS不确定性模型（如高斯分布建模）可能违反MOS的离散性和有限范围约束，clip等修补引入新偏差。

**本文方法的独特性**：不提出新指标，而是为两个最常用的既有指标（PCC、MSE）推导理论界。仅基于"投票期望等于真实质量"这一最基本假设，不需要MOS不确定性模型。界自然反映了MOS的离散性质和对投票数的依赖。

## 方法详解

### 整体框架
这篇论文不提新指标，而是回答一个被长期忽略的问题：在一次主观测试给定的条件下，PCC 究竟最高能到多少、MSE 最低能到多少？整套方法围绕「界由投票方差决定」这一主线展开。第一步借助一个能看见真实质量的 oracle 估计器，把 PCC 上界和 MSE 下界推导出来，二者都只由「投票方差」和「投票人数」决定；第二步用 BinoVotes 二项投票模型给出投票方差的解析公式，第三步把单票模型平均成 BinoMOS、得到与 MOS 数学性质自洽的分布，这两步一起补上「没记录原始投票时方差从哪来」这块短板；最后给出一套按数据丰富度递减的三条计算路径，让这套界能套用到几乎任何已发表的主观测试，并在 18 项跨域主观测试上验证理论界与实测高度吻合。

### 关键设计

**1. PCC 上界与 MSE 下界的推导：把「极限性能」交给能看见真值的 oracle 估计器**

问题在于任何客观估计器都只能逼近带噪的 MOS，而非真实质量 $Y$，所以它的 PCC/MSE 不可能无限好——但好到什么程度才是天花板，此前没有答案。本文的做法是设想一个「最佳可能」的 oracle 估计器，它能直接访问真实质量 $Y$；那么 $Y$ 与 MOS $X$ 之间的 PCC/MSE，就是任何现实估计器能逼近的极限。核心结论是 MSE 下界 $\mathbb{E}(D^2) = \frac{\mathbb{E}(v_r(Y))}{n_v}$，即下界只取决于投票方差的期望 $\mathbb{E}(v_r(Y))$ 与投票人数 $n_v$；PCC 上界同理，借助方差分解 $\text{Var}(X) = \frac{\mathbb{E}(v_r(Y))}{n_v} + \text{Var}(Y)$ 得到。

之所以严格可信，是因为整条推导只用到全期望定律、全方差定律与投票 i.i.d. 假设这些概率论基本工具，唯一的「领域假设」是 $\mathbb{E}(R_j|Y)=Y$——投票期望等于真实质量，这是所有主观测试默认成立的基石。换言之，界不依赖任何额外的 MOS 不确定性建模，自然就反映了 MOS 的离散性和对投票数 $n_v$ 的依赖：投票越多，下界越接近 0、上界越接近 1。

**2. BinoVotes 投票模型：当原始投票方差不可得时，用二项分布给出解析方差**

上面的界要落地，需要知道投票方差 $v_r(Y)$，但很多公开的主观测试只留下了 MOS、没保存逐人投票，方差无从算起。BinoVotes 把单次投票建模成一个平移缩放的二项分布来补上这块：真实质量 $Y \in [s_L, s_H]$ 先归一化为 $p = (Y-s_L)/(s_H-s_L) \in [0,1]$，投票则取 $R = \frac{s_H-s_L}{n_s-1}\text{Binom}(n_s-1, p) + s_L$，于是方差有闭式解 $v_r(Y) = \frac{(s_H-s_L)^2}{(n_s-1)} p(1-p)$。

选二项分布而非高斯，是因为它天然满足评分尺度的离散性和有限值域，不会像高斯那样跑出量纲再靠 clip 修补。更巧的是这个方差是质量的抛物线函数——在「明显好」或「明显差」的两端 $p(1-p)\to0$、投票高度一致，在中间质量处方差最大、投票最分歧，正好对应人类打分的直觉。这样一来，只要知道评分尺度 $n_s$ 和质量水平，就能反推出方差并代入界公式。

**3. BinoMOS 模型：把投票模型平均上去，得到与数学性质自洽的 MOS 分布**

有了单票模型，对 $n_v$ 个 BinoVotes 取平均就得到 MOS 的理论分布 BinoMOS。它的好处是从投票层自然推导，因而继承了 MOS 应有的全部性质：取值是离散的（共 $|M| = n_v(n_s-1)+1$ 个可能值），且随投票数 $n_v$ 增大依中心极限定理收敛到真实质量。相比之下，直接给 MOS 套高斯分布的做法常常违反离散性和有限范围，需要事后裁剪而引入新偏差；BinoMOS 把这些约束「内建」进了模型，无需额外修补。

**4. 界的三种计算路径：从有原始投票到完全没有，都能给出界**

落到实操，论文给出三条按数据丰富度递减的计算路径。最理想是完全数据驱动——主观测试直接提供逐题投票方差 $v_r(Y_i)$，把真值代进界公式即可；其次是借用方差信息——本测试没存方差，就从别的同尺度测试拟合出的经验方差函数借过来；最弱信息下则退到 BinoVotes 模型——仅凭评分尺度和投票人数，用解析方差公式估出界。三条路径让这套界对几乎任何已发表的主观测试都可用。

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

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] What Is the Optimal Ranking Score Between Precision and Recall? We Can Always Find It and It Is Rarely F₁](what_is_the_optimal_ranking_score_between_precision_and_recall_we_can_always_fin.md)
- [\[CVPR 2026\] FEAT: Federated Geometry-Aware Correction for Exemplar Replay under Continual Dynamic Heterogeneity](feat_federated_geometry_aware_correction_for_exemplar_replay_under_continual_dynamic_heterogeneity.md)
- [\[CVPR 2026\] Order Matters: 3D Shape Generation from Sequential VR Sketches](order_matters_3d_shape_generation_from_sequential_vr_sketches.md)
- [\[CVPR 2026\] IrisFP: Adversarial-Example-based Model Fingerprinting with Enhanced Uniqueness and Robustness](irisfp_adversarial-example-based_model_fingerprinting_with_enhanced_uniqueness_a.md)
- [\[CVPR 2026\] Your Classifier Can Do More: Towards Balancing the Gaps in Classification, Robustness, and Generation](your_classifier_can_do_more_towards_balancing_the.md)

</div>

<!-- RELATED:END -->
