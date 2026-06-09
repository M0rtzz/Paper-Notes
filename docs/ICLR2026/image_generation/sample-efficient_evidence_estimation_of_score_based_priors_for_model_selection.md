---
title: >-
  [论文解读] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection
description: >-
  [ICLR 2026][图像生成][模型证据] 提出 DiME，一种沿扩散后验时间边缘积分的模型证据估计器，无需先验评分或密度评估，仅用少量后验样本（如 20 个）即可准确估计扩散模型先验下的模型证据，用于先验选择和模型验证。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "模型证据"
  - "扩散先验"
  - "后验采样"
  - "模型选择"
  - "黑洞成像"
---

# Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection

**会议**: ICLR 2026  
**arXiv**: [2602.20549](https://arxiv.org/abs/2602.20549)  
**代码**: —  
**领域**: 贝叶斯推断 / 扩散模型  
**关键词**: 模型证据, 扩散先验, 后验采样, 模型选择, 黑洞成像

## 一句话总结

提出 DiME，一种沿扩散后验时间边缘积分的模型证据估计器，无需先验评分或密度评估，仅用少量后验样本（如 20 个）即可准确估计扩散模型先验下的模型证据，用于先验选择和模型验证。

## 研究背景与动机

在贝叶斯逆问题中，先验分布 $p(\boldsymbol{x})$ 对后验 $p(\boldsymbol{x}|\boldsymbol{y})$ 有决定性影响。选择一个不合适的先验会导致重建结果严重偏差。理想的做法是通过模型证据 $p(\boldsymbol{y}|M)$ 评估不同先验模型。

然而，对于扩散模型先验，直接计算模型证据是不可行的：
- 需要对完整先验积分 $\log p(\boldsymbol{y}|M) = \log \int p(\boldsymbol{y}|\boldsymbol{x}) p(\boldsymbol{x}|M) d\boldsymbol{x}$
- 现有方法（SMC、AIS、嵌套采样）要求干净先验得分 $\nabla_{\boldsymbol{x}} \log p(\boldsymbol{x})$ 或非归一化密度
- 扩散模型学习的是中间噪声先验的得分，干净先验得分不准确
- 密度估计方法方差高，需数千个后验样本

## 方法详解

### 整体框架

DiME 的核心思路是绕开"先验得分/密度不可得"这一拦路虎，把对数证据 $\log p(\boldsymbol{y})$ 拆成只依赖后验样本就能算的两项：一项是后验下的平均对数似然 $\mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{y})}[\log p(\boldsymbol{y}|\boldsymbol{x}_0)]$，另一项是后验与先验之间的 KL 散度 $D_{\text{KL}}(p(\boldsymbol{x}_0|\boldsymbol{y}) \| p(\boldsymbol{x}_0))$。关键观察是后一项并不需要先验本身——它可以沿逆向扩散的时间边缘积分写成 $D_{\text{KL}} \approx \sum_{i=1}^N c_{t_i} \Delta t_i \mathbb{E}_{\boldsymbol{x}_{t_i} \sim p(\boldsymbol{x}_{t_i}|\boldsymbol{y})} \|\nabla_{\boldsymbol{x}_{t_i}} \log p(\boldsymbol{y}|\boldsymbol{x}_{t_i})\|^2$（系数 $c_{t_i} = \sigma_{t_i}' \sigma_{t_i} - \sigma_{t_i}^2 \frac{a_{t_i}'}{a_{t_i}}$），从而把证据估计转化为对"逐时间步的似然得分平方"的积分。整套估计器与 DAPS 后验采样器协同运行，复用采样轨迹上自然产生的中间样本 $\boldsymbol{x}_{t_i}$，几乎不增加额外计算。

### 关键设计

**1. 无偏似然得分估计：用后验回采的干净样本绕开不可算的得分项。** 时间边缘积分里要的 $\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)$ 无法直接求值。DiME 转而利用 DAPS 在每个噪声层自然采出的后验干净样本 $\tilde{\boldsymbol{x}}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t, \boldsymbol{y})$，构造两个互补的无偏估计器：高噪声端方差更低的 $\Theta_{\text{high}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}(\tilde{\boldsymbol{x}}_0 - \mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t])$，以及低噪声端方差更低的 $\Theta_{\text{low}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} \nabla_{\tilde{\boldsymbol{x}}_0} \log p(\boldsymbol{y}|\tilde{\boldsymbol{x}}_0)$，按噪声水平自动切换以始终保持低方差。由于积分里要的是得分的平方，直接用单个样本的平方会引入偏差，因此对每个 $\boldsymbol{x}_t$ 独立采样两份 $\tilde{\boldsymbol{x}}_0^{(1)}, \tilde{\boldsymbol{x}}_0^{(2)}$ 再相乘，得到对平方得分的无偏估计。

**2. 改进的后验协方差：修掉高噪声端的方差高估。** 上述低噪声估计器依赖条件协方差 $\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t}$，而 DAPS 原本用的启发式 $\sigma_t^2$ 在高噪声时会把方差严重高估，导致估计偏差。DiME 引入一个带先验信息的近似 $\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} = \left[\boldsymbol{\Sigma}_0^{-1} + \frac{a_t^2}{\sigma_t^2}\mathbf{I}\right]^{-1}$，其中先验协方差 $\boldsymbol{\Sigma}_0$ 从训练数据经验估计。这相当于在高噪声端用先验把过宽的协方差"收紧"，从而在不增加样本的前提下显著压低偏差。

## 实验

### 高斯混合先验基准测试

| 方法 | 分布内 $\boldsymbol{x}^*$ 相对误差↓ | 分布外相对误差↓ | 鞍点处相对误差↓ |
|------|------|------|------|
| Naive MC (1000) | 2451% | 2357% | 2299% |
| 原始 DAPS 启发式 | 146% | 3.3% | 7.3% |
| TI | 3.2% | 5.6% | 1.2% |
| SMC | 2.6% | 1.2% | **0.7%** |
| **DiME** | **1.5%** | **0.6%** | 0.8% |

DiME 在不使用先验得分的情况下达到与 SMC 可比的精度。

### MNIST 模型选择

给定单个噪声测量值，从 10 个扩散模型中选择正确先验。DiME 一致选出正确的数字类别，而基线方法失败。

### M87* 黑洞成像

- DiME 表明 GRMHD 先验比 RIAF、空间图像、人脸和 MNIST 先验的似然更高
- 先验预测检验表明 M87* 观测与 GRMHD 先验统计相容

### 关键发现

- 仅需 20 个后验样本即可获得准确估计
- 高/低噪声估计器的自动切换策略有效降低方差
- 改进的协方差近似在高噪声时显著减少偏差
- DiME 可推广到任意退火路径下的模型证据估计

## 亮点

- 首个不依赖先验得分或密度的扩散模型证据估计器
- 样本效率极高（20 个样本 vs 基线方法需要数千个）
- 理论推导优雅，利用了扩散采样中自然产生的中间样本
- 真实科学应用（黑洞成像）验证了方法的实用价值

## 局限性

- 依赖高斯近似 $p(\boldsymbol{x}_0|\boldsymbol{x}_t) \approx \mathcal{N}$，在多模态先验下可能不准确
- 与特定后验采样方法（DAPS）耦合，泛化到其他方法需要额外推导
- 对角协方差近似在复杂高维问题中精度有限
- 估计器的方差随问题维度增加可能增大

## 相关工作

- **证据估计**：SMC、AIS、嵌套采样、调和均值估计器等
- **扩散后验采样**：DAPS、DPS、PnP-DM 等方法
- **模型选择**：贝叶斯因子、交叉验证等替代框架

## 评分

- 新颖性：⭐⭐⭐⭐⭐ — 完全新的扩散证据估计范式
- 理论性：⭐⭐⭐⭐⭐ — 推导严格，多个引理支撑
- 实验：⭐⭐⭐⭐ — 从玩具到真实科学应用的全面验证
- 实用性：⭐⭐⭐⭐ — 对科学成像和模型选择有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICML 2026\] DiScoFormer: Plug-In Density and Score Estimation with Transformers](../../ICML2026/image_generation/discoformer_plug-in_density_and_score_estimation_with_transformers.md)
- [\[ICLR 2026\] Monocular Normal Estimation via Shading Sequence Estimation](monocular_normal_estimation_via_shading_sequence_estimation.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[CVPR 2025\] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](../../CVPR2025/image_generation/diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)

</div>

<!-- RELATED:END -->
