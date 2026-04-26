---
title: >-
  [论文解读] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection
description: >-
  [ICLR 2026][图像生成][模型证据] 提出 DiME，一种沿扩散后验时间边缘积分的模型证据估计器，无需先验评分或密度评估，仅用少量后验样本（如 20 个）即可准确估计扩散模型先验下的模型证据，用于先验选择和模型验证。
tags:
  - ICLR 2026
  - 图像生成
  - 模型证据
  - 扩散先验
  - 后验采样
  - 模型选择
  - 黑洞成像
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

### 核心公式

**DiME 估计器**（沿标准边缘）：

$$\log p(\boldsymbol{y}) = \mathbb{E}_{\boldsymbol{x}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{y})}[\log p(\boldsymbol{y}|\boldsymbol{x}_0)] - D_{\text{KL}}(p(\boldsymbol{x}_0|\boldsymbol{y}) \| p(\boldsymbol{x}_0))$$

KL 散度通过逆向扩散的时间边缘积分估计：

$$D_{\text{KL}} \approx \sum_{i=1}^N c_{t_i} \Delta t_i \mathbb{E}_{\boldsymbol{x}_{t_i} \sim p(\boldsymbol{x}_{t_i}|\boldsymbol{y})} \|\nabla_{\boldsymbol{x}_{t_i}} \log p(\boldsymbol{y}|\boldsymbol{x}_{t_i})\|^2$$

其中 $c_{t_i} = \sigma_{t_i}' \sigma_{t_i} - \sigma_{t_i}^2 \frac{a_{t_i}'}{a_{t_i}}$。

### 关键创新一：无偏似然得分估计

直接计算 $\nabla_{\boldsymbol{x}_t} \log p(\boldsymbol{y}|\boldsymbol{x}_t)$ 不可行，但利用 DAPS 的后验样本 $\tilde{\boldsymbol{x}}_0 \sim p(\boldsymbol{x}_0|\boldsymbol{x}_t, \boldsymbol{y})$ 设计两个无偏估计器：

**高噪声估计器**（高噪声时方差低）：

$$\Theta_{\text{high}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}(\tilde{\boldsymbol{x}}_0 - \mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t])$$

**低噪声估计器**（低噪声时方差低）：

$$\Theta_{\text{low}}(\tilde{\boldsymbol{x}}_0) = \frac{a_t}{\sigma_t^2}(\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} \nabla_{\tilde{\boldsymbol{x}}_0} \log p(\boldsymbol{y}|\tilde{\boldsymbol{x}}_0))$$

对每个 $\boldsymbol{x}_t$ 采样两个独立的 $\tilde{\boldsymbol{x}}_0^{(1)}, \tilde{\boldsymbol{x}}_0^{(2)}$ 获得无偏的平方得分估计。

### 关键创新二：改进的后验协方差

DAPS 的协方差启发式 $\sigma_t^2$ 在高噪声时高估方差。DiME 引入先验近似：

$$\boldsymbol{\Sigma}_{\boldsymbol{x}_0|\boldsymbol{x}_t} = \left[\boldsymbol{\Sigma}_0^{-1} + \frac{a_t^2}{\sigma_t^2}\mathbf{I}\right]^{-1}$$

其中 $\boldsymbol{\Sigma}_0$ 从训练数据经验估计。

### 实现

DiME 与 DAPS 后验采样方法协同运行，利用采样过程中自然产生的中间样本，无需额外计算。

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

## 相关论文

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICLR 2026\] Monocular Normal Estimation via Shading Sequence Estimation](monocular_normal_estimation_via_shading_sequence_estimation.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[CVPR 2025\] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows](../../CVPR2025/image_generation/diverseflow_sample-efficient_diverse_mode_coverage_in_flows.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)

<!-- RELATED:END -->
