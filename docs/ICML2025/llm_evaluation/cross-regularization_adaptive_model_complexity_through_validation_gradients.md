---
title: >-
  [论文解读] Cross-regularization: Adaptive Model Complexity through Validation Gradients
description: >-
  [ICML2025][LLM评测][正则化] 提出 Cross-regularization（交叉正则化），通过验证集梯度直接优化正则化参数（权重范数、噪声尺度、增强强度），在单次训练中收敛到交叉验证最优解，消除手动调参需求。
tags:
  - "ICML2025"
  - "LLM评测"
  - "正则化"
  - "验证梯度"
  - "超参数优化"
  - "噪声注入"
  - "不确定性校准"
  - "数据增强"
---

# Cross-regularization: Adaptive Model Complexity through Validation Gradients

**会议**: ICML2025  
**arXiv**: [2506.19755](https://arxiv.org/abs/2506.19755)  
**代码**: 待确认  
**领域**: 正则化  
**关键词**: 正则化, 验证梯度, 超参数优化, 噪声注入, 不确定性校准, 数据增强

## 一句话总结

提出 Cross-regularization（交叉正则化），通过验证集梯度直接优化正则化参数（权重范数、噪声尺度、增强强度），在单次训练中收敛到交叉验证最优解，消除手动调参需求。

## 研究背景与动机

- **核心痛点**：传统正则化（weight decay、dropout 等）需要手动调超参 $\lambda$，通过交叉验证多次训练才能找到最优值，计算开销大且不灵活
- **已有方法的局限**：
    - 变分 dropout 等方法在训练数据上优化代理目标，而非直接面向泛化性能
    - 基于验证梯度的方法（Maclaurin et al. 2015）需要计算逆 Hessian 或维护完整参数历史，难以扩展到大网络
    - Luketina et al. (2016) 虽然只用最近一步参数更新计算验证梯度，但在两个数据分区上都训练正则化参数，无法保证收敛
    - Population Based Training (PBT) 需要并行训练多个模型，计算成本高
- **关键洞察**：与其通过间接超参 $\lambda$ 控制模型复杂度，不如直接优化正则化参数 $\rho$（如权重范数、噪声尺度），用验证集梯度提供连续的泛化反馈

## 方法详解

### 核心框架：参数与数据的双重分离

将模型参数分为两组，在不同数据集上优化：

- **特征学习参数 $\theta$**：在训练集上通过梯度下降优化
- **正则化参数 $\rho$**：在验证集（称为"正则化集"）上通过梯度下降优化

交替更新规则：

$$\theta_{t+1} = \theta_t - \eta_\theta \nabla_\theta \mathcal{L}_{\text{train}}(\theta_t, \rho_t)$$

$$\rho_{t+1} = \rho_t - \eta_\rho \nabla_\rho \mathcal{L}_{\text{val}}(\theta_{t+1}, \rho_t)$$

### L2 正则化实例化

将权重重参数化为幅度和方向：$\rho = \|w\|_2$，$\theta = w / \|w\|_2$。这样训练集优化方向 $\theta$，验证集优化幅度 $\rho$，等价于寻找最优 ridge 回归解。

**定理 3.1（L2 交叉验证等价性）**：在平滑和强凸条件下，交叉正则化收敛到与最优 ridge 回归相同的解 $\rho^* \theta^* = w_{\text{val}}(\lambda^*)$。

### 梯度分解扩展到任意正则化器

对于 L1 等不可微正则化器，通过将梯度投影到正则化方向和正交方向实现分解：

$$g = g_\rho + g_\perp, \quad g_\rho = \text{Proj}_{\nabla R}(g)$$

- 训练更新沿 $g_\perp$ 方向（保持当前复杂度）
- 验证更新沿 $g_\rho$ 方向（调节正则化强度）

### 随机正则化（噪声注入）

在神经网络的每层注入可学习尺度的噪声：

$$h_l = g(\hat{u}_l + \sigma_l \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

关键设计：训练时用单次噪声采样，验证时用 Monte Carlo 平均（K 次采样）：

$$f_{\text{val}}(x) = \frac{1}{K} \sum_{k=1}^K f(x, \epsilon_k)$$

验证用平均预测而非确定性预测（$\epsilon=0$），否则验证损失与噪声尺度 $\sigma$ 无关，无法优化。

## 实验关键数据

### 经典正则化验证

| 方法 | 任务 | 结果 |
|------|------|------|
| Cross-reg L2 | 合成数据 ridge | 收敛到最优 ridge 回归解 |
| Cross-reg L1 | 糖尿病预测 | 自动发现与 LASSO 最优交叉验证匹配的稀疏度 |
| Cross-reg 样条 | 函数拟合 | 自动学到合适的平滑度 |

### 神经网络噪声正则化 (CIFAR-10)

| 方法 | 准确率 | 说明 |
|------|--------|------|
| Baseline（无正则化） | 76.0% | — |
| Fixed noise ($\sigma=1$, 后5层) | — | 初期阻碍学习，后期仍过拟合 |
| PBT（多模型进化搜索） | 83.7% | 需并行训练多个模型 |
| **Cross-reg** | **83.7%** | 单次训练即达到 PBT 水平 |

### 不确定性校准 (ECE)

| 方法 | ECE ↓ | 准确率 |
|------|-------|--------|
| 未校准模型 | 0.163 | 67.4% |
| Temperature Scaling | 0.057 | 69.6% |
| Fixed Reg | 0.175 | 74.7% |
| Deep Ensemble (5模型) | 0.030 | 81.3% |
| **Cross-reg** | **0.038** | **79.5%** |

### 数据增强 (SVHN)

- 测试准确率从 82.8% 提升到 **86.3%**
- 泛化差距从 16.2% 降至 **7.3%**
- 自动学到：平移 1-2 像素、旋转 3°、剪切约 0

## 亮点与洞察

1. **惊人的高噪声容忍度**：VGG-16 在某些层学到 $\sigma \approx 13$ 的噪声尺度（等价于 dropout rate 99.87%），远超传统认知，但模型仍能正常工作。这与 Lottery Ticket Hypothesis 的发现吻合——VGG 在 CIFAR-10 上可被剪枝到 98%+ 稀疏度
2. **架构感知的正则化模式**：ResNet 的噪声集中在无法被 skip connection 绕过的早期层和最终层（$\sigma_2 = 10.4$），揭示了残差网络的信息流结构
3. **统一框架**：同一个方法可处理 L2/L1 范数正则化、噪声注入、数据增强、不确定性校准，仅需改变 $\rho$ 的定义
4. **计算高效**：相比 PBT 的 $O(PT)$ 前向传播，仅需 $O(T(1+K/r))$，约 10% 额外开销；验证集可小至训练集的 1%
5. **理论保证**：证明了线性收敛（定理 4.1）、与交叉验证等价（定理 4.5）、统计误差仅依赖正则化参数维度 $k$（定理 4.4）

## 局限与展望

1. **实验规模有限**：主要在 CIFAR-10/SVHN + VGG/WideResNet 上验证，缺少 ImageNet 等大规模实验和 Transformer 架构实验
2. **验证集依赖**：需要额外划分验证集用于正则化优化，在小数据场景下可能影响训练数据量（虽然论文说 1% 即可）
3. **理论收敛条件较强**：定理 4.3 假设验证损失对 $\rho$ 强凸，且梯度 Lipschitz 连续，这在深度网络中可能不严格成立
4. **噪声类型受限**：不支持标准 Dropout（不可微），仅支持高斯噪声等连续可微的随机正则化
5. **与现代训练技巧的交互**：未充分探讨与 AdamW、cosine schedule、混合精度等现代训练配置的兼容性
6. **单一作者/实验室**：来自 NightCity Labs，可能缺少大规模算力支持的充分验证

## 相关工作与启发

- **Luketina et al. (2016)**：最接近的前驱工作，也用验证梯度优化超参，但在两个分区都训练正则化参数导致无收敛保证
- **Concrete Dropout (Gal et al. 2017)**：变分推断方法学习 dropout rate，但优化代理目标而非直接面向泛化
- **PBT (Jaderberg et al. 2017)**：进化搜索验证性能最优的超参，但需多模型并行
- **Lottery Ticket Hypothesis**：Cross-reg 发现的高噪声容忍模式与剪枝稀疏度高度一致，提供了不同视角的互证

## 评分

- 新颖性: ⭐⭐⭐⭐ （将验证梯度直接作用于正则化参数的框架设计简洁优雅）
- 实验充分度: ⭐⭐⭐ （覆盖了多种正则化类型，但数据集和架构规模偏小）
- 写作质量: ⭐⭐⭐⭐ （理论与实验的衔接清晰，噪声动态分析有深度）
- 价值: ⭐⭐⭐⭐ （统一框架解决正则化调参痛点，实用性强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Uncertainty Weighted Gradients for Model Calibration](../../CVPR2025/llm_evaluation/uncertainty_weighted_gradients_for_model_calibration.md)
- [\[ICML 2025\] Random Registers for Cross-Domain Few-Shot Learning](random_registers_for_cross-domain_few-shot_learning.md)
- [\[ICML 2025\] CostFilter-AD: Enhancing Anomaly Detection through Matching Cost Filtering](costfilter-ad_enhancing_anomaly_detection_through_matching_cost_filtering.md)
- [\[ACL 2025\] Can External Validation Tools Improve Annotation Quality for LLM-as-a-Judge?](../../ACL2025/llm_evaluation/can_external_validation_tools_improve_annotation_quality_for_llm-as-a-judge.md)
- [\[ACL 2025\] CuLEmo: Cultural Lenses on Emotion - Benchmarking LLMs for Cross-Cultural Emotion Understanding](../../ACL2025/llm_evaluation/culemo_cultural_lenses_on_emotion_-_benchmarking_llms_for_cross-cultural_emotion.md)

</div>

<!-- RELATED:END -->
