---
title: >-
  [论文解读] Test-time Diverse Reasoning by Riemannian Activation Steering
description: >-
  [AAAI 2026][LLM评测][激活引导] 提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。 Best-of-N 推理策略通过采样 N…
tags:
  - "AAAI 2026"
  - "LLM评测"
  - "激活引导"
  - "推理多样性"
  - "黎曼优化"
  - "Best-of-N 采样"
  - "语言模型推理"
  - "流形优化"
  - "数学推理"
---

# Test-time Diverse Reasoning by Riemannian Activation Steering

**会议**: AAAI 2026  
**arXiv**: [2511.08305](https://arxiv.org/abs/2511.08305)  
**代码**: [https://github.com/lythk88/SPREAD](https://github.com/lythk88/SPREAD)  
**领域**: LLM评测  
**关键词**: 激活引导, 推理多样性, 黎曼优化, Best-of-N 采样, 语言模型推理, 流形优化, 数学推理  

## 一句话总结

提出 SPREAD 框架——一种无监督的测试时激活引导策略，通过在球面流形乘积上求解黎曼优化问题来最大化多条推理路径的隐藏激活张成的总体积，从而提升 Best-of-N 采样中的推理多样性和准确率，在数学推理基准上超越温度采样基线。

## 研究背景与动机

Best-of-N 推理策略通过采样 N 个候选答案再选最优来提升 LLM 在复杂任务上的准确率，但其效果受限于**输出多样性上限**：模型即使在随机采样下仍倾向生成高度相似的推理路径（多样性坍缩），反复犯相同错误。

现有提升多样性的方法面临三大挑战：

**随机解码方法**（温度采样、top-k、nucleus）：在 token 级引入随机性，但生成的推理路径常收敛到近似相同的推理链

**搜索策略**（对比搜索、diverse beam search）：联合考虑轨迹分布，计算成本高

**多样性度量难题**：词汇/语义多样性对文本长度、同义改写敏感，且需要额外神经网络计算

核心观察：隐藏激活（hidden activations）是模型的内部"思维空间"，不同的激活簇往往对应不同的"推理电路"。因此，**促进激活多样性可以诱导推理多样性**。

## 方法详解

### 整体框架

SPREAD（SPherical intervention for REAsoning Diversity）在自回归生成过程中，于指定的同步锚点位置提取所有 N 条生成路径的最后一个 token 的隐藏状态向量 $H = [h_1, ..., h_N] \in \mathbb{R}^{p \times N}$，计算加性引导向量 $V = [v_1, ..., v_N]$，使得干预后的激活 $H_{new} = H + V$ 在几何上尽可能"分散"。引导向量持续作用直到下一个同步锚点。

### 关键设计 1：体积最大化目标

SPREAD 最大化所有可能子集对应的平行六面体体积之和的平方：

$$\max_{\forall i: \|v_i\|_2^2 \leq \alpha_i} \sum_{\mathbb{I} \in 2^{[N]}} \text{Volume}(\mathcal{P}(\{h_i + v_i\}_{i \in \mathbb{I}}))^2$$

通过 Gram 矩阵行列式等价性将其转化为 log-det 最小化问题：

$$\min_{V} -\log\det[I + (H+V)^\top(H+V)]$$

约束 $\|v_i\|_2^2 = \alpha_i$（论文证明不等式约束在最优解处必然紧）。$\alpha_i = C \|h_i\|_2 / p$ 使得超参数缩减为单一的相对参数 $C > 0$。

**不计算所有 $2^N$ 个子集**：利用 DPP (Determinantal Point Process) 的性质，所有子集体积之和等于一个 $(N+1) \times (N+1)$ 矩阵的行列式。

### 关键设计 2：黎曼块坐标下降

等式约束 $\|v_i\|_2^2 = \alpha_i$ 将每个引导向量限制在半径为 $\sqrt{\alpha_i}$ 的球面上，产品流形 $\mathcal{M} = \mathcal{M}_1 \times ... \times \mathcal{M}_N$。算法：

1. 初始化：$v_i^{(0)} = \sqrt{\alpha_i} \cdot (h_i + \varepsilon_i - \bar{h}) / \|h_i + \varepsilon_i - \bar{h}\|_2$（减去均值 + 小噪声扰动）
2. 每次迭代对 $i = 1,...,N$ 依次更新：
    - 计算欧几里得梯度 $g_i$，投影到球面切空间得到黎曼下降方向 $d_i$
    - 通过指数映射沿测地线移动：$v_i^{(k)} = \cos(\cdot) v_i^{(k-1)} + \sin(\cdot) \frac{d_i}{\|d_i\|_2} \sqrt{\alpha_i}$

步长 $\eta_i = 1/L_i$ 由块光滑度常数确定（论文给出解析表达式），**无需调参**。

### 关键设计 3：收敛保证

论文证明算法以 $O(1/\sqrt{k})$ 的速率收敛到稳定点：

$$\min_{s \leq k} \|\text{grad}\ \ell(V^{(s)})\|_F \leq \sqrt{\frac{\ell(V^{(0)}) - \ell^*}{Ck}}$$

尽管目标函数非凸（论文通过反例证明），但 Riemannian 块坐标下降的性质保证任何极限点都是稳定点。

### 损失函数 / 训练策略

SPREAD 是纯推理时方法，**无需训练或微调**。不修改模型参数，仅通过加性干预 $\tilde{x}^{(l+1)} = x^{(l+1)} + v^{(l+1)}$ 作用于指定层的残差流。实验中在第 28 层（最后一层）应用引导，同步锚点设在 token 位置 $\tau \in \{100, 600, 1100, 1600\}$，迭代 $K=20$ 次。

## 实验

### 主实验表：Pass@N 准确率 (%)

| 模型 | Temp | AIME24 (SPREAD C=1) | AIME24 (Sampling) | MATH500 (SPREAD C=1) | MATH500 (Sampling) | OlympiadBench (SPREAD C=1) | OlympiadBench (Sampling) |
|------|:----:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | 1.0 | **3.3** | 0.0 | **43.2** | 42.8 | **21.5** | 19.0 |
| Qwen2.5-1.5B | 0.6 | **10.0** | 3.3 | **55.0** | 53.4 | 28.4 | 29.3 |
| Math-1.5B-Inst | 1.0 | **26.7** | 20.0 | 83.8 | 84.6 | 47.6 | 48.3 |
| Math-1.5B-Inst | 0.6 | **26.7** | 20.0 | **85.4** | 84.6 | 50.4 | 50.8 |

SPREAD 在绝大多数配置下表现优于或持平温度采样。

### 多样性指标表：Unique Solution Count

| 模型 | Temp | AIME24 (SPREAD C=1) | AIME24 (Sampling) | MATH500 (SPREAD C=1) | MATH500 (Sampling) |
|------|:----:|:---:|:---:|:---:|:---:|
| Qwen2.5-1.5B | 1.0 | **6.97** | 6.67 | **3.14** | 3.14 |
| Qwen2.5-1.5B | 0.8 | **6.83** | 3.60 | **3.03** | 2.97 |
| Math-1.5B-Inst | 1.0 | **6.63** | 3.67 | 1.92 | 1.93 |
| Math-1.5B-Inst | 0.8 | **6.63** | 3.47 | 1.87 | 1.89 |

SPREAD 在 AIME24 上显著提升唯一解数量（6.97 vs 6.67 或 3.67），表明其确实诱导了不同的推理路径。

### 关键发现

1. **假设验证**：统计假设检验（逻辑回归 + 聚类鲁棒标准差）证实激活体积与答案多样性正相关（$\hat{\beta}=0.88, p=0.001$），每增加一单位体积，产生独特解的 odds 比约 2.4 倍
2. **计算效率**：$N=32, p=16384$（对应 LLaMA-405B 级别）时算法运行时间仅 <1.8 秒
3. **层选择**：在最后一层（第 28 层）应用引导效果最佳；浅层引导效果不稳定
4. **超参数简洁**：实际仅需设置一个参数 $C$（引导强度的相对系数），$C=1$ 在多数情况下最优
5. **Pareto 前沿优势**：在准确率-多样性双轴图上，SPREAD 的点一致优于温度采样

## 亮点

- **数学基础扎实**：从体积最大化→log-det→球面流形优化的推导链条严谨，提供了收敛性理论保证
- **极致轻量**：不修改模型权重、不需额外神经网络、不需正负样本对；纯几何操作
- **无需对比样本**：不像传统激活引导需要正/负行为对比，SPREAD 是完全无监督的
- **首次将激活引导应用于数学推理**：传统激活引导避开数学推理（因为难以定义正负样本），SPREAD 绕开了这一限制
- **DPP 与几何的优雅连接**：利用行列式点过程的性质将指数级子集求和压缩为单次行列式计算

## 局限性

1. 激活多样性与推理多样性之间不存在一一对应关系——增加激活多样性不保证推理路径真正不同
2. 仅在 1.5B 参数模型上验证，未测试 7B+ 量级模型
3. 数学推理的进步幅度（几个百分点）在某些配置下处于噪声范围
4. 同步锚点位置（100, 600, 1100, 1600）的选择看似启发式，缺乏自适应策略
5. 多样性评估依赖 GPT-4.1-mini 作为 judge，可能引入评估偏差

## 相关工作

- **Activation Steering** (Turner et al., 2023): 对比式引导——用正负行为差异平均计算引导向量，不适用于数学推理
- **Inference-Time Intervention** (Li et al., 2023): 基于探针的引导，需要训练分类器来识别概念
- **Diverse Beam Search** (Vijayakumar et al., 2016): 在 beam 层面引入多样性，计算开销大
- **WLD-Reg** (Laakom et al., 2023): 层内激活多样性正则化用于训练而非推理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （黎曼流形优化用于推理多样性引导，构思独特且理论完整）
- 实验充分度: ⭐⭐⭐ （三个 benchmark 但仅 1.5B 模型，部分改进幅度较小）
- 写作质量: ⭐⭐⭐⭐⭐ （数学推导清晰严谨，图示直观）
- 价值: ⭐⭐⭐⭐ （提供了推理时多样性增强的新视角，轻量级方案具有实用潜力）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GuidedSampling: Steering LLMs Towards Diverse Candidate Solutions at Inference-Time](../../ICLR2026/llm_evaluation/guidedsampling_steering_llms_towards_diverse_candidate_solutions_at_inference-ti.md)
- [\[ICML 2026\] Spherical Steering: Geometry-Aware Activation Rotation for Language Models](../../ICML2026/llm_evaluation/spherical_steering_geometry-aware_activation_rotation_for_language_models.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[AAAI 2026\] OptScale: Probabilistic Optimality for Inference-time Scaling](optscale_probabilistic_optimality_for_inference-time_scaling.md)
- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](../../ICCV2025/llm_evaluation/batclip_bimodal_online_test-time_adaptation_for_clip.md)

</div>

<!-- RELATED:END -->
