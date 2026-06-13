---
title: >-
  [论文解读] Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models
description: >-
  [ICCV 2025][LLM 其他][大语言模型持续学习] 提出Analytic Subspace Routing（Any-SSR）框架，通过为每个任务分配独立的LoRA子空间消除任务间干扰，并利用递归最小二乘（RLS）闭式解训练一个零遗忘的解析路由器，实现LLM的无回放持续学习。 LLM持续学习的挑战 LLM在动态真实环…
tags:
  - "ICCV 2025"
  - "LLM 其他"
  - "大语言模型持续学习"
  - "灾难性遗忘"
  - "递归最小二乘"
  - "LoRA子空间路由"
  - "无回放学习"
---

# Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Models

**会议**: ICCV 2025  
**arXiv**: N/A (CVF OpenAccess)  
**代码**: [https://github.com/ZHUANGHP/Any-SSR](https://github.com/ZHUANGHP/Any-SSR)  
**领域**: LLM/NLP  
**关键词**: 大语言模型持续学习, 灾难性遗忘, 递归最小二乘, LoRA子空间路由, 无回放学习

## 一句话总结

提出Analytic Subspace Routing（Any-SSR）框架，通过为每个任务分配独立的LoRA子空间消除任务间干扰，并利用递归最小二乘（RLS）闭式解训练一个零遗忘的解析路由器，实现LLM的无回放持续学习。

## 研究背景与动机

### LLM持续学习的挑战

LLM在动态真实环境中需要不断吸收新领域知识，但微调过程本质上是一个持续学习过程，不可能重新访问所有预训练数据。这导致了**灾难性遗忘**——模型在接受新任务时快速丢失先前学习的知识。对于LLM而言，这个问题尤为严重，因为高维参数空间中编码的通用能力很容易被微调破坏。

### 现有方法的根本缺陷

**为什么现有方法不够好？**

**回放方法**（如Replay、SEEKR）：需要存储和回放历史数据，计算成本高且存在隐私风险，对于LLM海量预训练数据不切实际

**参数高效微调方法**（如LoRAMoE、O-LoRA）：使用固定数量的共享参数吸收所有任务知识，不同任务在共享模块上的顺序微调导致灾难性遗忘

**正则化方法**（如EWC）：在参数空间施加约束，但在LLM的巨大参数空间中确定参数重要性仍然困难

### 核心洞察

作者观察到现有PET方法的核心问题是任务间的**知识干扰**——所有任务共享同一参数空间。受皮层系统层次化处理的启发，提出假设：Transformer的低层编码跨任务的语义特征，高层处理任务特定的语义组合。基于此，可以冻结低层保持通用能力，在高层为每个任务分配独立的LoRA子空间。

## 方法详解

### 整体框架

Any-SSR的架构包含三个核心组件：

1. **冻结的通用特征提取器**：LLM的前$L_f$层完全冻结
2. **任务特定的LoRA Bank**：后续层中为每个任务维护独立的LoRA适配器
3. **递归解析学习（AL）路由器**：基于RLS闭式解的任务路由器

前向传播过程：
$$y_{t+1} = h_{\leq L_f}(x) \cdot f_{\theta_{k^*}}(h_{>L_f}(x))$$

其中$k^* = \arg\max_k g_k(h_{\leq L_f}(x))$是路由器选择的任务ID。

### 关键设计一：层次化特征解耦

**为什么要分层？** 基于"低层编码通用语义、高层处理任务特定信息"的假设，将预训练LLM的参数划分为：

- **冻结的低层**$h_{\leq L_f}$：保持预训练获得的通用语言理解能力
- **可适配的高层**$h_{>L_f}$：为每个新任务插入独立的LoRA适配器

每个任务$D_k$的LoRA通过低秩分解更新权重：
$$\Delta W_l^{(k)} = B_l^{(k)} A_l^{(k)}, \quad B_l^{(k)} \in \mathbb{R}^{d_{in} \times r}, A_l^{(k)} \in \mathbb{R}^{r \times d_{out}}$$

**为什么不同任务不会互相干扰？** 所有任务的LoRA适配器在冻结基础参数上独立训练，参数空间完全不相交。

### 关键设计二：解析路由机制

路由器的核心任务是处理低层特征$h_{\leq L_f}(x)$来预测任务归属分布$p(k|x)$，并且在引入新任务时无需访问历史数据就能更新权重。

**特征扩展**：首先将特征映射到高维空间以增强线性可分性（基于Cover定理）：
$$\tilde{h} = \phi(\text{mean-pool}(h_{\leq L_f}(X))) \in \mathbb{R}^E$$

其中$\phi: \mathbb{R}^d \rightarrow \mathbb{R}^E$是高斯初始化的固定投影（$E > d$），通过ReLU激活的线性变换实现。

**岭回归闭式解**：路由器权重通过求解一个凸优化问题获得：
$$\hat{W}_k^r = \left(\sum_{i=1}^{k} \tilde{h}_i^\top \tilde{h}_i + \lambda I\right)^{-1} \left(\sum_{i=1}^{k} \tilde{h}_i^\top Y_i\right)$$

其中$\lambda > 0$是正则化系数，$Y_i$是任务ID标签。

### 关键设计三：递归增量更新

**这是方法最核心的创新。** 定义自相关矩阵$R_k$和互相关矩阵$Q_k$：

$$R_k = \left(\sum_{i=1}^{k} \tilde{h}_i^\top \tilde{h}_i + \lambda I\right)^{-1}, \quad Q_k = \sum_{i=1}^{k} \tilde{h}_i^\top Y_i$$

当新任务$D_{k+1}$到来时，利用**Woodbury矩阵恒等式**进行递归更新：

$$R_{k+1} = R_k - R_k \tilde{h}_{k+1}^\top (I + \tilde{h}_{k+1} R_k \tilde{h}_{k+1}^\top)^{-1} \tilde{h}_{k+1} R_k$$

$$\hat{W}_{k+1}^r = (I - R_{k+1} X_{k+1}^\top X_{k+1}) \hat{W}_k^r + R_{k+1} X_{k+1}^\top Y_{k+1}$$

**为什么这保证了零遗忘？** 递归更新的数学等价性保证了：顺序训练每个任务得到的路由器权重，与将所有任务数据放在一起联合训练完全等价。因此，新任务的加入不会改变对旧任务的路由决策。

### 推理过程

1. 输入经过冻结特征提取器获取$h_{\leq L_f}(X_t)$
2. 特征扩展到高维空间
3. 通过路由器计算任务概率：$p(k|X_t) = \text{softmax}(\tilde{h}(X_t) W_k^r)$
4. 选择主导任务：$k^* = \arg\max_k p(k|X_t)$
5. 激活对应的LoRA适配器生成下一个token

## 实验关键数据

### 主实验

**TRACE Benchmark（LLaMA-2-7B-Chat）**：

| 方法 | Order1 OP(BWT) | Order2 OP(BWT) |
|------|----------------|----------------|
| AdaLoRA | 22.60 (-30.11) | 23.34 (-27.54) |
| LoRAMoE | 48.54 (-4.27) | 47.48 (-4.28) |
| SeqFT | 47.63 (-11.45) | 45.12 (-12.27) |
| EWC | 48.20 (-9.48) | 44.54 (-12.00) |
| O-LoRA | 44.64 (-4.20) | 42.83 (-9.11) |
| Replay (1%) | 48.47 (-9.69) | 47.04 (-10.24) |
| SEEKR (1%) | 54.99 (-2.61) | 54.69 (-2.53) |
| **Any-SSR** | **55.69 (0.00)** | **55.69 (0.00)** |
| *Upper-bound (MTL)* | *59.38* | - |

Any-SSR是唯一BWT=0的方法，且两种任务顺序结果完全一致，证明了其对任务顺序不变性。OP仅比联合训练上界低3.69%。

### 消融实验

**组件消融（热力图分析）**：

| 配置 | 特点 | 问题 |
|------|------|------|
| 单LoRA | 所有任务共享一个LoRA | 新任务学好后旧任务性能大幅下降 |
| Multi-LoRA + BP路由 | 独立LoRA但用反向传播训练路由 | 路由准确率逐步退化（每阶段平均下降21.7%） |
| **Multi-LoRA + AL路由** | **Any-SSR** | **路由100%准确，所有历史任务零遗忘** |

**超参数分析**：

| $L_f$ (冻结层数) | $E$ (扩展维度) | OP(BWT) |
|------------------|----------------|---------|
| 2 | 10000 | 54.79 (-0.19) |
| **4** | **10000** | **55.69 (0.00)** |
| 6 | 10000 | 53.21 (0.00) |

$L_f=4$、$E=10000$为最优配置。$L_f$过小导致低层特征过于通用影响路由精度；$L_f$过大则高层LoRA学习能力不足。

**通用能力保持**：

| 方法 | MMLU | GSM | BBH | GA (DeltaGA) |
|------|------|-----|-----|--------------|
| LLaMA-2-7B-Chat | 46.89 | 27.14 | 39.73 | 47.77 |
| SeqFT | 45.16 | 14.03 | 32.50 | 43.50 (-4.27) |
| SEEKR (1%) | 46.32 | 20.85 | 38.52 | 46.72 (-1.05) |
| **Any-SSR** | 45.77 | 25.43 | 37.01 | **46.51 (-1.26)** |

Any-SSR在无需回放的情况下，通用能力损失仅-1.26，接近使用1%回放数据的SEEKR（-1.05）。

### 关键发现

1. **BWT=0的数学保证**：Any-SSR是所有方法中唯一实现零后向迁移的方法，无论任务顺序如何
2. **路由精度是关键**：BP训练的路由器遭受灾难性遗忘（准确率逐步退化），而解析路由器始终保持100%
3. **存储效率**：每个任务仅需$O(rL_{adapt})$参数（<全参数微调的1%），路由器仅需存储自相关矩阵$R_k$
4. **任务顺序无关性**：两种不同的任务顺序产生完全相同的结果

## 亮点与洞察

1. **经典数学工具的创新应用**：将递归最小二乘（RLS）这一信号处理领域的经典工具应用到LLM持续学习中，提供了严格的数学保证
2. **参数解耦的优雅设计**：通过层次化分割和独立LoRA实现任务参数的完全隔离，从架构层面消除干扰
3. **代价极低的路由更新**：递归更新仅需存储自相关矩阵，对CPU即可完成，无需GPU
4. **理论与实践的统一**：不仅有理论证明零遗忘特性，实验也完美验证

## 局限与展望

1. NumGLUE-sm和NumGLUE-ds之间存在少量路由失败，原因是共享prompt（如"Solve the math problem"）导致路由模糊
2. 随着任务数量增长，LoRA Bank的存储需求线性增加
3. 通用能力评估中需要使用1%验证集训练路由器的通用任务分支，不完全满足"零数据访问"的理想
4. 仅在7B-9B规模LLM上验证，更大规模（70B+）的效果待确认
5. 关键应用场景（金融、医疗）中的路由错误可能带来风险

## 相关工作与启发

- **O-LoRA**：在正交子空间中实现LoRA持续学习，但仍使用固定参数数导致干扰
- **SEEKR**：通过注意力蒸馏减少遗忘，但需要存储回放样本
- **LoRAMoE**：引入LoRA专家混合，但路由训练不具备零遗忘保证
- **Cover定理**：特征升维增强线性可分性的理论基础
- **Woodbury矩阵恒等式**：递归更新的数学工具
- 启发：闭式解方法为持续学习提供了一条绕过梯度优化的数学严谨路径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](../../ACL2025/llm_nlp/recurrent_kif_continual_learning.md)
- [\[ICCV 2025\] VIM: Versatile Interactive Motion-Language Model](vim_versatile_interactive_motion_language_model.md)
- [\[ICCV 2025\] ShadowHack: Hacking Shadows via Luminance-Color Divide and Conquer](shadowhack_hacking_shadows_via_luminance-color_divide_and_conquer.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)

</div>

<!-- RELATED:END -->
