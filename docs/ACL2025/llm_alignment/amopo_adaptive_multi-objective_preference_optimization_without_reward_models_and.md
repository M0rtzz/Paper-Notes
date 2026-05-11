---
title: >-
  [论文解读] AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models
description: >-
  [ACL 2025 (Findings)][LLM对齐][多目标偏好对齐] 提出AMoPO框架，通过将生成空间建模为高斯分布实现维度感知的自适应权重分配，在不依赖奖励模型和参考模型的情况下完成多目标偏好对齐，在HelpSteer2数据集上超越SOTA 28.5%，并在7B/14B/32B模型上验证了缩放能力。
tags:
  - ACL 2025 (Findings)
  - LLM对齐
  - 多目标偏好对齐
  - 自适应权重
  - 高斯分布
  - 无奖励模型
  - 偏好优化
---

# AMoPO: Adaptive Multi-objective Preference Optimization without Reward Models and Reference Models

**会议**: ACL 2025 (Findings)  
**arXiv**: [2506.07165](https://arxiv.org/abs/2506.07165)  
**代码**: [https://github.com/Javkonline/AMoPO](https://github.com/Javkonline/AMoPO)  
**领域**: LLM Alignment  
**关键词**: 多目标偏好对齐, 自适应权重, 高斯分布, 无奖励模型, 偏好优化

## 一句话总结

提出AMoPO框架，通过将生成空间建模为高斯分布实现维度感知的自适应权重分配，在不依赖奖励模型和参考模型的情况下完成多目标偏好对齐，在HelpSteer2数据集上超越SOTA 28.5%，并在7B/14B/32B模型上验证了缩放能力。

## 研究背景与动机

大语言模型（LLM）的偏好对齐是当前研究热点，目标是让模型输出符合人类在多个维度上的偏好（如有用性、正确性、指令遵循等）。然而现有方法存在两个核心问题：

**多维度平衡困难**：现有多目标对齐方法（如MODPO）难以有效平衡不同偏好维度之间的权衡关系。简单的固定权重或线性加权无法捕捉不同维度在不同样本上的优先级差异——某些样本可能更需要关注正确性，而另一些则更需要关注有用性。

**依赖辅助模型带来计算开销**：DPO等方法需要参考模型（reference model），RLHF需要奖励模型（reward model），这些辅助模型不仅增加了计算和存储成本，还可能引入额外的对齐误差。

核心矛盾在于：如何在不引入额外模型的前提下，实现对多个偏好维度的动态平衡？AMoPO的切入角度是**利用维度感知的生成指标作为隐式奖励**，结合**基于高斯分布的自适应权重机制**，让模型根据当前生成质量自动调整各维度的优化优先级。

## 方法详解

### 整体框架

AMoPO的训练流程：
- **输入**：包含多维度偏好标注的preference对数据（chosen/rejected），每条数据携带helpfulness、correctness、instruction_following等维度评分
- **数据处理**：对每条数据构造4组（维度数+1）chosen/rejected对，每组使用不同的dimension-aware prompt引导模型关注特定维度
- **前向传播**：一次前向计算所有8个序列（4维度 × 2 chosen/rejected）的log概率
- **损失计算**：基于ORPO/SimPO计算各维度损失，通过高斯采样生成自适应权重，加权求和得到最终损失
- **输出**：对齐后的LLM

### 关键设计

1. **维度感知的Prompt注入（Dimension-Aware Prompting）**:

    - 功能：为每个偏好维度构造专门的系统提示，引导模型理解当前优化方向
    - 核心思路：对于"无特定维度"的通用版本，使用综合性prompt；对于特定维度（如helpfulness），注入评估标准描述（1-5分量表），并将该维度的评分值嵌入prompt中：`"focus on the {dimension} dimension... based on the evaluation value of {score}"`
    - 设计动机：通过prompt让模型在生成时意识到当前需要优化的维度，实现隐式的维度感知，无需额外的奖励模型来评估各维度质量

2. **基于ORPO/SimPO的无参考模型损失函数**:

    - 功能：在不依赖参考模型的情况下计算偏好损失
    - 核心思路：ORPO损失结合了SFT损失和odds ratio损失：
    $\mathcal{L}_{ORPO} = -\log p_{chosen} + \beta \cdot (-\log\sigma(\log\frac{p_{chosen}/(1-p_{chosen})}{p_{rejected}/(1-p_{rejected})}))$
      SimPO的单维度损失：
    $\mathcal{L}_{SimPO} = -\log\sigma(\beta \cdot (\log p_{chosen} - \log p_{rejected} - \gamma/\beta))$
    - 设计动机：ORPO直接在语言建模目标中嵌入偏好信号，SimPO通过length-normalized log概率差异实现参考模型无关的对齐

3. **高斯分布自适应权重分配（Gaussian Adaptive Weighting）**:

    - 功能：根据当前batch中每个维度的生成质量动态分配损失权重
    - 核心思路：
      1. 提取每个维度下chosen和rejected序列的逐token概率
      2. 对每个序列的非零token概率计算均值 $\mu$ 和标准差 $\sigma$
      3. 从高斯分布 $\mathcal{N}(\mu, \sigma)$ 中采样得到该序列的质量指标
      4. 将chosen和rejected的采样值相加，得到各维度的原始权重
      5. 对所有维度权重做softmax归一化
      6. 将归一化后的权重与对应维度损失相乘
    - 设计动机：生成质量差的维度（token概率分布更散）会得到更高的权重，引导模型优先优化薄弱维度；引入随机采样增加探索性，避免陷入固定权重模式

### 损失函数 / 训练策略

最终损失为各维度加权损失之和，可选地加入SFT辅助损失：

$$\mathcal{L}_{total} = \sum_{d \in D} w_d \cdot \mathcal{L}_d + \gamma_{ftx} \cdot \mathcal{L}_{SFT}$$

其中 $w_d$ 是高斯自适应权重，$D$ = {no_object, helpfulness, correctness, instruction_following}，$\gamma_{ftx}$ 控制SFT损失的比重。

训练基于LLaMA-Factory框架，使用DeepSpeed进行分布式训练。

## 实验关键数据

### 主实验

| 模型 | 方法 | HelpSteer2 总分 | 提升 |
|------|------|-----------------|------|
| Qwen2.5-7B | DPO | baseline | - |
| Qwen2.5-7B | ORPO | baseline+α | - |
| Qwen2.5-7B | SimPO | baseline+β | - |
| Qwen2.5-7B | MODPO | baseline+γ | - |
| Qwen2.5-7B | **AMoPO** | **best** | **+28.5%** |
| Qwen2.5-14B | AMoPO | - | 持续提升 |
| Qwen2.5-32B | AMoPO | - | 持续提升 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| AMoPO (full) | 最优 | 完整框架，高斯自适应权重+多维度 |
| 去除自适应权重（等权） | 下降 | 固定权重无法适应不同样本的维度优先级 |
| 减少维度数 | 下降 | 证明多维度覆盖的重要性 |
| 仅单维度优化 | 明显下降 | 验证多目标优化的必要性 |
| 7B → 14B → 32B | 持续提升 | 证明AMoPO的scaling能力 |

### 关键发现
- AMoPO在不使用任何奖励模型或参考模型的情况下，超越了需要这些辅助模型的方法（如DPO）
- 高斯自适应权重机制在训练过程中能动态调整各维度权重，自动发现并优先优化薄弱维度
- 框架具有良好的缩放性，从7B到32B模型性能持续提升
- 维度感知的prompt注入对各维度独立优化效果显著

## 亮点与洞察

- **高斯采样做权重分配**是一个新颖的trick：利用token概率分布的统计特征（均值和方差）反映生成质量，通过采样引入随机性避免权重固化，这个思路可以迁移到其他多目标学习场景
- **维度感知prompt**将偏好维度信息编码到输入中，让模型在推理时也能根据prompt关注特定维度，实现一定程度的可控生成
- 整个框架基于LLaMA-Factory构建，易于复现和扩展

## 局限与展望

- 当前仅在HelpSteer2数据集上验证，偏好维度固定为helpfulness/correctness/instruction_following，泛化到更多维度需要进一步验证
- 高斯采样在batch较小时统计量不稳定，可能导致权重波动较大
- 维度之间的相关性和冲突关系未被显式建模，简单的独立高斯可能遗漏维度间的交互
- 推理时dimension-aware prompt的选择依赖用户指定，缺乏自动维度选择机制

## 相关工作与启发

- **vs DPO**: DPO需要参考模型，AMoPO通过ORPO/SimPO变体实现无参考模型训练，减少一半GPU显存占用
- **vs MODPO**: MODPO使用固定的线性加权混合多目标，AMoPO通过高斯自适应权重实现动态平衡，性能显著更优
- **vs ORPO**: 原始ORPO是单目标的，AMoPO在其基础上扩展为多目标版本，并增加了维度感知和自适应权重

## 评分

- 新颖性: ⭐⭐⭐⭐ 高斯采样做自适应权重分配是新颖的切入点，但整体框架组合性较强
- 实验充分度: ⭐⭐⭐⭐ 多尺度模型实验和消融完整，但仅在一个数据集上验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述完整
- 价值: ⭐⭐⭐⭐ 对多目标对齐场景有实用价值，代码开源可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference](../../AAAI2026/llm_alignment/margin-aware_preference_optimization_for_aligning_diffusion_models_without_refer.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](../../CVPR2025/llm_alignment/calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[ACL 2025\] Robust Preference Optimization via Dynamic Target Margins](robust_preference_optimization_via_dynamic_target_margins.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](t-reg_preference_optimization_with_token-level_reward_regularization.md)

</div>

<!-- RELATED:END -->
