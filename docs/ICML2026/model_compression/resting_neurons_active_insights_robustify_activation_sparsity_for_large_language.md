---
title: >-
  [论文解读] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models
description: >-
  [ICML 2026][模型压缩][激活稀疏] 本文把激活稀疏导致 LLM 掉点的本质归因为"表示漂移"，并仿照生物自发放电向每层注入一个输入无关、训练后可吸收进 bias 的小向量（SPON），以接近零推理开销显著缩小稀疏模型与稠密模型的差距。
tags:
  - "ICML 2026"
  - "模型压缩"
  - "激活稀疏"
  - "表示稳定性"
  - "自发神经元"
  - "偏置吸收"
  - "知识保留"
---

# Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models

**会议**: ICML 2026  
**arXiv**: [2512.12744](https://arxiv.org/abs/2512.12744)  
**代码**: https://github.com/hxu105/SPON (有)  
**领域**: 模型压缩 / LLM 效率  
**关键词**: 激活稀疏、表示稳定性、自发神经元、偏置吸收、知识保留

## 一句话总结
本文把激活稀疏导致 LLM 掉点的本质归因为"表示漂移"，并仿照生物自发放电向每层注入一个输入无关、训练后可吸收进 bias 的小向量（SPON），以接近零推理开销显著缩小稀疏模型与稠密模型的差距。

## 研究背景与动机
**领域现状**：为了加速 LLM 推理，激活稀疏（activation sparsity）成为相对优雅的一条路线，其代表方法如 TEAL / LaRoSA / R-Sparse 通过幅度阈值 $\tau$ 将小幅激活置零，进而在 MLP/Attention 的线性变换中跳过相应权重列；这种"动态遮蔽"不改权重、不动激活函数，自然适合现有稠密权重的 LLM。

**现有痛点**：稀疏比一旦推到 50% 以上，几乎所有现有方案都会出现明显的 perplexity 上升和零样本任务掉点，必须靠重训练或结构调整才能挽回，与"零成本加速"的初衷相违。

**核心矛盾**：作者通过观察发现：随着序列变长，能在所有 token 上都被同时激活的神经元比例呈指数衰减（Figure 1）。也就是说，原本在稠密模型里充当"全局锚点"的那些常活神经元，在稀疏后被各 token 选择性地关掉，导致隐状态分布发生 token-dependent 的漂移，等价于丢掉了预训练时学到的"先验"。

**本文目标**：在不重训权重、不改架构、不增加推理 FLOPs 的前提下，恢复稀疏 LLM 的表示稳定性，从而把性能拉回稠密水平。

**切入角度**：把激活稀疏问题重新表述为"表示对齐"问题——稀疏引入的不是简单的信息丢失，而是缺少稳定的、输入无关的"基线活动"作为参考。生物神经系统中存在的自发放电（spontaneous activity）恰好扮演这种角色，提供静态先验。

**核心 idea**：在每一层注入少量可学习、输入无关的"自发激活向量" $\vec{\alpha}$，仅通过对稠密模型 logits 的 KL 蒸馏来训练这个向量；由于与输入无关，训练后可直接折进 bias，推理时零额外开销。

## 方法详解

### 整体框架
对 transformer 每个线性层 $Y = WX$，先做输入激活稀疏 $S(X)_i = \mathbf{1}\{|x_i|>\tau\}\cdot x_i$，然后并联一个"自发神经元"项 $W\vec{\alpha}$，整体写为 $Y = W\,S(X) + W\vec{\alpha}$。这里 $\vec{\alpha}$ 在训练后被吸收为 $b' = b + W\vec{\alpha}$，因此推理图与原始稀疏 LLM 完全一致，没有额外矩阵乘。训练阶段冻结整模型，只学每层一组 $\vec{\alpha}$，通过 KL 散度把稀疏模型与稠密模型在校准集上的 logits 分布对齐。

### 关键设计

1. **输入无关的自发激活注入**:

    - 功能：为每个被稀疏遮蔽的线性层提供一个静态的、与 token 无关的表示锚点，弥补 token-dependent 的常活神经元丢失。
    - 核心思路：在原本 $WS(X)$ 之后加一项 $W\vec{\alpha}$，其中 $\vec{\alpha}\in\mathbb{R}^d$ 是该层独有的可学习向量，与输入 $X$ 无关；因此 $W\vec{\alpha}$ 是常量，可在推理前算好并加到偏置上。论文显示，每层只需 **一个** 自发神经元（即 $\vec{\alpha}$ 等价于一个固定方向的激活）就足够把性能找回，体现了"极少量先验也能稳住表示"。
    - 设计动机：作者把稀疏视为对预训练统计先验的破坏，自发激活相当于把稠密模型隐含的"全局期望"显式写回稀疏图中，且不占新的算子，符合"零推理开销"的硬约束。

2. **分布匹配式的轻量校准**:

    - 功能：在不动 LLM 任何已有参数的前提下，只优化 $\mathcal{A} = \{\vec{\alpha}_\ell\}$ 来对齐稀疏与稠密模型的输出分布。
    - 核心思路：取一个小规模校准语料 $u\sim D$（WikiText 或 C4 均可），分别记稠密、稀疏模型的输出 logits 为 $z(u)$、$\tilde z(u;\mathcal{A})$，最小化 $\mathcal{L}(\mathcal{A}) = \mathbb{E}_u[\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))]$。由于只更新少量 $\vec{\alpha}$，校准成本远低于全量微调。
    - 设计动机：这种"输出层蒸馏 + 仅更新偏置项"的组合，让自发神经元充当对稀疏残差的全局补偿，且因为只蒸 logits、不强行匹配中间层，所以对校准数据集分布相对鲁棒（在 C4 上校准、WikiText 上评估，PPL 仍优于基线）。

3. **Fisher 加权的残差校正解释**:

    - 功能：从理论层面解释 SPON 为何能够稳定稀疏表示。
    - 核心思路：以最后一层投影为例，定义稀疏残差 $e(X) = WX - WS(X)$，对 KL 取一阶条件得到 $\mathbb{E}_u[W^\top H(W\vec{\alpha} - e(X))] = 0$，其中 $H$ 是 logits 处的 Hessian，恰好等价于输出分布的 Fisher 信息矩阵。换言之，最优 $\vec{\alpha}$ 把 $W\vec{\alpha}$ 推到 $e(X)$ 在 Fisher 度量下的最优近似——只在"输出分布最敏感"的方向上去补偿稀疏带来的偏差。
    - 设计动机：它把"为什么单个静态向量能挽救整个稀疏模型"讲清楚——KL 损失自带的 Fisher 几何让 SPON 把有限的容量优先花在影响输出最强的方向，从而以极小参数量稳住关键表示。

### 损失函数 / 训练策略
仅训练 $\mathcal{A}$，损失为 $\mathrm{KL}(\sigma(z)\|\sigma(\tilde z))$；校准集很小，训练完成后将 $W\vec{\alpha}$ 折入 bias，推理图保持不变。

## 实验关键数据

### 主实验

| 数据集 | 模型 | 稀疏度 | TEAL | SPON | 备注 |
|--------|------|--------|------|------|------|
| WikiText PPL | Llama3-8B | 50% | 8.34 | 7.83 | 接近稠密 6.75 |
| WikiText PPL | Mistral-7B | 50% | 6.00 | 5.86 | 稠密 5.49 |
| WikiText PPL | Qwen3-8B | 50% | 9.75 | 9.26 | 稠密 8.99 |
| WikiText PPL | Llama3-8B | 60% | 11.62 | 9.63 | 高稀疏增益最明显 |

与剪枝方法对比（Llama3-8B, 50%）SPON PPL=7.83，明显优于 SparseGPT (9.18)、Wanda (9.66)、MaskLLM (8.58)、ARMOR (10.10)。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| TEAL only | Llama3-8B 50% PPL 8.34 | 仅幅度阈值稀疏 |
| + 自发神经元(每层 1 个) | PPL 7.83 | 仅增加一个 $\vec{\alpha}$ |
| 校准在 C4、评估 WikiText | PPL 7.95 | 验证跨语料鲁棒 |
| 与 LaRoSA/WINA/R-Sparse 组合 | Llama3-8B 五任务均分 71.96% | 高于 LaRoSA(69.82)/WINA(70.97)/R-Sparse(69.56) |

### 关键发现
- 把每层"自发神经元"个数压到 1，性能依旧最好，说明 SPON 主要解决的是"方向"而非"容量"问题，与 Fisher 残差校正的理论解释一致。
- 越激进的稀疏（60% > 50% > 25%）SPON 的增益越大，提示自发激活实际在补偿"被强行关掉的常活神经元"。
- SPON 与现有稀疏方法（LaRoSA、WINA、R-Sparse、WAS）正交，可叠加获得进一步增益；在 Qwen3-32B 与 Llama3-70B 上也能稳定带来 0.75% / 0.96% 提升，说明并非仅对小模型奏效。

## 亮点与洞察
- "把缺失的常活神经元用静态偏置补回来"这一定义非常干净——既复用了 bias 的硬件路径，又把稀疏与表示稳定性挂钩，方法成本几乎为零。
- KL+Fisher 的推导让"一个向量为什么够用"这件事变成了可解释结论，而不是工程巧合；这种"用 Fisher 几何指导最小参数补偿"的思路可迁移到其他低 bit/低秩压缩里。
- 通常 LLM 设计倾向于忽略 bias，本文反其道而行之，说明在重度稀疏场景下"bias-like"参数实际充当不可或缺的表征支架，提示了一个被忽视的设计自由度。

## 局限与展望
- 仅在 7B–8B 主体上做了大量实验，70B 与 32B 上虽然有效但实验粒度较小，长上下文、推理链场景下自发向量是否依然稳定尚需更系统的验证。
- 自发向量是逐层独立学习的，没有显式建模层间相互作用，未来可探索按结构（如 attention vs MLP）共享或低秩耦合，以进一步减少校准成本。
- 训练仍需要稠密模型的 logits 作为教师，对完全无访问 dense 模型的部署场景（如只有量化权重）需要替代信号。

## 相关工作与启发
- **vs TEAL/LaRoSA/R-Sparse**：它们关注"如何更聪明地选要遮蔽的激活"，本文承认稀疏后的残差并主动补偿，因此与它们正交并可组合。
- **vs SparseGPT/Wanda/MaskLLM**：权重剪枝永久删除参数，SPON 完全在激活空间运作，权重原封不动，因此更易回滚和叠加。
- **vs Bias-only fine-tuning（如 BitFit）**：BitFit 是任务自适应，SPON 是稀疏自适应；两者形式相似但目标不同，提示"只动 bias"在 LLM 时代仍是一片值得挖掘的低成本调节空间。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把激活稀疏重新表述为表示对齐问题，并用 Fisher 残差解释，思路清晰但单点改动较小
- 实验充分度: ⭐⭐⭐⭐ 多模型多基线 + 与剪枝/SOTA 稀疏方法的全面对比，缺一些超长上下文场景验证
- 写作质量: ⭐⭐⭐⭐ 故事线（生物动机→经验观察→理论推导→工程实现）非常顺畅
- 价值: ⭐⭐⭐⭐ 几乎零成本即可叠加在现有稀疏方法上，工业部署友好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](../../NeurIPS2025/model_compression/duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ICCV 2025\] MixA-Q: Revisiting Activation Sparsity for Vision Transformers from a Mixed-Precision Quantization Perspective](../../ICCV2025/model_compression/mixa-q_revisiting_activation_sparsity_for_vision_transformers_from_a_mixed-preci.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](../../ICLR2026/model_compression/knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](../../AAAI2026/model_compression/failures_to_surface_harmful_contents_in_video_large_language_models.md)

</div>

<!-- RELATED:END -->
