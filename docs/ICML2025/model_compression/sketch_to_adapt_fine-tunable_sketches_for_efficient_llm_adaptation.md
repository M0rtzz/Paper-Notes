---
title: >-
  [论文解读] Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation
description: >-
  [ICML 2025][模型压缩][参数共享] SpaLLM 提出了一种基于 sketching 的参数共享方法来统一 LLM 的压缩和微调过程，将预训练权重压缩为查找表（LUT）后直接在表值上微调，避免了 QLoRA 等双塔架构的低秩假设和实现复杂性，在多项基准上以更少的训练参数取得了优于 QLoRA/LoftQ 的性能。
tags:
  - "ICML 2025"
  - "模型压缩"
  - "参数共享"
  - "Sketching"
  - "压缩自适应"
  - "查找表微调"
  - "统一压缩适配"
---

# Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation

**会议**: ICML 2025  
**arXiv**: [2410.06364](https://arxiv.org/abs/2410.06364)  
**代码**: 无公开代码  
**领域**: 模型压缩 / LLM高效微调  
**关键词**: 参数共享, Sketching, 压缩自适应, 查找表微调, 统一压缩适配

## 一句话总结
SpaLLM 提出了一种基于 sketching 的参数共享方法来统一 LLM 的压缩和微调过程，将预训练权重压缩为查找表（LUT）后直接在表值上微调，避免了 QLoRA 等双塔架构的低秩假设和实现复杂性，在多项基准上以更少的训练参数取得了优于 QLoRA/LoftQ 的性能。

## 研究背景与动机

**领域现状**：大语言模型的微调是下游应用的关键步骤。由于 LLM 参数量巨大，全精度微调不现实，因此"压缩式自适应"（compressive adaptation）方法应运而生——先压缩模型参数，再进行参数高效微调（PEFT）。QLoRA 是该领域最具代表性的方法。

**现有痛点**：
   - **低秩假设不成立**：QLoRA 和 LoftQ 都假设微调前后的权重差异是低秩的，但研究表明全量微调与基座模型的权重差异往往是高秩的
   - **双塔架构复杂**：QLoRA 等方法采用"压缩权重 + 全精度适配器"的双塔结构，推理时需要两条路径（量化权重的矩阵乘 + 适配器的矩阵乘），不同精度的运算需要分开处理，实现复杂
   - **低比特困境**：在 3-bit 或更低比特下，QLoRA 甚至无法收敛

**核心矛盾**：压缩和自适应被割裂为两个独立阶段——先量化再加适配器，这不仅引入了不必要的架构复杂性，还限制了适配器对量化损失的补偿能力。

**切入角度**：能否将模型压缩和微调统一为一个过程？即直接在压缩后的参数上微调，不需要额外的适配器？

**核心 idea**：用 sketching 算法将每行权重映射到一个小的查找表（一组 centroids），然后固定映射关系（sketching matrix $\Pi$），直接在查找表的值上微调。

## 方法详解

### 整体框架
SpaLLM 的流程分为两个阶段：
1. **压缩阶段**：对预训练 LLM 的每行权重执行参数 sketching，将其映射为查找表 $w \in \mathbb{R}^k$ 和 one-hot sketching 矩阵 $\Pi \in \mathbb{R}^{d \times k}$，使得 $\hat{\theta} = \Pi w$
2. **适配阶段**：固定 $\Pi$ 不变，直接在浮点数查找表 $w$ 上进行任务特定的微调

### 关键设计

1. **逐行参数 Sketching（Row-wise Parameter Sketching）**:

    - 对权重矩阵 $\Theta \in \mathbb{R}^{n \times d}$ 的每一行 $\theta \in \mathbb{R}^d$，近似为 $\hat{\theta} = \Pi w$
    - $\Pi$ 是 $d \times k$ 的 one-hot 矩阵，每行恰好有一个 1，将 $\theta$ 的每个元素映射到 $w$ 的某个条目
    - 由于 $k < d$，多个权重元素共享同一个条目值，实现了参数共享压缩
    - 设计动机：这种参数共享不依赖低秩假设——即使是满秩矩阵（如单位矩阵），也可以用少量共享值完美表示

2. **加权 Lloyd 算法学习 Sketching（Weighted Lloyd's Algorithm）**:

    - 不使用随机 hash 映射（LLM 权重没有 heavy-hitter 模式，随机映射会严重退化）
    - 改用加权 Lloyd 聚类算法（k-means 变体）学习最优的 $k$ 个 centroids
    - 权重由 Hessian 对角线 $\text{diag}(H^{-1}) = \text{diag}((XX^T)^{-1})$ 反比加权，赋予更敏感的参数更高的聚类精度
    - 学习完 centroids 后，使用迭代损失误差补偿框架（类似 GPTQ 的列式贪心）确定每个参数的映射
    - 设计动机：LLM 权重对扰动敏感，随机 sketching 会导致严重退化；利用 Hessian 信息进行加权聚类可以最小化对模型输出的影响

3. **Groups Per Row (GPR) 扩展**:

    - 为了增加可学习参数数量（提高表达能力），将每行权重分为多个连续组
    - 每组独立维护自己的查找表，映射矩阵大小不变，但总的可训练参数数增加
    - 如 GPR=8 表示每行分为 8 组，每组有自己的 LUT
    - 设计动机：通过控制 GPR 值，可以灵活调节"压缩率-精度"的权衡

4. **单塔统一架构**:

    - 推理时只需一次压缩矩阵乘：查找表索引 + 累加，无需双塔结构
    - 利用已有的高效内核（如 SqueezeLLM 的内核）实现推理加速
    - 不同任务只需存储和切换 LUT 值，映射关系 $\Pi$ 共享
    - 设计动机：避免 QLoRA 的双路径推理，减少系统复杂度和推理延迟

### 损失函数 / 训练策略
微调阶段使用标准的语言建模损失（下一词预测的交叉熵）。由于只更新 LUT 中的值，多个虚拟参数的梯度会汇聚到同一个可训练参数上取平均，这本身起到了正则化效果。学习率从 $(1\times10^{-4}, 5\times10^{-5}, ..., 2\times10^{-6})$ 中搜索，训练 10 个 epoch。

## 实验关键数据

### 主实验

| 模型/数据集 | 方法 | 比特 | 可训练参数 | WikiText-2 PPL | GSM8K Acc |
|------------|------|------|----------|---------------|-----------|
| LLaMA-2-7B | LoRA (r=64) | 16 | 160M | 5.08 | 36.9% |
| LLaMA-2-7B | QLoRA (r=64) | 4 | 160M | 5.70 | 35.1% |
| LLaMA-2-7B | LoftQ (r=64) | 4 | 160M | 5.24 | 35.0% |
| LLaMA-2-7B | SpaLLM (GPR=8) | 4 | - | 5.32 | **38.4%** |
| LLaMA-2-7B | QLoRA (r=64) | 2 | 160M | N.A. | N.A. |
| LLaMA-2-7B | LoftQ (r=64) | 2 | 160M | 7.85 | 20.9% |
| LLaMA-2-7B | SpaLLM (GPR=8) | 2 | - | 7.40 | **23.7%** |
| LLaMA-2-13B | SpaLLM (GPR=1) | 3 | 22M | **5.05** | - |
| LLaMA-3-70B | SpaLLM (GPR=4) | 4 | - | - | AVG 0.72 |

### 消融实验

| GPR 值 | 模型 | GSM8K Acc | 说明 |
|--------|------|-----------|------|
| GPR=1 | LLaMA-2-7B | ~30% | 最少参数，基础性能 |
| GPR=2 | LLaMA-2-7B | ~33% | 性能随 GPR 递增 |
| GPR=4 | LLaMA-2-7B | ~35% | 接近 QLoRA 基线 |
| GPR=8 | LLaMA-2-7B | 38.4% | 超过所有基线 |
| GPR=1 | LLaMA-2-13B | 超过基线 | 仅 1/5 训练参数 |

### 关键发现
- SpaLLM 在 13B 模型上、仅 GPR=1（22M 训练参数）就超过了所有基线；在 7B 上需要 GPR=8 才超过基线
- 在 2-bit 下 QLoRA 完全无法收敛，而 SpaLLM 仍能保持合理性能
- 推理效率：SpaLLM 比 QLoRA/LoftQ 快约 3 倍，显存也更低（单塔架构的优势）
- 在 LLaMA-3-70B 上，4-bit SpaLLM 微调版（39.8GB）优于全精度基座模型，且可放入单张 L40S-48GB GPU
- LLM-as-a-judge 评估中，SpaLLM 的 win-loss ratio 为 0.61（vs LoftQ），91%（vs Falcon-40B-Instruct）

## 亮点与洞察
- **"参数共享 = 正则化 + 压缩"的统一视角**：一个查找表值被多个权重位置共享，天然实现了正则化，避免过拟合
- **打破低秩假设的枷锁**：证明了参数共享比低秩分解更适合做压缩自适应，因为满秩矩阵也可以用共享参数完美表示
- **推理效率的本质优势**：单塔架构避免了双路径计算，这在多用户并发serving场景下尤为关键
- **可迁移的trick**：加权 Lloyd 聚类 + Hessian 加权的组合可应用到任何需要权重聚类的压缩方法中

## 局限与展望
- Sketching 矩阵 $\Pi$ 的存储也需要空间（one-hot 矩阵的索引），论文中用比特数编码但未详细讨论其开销
- 仅在 LLaMA 系列上验证，缺少对其他架构（如 Mistral、Qwen）的实验
- GPR 的选择目前是手动的，可以考虑自适应地为不同层分配不同的 GPR
- 与 GPTAQ 等更新的量化方法的组合效果未探索

## 相关工作与启发
- **vs QLoRA/LoftQ**：核心区别在于 SpaLLM 用参数共享替代了低秩适配器，消除了双塔架构和低秩假设；在低比特和大模型上优势明显
- **vs GPTQ**：GPTQ 只做压缩不做微调；SpaLLM 的压缩阶段借鉴了 GPTQ 的列式贪心框架，但目标是建立查找表而非量化
- **vs HashedNets/ROAST**：早期参数共享方法用随机 hash，适用于从头训练；SpaLLM 用学习式 sketching，适用于预训练模型的后压缩微调

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将 sketching 基于参数共享引入 LLM 压缩微调，统一了压缩和适配
- 实验充分度: ⭐⭐⭐⭐ 多数据集多模型多指标，GPR 消融详细，效率对比充分
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，与 QLoRA 的对比图非常直观
- 价值: ⭐⭐⭐⭐⭐ 提出了新的压缩微调范式，在低比特场景下优势显著，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sketch Down the FLOPs: Towards Efficient Networks for Human Sketch](../../CVPR2025/model_compression/sketch_down_the_flops_towards_efficient_networks_for_human_sketch.md)
- [\[ICML 2025\] BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference](blockdialect_block-wise_fine-grained_mixed_format_quantization_for_energy-effici.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[NeurIPS 2025\] RefLoRA: Refactored Low-Rank Adaptation for Efficient Fine-Tuning of Large Models](../../NeurIPS2025/model_compression/reflora_refactored_low-rank_adaptation_for_efficient_fine-tuning_of_large_models.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](../../NeurIPS2025/model_compression/data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)

</div>

<!-- RELATED:END -->
