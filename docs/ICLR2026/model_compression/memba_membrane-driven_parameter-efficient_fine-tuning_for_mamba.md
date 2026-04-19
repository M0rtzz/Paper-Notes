---
title: >-
  [论文解读] Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba
description: >-
  [ICLR 2026][模型压缩][Mamba] 提出 Memba，一种受生物神经元膜电位启发的参数高效微调方法，通过在 Mamba 门控分支引入泄漏积分膜（LIM）神经元实现时序自适应，结合 LoRA 放置优化和跨层膜传递，以极少参数在语言和视觉任务上超越现有 Mamba PEFT 方法。
tags:
  - ICLR 2026
  - 模型压缩
  - Mamba
  - PEFT
  - 膜电位
  - 泄漏积分
  - 状态空间模型
---

# Memba: Membrane-driven Parameter-Efficient Fine-Tuning for Mamba

**会议**: ICLR 2026  
**arXiv**: [2506.18184](https://arxiv.org/abs/2506.18184)  
**代码**: [GitHub](https://github.com/Intelligent-Computing-Lab-Panda/Memba)  
**领域**: 模型压缩  
**关键词**: Mamba, PEFT, 膜电位, 泄漏积分, 状态空间模型

## 一句话总结
提出 Memba，一种受生物神经元膜电位启发的参数高效微调方法，通过在 Mamba 门控分支引入泄漏积分膜（LIM）神经元实现时序自适应，结合 LoRA 放置优化和跨层膜传递，以极少参数在语言和视觉任务上超越现有 Mamba PEFT 方法。

## 研究背景与动机
状态空间模型（SSM）/ Mamba 以线性复杂度替代 Transformer 的注意力机制，随着模型规模增大，PEFT成为必要。但现有 PEFT 方法直接从 Transformer 迁移到 Mamba，忽略了 SSM 独特的时序处理动态：

关键痛点：
1. Mamba的门控机制仅是简单的线性变换+SiLU，缺乏像LSTM/GRU那样的多门控时序控制能力
2. 直接微调SSM核心组件（选择性扫描中的A, B, C, Δ）会导致性能退化（已有研究验证）
3. 如何在不破坏预训练SSM平衡动态的前提下引入时序适应能力？

核心idea：在Mamba的门控分支（而非SSM分支）引入生物启发的泄漏积分膜神经元。LIM神经元通过膜电位的累积-泄漏-重置动态，天然提供时序选择性记忆，且不需要额外可学习参数。

## 方法详解

### 整体框架
Memba在原始Mamba架构上做三处修改：①门控分支加入LIM神经元提供时序处理能力；②LoRA仅放在输入和输出投影层（非SSM组件）；③跨层传递平均膜电位。SSM分支完全不改动。

### 关键设计
1. **泄漏积分膜（LIM）神经元**:

    - 功能：在门控分支中引入时序动态
    - 核心思路：将输入序列分为 $T$ 个等大chunk，逐chunk处理：$\mathbf{u}[i+1]^l = r(\tau \mathbf{u}[i]^l + \mathbf{W}^l X[i])$，其中 $r(x) = 0$ if $x > V_{th}$, else $x$。$\tau \in (0,1]$ 控制泄漏率，$V_{th}$ 为重置阈值
    - 设计动机：LIM自然实现信息选择性保留——关键路径特征产生明显膜电位峰值，而基线电位跨chunk逐渐下降，模拟SSM对近期token的偏好。整个过程不引入额外可学习参数

2. **LoRA放置优化**:

    - 功能：确定LoRA应用于Mamba的哪些投影层
    - 核心思路：消融实验表明 in_proj 和 out_proj 最关键（移除分别降1.2%和0.8%），dt_proj和x_proj影响很小。仅用 in_proj + out_proj 上的 LoRA 即可超过全参数微调
    - 设计动机：in/out投影是Mamba的信息瓶颈，而dt/x是SSM内部参数不宜修改

3. **跨层膜电位传递**:

    - 功能：维持跨网络深度的时序一致性
    - 核心思路：第 $l$ 层处理完所有chunk后，计算平均膜状态 $\bar{\mathbf{u}}^l = \frac{1}{T}\sum_{i=1}^T \mathbf{u}^l[i]$，作为第 $l+1$ 层第一个chunk的初始膜电位：$\mathbf{u}^{l+1}[1] = \bar{\mathbf{u}}^l$
    - 设计动机：防止深层网络中时序上下文丢失，同时用平均值避免仅传递末状态造成的信息损失

### 理论分析
Theorem 1 表明LIM具有双重效果：均值膜成分通过泄漏动态提供时序上下文集成，波动成分引入有界正则化 $\mathcal{R}(\mathbf{y}_t, \bar{\mathbf{u}}_t) \leq \frac{\gamma}{2} \cdot \lambda_{\max} \cdot \epsilon^2$，使损失曲面更平滑。

## 实验关键数据

### 主实验 (常识推理, Mamba-130M)
| 方法 | #Params(%) | BoolQ | PIQA | SIQA | HellaS | WinoG | ARC-e | ARC-c | OBQA | Avg |
|------|-----------|-------|------|------|--------|-------|-------|-------|------|-----|
| Full FT | 100 | 56.1 | 65.3 | 38.7 | 35.3 | 52.0 | 46.4 | 25.7 | 32.8 | 43.8 |
| SLL LoRA | 1.45 | 56.3 | 63.3 | 38.2 | 34.6 | 51.6 | 43.5 | 23.6 | 30.6 | 42.7 |
| LoRA (in_proj) | 2.23 | 53.5 | 62.9 | 38.2 | 33.8 | 53.1 | 46.4 | 23.7 | 30.8 | 42.8 |
| LoRAp (X) | 2.67 | 61.7 | 64.0 | 39.5 | 34.3 | 52.2 | 43.5 | 25.3 | 29.4 | 43.7 |
| **Memba (in+out)** | **5.20** | **58.8** | **65.8** | **40.1** | **34.7** | 51.6 | 47.7 | **24.7** | **31.2** | **44.3** |

### 消融实验
| 配置 | Avg Acc(%) | 说明 |
|------|-----------|------|
| All projectors LoRA | 43.9 | 所有投影层 |
| -dt_proj | 43.9 | 移除dt影响极小 |
| -x_proj | 43.7 | 移除x影响小 |
| -out_proj | 43.1 | 输出投影重要 |
| -in_proj | 42.7 | 输入投影最重要 |
| Memba vs Full FT (790M) | Memba更高 | PEFT优于全微调 |
| Memba vs Full FT (1.4B) | Memba更高 | 全微调容易过拟合 |

### 关键发现
- Memba以5.2%参数超过全参数微调（130M/790M/1.4B均是如此），全微调容易过拟合
- LIM的膜电位可视化清晰展示关键特征的峰值和跨chunk的渐进衰减
- in_proj和out_proj是Mamba PEFT的关键位置，SSM组件（dt_proj, x_proj）不适合微调
- 跨层膜传递比无传递提升约0.5%，对深层网络更重要

## 亮点与洞察
- 生物启发的LIM设计与Mamba的SSM天然互补——SSM处理线性时序，LIM（在门控分支）提供非线性时序选择性
- "不触碰SSM"的设计哲学有说服力：已有研究证明直接微调SSM会退化
- 膜电位的chunking策略巧妙解决了逐token处理长序列的效率问题
- 理论正则化分析为膜电位波动的有益作用提供了解释

## 局限与展望
- chunk大小和chunk数T为超参数，需要调优
- LIM神经元的泄漏因子τ和阈值Vth的敏感性需要关注
- 未在最新的Mamba-2架构上验证
- 视觉任务的评测仅限于VTAB-1k，大规模视觉基准缺失

## 相关工作与启发
- **vs SLL LoRA**: Memba通过LIM提供更好的时序处理，平均准确率高1.6%
- **vs Affix-tuning**: 以5.2%对64.6%的参数量实现更好性能
- **vs 全参数微调**: 避免过拟合，PEFT反而更优

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 生物膜电位与SSM的结合是全新方向
- 实验充分度: ⭐⭐⭐⭐ 多尺度语言+视觉评测，但缺少大规模基准
- 写作质量: ⭐⭐⭐⭐ 膜电位可视化直观，结构清晰
- 价值: ⭐⭐⭐⭐ 为Mamba时代的PEFT开辟了生物启发的新路线

<!-- RELATED:START -->

## 相关论文

- [Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)
- [Generalized Tensor-based Parameter-Efficient Fine-Tuning via Lie Group Transformations](../../ICCV2025/model_compression/generalized_tensor-based_parameter-efficient_fine-tuning_via_lie_group_transform.md)
- [Trans-PEFT: Transferable Parameter-Efficient Fine-Tuning on Evolving Base Models](../../ACL2025/model_compression/trans_peft_transferable.md)

<!-- RELATED:END -->
