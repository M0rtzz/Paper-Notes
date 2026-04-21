---
title: >-
  [论文解读] Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging
description: >-
  [NeurIPS 2025][人体理解][持续学习] 提出测试时持续模型合并(TTCMM)新范式及Mingle框架，通过低秩专家混合架构和自适应零空间约束门控机制，在测试时利用少量无标签样本实现模型动态合并，在多个基准上以7-9%的优势超越SOTA，同时将遗忘降至接近零。
tags:
  - NeurIPS 2025
  - 人体理解
  - 持续学习
  - 模型合并
  - 测试时适应
  - 混合专家
  - 零空间约束
  - 低秩分解
---

# Mingle: Mixture of Null-Space Gated Low-Rank Experts for Test-Time Continual Model Merging

**会议**: NeurIPS 2025  
**arXiv**: [2505.11883](https://arxiv.org/abs/2505.11883)  
**作者**: Zihuan Qiu, Yi Xu, Chiyuan He, Fanman Meng, Linfeng Xu, Qingbo Wu, Hongliang Li (UESTC, DUT)
**代码**: [GitHub](https://github.com/zihuanqiu/MINGLE)  
**领域**: human_understanding  
**关键词**: 持续学习, 模型合并, 测试时适应, 混合专家, 零空间约束, 低秩分解

## 一句话总结

提出测试时持续模型合并(TTCMM)新范式及Mingle框架，通过低秩专家混合架构和自适应零空间约束门控机制，在测试时利用少量无标签样本实现模型动态合并，在多个基准上以7-9%的优势超越SOTA，同时将遗忘降至接近零。

## 研究背景与动机

### 问题背景
持续模型合并(Continual Model Merging, CMM)旨在将独立微调的模型在参数空间中顺序融合，无需访问原始训练数据。这一范式在数据隐私、分布式训练和可扩展性方面具有天然优势。然而，现有方法面临两个核心挑战：

**参数干扰与灾难性遗忘**：随着模型增量合并，重叠或冲突的参数更新不断累积

**测试分布适应性不足**：合并后的模型在面对未见或偏移的任务分布时泛化能力有限

### 已有工作的不足
- **Task Arithmetic (TA)**：简单加权的任务向量相加，对系数敏感且易受任务干扰
- **OPCM**：通过正交投影减少干扰，但忽略了对测试分布的适应
- **AdaMerging/WEMOE**：引入测试时适应，但假设所有模型和数据同时可用，不适用于持续场景
- 现有结构约束(正交投影、线性化、稀疏化)随任务数量增长效果递减

### 核心动机
将测试时适应(TTA)引入持续模型合并，利用少量无标签测试样本(每类5个)在推理时动态调整合并策略，同时通过零空间约束保护已学知识。

## 方法详解

### 整体框架
Mingle = Mixture of Null-Space Gated Low-Rank Experts，由三个核心组件构成。

### 1. 低秩专家混合(Low-Rank Expert Mixture)
当新任务t到达时，从任务向量中构建低秩专家：

- **正交投影**：先将任务向量投影到已学方向的正交补空间，消除与已有专家的重叠
- **截断SVD**：对投影结果做秩r截断SVD，得到紧凑的LoRA形式专家 f_t = BA
- **门控汇聚**：每层输出为预训练模型加上所有专家的门控加权求和

其中门控函数为可学习的线性投影，仅在测试时适应当前任务的门控参数。

### 2. 测试时适应(Test-Time Adaptation)
仅适应当前任务t的门控参数g_t，冻结所有专家和旧门控。优化目标为最小化合并模型与单独微调模型之间的KL散度，使用少量无标签seed样本驱动。

### 3. 自适应零空间约束门控(Adaptive Null-Space Constrained Gating)
核心创新，解决TTA引入的遗忘问题：

**硬零空间投影**：
- 缓存每个任务seed样本的特征协方差矩阵，提取top-k主方向
- 拼接历史任务方向并正交化
- 将梯度投影到旧任务特征子空间的正交补中

**自适应松弛策略**：
硬投影可能过度限制可塑性，因此引入自适应松弛：
- **干扰度量**：计算梯度与每个旧任务方向的对齐度
- **EMA平滑**：用指数移动平均(beta=0.99)抑制随机噪声
- **自适应衰减**：lambda = exp(-gamma * S)，干扰越大的方向保护越强
- **松弛投影器**：在无保护和硬投影间平滑插值

### 理论支撑
Theorem 1证明：只要路由误差足够小，动态MoE的风险严格低于任何静态平均策略。这从理论上支持了数据依赖门控的必要性。

## 实验关键数据

### 实验1：持续合并主实验(CLIP视觉模型)

在三种CLIP-ViT骨干、8/14/20任务设定下，10个随机种子取平均：

| 方法 | ViT-B/32 8任务ACC | ViT-B/32 14任务ACC | ViT-B/32 20任务ACC | ViT-B/16 8任务ACC |
|------|----------|----------|----------|----------|
| C. Task Arithmetic | 67.5% | 66.5% | 60.0% | 77.1% |
| OPCM | 75.5% | 71.9% | 65.7% | 81.8% |
| C. LoRA-WEMOE | 68.8% | 63.8% | 49.6% | 72.6% |
| **Mingle** | **85.8%** | **81.6%** | **77.1%** | **88.3%** |

BWT(遗忘度量，越高越好)：Mingle在所有设定下BWT接近零(-0.1%到-2.2%)，而OPCM为-4.3%到-7.8%，C. LoRA-WEMOE为-13.6%到-27.9%。

### 实验2：消融实验(ViT-B/16)

| TTA | 冻结旧门控 | 零空间约束 | 自适应松弛 | ACC(8) | BWT(8) | ACC(20) | BWT(20) |
|-----|---------|---------|---------|--------|--------|---------|---------|
| 否 | - | - | - | 78.7% | -0.5% | 70.6% | -1.3% |
| 是 | 否 | 否 | 否 | 86.4% | -6.0% | 76.7% | -12.8% |
| 是 | 是 | 否 | 否 | 87.4% | -2.3% | 76.2% | -6.8% |
| 是 | 是 | 是 | 否 | 86.0% | -0.1% | 78.3% | -0.2% |
| 是 | 是 | 是 | 是 | **88.3%** | **-0.4%** | **81.9%** | **-1.9%** |

自适应松弛在保持近零遗忘的同时显著提升了准确率(vs硬约束提升2-4%)。

### 其他关键发现
- **NLP任务**：在Flan-T5-base的8个GLUE任务上，Mingle以83.3%平均准确率超越所有基线
- **鲁棒性**：在7种图像损坏下，Mingle平均ACC 73.2%(BWT -0.2%)，远超OPCM的67.3%(BWT -5.1%)
- **样本效率**：每类仅1个样本即可将ACC从静态的70-79%提升至81-88%
- **计算开销**：每个任务适应约10秒，rank 64下仅36.9k可训练参数

## 亮点

- **新范式定义**：首次形式化测试时持续模型合并(TTCMM)，将TTA引入持续模型合并场景，填补两个领域的交叉空白
- **零空间约束门控**：将梯度更新限制在旧任务特征子空间的正交补中，从机制上阻止遗忘；自适应松弛通过实时监测干扰信号动态调节约束强度
- **高效实用**：仅需每类5个无标签样本、50步适应、约10秒/任务；推理时门控固定无需TTA，保证低延迟部署
- **全面验证**：覆盖视觉(CLIP)+语言(Flan-T5)、3种骨干、8/14/20任务、10个随机序列、7种图像损坏、MTIL基准

## 局限与展望

- **需要独立微调模型**：假设每个任务模型已从共享预训练模型独立微调，不适用于异构模型
- **线性门控表达能力有限**：门控为简单线性投影，在高度重叠的任务分布下可能不足
- **零空间维度增长**：随任务数量增加，累积的零空间维度线性增长(每任务k个方向)，可能挤压新任务的可用梯度子空间
- **仅限分类任务**：实验主要在分类任务上验证，生成、检测等任务的适用性未知
- **seed样本假设**：需要少量无标签测试样本，在完全无数据场景中不适用

## 与相关工作的对比

- **vs OPCM**：OPCM仅通过正交投影减少参数干扰但缺乏测试分布适应；Mingle通过TTA+MoE实现输入依赖的动态合并，ACC领先7-10%
- **vs WEMOE**：WEMOE假设所有模型和数据同时可用(多任务设定)；直接用于持续设定时遗忘严重(BWT < -20%)
- **vs AdaMerging**：在持续设定中也有严重遗忘(BWT -22%到-33%)，通用TTA方法不能直接移植到持续场景
- **vs 传统CL方法**：Mingle在独立微调的合并范式下达到了可比性能(MTIL基准83.0% vs传统CL最佳86.8%)

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ - 首次形式化TTCMM任务，零空间约束门控与自适应松弛设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ - 覆盖视觉+NLP、3种骨干、多任务数、10种序列、鲁棒性、消融、效率分析
- 写作质量: ⭐⭐⭐⭐ - 结构清晰，理论与实验紧密配合，部分符号较繁重
- 价值: ⭐⭐⭐⭐⭐ - 开辟TTCMM新范式，7-9%全面领先极具说服力，实用门槛低

<!-- RELATED:START -->

## 相关论文

- [FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/human_understanding/free-merging_fourier_transform_for_efficient_model_merging.md)
- [FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](../../ICCV2025/human_understanding/fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](../../ICCV2025/human_understanding/a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/human_understanding/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [MoEE: Mixture of Emotion Experts for Audio-Driven Portrait Animation](../../CVPR2025/human_understanding/moee_mixture_of_emotion_experts_for_audio-driven_portrait_animation.md)

<!-- RELATED:END -->
