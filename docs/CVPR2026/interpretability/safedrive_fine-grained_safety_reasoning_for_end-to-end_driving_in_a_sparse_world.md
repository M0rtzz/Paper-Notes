---
title: >-
  [论文解读] SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World
description: >-
  [CVPR 2026][端到端驾驶] 提出 SafeDrive 端到端规划框架，通过轨迹条件化的稀疏世界模型（SWNet）模拟关键实体的未来行为，再由细粒度推理网络（FRNet）进行逐实例碰撞评估和逐时刻可行驶区域合规评估，在 NAVSIM 上 PDMS 达 91.6、仅 0.5% 碰撞率，Bench2Drive 驾驶分 66.8%。
tags:
  - CVPR 2026
  - 端到端驾驶
  - 安全推理
  - 稀疏世界模型
  - 轨迹评估
  - 碰撞预测
---

# SafeDrive: Fine-Grained Safety Reasoning for End-to-End Driving in a Sparse World

**会议**: CVPR 2026  
**arXiv**: [2602.18887](https://arxiv.org/abs/2602.18887)  
**代码**: 将公开  
**领域**: others / 自动驾驶  
**关键词**: 端到端驾驶, 安全推理, 稀疏世界模型, 轨迹评估, 碰撞预测

## 一句话总结

提出 SafeDrive 端到端规划框架，通过轨迹条件化的稀疏世界模型（SWNet）模拟关键实体的未来行为，再由细粒度推理网络（FRNet）进行逐实例碰撞评估和逐时刻可行驶区域合规评估，在 NAVSIM 上 PDMS 达 91.6、仅 0.5% 碰撞率，Bench2Drive 驾驶分 66.8%。

## 研究背景与动机

**领域现状**：端到端自动驾驶（E2E）将感知-预测-规划统一为一个模型，减少模块间误差传播。近期方法通过轨迹评估（如 Hydra-MDP）或世界模型（如 OccWorld, WoTE）来增强安全性。

**现有痛点**：
   - 轨迹评估方法仅做场景级整体安全评分，缺乏对"为什么安全/不安全"的显式推理，无法精确区分细微差异的轨迹
   - 稠密世界模型（BEV/occupancy）以网格为中心，缺乏对象间的交互关系建模，难以捕捉动态交互风险

**核心矛盾**：安全评估需要**实例级、时序化**的细粒度推理，而现有方法只能提供粗粒度的整体评分

**切入角度**：借鉴人类驾驶员对风险的推理方式——先识别可能碰撞的对象，再评估每个对象在未来各时刻的碰撞风险

**核心 idea**：构建**稀疏世界模型**聚焦关键动态实体，实现逐对象、逐时刻的精细安全推理

## 方法详解

### 整体框架

SafeDrive 由三个核心模块组成：
- **ProposalNet**：编码多模态传感器输入（相机+LiDAR）为 BEV 特征，检测周围实例，筛选安全感知的候选轨迹
- **SWNet（稀疏世界网络）**：为每条候选轨迹构建轨迹条件化的稀疏世界，模拟周围实体的未来行为
- **FRNet（细粒度推理网络）**：基于稀疏世界表示进行逐对象碰撞风险和逐时刻可行驶区域合规评估

输入为多帧多模态传感器数据，输出为最安全的驾驶轨迹。

### 关键设计

1. **ProposalNet — 候选轨迹生成与粗筛**:

    - 功能：从 BEV 特征中生成并粗筛安全候选轨迹
    - 核心思路：用 K-means 聚类训练集中的 ego 轨迹得到 $K$ 个锚轨迹，通过轨迹引导的可变形注意力从 $F_\text{BEV}$ 中沿轨迹采样特征：$\hat{\mathcal{Q}}_\text{plan} = \text{FFN}(\text{Deform-Attn}_\text{traj}(\mathcal{Q}_\text{plan}, \mathcal{A}, F_\text{BEV}))$
    - 对每条轨迹预测模仿分和 5 项安全指标（NC/DAC/TTC/C/EP），选取 top-$K'$ 候选
    - 设计动机：先粗筛减少后续稀疏世界构建的计算量

2. **SWNet — 稀疏世界网络**:

    - 功能：为每条候选轨迹构建一个包含周围关键实体的"稀疏世界"，模拟未来状态
    - **稀疏世界生成器**：将检测到的实例查询 $\mathcal{O}_\text{ins}$ 复制 $K'$ 份，与对应候选轨迹查询拼接，形成稀疏世界集合 $\mathcal{W} = \{c_\text{plan}^j, o_\text{ins}^1, ..., o_\text{ins}^N\}_{j=1}^{K'}$
    - **世界交互模块**：先用自注意力建模 ego 与周围实体的交互，再用轨迹引导可变形注意力聚合 BEV 时空特征：
    $\bar{\mathcal{W}} = \text{World-SelfAttn}(\mathcal{W}), \quad \hat{\mathcal{W}} = \text{FFN}(\text{Deform-Attn}_\text{traj}(\bar{\mathcal{W}}, \mathcal{T}_\text{world}, F_\text{BEV}))$
    - 设计动机：不同于稠密世界模型预测整个场景，稀疏世界只关注有限个关键实体，计算高效且保留了实例级交互信息

3. **FRNet — 细粒度推理网络**:

    - 功能：基于稀疏世界表示进行两种细粒度安全评估
    - **逐对象无过错碰撞评分 (PwNC)**：将每条轨迹查询与每个实例查询拼接，通过 MLP+sigmoid 预测逐时刻碰撞概率 $p_\text{pwnc}^{i,j} \in [0,1]^H$，总评分为所有实例所有时刻的乘积 $P_\text{PwNC}^j = \prod_{i=1}^N \prod_{h=1}^H p_\text{pwnc}^{i,j}(h)$
    - **逐时刻可行驶区域合规 (TwDAC)**：用 ConvNeXt-v2 从 BEV 生成静态分割图，沿轨迹采样可行驶区域概率，在未来 ego box 的 9 个关键点采样后相乘
    - 设计动机：逐对象评估碰撞风险，比整体评分更精细——能回答"会和哪个对象在什么时刻碰撞"

### 损失函数 / 训练策略

通过加权对数求和整合 PwNC、TwDAC 与场景级安全评分，选择综合得分最高的轨迹作为最终驾驶计划。训练采用模仿学习结合安全评分的多任务损失。

## 实验关键数据

### 主实验 — NAVSIM 开环

| 方法 | NC | DAC | TTC | EP | PDMS |
|------|-----|-----|-----|-----|------|
| GoalFlow (之前SOTA) | 98.4 | 98.3 | 94.6 | 85.0 | 90.3 |
| SeerDrive | 98.4 | 97.0 | 94.9 | 83.2 | 88.9 |
| **SafeDrive** | **99.5** | **99.0** | **97.2** | 84.3 | **91.6** |

NAVSIM EPDMS 排行：

| 方法 | NC | DAC | EP | EPDMS |
|------|-----|-----|-----|-------|
| GaussianFusion | 98.3 | 97.3 | 87.5 | 85.0 |
| DiffusionDrive | 98.2 | 96.2 | 87.4 | 84.8 |
| **SafeDrive** | **99.5** | **99.0** | 88.6 | **87.5** |

### Bench2Drive 闭环

SafeDrive 达 66.8% 驾驶分，且 NAVSIM 12146 场景中仅 61 次碰撞（0.5%），碰撞率极低。

### 关键发现
- 稀疏世界模型比稠密 BEV/occupancy 世界模型更适合安全推理：聚焦实例级交互
- 逐对象碰撞评估（PwNC）的贡献大于场景级整体评分
- NC（无过错碰撞）指标提升最显著（99.5% vs 98.4%），体现了细粒度安全推理的优势
- TwDAC 通过 9 关键点采样可行驶区域概率图捕捉边界精细过渡

## 亮点与洞察
- **"稀疏世界"概念的提出**：用有限个关键实体替代稠密场景表示，既降低计算量又保留交互信息  
- **安全推理从"整体评分"升级为"逐对象逐时刻"**：可以精确判断"和谁在什么时候可能碰撞"，可解释性强
- **轨迹条件化的世界模型**：同一场景下不同候选轨迹对应不同的交互预测，比无条件预测更准确
- **人类驾驶直觉的形式化**：先识别风险对象再评估碰撞，符合人类认知过程

## 局限与展望
- 仅在 NAVSIM 和 Bench2Drive 验证，实际部署场景（如恶劣天气、施工区域）未测试
- 稀疏世界依赖检测模块的实例发现质量，漏检关键对象会导致安全推理失败
- PwNC 的乘积评分方式在实例很多时可能过度保守（所有概率相乘趋近 0）
- 目前使用 ResNet-34 作为图像骨干，更强的视觉骨干可能进一步提升性能

## 评分
- 新颖性: ⭐⭐⭐⭐ 稀疏世界模型+细粒度安全推理的组合有较好创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 开环闭环双验证，与多种方法充分对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 推进了端到端自动驾驶安全性的 SOTA

<!-- RELATED:START -->

## 相关论文

- [TDATR: Improving End-to-End Table Recognition via Table Detail-Aware Learning and Cell-Level Visual Alignment](tdatr_improving_end-to-end_table_recognition_via_table_detail-aware_learning_and.md)
- [FINER: MLLMs Hallucinate under Fine-grained Negative Queries](finer_mllms_hallucinate_under_fine-grained_negative_queries.md)
- [Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)
- [Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](../../ACL2025/interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/interpretability/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)

<!-- RELATED:END -->
