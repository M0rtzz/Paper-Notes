---
title: >-
  [论文解读] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency
description: >-
  [CVPR 2026][自监督学习][3D形状补全] 提出 LaS-Comp，一种零样本、类别无关的 3D 形状补全框架，通过 Explicit Replacement Stage 在空间域注入已知几何 + Implicit Alignment Stage 在隐空间梯度优化边界一致性，桥接了预训练 3D 基础模型的隐空间与空间域之间的 gap，在多种部分观测模式下达到 SOTA。
tags:
  - CVPR 2026
  - 自监督学习
  - 3D形状补全
  - 零样本
  - 3D基础模型
  - 隐空间-空间一致性
  - 点云补全
---

# LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency

**会议**: CVPR 2026  
**arXiv**: [2602.18735](https://arxiv.org/abs/2602.18735)  
**代码**: https://github.com/wylyan/LaS-Comp (有)  
**领域**: 自监督学习  
**关键词**: 3D形状补全, 零样本, 3D基础模型, 隐空间-空间一致性, 点云补全

## 一句话总结
提出 LaS-Comp，一种零样本、类别无关的 3D 形状补全框架，通过 Explicit Replacement Stage 在空间域注入已知几何 + Implicit Alignment Stage 在隐空间梯度优化边界一致性，桥接了预训练 3D 基础模型的隐空间与空间域之间的 gap，在多种部分观测模式下达到 SOTA。

## 研究背景与动机
3D 形状补全是计算机视觉和图形学的基础问题，目标是从部分观测重建完整 3D 形状，广泛应用于机器人、自动驾驶和 AR/VR。一个理想的补全方法需要：(i) 鲁棒处理多样化的部分缺失模式（单视角扫描、随机裁剪、语义部件缺失）；(ii) 跨类别泛化；(iii) 不依赖配对数据；(iv) 支持文本引导和自动补全。

传统监督方法依赖配对数据且无法泛化到未见类别。近期利用生成先验的方法（SDS-Complete, ComPC, GenPC）依赖"部分输入至少能渲染出一张完整图像"的假设——当缺失区域从任何视角都可见时，不完整的渲染导致结果退化。

而最新的 3D 基础模型（TRELLIS, Direct3D-S2）采用"隐空间生成"pipeline：先用 VAE 将形状编码到紧凑隐空间，再在隐空间训练扩散/flow-matching 模型。这产生了一个独特挑战：**完整形状和部分输入即使在重叠区域几何完全相同，其隐空间编码也存在显著差异**。因此，直接在隐空间补全不可靠。

LaS-Comp 的核心 idea 是：通过显式空间域替换 + 隐式隐空间对齐，桥接 latent 与 spatial 之间的 domain gap，释放 3D 基础模型的补全潜力。

## 方法详解

### 整体框架
从高斯噪声出发，通过多步去噪迭代，在部分输入 $\boldsymbol{S}_p$ 的引导下逐步恢复完整几何。每一步 $t \in [0,1]$ 包含两个互补阶段：
1. **Explicit Replacement Stage (ERS)**：在空间域显式注入 $\boldsymbol{S}_p$ 的几何信息 → 产生更新的隐特征 $\boldsymbol{x}_t^*$
2. **Implicit Alignment Stage (IAS)**：通过几何对齐损失梯度优化 $\boldsymbol{x}_t^*$ → 产生空间对齐的 $\boldsymbol{x}_{t-dt}$

最终解码 $\boldsymbol{S}_c = \mathcal{D}(\boldsymbol{x}_0)$ 得到完整形状。

### 关键设计
1. **Explicit Replacement Stage (ERS)**:

    - 功能：将部分输入的几何信息显式注入到生成过程的隐特征中，确保对已知区域的忠实保留
    - 核心思路：分为 Clean Branch 和 Noisy Branch 两个并行分支
        - **Clean Branch**：用生成器预测无噪声隐特征 $\hat{\boldsymbol{x}}_{0|t} = \boldsymbol{x}_t - t \cdot \mathcal{G}(\boldsymbol{x}_t, t)$，解码为完整形状 $\boldsymbol{S}_{0|t}$，然后通过空间掩码替换：$\boldsymbol{S}'_{0|t} = \boldsymbol{S}_p \odot \boldsymbol{M} + \boldsymbol{S}_{0|t} \odot (1-\boldsymbol{M})$，再编码回隐空间 $\boldsymbol{x}^*_{0|t} = \mathcal{E}(\boldsymbol{S}'_{0|t})$
        - **Noisy Branch** + Partial-aware Noise Schedule (PNS)：对观测区域（$\boldsymbol{M}=1$）施加时间依赖递减的扰动以保持稳定，对缺失区域（$\boldsymbol{M}=0$）用纯高斯噪声鼓励多样探索：$\boldsymbol{x}^*_{1|t} = \boldsymbol{M} \odot (\sqrt{1-t} \cdot \hat{\boldsymbol{x}}_{1|t} + \sqrt{t} \cdot \boldsymbol{\epsilon}_1) + (1-\boldsymbol{M}) \odot \boldsymbol{\epsilon}_2$
    - 最终通过 flow 插值合成：$\boldsymbol{x}^*_t = (1-t) \cdot \boldsymbol{x}^*_{0|t} + t \cdot \boldsymbol{x}^*_{1|t}$
    - 设计动机：直接在隐空间替换会因 domain gap 失败，而在空间域显式替换后重编码可绕过这一问题；PNS 使已知/未知区域获得不同程度的随机性

2. **Implicit Alignment Stage (IAS)**:

    - 功能：修复 ERS 在观测区域与生成区域边界处可能引入的不连续性
    - 核心思路：从 $\boldsymbol{x}_t^*$ 预测无噪隐特征，解码后计算几何对齐损失：$\mathcal{L}_{\text{align}} = \text{BCE}(\boldsymbol{S}_{0|t} \odot \boldsymbol{M}, \boldsymbol{S}_p \odot \boldsymbol{M})$，然后单步梯度更新隐特征：$\boldsymbol{x}^{\text{aligned}}_{0|t} = \hat{\boldsymbol{x}}_{0|t} - \eta \cdot \nabla_{\hat{\boldsymbol{x}}_{0|t}} \mathcal{L}_{\text{align}}$
    - 设计动机：ERS 的空间替换保证了忠实度但可能在边界产生伪影，IAS 通过在隐空间的梯度优化平滑这些不一致，且只需一步更新，计算开销极低
    - 注意：这个损失不更新模型参数，只更新隐特征本身

3. **Omni-Comp Benchmark**:

    - 功能：提出新的综合评估基准，包含 30 个不同类别对象、3 种部分缺失模式（单视角扫描、随机裁剪、语义部件缺失）、共 180 个样本
    - 数据来源：10 个 Redwood 真实扫描 + 10 个 YCB 日常物体 + 10 个合成形状
    - 设计动机：现有 benchmark 规模小（Redwood 仅 10 个）、类别少（KITTI/ScanNet ≤2 类）且仅限单一部分模式

### 损失函数 / 训练策略
LaS-Comp 是 **training-free** 的——不需要任何额外训练，直接利用预训练 3D 基础模型（TRELLIS 或 Direct3D-S2）进行推理。IAS 中的梯度更新学习率 $\eta = 1 \times 10^{-5}$。每个形状补全仅需 ~20 秒，比现有零样本方法快 3 倍以上。

## 实验关键数据

### 主实验
| 数据集/指标 | 本文 (TRELLIS) | ComPC (之前SOTA) | 提升 |
|------------|---------------|-----------------|------|
| Redwood CD↓/EMD↓ | **1.42/1.84** | 1.95/2.59 | 27.2%/29.0% |
| Synthetic CD↓/EMD↓ | **1.11/1.41** | 1.61/2.09 | 31.1%/32.5% |
| ScanNet-Chair UCD↓/UHD↓ | **0.8/2.0** | 2.0/5.3 | 60%/62% |
| KITTI-Car UCD↓/UHD↓ | **1.4/4.5** | 1.1/5.7 | - |
| Omni-Comp Single Scan CD↓ | **2.21** | 4.24 | 47.9% |
| Omni-Comp Random Crop CD↓ | **2.60** | 5.48 | 52.6% |
| Omni-Comp Semantic Part CD↓ | **3.30** | 6.37 | 48.2% |

### 消融实验
| 配置 | Redwood CD↓ | 说明 |
|------|-------------|------|
| 完整 LaS-Comp (TRELLIS) | 1.42 | 最优 |
| 仅 ERS（无 IAS） | 1.68 | IAS 提供~15%改善 |
| 仅 IAS（无 ERS） | 2.31 | ERS 是核心 |
| 无 PNS（均匀噪声调度） | 1.89 | PNS 重要 |
| Direct3D-S2 backbone | 1.64 | TRELLIS 更优 |

### 关键发现
- 在 Omni-Comp 上，之前方法对 Random Crop 和 Semantic Part 模式性能急剧下降（因为依赖"至少一个视角完整"假设），而 LaS-Comp 表现稳健
- 兼容不同 3D 基础模型（TRELLIS 和 Direct3D-S2），且都显著优于 baseline
- 支持 text-guided 补全（通过基础模型的 CFG 机制），可控制生成结果的语义
- 推理速度 ~20 秒/形状，比 ComPC（~60s）和 SDS-Complete（>5min）快很多

## 亮点与洞察
- **Training-free**：不需要任何额外训练，直接利用预训练模型，部署成本极低
- **解决了核心 domain gap**：发现并解决了"隐空间编码的部分形状 vs 完整形状差异"这一之前未被重视的关键问题
- **ERS+IAS 互补设计精妙**：显式替换保证忠实度，隐式对齐保证平滑性，二者缺一不可
- **Partial-aware Noise Schedule**：对已知/未知区域采用不同噪声策略，体现了对补全任务不对称性的深刻理解
- **Omni-Comp benchmark** 填补了多模式补全评估的空白

## 局限与展望
- 依赖预训练 3D 基础模型的质量，若基础模型对某类别生成能力弱，补全效果也会受限
- 每步需要编码-解码往返（ERS），增加了推理开销
- IAS 仅做单步梯度更新，更多步优化可能进一步提升边界质量但增加时间
- 当前仅处理刚性物体，对铰接物体、变形物体等未验证

## 相关工作与启发
- **RepPaint/FlowDPS** 启发了 ERS 的 clean-noisy 双分支设计，从 2D 图像修复迁移到 3D
- **TRELLIS/Direct3D-S2** 等 3D 基础模型的隐空间生成范式为本工作提供了基础
- 与 ComPC/GenPC 的关键区别：不依赖 2D 渲染假设，直接在 3D 隐空间操作
- 启发：在其他隐空间生成任务（如 3D 编辑、风格迁移）中，latent-spatial 一致性可能同样是关键挑战

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统解决隐空间3D基础模型的补全问题，ERS+IAS设计原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 4个已有benchmark+自建Omni-Comp，2个backbone，多种部分模式
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细，图示直观
- 价值: ⭐⭐⭐⭐⭐ training-free方案实用性强，Omni-Comp benchmark有持续价值

<!-- RELATED:START -->

## 相关论文

- [CObL: Toward Zero-Shot Ordinal Layering without User Prompting](../../ICCV2025/self_supervised/cobl_toward_zero-shot_ordinal_layering_without_user_prompting.md)
- [Escaping Plato's Cave: Towards the Alignment of 3D and Text Latent Spaces](../../CVPR2025/self_supervised/escaping_platos_cave_towards_the_alignment_of_3d_and_text_latent_spaces.md)
- [Alpha-SQL: Zero-Shot Text-to-SQL using Monte Carlo Tree Search](../../ICML2025/self_supervised/alpha-sql_zero-shot_text-to-sql_using_monte_carlo_tree_search.md)
- [TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [Regularized Latent Dynamics Prediction is a Strong Baseline for Behavioral Foundation Models](../../ICLR2026/self_supervised/regularized_latent_dynamics_prediction_is_a_strong_baseline_for_behavioral_found.md)

<!-- RELATED:END -->
