---
title: >-
  [论文解读] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
description: >-
  [NeurIPS 2025][多模态VLM][VLA] 提出 ForceVLA，在 VLA 框架中将 6 轴力/力矩传感引入为一等模态，通过 FVLMoE（力感知混合专家）模块在动作解码阶段动态融合视觉-语言嵌入与实时力反馈，在 5 项接触密集操作任务上平均成功率提升 23.2%，个别任务达 80%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "VLA"
  - "力反馈"
  - "MoE"
  - "接触密集操作"
  - "机器人操作"
---

# ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation

**会议**: NeurIPS 2025  
**arXiv**: [2505.22159](https://arxiv.org/abs/2505.22159)  
**代码**: 待开源  
**领域**: 多模态VLM  
**关键词**: VLA, 力反馈, MoE, 接触密集操作, 机器人操作

## 一句话总结
提出 ForceVLA，在 VLA 框架中将 6 轴力/力矩传感引入为一等模态，通过 FVLMoE（力感知混合专家）模块在动作解码阶段动态融合视觉-语言嵌入与实时力反馈，在 5 项接触密集操作任务上平均成功率提升 23.2%，个别任务达 80%。

## 研究背景与动机

**领域现状**：VLA（Vision-Language-Action）模型如 OpenVLA、π₀ 通过大规模多模态预训练实现了通用机器人操作，在语义理解和零样本泛化上表现出色。

**现有痛点**：VLA 模型严重依赖视觉和语言线索，忽略了力觉感知——这在接触密集任务（插入、工具使用、装配）中至关重要。遮挡或视觉条件差时，纯视觉策略容易失败。力的需求还会随任务阶段变化（抓取→插入→表面接触各需不同力控），现有方法缺乏感知和适应这种动态变化的机制。

**核心矛盾**：如何将力觉信号有效整合进 VLA 系统，使之能根据触觉反馈动态调整操作策略，而不是简单地将力信号拼接到输入（实验证明简单拼接甚至会让 π₀-fast 性能从 31% 暴跌到 14.2%）。

**本文目标**：设计一种力感知融合机制，使 VLA 模型能在接触密集任务中根据实时力反馈进行上下文感知的动作生成。

**切入角度**：6D 外部力应视为"一等模态"，在动作专家模块中正式集成以实现基于力反馈的阶段感知动作生成。关键洞察是力信号应在 VLM 编码之后引入（保留预训练表示），并通过 MoE 实现自适应融合。

**核心 idea**：用 Mixture-of-Experts 在 VLM 编码后动态融合力觉、视觉、语言三种模态，使动作生成能感知接触动态。

## 方法详解

### 整体框架
ForceVLA 基于 π₀ 架构构建。输入包括双目 RGB 图像（基座+手腕摄像头）、语言指令、本体感受（7D TCP 位姿+夹爪宽度）和 6 轴力/力矩。视觉和语言输入经 SigLIP (PaliGemma) 编码为上下文嵌入，与力觉信号一起送入 FVLMoE 融合后，指导 flow-based 动作头生成动作序列 $A_t = \{a_t, \ldots, a_{t+H-1}\}$。

### 关键设计

1. **FVLMoE 输入映射**:

    - 功能：将力信号投射为与 VL 嵌入同维的 token
    - 核心思路：对原始 6 轴力/力矩 $f_{raw} \in \mathbb{R}^6$ 通过线性投射 $E_F = \phi_F(f_{raw}) \in \mathbb{R}^{D_{model}}$ 生成力 token，与 VL 特征 $E_{VL} \in \mathbb{R}^{N_{VL} \times D_{model}}$ 拼接为 $E_{in} = [E_{VL}; E_F] \in \mathbb{R}^{(N_{VL}+1) \times D_{model}}$
    - 设计动机：力信号必须在 VLM 编码之后引入——消融实验证明在 VLM 之前引入力（"MoE before VLM"）会导致 0% 成功率，因为破坏了预训练视觉-语言特征分布

2. **多模态路由与融合**:

    - 功能：通过编码器层和稀疏 MoE 层实现自适应跨模态融合
    - 核心思路：$E_{in}$ 先通过多头自注意力编码器层进行全局交互得到 $E_{enc}$，再送入稀疏 MoE 层（4 个专家 MLP，top-1 路由，即每个 token 只激活 1 个专家）。门控网络根据 token 内容动态路由，残差连接后得到 $E_{fused}$，最后线性投射匹配动作空间维度
    - 设计动机：MoE 的稀疏激活使不同模态/任务阶段可以由不同专家处理，天然适合力觉-视觉-语言这种高度异质的多模态融合

3. **融合特征注入动作流头**:

    - 功能：将融合特征注入 flow-based 去噪动作生成器
    - 核心思路：从 $E_{FVLMoE}$ 中提取最后 $H_{action}$ 个 token 组成引导信号 $G_{FVLMoE} \in \mathbb{R}^{H_{action} \times D_a}$，与 VLM 对当前状态和噪声动作轨迹的编码 $S_{suffix}$ 逐元素相加，送入 flow matching 去噪过程
    - 设计动机：逐元素加法使力感知上下文直接调制动作序列的每个时间步，实现精细的力感知动作生成

### 损失函数 / 训练策略
基于 π₀ 的 flow matching（rectified flow）训练框架。使用 ForceVLA-Data（244 条轨迹，5 个任务，约 14 万同步时间步）进行端到端训练，约 50 条示范/任务。

## 实验关键数据

### 主实验

| 任务 | π₀-base w/o F | π₀-base w/ F | ForceVLA |
|------|--------------|-------------|----------|
| Insert USB | 30% | 35% | 55% |
| Pump Bottle | 55% | 60% | 70% |
| Insert Plug | 45% | 40% | 80% |
| Wipe Board-1 | 30% | 40% | 40% |
| Peel Cucumber | — | — | — |
| **平均** | **37.3%** | **40.2%** | **60.5%** |

| 模型 | 平均剥皮长度(cm)↑ | 最少剥皮次数↓ |
|------|-------------------|--------------|
| π₀-base w/o F | 10.27 | 14 |
| π₀-base w/ F | 13.17 | 10 |
| ForceVLA | **14.12** | **7** |

### 消融实验

| 配置 | 成功率 |
|------|-------|
| π₀ baseline (无力) | 45% |
| Linear before VLM | 55% |
| MoE before VLM | 0% |
| Concatenate after VLM | 60% |
| ForceVLA (FVLMoE after VLM) | **80%** |

### 关键发现
- 简单加入力信号仅提升 2.9pp（37.3%→40.2%），而通过 FVLMoE 融合可提升 23.2pp，说明"怎么融合"比"有没有力"更关键
- 在 VLM 之前引入力的 MoE 直接崩溃（0%），验证了后融合策略的必要性——预训练 VLM 的特征空间不能被打扰
- π₀-fast 架构加入力信号后反而从 31% 降到 14.2%，其紧凑 token 空间被未经预训练的力 token 干扰
- 泛化实验中，ForceVLA 在视觉遮挡下仍达 90% 成功率，说明力反馈有效弥补了视觉信息缺失

## 亮点与洞察
- **力作为一等模态**：不是简单附加力信号，而是系统性地回答了"在哪里融合""怎么融合"的问题，消融实验构成了完整的设计空间探索
- **MoE 用于异质模态融合**：利用 MoE 的稀疏路由天然地让不同模态/任务阶段由专门的专家处理，比统一的 MLP 融合更符合多模态的异质性
- **后融合设计原则**：保护预训练 VLM 不被新模态干扰的设计可迁移到其他新模态引入场景（如触觉、声音等传感器信号融入 VLA）

## 局限与展望
- 使用估计的外部力而非直接测量，在需要极致触觉灵敏度的场景可能精度不足
- 训练数据仅 244 条轨迹/5 个任务，规模较小，泛化到更多任务类型待验证
- 实验依赖高成本的 Flexiv Rizon 机器人（内置力传感器），低成本平台的适配尚未验证
- 仅考虑 6 轴力/力矩，未引入触觉传感器（如 GelSight）提供更丰富的接触信息

## 相关工作与启发
- **vs π₀**: ForceVLA 直接在 π₀ 基础上增加力觉模态，验证了力觉对接触密集任务的关键价值，π₀ 本身缺乏力觉感知是其短板
- **vs TLA/Tac-Man**: 这些触觉方法关注指尖触觉传感，而 ForceVLA 使用手腕处的 6 轴力传感器，两者互补——可考虑同时引入触觉和力觉
- **vs ForceMimic**: ForceMimic 使用力作为额外输入但采用静态融合，ForceVLA 的 MoE 动态路由是关键改进

## 评分
- 新颖性: ⭐⭐⭐⭐ 力觉引入 VLA 的方向有价值，FVLMoE 设计合理但核心技术（MoE 融合）并非全新
- 实验充分度: ⭐⭐⭐⭐ 5 个真实任务+泛化测试+详细消融，但数据规模和任务多样性可更丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机充分，消融实验有说服力
- 价值: ⭐⭐⭐⭐ 为 VLA 系统引入力觉感知开辟了新方向，23.2% 的提升具有实际意义

## 与相关工作的对比
- **vs π₀/π₀.5**: π₀ 框架强大但完全缺乏力觉感知，ForceVLA 证明在接触密集任务中力觉是不可或缺的模态，23.2% 的提升说明纯视觉-语言策略的天花板
- **vs OpenVLA/OpenVLA-OFT**: 这些方法专注于视觉-语言-动作的对齐和微调效率，但均未考虑触觉/力觉信号，在接触密集场景下表现受限
- **vs FORGE**: FORGE 研究力反馈在控制中的角色但未集成到 VLA 框架中，ForceVLA 首次在 VLA 系统中系统性地解决了力觉融合问题
- **vs TLA/Tac-Man**: 这些方法使用指尖触觉传感（视触觉），ForceVLA 使用腕部 6 轴力传感器——两者感知的物理量不同，未来可考虑结合

## 启发与关联
- 后融合设计原则（新模态在 VLM 编码后引入）可迁移到任何需要向 VLA 添加新传感模态的场景（如声音、温度、振动传感器）
- MoE 稀疏路由在多模态融合中的优势值得在更多领域验证——不同专家自动学会处理不同模态/任务阶段
- ForceVLA-Data 的数据采集流程（VR 遥操作 + 多模态同步记录）可作为构建力感知数据集的标准范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] What Can RL Bring to VLA Generalization? An Empirical Study](what_can_rl_bring_to_vla_generalization_an_empirical_study.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](../../ICML2026/multimodal_vlm/any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)
- [\[AAAI 2026\] FT-NCFM: An Influence-Aware Data Distillation Framework for Efficient VLA Models](../../AAAI2026/multimodal_vlm/ft-ncfm_an_influence-aware_data_distillation_framework_for_efficient_vla_models.md)
- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching VLA Models](../../CVPR2026/multimodal_vlm/flowhijack_dynamics_aware_backdoor_attack_on_flow_matching_vla_models.md)

</div>

<!-- RELATED:END -->
