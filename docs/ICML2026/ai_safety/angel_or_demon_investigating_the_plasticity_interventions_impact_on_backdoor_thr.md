---
title: >-
  [论文解读] Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning
description: >-
  [ICML 2026][AI安全][DRL backdoor] 作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)…
tags:
  - "ICML 2026"
  - "AI安全"
  - "DRL backdoor"
  - "plasticity intervention"
  - "SAM"
  - "loss landscape sharpness"
  - "robust backdoor injection"
---

# Angel or Demon: Investigating the Plasticity Interventions' Impact on Backdoor Threats in Deep Reinforcement Learning

**会议**: ICML 2026  
**arXiv**: [2605.14587](https://arxiv.org/abs/2605.14587)  
**代码**: <https://github.com/maoubo/Plasticity>  
**领域**: AI 安全 / 深度强化学习后门攻击 / 可塑性干预  
**关键词**: DRL backdoor, plasticity intervention, SAM, loss landscape sharpness, robust backdoor injection

## 一句话总结
作者首次系统评估 7 种主流可塑性干预 (SAM/Shrink&Perturb/Weight Clip/SN/WD/LN/ReDo) 对深度强化学习 (DRL) 后门攻击的影响 (14,664 个实验)，发现只有 SAM 是"恶魔"——能显著加剧后门威胁；据此提出"Sweeper-Converter-Connector" 鲁棒后门注入框架并给出基于 loss landscape 锐度的检测信号。

## 研究背景与动机

**领域现状**：DRL 在机器人控制、无人机导航、自动驾驶中应用广泛；同时被发现易受后门攻击 (TrojDRL/BadRL/SleeperNets/UNIDOOR 等)。另一方面，DRL 训练存在"可塑性丢失" 问题 (非平稳输入 + 优化目标漂移会让 agent 渐失学习能力)，因此现代 DRL pipeline 普遍内置 plasticity 干预：Shrink & Perturb、Weight Clipping、Spectral Normalization、Weight Decay、Layer Normalization、ReDo、SAM 等。

**现有痛点**：(1) 后门研究和可塑性研究两条线长期井水不犯河水，从来没人系统问过"可塑性干预到底会让后门更容易还是更难"；(2) 实际部署 DRL agent 时这两类技术几乎总是同时存在，但缺乏指引会导致"以为加了 LN/SAM 是性能改进，实际却是安全漏洞"。

**核心矛盾**：plasticity 干预的设计初衷是稳定训练，对"恶意触发器→目标动作" 的映射学习是不是也有副作用？如果某些干预反而帮助后门更稳更猛，那它们就成了无意中的"攻击放大器"。

**本文目标**：(1) 量化每种干预在两种威胁模型 (TM-Scratch 从头训练时注入 / TM-Post 拿到模型后注入) 下对 ASR (攻击成功率) 与 BTP (良性任务表现) 的影响；(2) 找出影响背后的内在机理；(3) 据机理设计更鲁棒的后门注入框架，并提出后门检测信号。

**切入角度**：把可塑性领域已经成熟的三个 pathological 指标——**权重幅值 / 有效秩 / loss landscape 锐度**——直接挪来当后门内部属性的诊断仪表盘，对每种干预下后门 agent 进行排名分析。

**核心 idea**：用大规模 (14,664 cases) 受控实验 + 三指标病理诊断，把"干预效果"分解到三种机理 (M1 激活通路扰动 / M2 表征空间压缩 / M3 后门梯度放大)，再用机理反推鲁棒攻击与检测策略。

## 方法详解

### 整体框架
这是一篇"系统性实证+机理分析+派生设计" 三段式工作：(1) **实证 RQ1**——构造 2 (TM-Scratch/Post) × 8 (干预) × 47 (后门任务) × 4 (攻击算法) × 3 (seed) = 9,024 cases，再加 5,640 cases 评估干预组合，总计 14,664 cases，绘出 ASR/BTP 变化谱；(2) **机理 RQ2**——对每个干预下 backdoored agent 测三种病理指标 (weight magnitude / effective rank / loss landscape sharpness)，建立 8×3 的病理向量 $\mathbf{v}(p_i)$ 并排名；(3) **设计 RQ3**——基于机理设计 SCC 注入框架与 sharpness-based 检测；并用 Pathological Distance 量化干预组合的协同程度。

### 关键设计

1. **三类病理机理的实证拆解 (M1 / M2 / M3)**:

    - 功能：把杂乱的"某干预 ASR ±x%" 现象归约为可解释的三种内在机制。
    - 核心思路：(M1) **激活通路扰动**——Shrink&Perturb / Weight Clipping / ReDo 通过裁剪或重置权重让"后门 pathway" 和"良性 pathway" 互相竞争资源；Fig.6 显示后门攻击会让 actor 网第二层少量权重幅值剧增 (后门通路稀疏)，Weight Clip 一刀切让它们被压回，导致重建竞争。(M2) **表征空间压缩**——Spectral Norm / Weight Decay / Layer Norm 通过限制 Lipschitz 常数或平滑激活，把原本与良性梯度近乎正交的后门梯度 (dot product≈0) 拉到几乎完全对齐 (≈1.0)，把后门从稀疏单通路变成与良性共享的多通路，反而在非平稳训练下更不稳定。(M3) **后门梯度放大**——SAM 通过对抗扰动捕捉损失尖锐方向，正好对应后门方向 (后门样本导致 loss landscape sharpness 范围扩 6 倍多)；SAM 把这些梯度放大并把后门通路引向 flat-minimum，让它对参数扰动鲁棒。
    - 设计动机：单看 ASR/BTP 数字无法解释"为什么 SAM 反向" 这种反直觉结果；引入病理诊断把统计结果与具体网络变化关联，使结论可推广 (而非只是某个超参下的偶然)。

2. **SCC 鲁棒后门注入框架 (Sweeper-Converter-Connector)**:

    - 功能：把"哪些干预对攻击有利"的发现反向用作攻击设计 cookbook，构造 TM-Post 下的强鲁棒后门。
    - 核心思路：观察到组合干预 (Plastic/SLac/SSW) 比单 SAM 更猛 (ASR 0.178→0.418, BTP 0.745→0.915)，提炼出三步注入流程——(a) **Sweeper**：用 Shrink&Perturb / Weight Clip / ReDo 类干预清空一部分良性 pathway，为后门腾位置 (利用 M1)；(b) **Converter**：用 Spectral Norm / Weight Decay / LN 把后门梯度从正交拉向对齐良性梯度，让后门变成多 pathway 结构 (利用 M2)；(c) **Connector**：用 SAM 把多 pathway 联合优化到 flat minima，使表征稳定共存 (利用 M3)。再定义 Pathological Distance $PD(A)=\sum_{i<j}\|\mathbf{v}(p_i)-\mathbf{v}(p_j)\|_2$ 衡量组合中干预间的病理差异；实验证实 $PD$ 越大 (e.g. SSW=18.64)，后门威胁越强。
    - 设计动机：现实部署常用多干预组合 (Plastic/Swiss Cheese 等)，attacker 只要按 SCC 模板挑互补干预即可获得免费攻击放大；这是给"plasticity-aware 安全评估" 提供的具体威胁模型。

3. **基于 loss landscape sharpness 的后门检测信号**:

    - 功能：把后门最显著的外在表象 (loss 锐度异常) 转化为防御端可监控的指标。
    - 核心思路：观察后门攻击让 loss landscape sharpness 的波动范围扩大 635.22%，是三大病理中差异最显著的；除 SAM 之外的所有干预都会进一步加剧这一异常 ($v_{i3}>v_{13}$)。Defender 可在 agent 训练全过程实时监测 sharpness 时间序列；显著异常尖峰 / 急降都可视为可疑。配合任务自适应阈值与多源噪声去耦，可作为通用 DRL 后门预警。
    - 设计动机：现有 DRL 后门检测大多依赖触发器或专门 probe，本文提出的 sharpness 信号无须知道触发器，且可与任何 DRL 训练流程兼容；缺点是任务间 sharpness 基线差异大、误报源待研究——作者明确把这两点列为开放问题。

### 损失函数 / 训练策略
本文没有提出新损失，主体是评测协议设计——攻击端用 transition tampering 把 trigger 注入 (state,action,reward) 三元组、用 backdoor reward 强化绑定；防御端不施加任何 (本文只研究干预副作用)。任务覆盖 OpenAI Gym 4 经典控制 + 2 物理控制 + PyBullet 3 机器人，含离散/连续动作、稀疏/稠密奖励、冷启动/非冷启动条件；4 种攻击 (TrojDRL/BadRL/SleeperNets/UNIDOOR)，47 backdoor tasks，包含单后门/多后门。每个干预的超参遵循对应原论文。

## 实验关键数据

### 主实验
TM-Post 场景下 (干预对已训练好的 agent 更显著影响)，机器人控制任务的代表性 ASR / BTP 变化：

| 干预 | ASR (机器人) | BTP (机器人) | 主要病理影响 |
|------|--------------|---------------|-----------------|
| None (baseline) | 0.178 ± 0.157 | 0.745 ± 0.230 | — |
| Weight Clipping | ↓ 17.46% | ↓ 20.19% | M1 通路扰动 |
| Spectral Norm | ↓ 11.78% | ↓ 中等 | M2 表征压缩 |
| Layer Norm | ↓ 中等 | ↓ 11.93% | M2 表征压缩 |
| Weight Decay | ↓ 轻度 | ↓ 轻度 | M2 表征压缩 |
| Shrink & Perturb | ↓ 轻度 | ↓ 轻度 | M1 软扰动 |
| ReDo | ↓ 轻度 | ↓ 轻度 | M1 神经元重置 |
| **SAM** | **↑ 0.326 (+83%)** | **↑ 0.814 (+9%)** | **M3 梯度放大** |

干预组合对比 (机器人控制 + SAM 系列)：

| 组合 | 含 SAM? | ASR | BTP | Pathological Distance |
|------|---------|------|------|------------------------|
| None | — | 0.178 ± 0.157 | 0.745 ± 0.230 | N/A |
| Plastic | ✓ | 0.368 ± 0.144 | 0.724 ± 0.362 | 9.43 |
| SLac | ✓ | 0.417 ± 0.146 | 0.816 ± 0.276 | 17.42 |
| **SSW** | ✓ | **0.418 ± 0.092** | **0.915 ± 0.131** | **18.64** |
| Swiss Cheese (WD+LN) | ✗ | ≈ LN 单独 | ≈ LN 单独 | 0.52 |

### 消融实验

| 配置 | 现象 | 解读 |
|------|------|------|
| TM-Scratch (注入与训练同时) | ASR 仅微动 (LN 最大 -8.84%) | 表征尚未稳定，干预效果被训练动态稀释 |
| TM-Post (后训练注入) | ASR/BTP 变化显著 | 已稳定模型才显出干预影响 |
| 后门 vs 普通训练 (Fig.4) | weight magnitude 范围 +98.63%, effective rank +19.16%, sharpness +635.22% | sharpness 是最强的后门外在标志 |
| 仅 v.s. 多干预组合 | $PD$ 越大攻击越强；同机理组合 (Swiss Cheese) 几乎无增益 | 互补机理才能联合放大威胁 |
| Spectral Norm 后梯度对齐分析 (Fig.7) | 后门-良性梯度从 ≈0 升到 ≈1.00 | 验证 M2 表征压缩→通路共享 |
| Weight Clipping 3D 权重可视化 (App. Fig.13) | TM-Scratch 影响小、TM-Post 强 | 参数灵活性是关键变量 |

### 关键发现
- **反直觉**：SAM (本意稳定训练) 是唯一加剧后门的干预，因为它正好对后门带来的尖锐 loss 方向敏感并把它放大并压扁。
- **TM-Post 比 TM-Scratch 更敏感**：已收敛的良性表征要被后门"挤出空间" 才能注入，此时干预对参数灵活性的限制效应放大。
- **BTP 比 ASR 更敏感**：良性表征复杂、依赖大量协同参数，干预破坏后难以重建；后门表征稀疏、局部 pathway 易于快速重建。
- **干预组合非加性**：同机理组合 (Swiss Cheese = WD+LN, $PD$=0.52) 几乎无叠加效应；异机理组合 (SSW, $PD$=18.64) 攻击放大显著——SCC 的 Pathological Distance 是有效设计指标。
- **三大病理中 sharpness 最具检测价值**：后门攻击使 sharpness 波动范围扩 6 倍，其他干预 (除 SAM) 会进一步加剧这种异常，可被防御端利用。

## 亮点与洞察
- **大规模受控实验设计**：14,664 cases 覆盖 2 威胁模型 × 8 干预 × 5 组合 × 4 攻击 × 9 任务 × 多 seed 的笛卡尔积；这种规模在 DRL 安全文献里罕见，结论可信度高。
- **跨领域桥接**：把"可塑性" 与"后门安全" 两个看似无关的子社区用三大病理指标对接起来，是少有的"跨子领域诊断" 工作；对其他安全方向 (公平性、隐私) 都有方法论启发。
- **"角色复用" 思维**：SAM 在防御文献里被推崇为提升泛化的良药，本文揭示它同时是攻击放大器，提醒人们任何 generalization 工具都可能是双刃剑。
- **从机理到设计**：SCC 三角 (Sweeper-Converter-Connector) 把诊断结果直接转译为攻击设计 cookbook，并提供 PD 这种可量化的协同度指标——这种"机理→流程→指标" 三件套套路可借鉴。
- **sharpness 检测的可行性**：因 sharpness 本身就是优化器常规监控量，部署成本极低；这是一个被低估的免费防御信号。

## 局限与展望
- 实验任务集中在低维状态空间的控制任务 (Gym/PyBullet)，对 Atari / StarCraft 等高维像素观察任务是否依然成立未知。
- SCC 框架仅给出概念性设计，没有正式实现 + 对比真实"统一注入算法"——读者得自己拼装。
- Sharpness-based 检测面临两大挑战 (作者自承)：(1) 任务间 sharpness 基线方差大，难以设统一阈值；(2) 其他训练异常 (奖励 hacking、unstable critic) 也可能引起异常 sharpness。
- 干预超参敏感性虽在 App.E 做了 ablation，但只验证 "趋势一致"，未深入找出最坏情况组合。
- "组合干预放大攻击" 的结论倚赖现成的 5 种组合 (Plastic/Swiss Cheese/Lac/SLac/SSW)，未做系统的组合搜索；理论上还有更猛的组合存在。
- 本文不提防御对策 (除 sharpness 检测建议)，距离闭环安全方案还有距离。

## 相关工作与启发
- **vs TrojDRL/BadRL/SleeperNets/UNIDOOR**：这些工作只研究 vanilla DRL 下的攻击；本文把"现代 DRL pipeline 标配" 加进来评估，揭示了攻击 / 干预的复合效应。
- **vs Klein et al. 2024 (可塑性 survey)**：提供了四类干预动机分类；本文沿用其框架但调换视角，从"保塑" 改到"安全副作用"。
- **vs Lee et al. 2023 (SAM for DRL)**：把 SAM 当 plasticity-preserving 良药；本文相当于给出 SAM 的"安全警告标签"。
- **vs deep learning 后门防御 (Li et al. 2024b)**：DL 后门可通过 finetune-pruning 缓解；本文显示 DRL 中"裁剪" 类干预 (Weight Clip) 也有类似效果但代价是 BTP 下降，trade-off 仍开。
- **vs Lyle et al. 2024 (Swiss Cheese / multi-intervention plasticity)**：他们提倡组合干预提升泛化；本文显示同样的组合在 backdoor 视角下可能成为"漏洞放大器"，给"plasticity-aware 安全评测" 留下作业。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次系统刻画 plasticity 干预与 DRL 后门的交互关系，提出 SCC + sharpness 检测两条新思路。
- 实验充分度: ⭐⭐⭐⭐⭐ 14,664 cases 的笛卡尔积评估，配合三大病理诊断分析，论据极扎实。
- 写作质量: ⭐⭐⭐⭐ RQ 驱动结构、概念命名清楚 (M1/M2/M3 → SCC)，公式与图配合到位。
- 价值: ⭐⭐⭐⭐⭐ 直接影响所有部署带 plasticity 干预的 DRL 系统的安全实践，给攻防双方都提供了可操作信号。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning](../../NeurIPS2025/ai_safety/impact_of_dataset_properties_on_membership_inference_vulnerability_of_deep_trans.md)
- [\[ICML 2026\] Regret-Based Federated Causal Discovery with Unknown Interventions](regret-based_federated_causal_discovery_with_unknown_interventions.md)
- [\[ICLR 2026\] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning](../../ICLR2026/ai_safety/beware_untrusted_simulators_--_reward-free_backdoor_attacks_in_reinforcement_lea.md)
- [\[ICML 2025\] Adversarial Inception Backdoor Attacks against Reinforcement Learning](../../ICML2025/ai_safety/adversarial_inception_backdoor_attacks_against_reinforcement_learning.md)
- [\[CVPR 2026\] Monte Carlo Stochastic Depth for Uncertainty Estimation in Deep Learning](../../CVPR2026/ai_safety/mcsd_uncertainty_estimation.md)

</div>

<!-- RELATED:END -->
