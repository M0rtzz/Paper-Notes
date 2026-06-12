---
title: >-
  [论文解读] CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning
description: >-
  [CVPR 2026][机器人][连续隐运动] 提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。
tags:
  - "CVPR 2026"
  - "机器人"
  - "连续隐运动"
  - "伪动作标签"
  - "逆动力学模型"
  - "时序对比学习"
  - "视频-机器人联合训练"
---

# CoMo: Learning Continuous Latent Motion from Internet Videos for Scalable Robot Learning

**会议**: CVPR 2026  
**arXiv**: [2505.17006](https://arxiv.org/abs/2505.17006)  
**代码**: [github.com/MCG-NJU/CoMo](https://github.com/MCG-NJU/CoMo)  
**领域**: 机器人学习  
**关键词**: 连续隐运动, 伪动作标签, 逆动力学模型, 时序对比学习, 视频-机器人联合训练

## 一句话总结

提出 CoMo，通过早期时序差分（Td）和时序对比学习（Tcl）两个机制协同解决连续隐运动学习中的捷径学习问题，从互联网视频中提取精细的连续伪动作标签，使视频数据与机器人动作在统一连续分布下联合训练，显著提升策略性能。

## 研究背景与动机

从互联网视频中学习隐运动（latent motion）是扩展机器人学习的关键方向，但现有方法面临根本瓶颈：

**离散化的信息损失**：LAPA、UniVLA、Moto-GPT 等先驱工作采用 VQ-VAE 将隐运动离散化（codebook 大小仅 8-16），虽然有效抑制了捷径学习（shortcut learning），但离散化导致大量运动信息丢失，难以捕获复杂和精细的动态。

**捷径学习问题**：如果直接学习连续隐运动（去掉 VQ），逆动力学编码器（IDM）会倾向于从未来帧中提取大量静态背景信息（而非前景运动），因为解码器可以更容易地重建像素细节。模型退化为无效的未来帧预测器。

**离散 vs 连续的分布不匹配**：离散隐运动与连续机器人动作之间存在天然的分布鸿沟，阻碍了统一策略的联合学习，需要复杂的多阶段训练流程。

核心问题：**能否在不用 VQ 的情况下，从无动作标注的视频中学习连续隐运动？** 连续表示更适合精细的帧间变化建模，且与连续机器人动作的分布天然一致。

## 方法详解

### 整体框架

CoMo 想从无动作标注的互联网视频里学出**连续**的隐运动表示，又要绕开"直接学连续表示会被捷径学习毁掉"的陷阱。它沿用逆动力学编码器-前向动力学解码器（IDM-FDM）的老框架：IDM 看当前帧 $O_t$ 和未来帧 $O_{t+n}$，吐出连续隐运动 $Z_{t,t+n}$；FDM 再拿 $Z_{t,t+n}$ 和 $O_t$ 去重建未来帧 $\hat{O}_{t+n}$。真正的新东西是用早期时序差分（Td）和时序对比学习（Tcl）两个机制替掉 VQ 离散化，逼 IDM 把注意力放在前景运动而不是背景上；训好的 IDM 再为视频生成连续伪动作标签，与机器人动作在统一策略里联合训练。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["当前帧 O_t + 未来帧 O_t+n"] --> B["共享 MAE-ViT 编码<br/>token 特征 F_t, F_t+n"]
    B --> C["早期时序差分 Td<br/>D_t=F_t+n−F_t，仅输入 F_t 与 D_t、抽走未来帧"]
    C --> D["Motion Q-former<br/>可学习 query 输出连续隐运动 Z"]
    D --> E["前向动力学解码器 FDM<br/>Z + O_t 重建未来帧（训练 IDM）"]
    D -. InfoNCE 约束 .-> F["时序对比学习 Tcl<br/>正样本时序偏移 / 负样本时序反转"]
    D --> G["联合统一策略学习<br/>视频伪标签 + 机器人动作 co-train"]
    D -. 免策略诊断 .-> H["MSE 与 S-PCFC<br/>评估隐运动质量"]
```

### 关键设计

**1. 早期时序差分 Td：把未来帧的"答案"从输入里抽走**

连续表示最大的隐患是捷径学习——IDM 一旦能直接看到未来帧的完整特征，解码器就会偷懒去抄背景像素重建画面，整个模型退化成一个无效的未来帧预测器。Td 用两步堵住这条捷径。第一步借共享的 MAE 预训练 ViT 把当前帧和未来帧编成 token 级特征 $F_t, F_{t+n}$，再做元素级差分 $D_t = F_{t+n} - F_t$，这个稀疏差分天然放大了帧间真正在动的部分。第二步是关键：送进 Motion Q-former 的是 $[F_t, D_t]$ 而不是 $[F_t, F_{t+n}]$——未来帧的完整表示被拿掉了，编码器再想抄背景也无从下手，捷径从源头被切断。

**2. 时序对比学习 Tcl：用正负样本逼模型聚焦"怎么动"**

光有 Td 还不够：稀疏差分里仍可能混入无关信息，而单纯的对比学习又容易只学到"前景是什么、在哪"这种身份信息，漏掉"怎么动"。Tcl 因此设计了一组特殊样本对——正样本是时序略有偏移的 $Z_{t,t+n+\delta}$ 与 $Z_{t,t+n}$（$\delta \in [-n/5, n/5]$，运动方向一致只是早晚一点），负样本是时序方向反转的 $Z_{t+n,t}$ 与 $Z_{t,t+n}$（同一段画面但运动倒着走）。再用 InfoNCE 拉近正样本、推开负样本：

$$\mathcal{L}_{\text{tcl}} = -\log\frac{e^{S_1}}{e^{S_1} + e^{S_2} + e^{S_3}}$$

因为负样本是"倒放"，模型要把它和正样本分开，就不得不真正编码运动的方向性而非静态外观。于是 Td 管"从哪里学"（抽走未来帧）、Tcl 管"学什么"（聚焦前景运动方向），两者刚好互补。

**3. 联合统一策略学习：连续分布让视频和动作能一锅烩**

离散隐运动和连续机器人动作之间隔着一道分布鸿沟，过去得靠多阶段、先 motion 后 action 的复杂流程来弥合。CoMo 的连续隐运动和连续机器人动作天生共享同一个连续分布，于是可以在一个统一策略模型里直接联合训练，只需给两者各分配一个轻量 head，省掉了显式的两阶段预训练。

**4. MSE 与 S-PCFC：不跑策略就能诊断隐运动好坏**

评估隐运动质量通常要把整套策略训完再看成功率，代价很高。CoMo 给了两个轻量代理指标：MSE 是训练一个 MLP 从隐运动回归真实动作的误差，越低说明隐运动里编码的动作相关信息越多；S-PCFC 是"过去→当前"与"未来→当前"两段运动的余弦相似度，越高说明隐运动里混进了越多与动作无关的背景噪声。两个指标合起来，能在不跑策略的情况下快速判断 Td/Tcl 各自起了什么作用。

### 损失函数 / 训练策略

- CoMo 联合最小化加权 InfoNCE 损失、像素级重建损失和感知损失
- 使用 120K 互联网视频（SAM-V 40K + EgoVid 40K + Droid 40K）训练 IDM-FDM
- 策略训练支持 Diffusion（LIBERO）和自回归（CALVIN）两种架构

## 实验关键数据

### 主实验

LIBERO 基准（每任务仅 10 条机器人轨迹 + 视频伪标签）：

| 方法 | Spatial | Object | Goal | Long | 平均SR |
|------|---------|--------|------|------|--------|
| DP (无视频) | 72.3 | 82.3 | 70.3 | 56.7 | 70.4 |
| GR2-like（未来帧特征） | 76.0 | 92.0 | 73.3 | 53.7 | 73.8 |
| GR00T（离散隐运动） | 80.7 | 83.3 | 80.0 | 59.7 | 75.9 |
| Dynamo（协方差正则化） | 75.3 | 92.7 | 80.7 | 46.0 | 73.7 |
| **CoMo** | **80.3** | **97.0** | **81.0** | **62.0** | **80.1** |

CALVIN ABC→D 基准：

| 方法 | 1步 | 2步 | 3步 | 4步 | 5步 | 平均 |
|------|-----|-----|-----|-----|-----|------|
| 无运动 | 0.772 | 0.494 | 0.307 | 0.191 | 0.114 | 1.878 |
| 离散(Moto) | 0.801 | 0.575 | 0.409 | 0.283 | 0.187 | 2.255 |
| **CoMo** | **0.882** | **0.732** | **0.589** | **0.490** | **0.377** | **3.070** |
| CoMo ×4维度 | 0.891 | 0.758 | 0.646 | 0.529 | 0.423 | 3.247 |

### 消融实验

| 配置 | 平均SR | MSE↓ | S-PCFC↓ | 说明 |
|------|--------|------|---------|------|
| 未来帧特征 | 73.8 | 2.14 | 1.000 | 大量背景噪声 |
| 离散(Dis.) | 75.9 | 5.67 | 0.481 | 信息损失严重 |
| 连续基线(Con.) | 75.2 | 1.63 | 0.927 | 捷径学习问题 |
| Con.+Td | 76.9 | 1.52 | 0.889 | Td 降低 S-PCFC |
| Con.+Tcl | 77.9 | 1.31 | 0.623 | Tcl 大幅降低 S-PCFC |
| **Con.+Td+Tcl (CoMo)** | **80.1** | **1.26** | **0.550** | 两者协同最优 |

### 关键发现

- Td 和 Tcl 各自有效但互补：Td 主要降低 MSE（增强运动线索），Tcl 主要降低 S-PCFC（抑制背景噪声）
- 连续表示优于离散表示：CoMo 平均 SR 80.1% vs 离散 75.9%，尤其在精细操作任务（Object 97.0% vs 83.3%）
- 维度可扩展：CoMo 从 128→512 维持续提升（CALVIN 3.070→3.247），而仅 Td 会随维度增加引入更多噪声
- 真实世界实验同样验证连续隐运动优势，尤其在低容错精细操作（开抽屉、插面包）上

## 亮点与洞察

- **连续 > 离散**的论证充分且有说服力：离散化虽能抑制捷径学习但代价太高
- **Td + Tcl 的协同设计**非常精巧：Td 解决"从哪里学"（移除未来帧直接信息），Tcl 解决"学什么"（正负样本引导前景聚焦 + 运动方向性）
- **MSE + S-PCFC 评估体系**提供了无需昂贵策略评估的快速诊断工具
- 联合训练的简洁性是重要工程价值：一阶段统一训练 vs 多阶段复杂流程

## 局限与展望

- 仅在 LIBERO 和 CALVIN 上验证，泛化到更复杂的真实世界场景有待探索
- 隐运动维度扩展的上限和最优维度选择缺乏理论指导
- Tcl 的正负样本构建依赖时序偏移假设（$\delta$ 范围），在快速运动场景可能不理想
- 真实世界实验规模较小（每任务 25 条轨迹 + 25 条人手视频）

## 相关工作与启发

- 与 LAPA/GR00T 等离散隐运动方法的核心对比：CoMo 证明连续化是更好的方向
- 时序差分来自视频理解社区（TDN），跨领域迁移到机器人学习效果显著
- 对比学习的正负样本设计有创意：用时序反转作为负样本，天然适合运动方向区分

## 评分

- **新颖性**: ⭐⭐⭐⭐ Td+Tcl 替代 VQ 的思路清晰有新意，但各组件并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融极其全面，MSE/S-PCFC 分析深入，跨架构验证充分
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，实验分析有深度，但部分表述略显冗余
- **价值**: ⭐⭐⭐⭐⭐ 为从互联网视频扩展机器人学习提供了实用且原理清晰的方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] IGen: Scalable Data Generation for Robot Learning from Open-World Images](igen_scalable_data_generation_for_robot_learning_from_open-world_images.md)
- [\[ICCV 2025\] Moto: Latent Motion Token as the Bridging Language for Learning Robot Manipulation from Videos](../../ICCV2025/robotics/moto_latent_motion_token_as_the_bridging_language_for_learning_robot_manipulatio.md)
- [\[CVPR 2026\] Chain of World: World Model Thinking in Latent Motion (CoWVLA)](chain_of_world_world_model_thinking_in_latent_motion.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](dawn_pixel_motion_diffusion_robot_control.md)
- [\[CVPR 2026\] Lifelong Imitation Learning with Multimodal Latent Replay and Incremental Adjustment](lifelong_imitation_learning_multimodal_latent_rep.md)

</div>

<!-- RELATED:END -->
