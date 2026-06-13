---
title: >-
  [论文解读] ProDrive: Proactive Planning for Autonomous Driving via Ego-Environment Co-Evolution
description: >-
  [CVPR 2026][自动驾驶][主动规划] ProDrive 把一个 query-centric 轨迹规划器和一个 BEV 世界模型端到端联合训练，让规划器输出的候选轨迹和"决策语义 token"驱动世界模型预测未来场景演化，再把基于未来结果的奖励梯度回传给规划器，从而把"只看当前观测的反应式规划"升级为"预判未来的主动式规划"，在 NAVSIM v1 上把 PDMS 刷到 86.6（优于所有对比基线）。
tags:
  - "CVPR 2026"
  - "自动驾驶"
  - "主动规划"
  - "世界模型"
  - "BEV"
  - "轨迹评估"
  - "自我-环境协同进化"
---

# ProDrive: Proactive Planning for Autonomous Driving via Ego-Environment Co-Evolution

**会议**: CVPR 2026  
**arXiv**: [2604.25329](https://arxiv.org/abs/2604.25329)  
**代码**: 无（论文提及 released implementation，未给出明确链接，⚠️ 以原文为准）  
**领域**: 自动驾驶 / 端到端规划 / 世界模型  
**关键词**: 主动规划、世界模型、BEV、轨迹评估、自我-环境协同进化

## 一句话总结
ProDrive 把一个 query-centric 轨迹规划器和一个 BEV 世界模型端到端联合训练，让规划器输出的候选轨迹和"决策语义 token"驱动世界模型预测未来场景演化，再把基于未来结果的奖励梯度回传给规划器，从而把"只看当前观测的反应式规划"升级为"预判未来的主动式规划"，在 NAVSIM v1 上把 PDMS 刷到 86.6（优于所有对比基线）。

## 研究背景与动机
**领域现状**：端到端自动驾驶（UniAD、VADv2、PARA-Drive、TransFuser 等）已成主流范式——直接把多视角传感器观测映射成规划轨迹，靠联合优化减少模块级联误差。与此并行，世界模型也被引入驾驶系统，用来预测未来场景。

**现有痛点**：这两条线没真正打通。世界模型在多数系统里只是"配角"——要么当数据引擎合成长尾场景，要么当强化学习的环境模拟器，要么当一个**外挂的轨迹评估器**对候选轨迹做事后重排（reranking）。而轨迹生成本身，仍然几乎只由当前观测决定。

**核心矛盾**：规划本质上还是**反应式（reactive）**的——未来推理顶多帮你给候选轨迹打分挑一个，却很少作为"一等公民"的训练信号去直接塑造规划器内部的决策过程。在动态多智能体环境里，这种"只看当下、不预判演化"的规划容易做出短视决策，酿成安全事故。即便是近期引入预测/想象的 future-aware 框架（SeerDrive、PWM、FutureSightDrive、ImagiDrive），也大多没把**规划侧的决策表示**和**预测的未来场景**紧耦合起来。

**本文目标**：让"未来推理"成为规划器训练的内在组成部分，而不是事后的评估外挂；具体要解决（1）世界模型怎么拿到规划器的决策语义、（2）未来评估结果怎么反向塑造规划器。

**切入角度**：作者观察到，规划器在多轮 query refinement 中累积的 ego token 里，编码了"为什么这么开"的丰富语义；如果把这些 token 注入世界模型，未来预测就能 condition 在"规划意图"上，而不只是"轨迹坐标"上。

**核心 idea**：用"自我模块 ↔ 环境模块"的双向耦合（ego token 前向注入 + 未来奖励梯度反向回传），让规划意图建模和环境演化建模在端到端训练中**协同进化（co-evolution）**，实现主动式规划。

## 方法详解

### 整体框架
ProDrive 接收多视角相机图像、LiDAR 点云和自车状态（速度/加速度/横摆角速度），输出未来 4 秒、8 个 waypoint 的轨迹。整个系统由两个紧耦合模块组成：**Ego Module**（query-centric 规划器）负责从观测里生成 K=64 条多样化候选轨迹及其"轨迹 token / ego token"；**Environment Module**（BEV 世界模型）以这些 token 和当前 BEV 状态为条件，递归预测未来 BEV 场景演化，并对所有候选并行地算奖励、选出最优轨迹。关键在于二者不是"先规划再重排"的串行流水线，而是在**特征层**（ego token 注入世界模型）和**优化层**（世界模型的 future-aware 梯度回传规划器）双向打通，使规划器不再只被当前观测优化，而是被"预测出来的未来"显式塑造。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["多视角图像 + LiDAR<br/>+ 自车状态"] --> B["Ego Module：query-centric 规划器<br/>L 层迭代交叉注意力 refine ego token"]
    B --> C["K=64 条候选轨迹<br/>+ refined ego token Q(L)"]
    C -->|ego token 注入 s_i = MLP(Q_plan)| D["Environment Module：BEV 世界模型<br/>递归预测未来 BEV + 奖励评估"]
    D --> E["按总分 R^k argmax<br/>选出最终轨迹"]
    D -->|主动梯度反向回传| B
```

### 关键设计

**1. Ego Module：用迭代交叉注意力把 ego token 逐层 refine 成多样候选轨迹**

针对"端到端规划器怎么生成既准又多样的候选、同时积累可供世界模型使用的决策语义"这个问题，Ego Module 基于 BEVFormer 架构：ResNet-34 抽多视角图像特征，自车状态投影成 ego-conditioned 特征加到一组可学习的 ego token $\mathbf{Q}^{(0)}\in\mathbb{R}^{(K\cdot T)\times d}$ 上，然后经 $L$ 层**共享参数**的级联 refinement。每一层做两步：先用 MLP 把当前 token 解码成轨迹 $\boldsymbol{\tau}^{(l)}=\mathrm{MLP}_{\text{traj}}(\mathbf{Q}^{(l)})\in\mathbb{R}^{K\times T\times 3}$（K 条候选），再把这些轨迹 waypoint 当作 3D 参考点，做 deformable cross-attention 反向去采样多视角图像特征：$\mathbf{Q}^{(l+1)}=\mathrm{EgoRefiner}(\mathbf{Q}^{(l)},\boldsymbol{\tau}^{(l)},\mathbf{F}_{\text{img}})$。这种"轨迹位置当 query 索引去看图"的循环，让 token 逐层在"与各候选轨迹相关的空间位置"上吸收越来越丰富的场景上下文，$L$ 层后得到最终候选 $\{\boldsymbol{\tau}^{k}\}_{k=1}^{K}$ 和富含规划语义的 refined ego token $\mathbf{Q}^{(L)}$——后者正是下游耦合的关键载体。

**2. Environment Module：BEV 世界模型递归预测未来、对候选并行打分**

针对"怎么真正在未来交互下评估每条候选轨迹"的问题，Environment Module 用一个独立的 camera-LiDAR backbone（仿 TransFuser）产出初始 BEV 状态 $\mathbf{B}_0$，并为每条候选 $\boldsymbol{\tau}^k$ 编码动作 token $\mathbf{a}^k$。世界模型在 $N$ 次迭代里递归预测未来 BEV：每步把动作 token、ego token、当前 BEV 拼成序列 $\mathbf{F}_i=[\mathbf{a}_i^k;\mathbf{s}_i^k;\mathbf{B}_i^k]$，加场景位置编码后过 Transformer encoder，输出下一步动作 token、增强后的状态 token 和**预测的未来 BEV** $\mathbf{B}_{i+1}^k$。评估时把多步 BEV 特征和动作 token 聚合成奖励表示，两个 head 分别预测模仿奖励（监督自"到专家轨迹的距离"）和仿真奖励（碰撞规避 NC、可行驶区域 DAC、行进度 EP、TTC、舒适度 C）。最终得分

$$R^{k}=w_{0}\log R_{\text{im}}^{k}+w_{1}\log S_{\text{NC}}^{k}+w_{2}\log S_{\text{DAC}}^{k}+w_{3}\log(5S_{\text{TTC}}^{k}+2S_{\text{C}}^{k}+5S_{\text{EP}}^{k})$$

把安全项（NC/DAC）当强约束、效率与舒适软折中；训练和推理都用 $\arg\max$ 选最高分轨迹。与 LAW、World4Drive 那种"外挂世界模型重排"不同，这里世界模型显式 rollout 出未来场景，去掉它（消融见下）会让安全指标大幅崩塌，说明它真在"模拟未来危险"而非只当个奖励回归器。

**3. Ego-Environment Coupling：ego token 注入（前向）+ 主动梯度回传（反向）实现协同进化**

这是 ProDrive 的灵魂，把上面两个模块从"串行"变成"双向耦合"。**前向——ego token 注入**：耦合规划器与世界模型的核心难点是"未来预测会丢掉规划器的内部决策语义"（已有方法如 LAW 只用通用 latent 或裸轨迹坐标 condition 世界模型，把规划过程里积累的上下文都扔了）。ProDrive 把 refined 规划特征直接注入世界模型——$\mathbf{Q}^{(L)}$ reshape 成 $\mathbf{Q}_{\text{plan}}\in\mathbb{R}^{K\times T\times d}$，在世界模型第 $i$ 步把对齐时刻 $t_i$ 的特征投影成 ego token $\mathbf{s}_i^k=\mathrm{MLP}_{\text{state}}(\mathbf{Q}_{\text{plan}}[k,t_i,:])$，使未来预测不仅依赖"自车会往哪走"，还依赖"规划器为什么这么提议"。**反向——主动梯度回传**：因为整套是端到端联合优化，世界模型基于未来结果算出的奖励梯度会回传到规划器（消融里"w/o Proactive Gradient"就是把这条梯度截断），让规划器的内部决策被"预测的未来"直接塑造。前向给未来预测灌入规划语义、反向用未来结果重塑规划，二者构成闭环，就是论文反复强调的"自我-环境协同进化"。

### 损失函数 / 训练策略
端到端多任务总目标为 $\mathcal{L}=\mathcal{L}_{\text{traj}}+\mathcal{L}_{\text{score}}+\mathcal{L}_{\text{reward}}+\mathcal{L}_{\text{wm}}+\mathcal{L}_{\text{aux}}$（正文省略各项系数）。其中：
- **$\mathcal{L}_{\text{traj}}$**：对所有 refinement 阶段加权（$\gamma^{L-l}$）的 winner-take-all 回归损失 + 多样性正则 $\mathcal{L}_{\text{div}}$，鼓励候选既贴近专家又彼此多样。
- **$\mathcal{L}_{\text{score}}$**：规划侧 scorer 用在线 proposal-wise PDM 目标 + 辅助标签（关键智能体状态、有效性、ego 占据区域）监督。
- **$\mathcal{L}_{\text{reward}}=\lambda_{\text{im}}\mathcal{L}_{\text{im}}+\lambda_{\text{sim}}\mathcal{L}_{\text{sim}}+\lambda_{\text{align}}\mathcal{L}_{\text{align}}$：模仿奖励用"到专家轨迹距离"做 softmax 软标签 $q^k\propto\exp(-\|\boldsymbol{\tau}^k-\boldsymbol{\tau}^*\|_2)$；仿真奖励用最近 anchor 轨迹的预算仿真指标做 BCE；对齐项 $\mathcal{L}_{\text{align}}$ 把规划侧 score $\sigma(\ell^{\text{ego}})$ 拉向归一化的世界模型分 $\tilde{R}^{\text{wm}}$，进一步把两模块的打分对齐。
- **$\mathcal{L}_{\text{wm}}$**：用 Focal loss 监督世界模型预测的当前和未来 BEV 语义图，未来 target 以"在未来 BEV 画布上按候选位置渲染 ego box"的 proposal-conditioned 方式构造。

训练细节：两个模块都用 ResNet-34 backbone，4 秒规划窗口、0.5 秒采样 → $T=8$ waypoint，K=64 候选；Adam 双参数组（Ego Module lr $10^{-4}$、Environment Module lr $10^{-5}$，稳定联合优化）；8 卡分布式混合精度训练 15 epoch，per-GPU batch 16。

## 实验关键数据

### 主实验
NAVSIM v1（基于 nuPlan/OpenScene，强调困难交互场景；Navtrain 1192 个、Navtest 136 个场景）。主指标 PDMS 由五因子合成：$\mathrm{PDMS}=\mathrm{NC}\times\mathrm{DAC}\times\frac{5\cdot\mathrm{EP}+5\cdot\mathrm{TTC}+2\cdot\mathrm{C}}{12}$，其中 NC=无责碰撞、DAC=可行驶区域合规、TTC=碰撞时间、C=舒适、EP=自车行进度。

| 方法 | 规划范式 | NC↑ | DAC↑ | TTC↑ | EP↑ | PDMS↑ |
|------|----------|------|------|------|------|--------|
| Human | – | 100.0 | 100.0 | 100.0 | 87.5 | 94.8 |
| TransFuser | Reactive | 97.7 | 92.8 | 92.8 | 79.2 | 84.0 |
| LAW | Proactive（世界模型联训） | 96.4 | 95.4 | 88.7 | 81.7 | 84.6 |
| World4Drive | Reactive（世界模型重排） | 97.4 | 94.3 | 92.8 | 79.9 | 85.1 |
| FSDrive | Proactive（视觉时空 CoT） | 98.2 | 93.8 | 93.3 | 80.1 | 85.1 |
| Epona | Reactive（世界模型联训） | 97.9 | 95.1 | 93.8 | 80.4 | 86.2 |
| Hydra-MDP | Reactive（model-free 重排） | 98.3 | 96.0 | 94.6 | 78.7 | 86.5 |
| **ProDrive（本文）** | **Proactive（世界模型重排）** | 98.0 | 95.4 | 93.7 | **80.7** | **86.6** |

ProDrive 取得 86.6 PDMS，超过所有对比基线。值得注意的是它在 EP（80.7）上明显领先安全型强基线 Hydra-MDP（78.7），说明"预判未来"不仅提安全也提**长程规划效率**，而不是靠极度保守换安全。

### 消融实验
| 配置 | NC↑ | DAC↑ | TTC↑ | EP↑ | PDMS↑ | 说明 |
|------|------|------|------|------|--------|------|
| **ProDrive（Full）** | 98.0 | 95.4 | 93.7 | 80.7 | **86.6** | 完整模型 |
| w/o World Model | 94.9 | 93.9 | 90.8 | 79.5 | 83.5 | 用 MLP 奖励预测器替掉世界模型，掉 3.1 |
| w/o Ego Token Injection | 97.8 | 94.3 | 93.0 | 80.3 | 85.5 | 只靠轨迹 token 连接两模块，掉 1.1 |
| w/o Proactive Gradient | 97.6 | 94.9 | 93.1 | 80.3 | 85.8 | 截断奖励梯度回传规划器，掉 0.8 |

### 关键发现
- **显式建模未来最关键**：去掉世界模型（换成 256×1024×1024×6 的 MLP 直接回归奖励）NC 暴跌到 94.9、PDMS 掉到 83.5（−3.1），证明 Environment Module 不是个可有可无的奖励回归器，而是真在"模拟未来危险"。
- **ego token 注入对稳定性也重要**：去掉它 PDMS 掉到 85.5，且作者观察到世界模型相关损失训练一段后会**急剧上升**，说明注入规划语义还对两模块的协同优化稳定性有帮助。
- **双向耦合缺一不可**：前向（ego token 注入）与反向（主动梯度）各去掉一个都会明显掉点，二者同时存在才实现真正的协同进化。

## 亮点与洞察
- **把世界模型从"事后评估外挂"变成"塑造规划器的训练信号"**：这是相对 World4Drive/Hydra-MDP 这类 reranking 思路的本质区别——未来推理第一次成了直接回传梯度、改写规划器内部决策的一等公民。
- **ego token 注入很巧**：作者抓住"规划器多轮 refine 累积的 ego token 里藏着'为什么这么开'的语义"，把它注入世界模型让未来预测 condition 在"意图"而非"坐标"上，这个视角可迁移到任何"规划器+预测器"耦合系统（如机器人 manipulation 的 plan-and-imagine）。
- **proposal-conditioned 的未来 BEV 监督**：把每条候选轨迹的 ego box 渲染到未来 BEV canvas 上当 target，让世界模型学到"候选特定（candidate-specific）"的未来，而非一个笼统的未来，这让并行评估 64 条候选有了细粒度依据。

## 局限性 / 可改进方向
- **作者自承**：未来工作要扩展到更具表达力的世界模型（当前 BEV 结构化世界模型相对生成式像素级世界模型在场景细节上仍受限）。
- **评测范围有限**：只在 NAVSIM v1（Navtest 仅 136 场景）上验证，且是开环/非反应式仿真评测，未在 CARLA 闭环或真实路测上验证主动规划的实际收益。
- **算力代价**：对 K=64 条候选各做 N 步递归 BEV 预测并行评估，世界模型分支显著增加训练/推理开销，论文未给出明确的延迟/FLOPs 对比，部署成本存疑。
- **改进思路**：可探索"候选数自适应"（先粗筛再对少量候选精细 rollout）以摊薄世界模型成本；或把生成式世界模型蒸馏进 BEV 世界模型以增强长尾交互预测。

## 相关工作与启发
- **vs LAW / Epona（世界模型联合训练，Proactive/Reactive）**：它们把世界模型 condition 在通用 latent 或解耦的动作 token 上，丢掉了规划语义；ProDrive 直接注入规划器 refined 的轨迹/ego token，提供更丰富的 planning-aware 语义，PDMS 从 84.6/86.2 提到 86.6。
- **vs World4Drive / Hydra-MDP（世界模型/model-free 重排，Reactive）**：它们仍是"先规划、世界模型事后挑一个"，规划器本身不被未来塑造；ProDrive 通过主动梯度回传让规划器被未来训练，EP（行进效率）明显更高（80.7 vs 78.7/79.9）。
- **vs PRECOG / PiP / PRIME（planning-conditioned 预测）**：这些工作意识到"周车未来该 condition 在自车意图上"，但仍是 prediction-centric，规划器自身不被未来推理塑造；ProDrive 让未来 BEV rollout 与主动梯度成为规划的内在部分。
- **启发**：把"决策语义 token 注入预测/世界模型 + 预测结果梯度回传决策器"抽象出来，是一个通用的"让下游评估反塑上游决策"的端到端耦合范式，可用于具身规划、对话规划等任何"生成候选→评估→选择"被割裂训练的任务。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把世界模型从评估外挂升级为塑造规划器的训练信号，双向耦合视角清晰且有说服力，但各组件（query-centric 规划、BEV 世界模型、reranking）多为已有积木的精巧组合。
- 实验充分度: ⭐⭐⭐ NAVSIM 上对比充分、三组消融干净有力，但仅单一 benchmark、开环评测、缺延迟/算力对比与闭环验证。
- 写作质量: ⭐⭐⭐⭐ 动机层层递进、公式完整、框架与消融对应清楚，易读。
- 价值: ⭐⭐⭐⭐ "预测结果反塑规划器"的端到端耦合范式对自动驾驶乃至更广的规划任务有迁移价值，PDMS 达到当前 NAVSIM 强基线之上。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Future-Aware End-to-End Driving: Bidirectional Modeling of Trajectory Planning and Scene Evolution](../../NeurIPS2025/autonomous_driving/future-aware_end-to-end_driving_bidirectional_modeling_of_trajectory_planning_an.md)
- [\[CVPR 2026\] ColaVLA: Leveraging Cognitive Latent Reasoning for Hierarchical Parallel Trajectory Planning in Autonomous Driving](colavla_leveraging_cognitive_latent_reasoning_for_hierarchical_parallel_trajecto.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[AAAI 2026\] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving](../../AAAI2026/autonomous_driving/decoupling_scene_perception_and_ego_status_a_multi-context_fusion_approach_for_e.md)
- [\[CVPR 2026\] DLWM: Dual Latent World Models enable Holistic Gaussian-centric Pre-training in Autonomous Driving](dlwm_dual_latent_world_models_enable_holistic_gaussian-centric_pre-training_in_a.md)

</div>

<!-- RELATED:END -->
