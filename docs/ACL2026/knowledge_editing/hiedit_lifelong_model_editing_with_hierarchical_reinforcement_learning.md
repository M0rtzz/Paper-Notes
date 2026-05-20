---
title: >-
  [论文解读] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning
description: >-
  [ACL 2026][知识编辑][lifelong model editing] HiEdit 用分层强化学习把"终身模型编辑"拆成 high-level 选层 + low-level 算梯度更新两个子任务，让 hypernetwork 按知识自适应地只动一半的层，把强基线 RLEdit 平均再提 8.48%…
tags:
  - "ACL 2026"
  - "知识编辑"
  - "lifelong model editing"
  - "hierarchical RL"
  - "hypernetwork"
  - "layer selection"
  - "sparse update"
---

# HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning

**会议**: ACL 2026  
**arXiv**: [2604.11214](https://arxiv.org/abs/2604.11214)  
**代码**: https://github.com/yangfanww/hiedit  
**领域**: 知识编辑 / 终身学习 / 分层强化学习  
**关键词**: lifelong model editing, hierarchical RL, hypernetwork, layer selection, sparse update

## 一句话总结
HiEdit 用分层强化学习把"终身模型编辑"拆成 high-level 选层 + low-level 算梯度更新两个子任务，让 hypernetwork 按知识自适应地只动一半的层，把强基线 RLEdit 平均再提 8.48%。

## 研究背景与动机

**领域现状**：终身模型编辑（Lifelong Model Editing, LME）要在不重训的前提下对部署中的 LLM 持续注入新知识。主流范式是"先定位再编辑"：先确定影响层 $\mathcal{W}$，再施加扰动 $\tilde{\nabla}_{\mathcal{W}}$；近期 RLEdit 把整个序列编辑建模成 RL 任务，让 hypernetwork 通过 PPO 风格优化跨越上万次编辑。

**现有痛点**：所有现存方法都把扰动施加在"静态、稠密"的一组层上——不管要编的是哪条知识，都对同一批层（往往是 5–7 层 MLP）做更新。在 Llama-3-8B 上跑 ZsRE 序列编辑，5000 步后泛化能力和过往编辑都开始崩塌（catastrophic forgetting）。

**核心矛盾**：研究已表明不同知识激活 LLM 中不同组件，但"locating"阶段给所有 instance 用同一套层，相当于过度修改了与当前知识无关的参数；同时也把 hypernetwork 的优化空间不必要地收紧到一个次优解上。

**本文目标**：把"该编哪些层"从离线一次定好的静态决策，变成对每条知识的可学习动态决策。

**切入角度**：把 LME 从扁平的 MDP 升级为 Hierarchical MDP——高层 option 选层，低层 action 算参数更新，离散选层和连续更新分离开训练。

**核心 idea**：用分层 RL 把"where to edit"和"how to edit"解耦，并加一个 intrinsic reward 鼓励稀疏选层，做到 instance-aware、localized 的精准编辑。

## 方法详解

### 整体框架
HiEdit 把每一步编辑建模成 Hierarchical MDP $(\mathcal{S}, \mathcal{A}, \Omega, \mathcal{P}, r, \gamma)$。在第 $t$ 步：

1. 拿一对待编知识 $(x_t, y_t)$，做一次标准 SFT 得到全部影响层的梯度矩阵 $\nabla \mathcal{W}_t = \{\nabla \mathcal{W}_{t,1}, \dots, \nabla \mathcal{W}_{t,L}\}$，对每层做低秩分解 $\nabla \mathcal{W}_{t,l} = v_l u_l^\top$。
2. 高层 hypernetwork $\pi_\phi$ 读全部 $(u_l, v_l)$，输出 option $\omega_t \in \{0,1\}^L$（一个 mask），只激活 $K$ 个层。
3. 低层 hypernetwork $\mathcal{H}_\theta$ 只在被激活的层上跑 MEND/MALMEN 风格的编辑网络，把 $(u_l, v_l)$ 映射成伪向量 $(\tilde u_l, \tilde v_l)$，得到参数更新 $\tilde{\nabla} \mathcal{W}_{t,l} = \tilde v_l \tilde u_l^\top$。
4. 整条编辑轨迹结束后，按高/低层 reward 联合优化两套 hypernetwork。

### 关键设计

1. **高层 importance router（动态选层）**：

    - 功能：根据当前知识动态决定要改哪些层。
    - 核心思路：把每层 $(u_l \| v_l)$ 过一个 layer-shared gradient encoder $\mathbf{W}_{\text{GradEnc}}$ + layer-specific scale & offset $\mathbf{SPE}_l$ 得到 $h_l$，再 concat 起来过 gate network 输出 $z_t \in \mathbb{R}^L$，最后用 $\mathbf{TopK}(z_t, K)$ 产生 mask $m_t$。为了让离散 TopK 仍可微，借鉴 MoE 的 straight-through estimator：$m_t = \mathbf{sg}(m_t - z_t) + z_t$，前向用硬 mask、反向把梯度透传到 $z_t$。
    - 设计动机：让 layer selection 也成为可学习的 high-level action，而不是离线一次定死，这样不同知识可以走不同层路径。

2. **Intrinsic reward 鼓励稀疏**：

    - 功能：让高层倾向于"用尽量少的层完成编辑"。
    - 核心思路：高层 reward 不是绝对值，而是部分选层和全选层的相对优势 $r_{\text{high},t} = r_{\text{low}}(s_t, \omega_t, a_t) - r_{\text{low}}(s_t, \mathbf{1}, a_t)$，其中 $\mathbf{1}$ 表示全选。若部分选层的编辑损失比全选还低，高层就能拿正 reward。
    - 设计动机：直接监督"选少了好"很难，把信号建模成"部分 vs. 全部"的差异，等价于学一个"该层对本条知识到底有没有用"的因果对比信号，天然鼓励稀疏且不损失性能。

3. **分层联合优化 + 损失正则**：

    - 功能：把高低层 hypernetwork 端到端协同训练。
    - 核心思路：低层 reward 取负总损失 $r_{\text{low},t} = -\mathcal{L}_t$，其中 $\mathcal{L}_t = \eta \|\tilde{\nabla} \mathcal{W}_t\|^2 + \sum_{i=t-k}^t \mu^{t-i} \mathcal{L}_{t,i}$，$\mathcal{L}_{t,i} = -\log p_{\mathcal{W}_t}(y_i|x_i) + \tilde\lambda \mathrm{KL}[p_{\mathcal{W}_{t-1}}(\cdot|\tilde x_i) \| p_{\mathcal{W}_t}(\cdot|\tilde x_i)]$，同时回看 $k$ 步过去编辑做 memory backtracking 防遗忘；KL 项约束无关输入分布尽量不变。两套 hypernetwork 各按累积折扣 reward $\sum \gamma^t r_{\beta,t}$ 优化，$\gamma=1$ 保证整序列权重一致。
    - 设计动机：稀疏 mask 阻断了普通梯度回传，需要 straight-through + 联合优化才能保证 high-level 探索和 low-level 利用能互相校正。

### 损失函数 / 训练策略
- 训练时 TopK 的 $K$ 与推理一致，强制 sparsity-consistency，避免训练用稠密推理却稀疏导致 distribution shift。
- 每条编辑序列上跑完整 trajectory 后再回传梯度（trajectory-level update），更稳定。

## 实验关键数据

### 主实验
在 Llama-3-8B 与 Gemma-2-9B 两个底座、ZsRE 与 CounterFact 两个数据集、20000 次长程序列编辑下，HiEdit 对比 11 种基线（FT、ROME、MEMIT、PRUNE、RECT、AlphaEdit、MEND、MALMEN、DAFNet、RLEdit 等）。指标包括 Efficacy / Generalization / Specificity / Retention。

| 模型 | 方法 | ZsRE-Eff. | ZsRE-Gen. | ZsRE-Spe. | ZsRE-Ret. | CounterFact-Eff. |
|------|------|-----------|-----------|-----------|-----------|------------------|
| Llama-3-8B | RLEdit | 81.43 | 79.49 | 42.73 | 70.72 | 66.35 |
| Llama-3-8B | HiEdit (rand) | 81.95 | 79.63 | 47.97 | 74.66 | 66.40 |
| Llama-3-8B | HiEdit (full) | **82.10** | **79.99** | **48.42** | **75.16** | **66.53** |
| Gemma-2-9B | AlphaEdit | 15.79 | 15.32 | 20.21 | 13.19 | 38.17 |
| Gemma-2-9B | RECT | 11.26 | 11.25 | 16.19 | 9.62 | 30.72 |

在长程编辑设置下，AlphaEdit / ROME / MEMIT 在 Gemma-2-9B 上多数掉到 0，HiEdit 在 ZsRE 上把 RLEdit 的平均得分再提 8.48%，且每次只动一半层（$K=L/2$）。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| HiEdit-full | 82.10 / 75.16 | 训练 + 推理都用学到的 TopK mask |
| HiEdit-rand | 81.95 / 74.66 | 推理时换成随机 mask（同 K）→ 仍优于 RLEdit，但 Retention 掉 0.5 |
| RLEdit (dense, all layers) | 81.43 / 70.72 | 不做选层，Retention 显著下降 |
| Static fixed K layers | < 81 | 固定一组层（非学得），明显劣于 HiEdit |

### 关键发现
- intrinsic reward 是关键：去掉相对优势奖励（直接最大化绝对 reward），高层倾向选所有层退化为 RLEdit。
- 随机 mask 也能拿到大部分增益，说明稀疏更新本身就缓解了过修改；而学得的 mask 在 Retention（过往编辑保留率）上额外多 0.5–1 分。
- 在 20k 步长程编辑下，传统 closed-form 方法（PRUNE/RECT）直接归零，hypernetwork 系（RLEdit、HiEdit）能撑住——印证了"参数更新空间需要更结构化的探索"。

## 亮点与洞察
- **把 MoE 的 router 思想搬进 LME**：编辑器和 MoE 一样面对"该路由到哪个专家/层"问题，gate + straight-through 几乎可以无缝迁移，是个很优雅的跨任务复用。
- **Intrinsic reward 的设计**：用"部分 vs. 全部"的相对优势替代绝对稀疏惩罚，避免了 hand-tune sparsity coefficient，给所有"想学稀疏 mask"的任务提供了模板。
- **解耦 where / how 的训练范式**：分层 MDP 把原本扁平的 $\{0,1\}^L \times \mathbb{R}^{d \times d}$ 巨大动作空间拆开，让 high-level 用 RL，low-level 走梯度，是 RL+SL 混合优化的一个干净例子。

## 局限与展望
- 高层 router 只看本步梯度信号，没有显式的"过往编辑历史"输入；长程下能否一直保持对历史 mask 分布的稳定，论文只跑到 20k 步。
- $K$ 是固定超参（实验用 $K=L/2$），没有让模型自适应决定每步该激活多少层；理想方法应该让 $K$ 也是 learnable。
- 只在 hypernetwork 系上验证；和 closed-form 方法（如 AlphaEdit）能否结合，尚未尝试。

## 相关工作与启发
- **vs RLEdit**：把扁平 RL 升级为分层 RL，新增可学习的 layer selection；同样的 hypernetwork 主干换上 HiEdit 框架就直接涨点。
- **vs AlphaEdit / MEMIT (closed-form)**：闭式解方法在长程编辑下容易爆炸，HiEdit 走的是可学习路径，更适合 LME 这种 streaming 场景。
- **vs MoE routing**：思想同源（gate + TopK + straight-through），但目标从"专家选择"换成"知识相关层选择"。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个把 LME 建模成分层 MDP 的工作，intrinsic reward 设计也有巧思。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 2 个底座 × 2 个数据集 × 11 个基线，长程编辑设置足够严苛。
- 写作质量: ⭐⭐⭐⭐ HRL 的 motivation 讲得清晰，公式编号略多但逻辑顺畅。
- 价值: ⭐⭐⭐⭐ 长程模型编辑是部署侧刚需，能稳过 5k 步遗忘点的方法不多。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Representation Interventions Enable Lifelong Knowledge Memory Control in LLMs](representation_interventions_enable_lifelong_knowledge_memory_control_in_llms.md)
- [\[NeurIPS 2025\] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs](../../NeurIPS2025/knowledge_editing/memoir_lifelong_model_editing_with_minimal_overwrite_and_informed_retention_for_.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)
- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Training Examples](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_training_examples.md)
- [\[ICML 2025\] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs](../../ICML2025/knowledge_editing/wikibigedit_understanding_the_limits_of_lifelong_knowledge_editing_in_llms.md)

</div>

<!-- RELATED:END -->
