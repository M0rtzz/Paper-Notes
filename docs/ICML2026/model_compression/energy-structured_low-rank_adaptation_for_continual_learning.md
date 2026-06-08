---
title: >-
  [论文解读] Energy-Structured Low-Rank Adaptation for Continual Learning
description: >-
  [ICML 2026][模型压缩][continual learning] E2-LoRA 不在参数空间或输入特征空间做正交约束，而是把视角换到"任务引起的输出特征漂移" $\Delta \mathbf{Y}_t = \Delta \mathbf{W}_t \mathbf{X}_t$…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "continual learning"
  - "LoRA"
  - "正交子空间"
  - "能量集中"
  - "动态秩分配"
---

# Energy-Structured Low-Rank Adaptation for Continual Learning

**会议**: ICML 2026  
**arXiv**: [2605.27482](https://arxiv.org/abs/2605.27482)  
**代码**: 暂未公开  
**领域**: 模型压缩 / LoRA / 持续学习  
**关键词**: continual learning, LoRA, 正交子空间, 能量集中, 动态秩分配  

## 一句话总结
E2-LoRA 不在参数空间或输入特征空间做正交约束，而是把视角换到"任务引起的输出特征漂移" $\Delta \mathbf{Y}_t = \Delta \mathbf{W}_t \mathbf{X}_t$，对它做 SVD 后把 LoRA 参数重排到能量集中且按秩有序的基上，从而能丢掉低能量秩、把容量回收给新任务，并配合按能量保留率自适应分配秩的策略，在多个持续学习基准上拿到 SOTA。

## 研究背景与动机

**领域现状**：基于预训练大模型 (PTM) 的持续学习（CL）主流做法是冻结 backbone，给每个新任务加 PEFT 模块（prompt / adapter / LoRA）。为了减少任务间干扰，又派生出一族"正交化"方法：要么强制不同任务的 LoRA 参数互相正交（O-LoRA，Param-Param），要么用历史输入特征做 SVD 后让新任务参数与主奇异向量正交（GPM / DualGPM / InfLoRA，Input-Param）。

**现有痛点**：作者观察到两条正交化路线都存在"能量过度弥散"问题——参数空间里旧任务知识被无规律打散到 $\mathbf{B}$ 的各列上，子空间随任务线性增长，新任务可用容量被严重挤压；输入空间里 PTM 的特征本身就高维多样，按输入主方向做约束等于把新任务可走的方向限制得太死，可塑性骤降（见原文图 1）。本质上：受限子空间是"刚性"的，无法回收。

**核心矛盾**：稳定性（不忘旧）与可塑性（学新）之间的 trade-off 被"正交子空间不可回收"这一约束放大了——子空间一旦分给某任务就锁死了。

**本文目标**：找一个低维、能量集中、有内在排序的子空间表示旧任务知识，使得低能量方向可以随时被释放、回收给新任务，同时保留旧任务输出几乎不变。

**切入角度**：与其在参数或输入这两端做文章，不如盯住 LoRA 真正影响模型的中间产物——输出特征的漂移 $\Delta y_t(x) = \Delta \mathbf{W}_t x$。作者实证 + 理论都发现：尽管输入 $\mathbf{X}_t$ 可能是高秩的，由 $\Delta \mathbf{W}_t$ 引起的 $\Delta \mathbf{Y}_t$ 通常集中在极少数主方向上（任务语义本身就是低维的）。

**核心 idea**：对 $\Delta \mathbf{Y}_t$ 做 PCA / SVD，把 LoRA 的 $\mathbf{B}_t, \mathbf{A}_t$ 重写到这组按能量降序排列的正交基上——这样"前 $r$ 秩"就是该任务知识的最佳低秩近似（理论可证，详见 Prop 3.1/3.2），剩下的低能量秩可以安全砍掉、释放容量。

## 方法详解

### 整体框架
设 $\mathbf{W}_0$ 是预训练线性层权重，第 $t$ 个任务到来时为它分配一个 LoRA 模块 $\Delta \mathbf{W}_t = \mathbf{B}_t \mathbf{A}_t$，整套方法的关键不在于"加什么参数"，而在于"约束哪个空间"——E2-LoRA 把正交约束从参数空间、输入空间挪到了输出漂移空间。每个任务上它走四步闭环：先按能量阈值 $\rho$ 把旧任务的 LoRA 剪到合适秩、把腾出的列拼成新任务 $\mathbf{B}_t$ 的初始基并冻结（动态秩分配）；然后只优化 $\mathbf{A}_t$ 并配 self-distillation 防遗忘（训练）；训练完对输出漂移 $\Delta \mathbf{Y}_t$ 做 SVD，把 $(\mathbf{B}_t, \mathbf{A}_t)$ 旋转到按能量降序排好的基上（能量结构化变换）；最后用类统计合成特征对齐分类头。这样旧任务知识被压成"前几秩承载几乎全部能量"的形式，尾部低能量秩随时可释放给后来者。

### 关键设计

**1. 输出漂移诱导的正交化：把"该正交于谁"换到能量最集中的参考空间。**

前面指出两条旧路线都吃了"参考空间选错"的亏：参数级正交（O-LoRA）盯着 $\mathbf{B}$ 的列做约束 $\|\mathbf{B}_i^\top \mathbf{B}_t\|_F^2$，但 $\mathbf{B}$ 的列本身没有任何能量含义，等于"盲约束"；输入级正交（DualGPM / InfLoRA）盯着 PTM 的输入特征，可这些特征天然高秩、能量分散，约束下去等于把新任务大半个可走方向都关死。E2-LoRA 改盯 LoRA 真正影响模型的中间产物——输出特征的漂移。它在一个 proxy 输入 batch $\mathbf{X}_t$ 上算出漂移矩阵 $\Delta \mathbf{Y}_t = \mathbf{B}_t \mathbf{A}_t \mathbf{X}_t \in \mathbb{R}^{d_\text{out}\times n}$，做 SVD 得 $\Delta \mathbf{Y}_t = \mathbf{U}_t \mathbf{\Sigma}_t \mathbf{V}_t^\top$，$\mathbf{U}_t$ 的列就是按能量降序排列的输出方向。新任务的 $\mathbf{B}_t$ 直接拿旧任务的"低能量列"拼起来再冻结，于是 $\mathbf{B}_t \mathbf{A}_t$ 的输出必然落在旧任务漂移的零能量子空间，等价于在输出空间做了一次硬正交。之所以这个空间最合适：$\Delta \mathbf{Y}_t$ 既如实反映"任务到底改变了模型输出的哪些方向"，又因为单任务语义本就低维而能量高度集中——既不盲，又不浪费可塑性。

**2. 能量结构化变换 + 秩截断最优性：让前几秩扛住几乎全部任务知识。**

光知道"该在输出空间正交"还不够，得让 $\mathbf{B}_t$ 的列真的按能量排好序，否则没法精准地"剪尾部"。这一步在不改变整体 $\Delta \mathbf{W}_t = \mathbf{B}_t \mathbf{A}_t$ 数学等价性的前提下旋转坐标：训练完当前任务后做 $\mathbf{B}_t \leftarrow \mathbf{U}_t[:,:r_t]$、$\mathbf{A}_t \leftarrow (\mathbf{U}_t[:,:r_t])^\top \mathbf{B}_t \mathbf{A}_t$。论文把"低能量秩可丢"从经验观察升级成了带证明的最优截断：Prop 3.1 表明在所有秩 $\le r$ 的更新里，$\mathbf{B}_t[:,:r]\mathbf{A}_t[:r,:]$ 让期望输出重构误差 $\mathbb{E}_x \|\mathbf{B}_t[:,:r]\mathbf{A}_t[:r,:]x - \mathbf{B}_t\mathbf{A}_t x\|^2$ 最小；Prop 3.2 进一步给出截到前 $r$ 秩后的误差正好是被丢掉的奇异值平方和 $\sum_{i=r+1}^{d_\text{out}} \sigma_i^2$。有了这条闭式上界，"剪掉尾部、把列让给下个任务"就不再是冒险，而是误差可控、不伤已学知识的操作——它也是把上一个设计落地为可执行步骤的关键齿轮。

**3. 动态秩分配策略：在固定容量池里让每个任务按需占地。**

固定秩分配的尴尬是：任务一多，要么早早压缩损精度，要么前面任务占太满、后面任务可塑性塌陷。E2-LoRA 把"剪多少"直接绑到能量阈值 $\rho$ 上，在共享的 $d_\text{out}$ 容量池里动态调度。对每个旧任务 $k$，按能量保留比例选满足 $\sum_{i=1}^{r_k^{(t)}} \sigma_{k,i}^2 / \sum_{i=1}^{r_k} \sigma_{k,i}^2 \ge \rho$ 的最小秩 $r_k^{(t)}$；同时给新任务设最低门槛 $r_t^\text{min} = \lceil d_\text{out}/t \rceil$；万一按能量剪完仍不够分给新任务，就再均匀地从所有旧任务尾部继续砍能量最低的秩，直到腾够。效果是简单任务只占 1~2 秩、复杂任务多占几秩，整个 $d_\text{out}$ 被精打细算地反复复用——这正好对上前面"低能量方向应当可回收"的目标，把"正交子空间不可回收"这条老约束彻底打破。

### 损失函数 / 训练策略
总损失 $\mathcal{L} = \mathcal{L}_\text{ce} + \lambda \mathcal{L}_\text{distill}$。其中 $\mathcal{L}_\text{distill}$ 是温度 $T=2$ 下旧类 logits 的 KL 自蒸馏（teacher = 冻结历史 LoRA 的同一网络），$\mathcal{L}_\text{ce}$ 是新类的交叉熵。最后一步分类头对齐用类内统计合成特征。

## 实验关键数据

### 主实验
ViT-B/16-IN21K 为骨架，在 ImageNet-R / CIFAR-100 / CUB-200 / Cars-196 上做 class-incremental 持续学习（Last-Acc = 学完所有任务的精度，Inc-Acc = 整个增量过程平均精度）。摘录原文 Table 1 的关键 baseline：

| 方法 | ImageNet-R Last-Acc | CIFAR-100 Last-Acc | CUB-200 Last-Acc | Cars-196 Last-Acc |
|------|--------------------|--------------------|------------------|-------------------|
| L2P | 66.49 | 82.76 | 62.21 | 38.18 |
| DualPrompt | 68.50 | 85.56 | 66.00 | 40.14 |
| SLCA | 77.00 | 91.53 | 84.71 | 67.73 |
| EASE | 77.75 | 86.54 | 79.90 | 35.46 |
| BiLoRA | 77.95 | 87.46 | — | — |
| SSIAT | 79.38 | 91.35 | 88.75 | 71.02 |
| MOS | 78.10 | 90.53 | 89.91 | 67.76 |
| **E2-LoRA（本文）** | **新 SOTA** | **新 SOTA** | **新 SOTA** | **新 SOTA** |

> 所有基准上 Last-Acc 与 Inc-Acc 均优于已知 SOTA（具体数值见原文，提升幅度在 Cars-196 这类细粒度长任务序列上最显著）。

### 消融实验

| 配置 | 行为说明 |
|------|----------|
| Param-Param 正交（O-LoRA 风格） | 子空间被打散、新任务可用方向迅速耗尽，长序列上掉点 |
| Input-Param 正交（InfLoRA 风格） | PTM 输入特征高秩，约束太强，可塑性降低 |
| Output-Drift 正交（本文） | 单任务能量集中在少数主方向，旧任务保留 + 新任务空间充足 |
| 去掉能量结构化变换 | $\mathbf{B}$ 列无序，无法按秩剪枝，退化为固定秩 LoRA |
| 去掉动态秩分配（固定 $r_t$） | 后期任务可用容量不足，可塑性下降 |
| 去掉 self-distillation | 旧类 logits 漂移，遗忘加重 |

### 关键发现
- 视角的转换比额外正则更重要：仅仅把约束空间从"参数 / 输入"换成"输出漂移"，并不增加可训练参数，但带来一致涨点——说明问题症结在子空间的能量分布，而非约束方式本身。
- 截断误差有闭式上界 $\sum_{i>r} \sigma_i^2$，意味着能量保留率 $\rho$ 直接对应"输出重构误差上限"，调参可解释。
- 在 Cars-196（细粒度 + 任务多）这种最难的设定上，差距相对其他 SOTA 拉得最大，呼应了"动态容量回收"在长任务序列上的价值。

## 亮点与洞察
- "知识藏在输出而非参数"的观察 + 一个简洁的 PCA on $\Delta \mathbf{Y}_t$，几乎不增加运行开销就把 LoRA 持续学习推到新一档。这种"换问题表述"的设计常常比"加一个新正则"更有威力。
- 把能量保留率 $\rho$ 这一物理可解释的旋钮直接挂到秩分配上——这点对调参非常友好，业务上知道"我能容忍 5% 输出误差"就能反推 $\rho$。
- 能量结构化变换其实是 LoRA 版本的"主成分对齐"：它启示我们任何 PEFT 方法（IA³、AdaLoRA、prefix tuning）在持续学习场景下，都可以做完任务后"对齐到任务内禀低维基"再保存，这种思路应可推广。
- 释放低能量基给后续任务的设计，本质上把持续学习当成"有限容量池的拍卖"，比"线性增长子空间"更接近大脑容量约束下的学习模式。

## 局限与展望
- 输出漂移的 SVD 在每个任务结束时跑一次，proxy batch 越大越准但越慢；如果 $d_\text{out}$ 很大（如 LLM 隐藏维 4096+），SVD 成本不可忽略。
- "输出漂移低秩"这条经验观察来自 ViT 分类任务，在 LLM 生成任务、检索任务、RLHF 微调等输出分布更复杂的场景能否同样集中尚未验证。
- 动态秩分配假设所有任务共享一个 $d_\text{out}$ 容量池，但实际多层 LoRA 跨层秩耦合 / 跨模块平衡没单独建模。
- 阈值 $\rho$ 全网络全任务共用一个值，没探索分层 / 分任务自适应 $\rho$ 是否能进一步提升。

## 相关工作与启发
- **vs O-LoRA**：O-LoRA 在参数空间做正交 $\|\mathbf{B}_i^\top \mathbf{B}_t\|_F^2$，子空间随任务线性增长且不可回收；E2-LoRA 让每个任务的 LoRA 按能量"压紧 + 排序"，低能量部分可直接腾给下一个任务。
- **vs GPM / DualGPM / InfLoRA**：它们在输入特征上做 SVD，但 PTM 的输入特征能量分散；E2-LoRA 用输出漂移的低秩性质，约束更精准、对可塑性的伤害更小。
- **vs AdaLoRA**：AdaLoRA 也在 LoRA 上做能量级裁剪，但目标是单任务下的参数效率；E2-LoRA 把同样的思想扩到多任务，并加上"释放给后续任务"的回收机制。
- **vs EASE / MOS / TUNA**（adapter 路线）：它们靠任务专属 adapter + 路由 / 融合解耦任务，参数随任务数线性增长；E2-LoRA 在固定 $d_\text{out}$ 容量内复用秩，更节省。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把正交化的参考空间换到"输出漂移"上是一个干净的视角转换，并配套了 PCA + 秩截断最优性证明
- 实验充分度: ⭐⭐⭐⭐ 四个分类基准全面对比 + class/domain-incremental 双设定 + 多种正交化路线消融
- 写作质量: ⭐⭐⭐⭐ 用 Prop 3.1/3.2 把直觉写成定理，图 1 把三种正交化空间的能量分布对比讲得清楚
- 价值: ⭐⭐⭐⭐ 在持续学习 + PEFT 这条主流路线上拿 SOTA，方法本身可即插即用到任何 LoRA-CL 框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] ScaLoRA: Optimally Scaled Low-Rank Adaptation for Efficient High-Rank Fine-Tuning](scalora_optimally_scaled_low-rank_adaptation_for_efficient_high-rank_fine-tuning.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[NeurIPS 2025\] Gated Integration of Low-Rank Adaptation for Continual Learning of Large Language Models](../../NeurIPS2025/model_compression/gated_integration_of_low-rank_adaptation_for_continual_learning_of_large_languag.md)
- [\[ACL 2026\] Not All Directions Matter: Towards Structured and Task-Aware Low-Rank Model Adaptation](../../ACL2026/model_compression/not_all_directions_matter_towards_structured_and_task-aware_low-rank_model_adapt.md)
- [\[CVPR 2025\] CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning](../../CVPR2025/model_compression/cl-lora_continual_low-rank_adaptation_for_rehearsal-free_class-incremental_learn.md)

</div>

<!-- RELATED:END -->
