---
title: >-
  [论文解读] Vision Transformers Need More Than Registers
description: >-
  [CVPR 2026][自监督学习][Transformer] 这篇论文认为 ViT 在标签监督、文本监督和自监督下普遍存在的 dense feature 伪影，本质上不是单纯的 high-norm token 问题，而是模型在粗粒度监督和全局注意力共同作用下学会了用背景 patch 充当全局语义捷径；作者据此提出 LaSt-ViT，用频域稳定性引导的选择性聚合替代原始 CLS 聚合，在 12 个基准上稳定改善定位、分割和开放词汇任务。
tags:
  - CVPR 2026
  - 自监督学习
  - Transformer
  - Lazy Aggregation
  - Register Token
  - DINO
  - Dense Feature Alignment
---

# Vision Transformers Need More Than Registers

**会议**: CVPR 2026  
**arXiv**: [2602.22394](https://arxiv.org/abs/2602.22394)  
**代码**: https://github.com/ChengShiest/LAST-ViT  
**领域**: self_supervised  
**关键词**: Vision Transformer, Lazy Aggregation, Register Token, DINO, Dense Feature Alignment

## 一句话总结
这篇论文认为 ViT 在标签监督、文本监督和自监督下普遍存在的 dense feature 伪影，本质上不是单纯的 high-norm token 问题，而是模型在粗粒度监督和全局注意力共同作用下学会了用背景 patch 充当全局语义捷径；作者据此提出 LaSt-ViT，用频域稳定性引导的选择性聚合替代原始 CLS 聚合，在 12 个基准上稳定改善定位、分割和开放词汇任务。

## 研究背景与动机
ViT 现在已经不只是分类 backbone，而是很多下游视觉系统的通用特征提取器。问题在于，一旦把这些特征拿去做定位、分割、开放词汇检测这类需要 spatially aligned dense feature 的任务，就会发现它们经常“看错地方”。

已有工作分别从不同现象切入过这个问题。
- 在 label supervision 下，有工作指出 ViT 的 dense feature 对前景不敏感。
- 在 text supervision 下，CLIP 式 ViT 的 patch-text 对齐往往很差，影响 zero-shot dense prediction。
- 在 self-supervision 下，DINO 系列又暴露出 high-norm token / artifact token，会破坏 object discovery。

作者认为，这几类问题虽然表面上不一样，但背后很可能是同一个机制在不同训练范式下的不同表现。现有方法大多只是在某一个设置里“打补丁”，例如给 DINO 加 register token，可以把一部分异常全局信息转移出去，却没有解释清楚为什么这些异常 token 会出现，也没有解释为什么换成文本监督或标签监督时还会出现类似现象。

论文的核心动机可以概括为三层。
- 第一层，作者想给不同监督范式下的 artifact 找到一个统一可比较的定义，而不是每个社区各说各话。
- 第二层，作者想判断 register token 到底是在解决根因，还是只是在搬运症状。
- 第三层，作者希望提出一个不依赖特定训练目标、能直接在预训练阶段抑制 artifact 的统一方案。

作者的关键观察是：对于只接受图像级监督信号的 ViT，CLS token 只需要对整张图的语义负责，却不需要显式保证每个 patch 都对得上前景语义。在这种设定下，模型最省事的做法不是老老实实从前景 patch 里抽取全局语义，而是借助全局自注意力，把少量前景信息扩散到大量背景 patch 上，再让 CLS 从这些“背景捷径”里完成语义聚合。作者把这种行为称为 lazy aggregation。

这个解释之所以有说服力，是因为它同时解释了两个现象。
- 为什么背景 patch 往往拥有很高的 patch score：因为它们被模型当成了全局语义的代存点。
- 为什么 register 只能缓解不能根治：因为它只是提供了新的全局信息存储位置，但没有改变模型依赖捷径聚合的倾向。

用一句更精炼的话说，本文要解决的问题不是“怎么把异常 token 挪走”，而是“怎么阻止 ViT 从一开始就学会用背景作为全局语义捷径”。

## 方法详解
本文的方法分成两部分：先用统一诊断工具证明 lazy aggregation 的存在，再设计一个新的 CLS 聚合机制，让全局表征更多依赖真正稳定、与前景相关的 patch。

作者并没有直接从改 loss 开始，而是先重新定义“我们到底在看什么”。这一点很重要，因为如果没有统一度量，就无法把 supervised / text-supervised / self-supervised 三个体系放在一起分析。

### 整体框架
整体 pipeline 可以概括成下面四步。

1. 用标准 ViT encoder 得到所有 patch token 表示 $\mathbf{x}_{patch} \in \mathbb{R}^{N \times D}$。
2. 定义 Patch Score，也就是每个 patch 与全局表征 CLS 的相似度，用它衡量该 patch 是否被模型视为“代表整张图语义”的关键位置。
3. 发现高分 patch 常常落在背景而不是前景，于是进一步提出 Point-in-Box (PiB) 指标，直接统计最高分 patch 是否位于标注框内。
4. 在训练时不再让 CLS 无差别地吸收所有 patch，而是先为每个 patch 的每个通道计算稳定性分数，再按通道选择 Top-K 最稳定 token 做聚合，得到新的 CLS 表征。

从输入输出角度看，LaSt-ViT 没有重写 ViT 主干，也没有引入复杂的新监督分支。它做的事情更像是“重新定义 CLS 从哪些 patch 取信息、按什么原则取信息”。因此它可以迁移到监督、文本监督和自监督三类预训练范式中。

### 关键设计

1. **Patch Score 与 Point-in-Box：统一诊断 ViT artifact**
	- 功能：用 CLS 与 patch 的相似度作为统一探针，分析 ViT 的全局语义到底落在前景还是背景上。
	- 核心思路：作者定义 patch score 为 $\mathcal{S}_p = \frac{\mathbf{x}_{patch} \cdot Q_{CLS}}{\lVert \mathbf{x}_{patch} \rVert_2 \lVert Q_{CLS} \rVert_2}$。分数越高，说明该 patch 越接近全局语义表示。
	- 设计动机：如果 dense feature 是健康的，那么最能代表整图语义的区域通常应该落在主体对象附近；如果高分 patch 总在背景，说明模型在走捷径。
	- 进一步量化：作者提出 PiB，统计最高 patch score 是否落在前景框内。这个指标的好处是足够直接，而且能在不同架构和不同预训练范式间对齐。

2. **Lazy Aggregation 假设：artifact 的根因不是 norm，而是背景捷径**
	- 功能：用训练动态、掩码实验和结构干预验证 artifact 形成机制。
	- 核心思路：作者发现，去掉高分 patch 并不会伤害分类精度，反而有时还能微幅提升；相反，去掉低分 patch 会让精度大幅下降。这说明高分 patch 并不等于关键语义区域。
	- 设计动机：如果 highest-score patch 真是前景关键位置，那么删掉它们应该伤害性能；现实结果相反，意味着这些位置更像冗余但高相关的背景捷径。
	- 进一步验证：作者观察到 ViT 的 PiB 在训练很早期就偏低，并在整个训练过程中几乎维持不变，而分类精度还在继续上升。这说明 artifact 不是后期偶发副产物，而是早期就被学到的稳定策略。

3. **频域稳定性评分：用“通道稳定”刻画前景候选 token**
	- 功能：为每个 patch 的每个通道分配稳定性分数，估计哪些特征在低通滤波后依然稳定。
	- 核心思路：作者的直觉是，前景区域在深层特征上语义更一致，沿通道维度的变化更平滑；背景往往语义杂，经过低通滤波后变化更大。于是先对每个 patch 在通道维度做一维 FFT，再乘高斯低通权重 $\mathbf{g}$，逆变换得到低频版本 $\hat{\mathbf{x}}_{patch}$。
	- 公式上，稳定性分数写成 $\mathbf{S}_{i,j} = \frac{\hat{\mathbf{x}}_{patch}[i,j]}{|\hat{\mathbf{x}}_{patch}[i,j] - \mathbf{x}_{patch}[i,j]| + \varepsilon}$。
	- 设计动机：如果某个 patch 的某个通道在低通后变化很小，说明该通道信息更“稳定”，更可能属于被持续共享的主体语义，而不是背景中的杂散高频线索。

4. **按通道 Top-K 聚合：让 CLS 只吸收最可靠的局部证据**
	- 功能：不再对所有 patch 做平均，也不靠单一 attention pooling，而是对每个通道单独选择稳定性最高的 K 个 patch，再求均值形成 CLS 的该通道值。
	- 核心思路：对第 $j$ 个通道，先找出稳定性分数最高的集合 $\mathcal{I}_K(j)$，再计算 $\mathcal{Q}_{CLS}[j] = \frac{1}{K}\sum_{i \in \mathcal{I}_K(j)} \mathbf{x}_{patch}[i,j]$。
	- 设计动机：同一个 patch 不一定在所有通道都可靠，因此作者不是整 token 地筛选，而是做 channel-wise 选择。这样既保留表征细粒度，又避免整图平均把大量背景噪声灌进 CLS。
	- 和 register 的区别：register 是新增存储位；LaSt-ViT 是直接限制 CLS 的信息来源。前者是“给捷径换个容器”，后者是“减少走捷径的机会”。

5. **Vote Count 可视化：解释 CLS 实际在看哪里**
	- 功能：统计每个 patch 在多少个通道的 Top-K 集合中被选中，得到 vote count。
	- 核心思路：$v_i = \sum_{j=1}^{D} \mathbf{1}\{i \in \mathcal{I}_K(j)\}$，票数越高，说明这个 patch 在更多语义通道上被认为是可靠证据。
	- 设计动机：作者希望证明 LaSt-ViT 不是黑盒 trick，而是真的让 CLS 的注意重心迁回前景。可视化结果显示，高票 patch 与前景区域高度对齐，而且会随着前景证据多少自适应增减。

### 损失函数 / 训练策略
本文没有引入一个全新的监督目标，而是保持原训练范式不变，把改动集中在 CLS 聚合方式上。

- 在 fully supervised 场景下，仍然使用标准图像分类监督。
- 在 text-supervised 场景下，仍然使用 CLIP 式图文对比学习。
- 在 self-supervised 场景下，仍然沿用 DINO 类自监督训练。

因此，LaSt-ViT 更像一种通用 aggregation module，而不是某个只服务于单一任务的新 loss。这个设计很实用，因为它把方法收益集中在“前景语义对齐”这件事上，而不是依赖额外标注或复杂多任务训练。

作者还做了两类关键验证，进一步支撑动机。
- 增大 patch size，减少背景 token 占比后，PiB 从 0.44 升到 0.52，但分类 top-1 从 62% 降到 55%。这说明粗粒度监督确实促成了背景捷径，同时也说明直接粗化 patch 不是好解法。
- 把全局注意力换成 window attention 后，PiB 持续上升，但 top-1 持续下降。例如 ViT-Small 在全部层都用窗口注意力时，PiB 从 50.1 提到 59.8，而 top-1 从 72.3 降到 63.9。说明全局依赖既带来识别收益，也让背景更容易吸收前景语义。

## 实验关键数据
论文的实验不是只围绕一个任务展开，而是覆盖三种训练范式和多个 dense downstream task。作者最想证明的不是“又一个 benchmark 上刷了分”，而是“只要 artifact 被抑制，ViT 的 dense behavior 会在多个任务上同步变好”。

### 主实验
先看作者最核心的 artifact 指标 Point-in-Box。这个表最能直接说明 LaSt-ViT 是否真的打到了根因。

| 训练范式 / 模型 | 基线 PiB | LaSt-ViT PiB | 提升 |
|------|------:|------:|------:|
| Supervised ViT | 42.7 | 55.1 | +12.4 |
| DINO-v1 | 44.5 | 69.7 | +25.2 |
| CLIP ViT | 39.8 | 50.1 | +10.3 |
| ResNet 参考 | 68.4 / 71.1 / 53.9 | - | - |

这个结果很关键，因为它同时说明三件事。
- 第一，artifact 不是某个单独训练范式的问题。
- 第二，LaSt-ViT 带来的不是边角收益，而是对前景对齐能力的显著修正。
- 第三，DINO-v1 上的提升最大，说明自监督 ViT 在 object-centric dense feature 上尤其容易受 lazy aggregation 影响。

再看 self-supervised 领域最相关的 object discovery 结果。

| 方法 | FPS | VOC07 CorLoc | VOC12 CorLoc | COCO CorLoc |
|------|------:|------:|------:|------:|
| DINO-seg | 29.4 | 45.8 | 46.2 | 42.1 |
| LOST | 29.4 | 61.9 | 64.0 | 50.7 |
| DINO + LaSt-ViT | 55.9 | 64.4 | 67.6 | 51.6 |

这张表说明 LaSt-ViT 的收益不只是“Patch Score 看起来更合理”，而是真的能转化为无监督目标发现性能。尤其值得注意的是，它不仅比 DINO-seg 高很多，也超过了 LOST，同时速度更快，不需要额外的重型特征分解步骤。

作者还展示了 fully supervised 和 text-supervised 下的 dense task 提升，说明方法是统一有效的。

| 任务 | 基线 | LaSt-ViT | 提升 |
|------|------:|------:|------:|
| VOC12 coarse segmentation, ViT-B/16 supervised | 22.3 mIoU | 32.8 | +10.5 |
| VOC12 coarse segmentation, ViT-S/16 supervised | 29.5 mIoU | 41.9 | +12.4 |
| VOC12 coarse segmentation, ViT-S/16 DINO | 47.7 mIoU | 55.1 | +7.4 |
| CLIP ViT-B/16 on VOC20 segmentation | 49.0 mIoU | 75.0 | +26.0 |
| F-ViT ViT-B/16 on OV-COCO novel AP50 | 117.5 | 133.3 | +15.8 |

虽然这些结果跨任务跨指标，但它们指向同一个结论：一旦 CLS 不再过度依赖背景捷径，ViT 的 dense representation 会整体变得更可用。

### 消融实验
本文的消融围绕两个问题展开：一是 K 应该取多少，二是 LaSt-ViT 的提升是否只是某种 pooling 偏置。

先看 label-supervised 场景下的 K 消融。

| 配置 | IN1K Top-1 | VOC07 CorLoc | VOC12 CorLoc | 说明 |
|------|------:|------:|------:|------|
| Attention-Pool | 59.1 | 14.1 | 28.7 | 原始聚合 |
| Mean-Pool | 64.3 | 15.3 | 29.6 | 简单平均 |
| LaSt-ViT, K=1 | 64.6 | 30.4 | 35.6 | 极强筛选 |
| LaSt-ViT, K=7 | 64.8 | 32.1 | 37.6 | 最优折中 |
| LaSt-ViT, K=49 (Full) | 64.9 | 15.8 | 30.3 | 接近退化为全量聚合 |

这里最有意思的点是：当 K 变得过大时，定位收益明显退化。也就是说，作者的方法有效并不是因为换了一种 pooling 公式，而是因为“选择性”本身在起作用。K 太大以后，又把背景 token 放回来了。

再看 text-supervised 场景下的消融。

| 配置 | IN1K | VOC mIoU | COCO mIoU | 说明 |
|------|------:|------:|------:|------|
| Attention-Pool | 55.8 | 10.7 | 3.3 | 原始 CLIP 聚合 |
| Max-Pool | 53.1 | 71.9 | 12.2 | 天然压背景，但分类变差 |
| LaSt-ViT, K=1 | 53.5 | 72.7 | 13.5 | 过于激进 |
| LaSt-ViT, K=49 | 55.8 | 75.8 | 18.5 | 最优 |
| LaSt-ViT, K=98 | 56.2 | 75.9 | 18.0 | 接近最优 |
| LaSt-ViT, K=196 (Full) | 55.3 | 13.5 | 4.8 | 基本失效 |

这组结果进一步表明，LaSt-ViT 不是简单地做“更稀疏的 pooling”，而是通过筛掉不稳定背景 token，保留了对密集预测更有价值的局部证据。尤其 K=196 时几乎回到失败状态，说明 full aggregation 本身就是问题来源。

### 关键发现
- 去掉 high-score patch 对分类几乎没伤害，说明高分 patch 不等于关键前景，而更像被 CLS 利用的背景捷径。
- artifact 从训练早期就出现，PiB 几乎不随训练后期改善，说明这是优化路径里的早期习惯，而不是后期过拟合副作用。
- register token 能去掉 high-norm 现象，但 PiB 仍然很低，因此 high-norm 只是 lazy aggregation 的表现，不是根因。
- 限制全局依赖或减少背景 token 都能提高 PiB，但会牺牲分类精度，说明问题不能靠硬性削弱模型能力来解决，而要改聚合机制。
- Top-K 不是越大越好；当 K 接近全量 token 时，dense task 性能显著退化，证明选择性聚合是核心。

## 亮点与洞察
- 这篇论文最强的地方不是提出了一个复杂新模块，而是把多个社区分散讨论的 artifact 现象统一到了 lazy aggregation 这个解释框架里。这个框架足够简单，却能同时解释 supervised、CLIP、DINO 三个体系的异常。
- 作者把“register 是否有效”这个问题重新表述成“ViT 是否还在走背景捷径”。这个视角比只盯着高范数 token 更深入，因为它直接关注表征形成机制。
- 频域稳定性这个切入点很巧妙。它不是显式做前景监督，却利用前景语义通常更一致、背景语义更杂这一统计特征，给 CLS 聚合引入了一个自然的筛选依据。
- channel-wise Top-K 也很值得借鉴。很多 token 筛选方法按 token 整体打分，这篇工作提醒我们：不同通道可能对应不同语义子空间，逐通道选择会更细腻。
- 从研究方法上看，这篇论文是“先诊断、再建模、最后统一验证”的典型范式，论证链相对完整，适合作为分析型 paper 的写法参考。

## 局限与展望
- 论文的理论解释仍然偏经验归纳。作者用大量现象支持 lazy aggregation，但没有给出更严格的优化层面推导，例如为什么 CLS 在早期训练中会优先收敛到背景捷径。
- 频域稳定性假设默认“前景语义更平滑、背景更杂”，这在自然图像里通常成立，但在纹理主导或多实例复杂场景下未必总成立。
- 方法主要围绕 CLS 聚合展开，因此更适合依赖全局 token 的 ViT 变体。对于没有显式 CLS、或 heavily token-mixing 的架构，能否直接迁移还需要验证。
- 实验虽然广，但真正针对 self-supervised 的核心结果主要集中在 DINO-v1 object discovery。若能进一步覆盖 DINOv2、MAE、iBOT 等更强自监督骨干，会更有说服力。
- 文中强调 register 不够，但没有系统比较“register + LaSt-ViT”是否互补。若两者能配合，可能会得到更强的工程方案。

## 相关工作与启发
- **vs Register**: Register 的思路是给模型额外的全局存储槽，减少异常高范数 token 留在 patch map 上；本文则认为这只是转移症状，没有改变背景捷径形成机制。LaSt-ViT 直接改 CLS 的聚合来源，因此更接近治本。
- **vs CLIP dense alignment 系列方法**: 很多方法在下游额外做 patch-text 对齐或改最后几层 attention；本文把问题前移到预训练表征形成阶段，思路更基础，也更统一。
- **vs token pruning / token selection**: pruning 关注的是冗余和效率，而本文关注的是语义来源是否健康。两者可以结合，先用 LaSt-ViT 矫正语义，再做更可信的 token 裁剪。
- **对 self-supervised 的启发**: 自监督 ViT 的 object-centric 性能不一定只由 teacher-student loss 决定，CLS 如何聚合 patch 同样关键。今后做 DINO 类方法时，完全可以把 aggregation bias 当成独立设计维度。
- **对我的启发**: 如果一个 backbone 的全局表征是通过不健康的 shortcut 形成的，那么下游很多 dense task 问题都只是症状。以后分析 foundation model 时，应该优先审视“全局语义从哪里来”，而不是只看下游头部设计。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 统一提出 lazy aggregation 这一根因解释，把 register 现象和跨范式 artifact 串成了一个更完整的故事。
- 实验充分度: ⭐⭐⭐⭐☆ 覆盖 12 个 benchmark、三种监督范式，范围很广；但自监督主线还可以再补更现代的 backbone。
- 写作质量: ⭐⭐⭐⭐☆ 诊断逻辑清楚，实验组织也有说服力；不过部分方法直觉仍略强于严格推导。
- 价值: ⭐⭐⭐⭐⭐ 这不仅是一个提升 dense feature 的技巧，更是一个理解 ViT 全局语义形成机制的分析框架。

<!-- RELATED:START -->

## 相关论文

- [DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [TALO: Pushing 3D Vision Foundation Models Towards Globally Consistent Online Reconstruction](talo_pushing_3d_vision_foundation_models_towards_globally_consistent_online_reco.md)
- [TrackMAE: Video Representation Learning via Track, Mask, and Predict](trackmae_video_representation_learning_via_track_mask_and_predict.md)
- [Suppressing Non-Semantic Noise in Masked Image Modeling Representations](suppressing_non-semantic_noise_in_masked_image_modeling_representations.md)
- [LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency](las-comp_zero-shot_3d_completion_with_latent-spatial_consistency.md)

<!-- RELATED:END -->
