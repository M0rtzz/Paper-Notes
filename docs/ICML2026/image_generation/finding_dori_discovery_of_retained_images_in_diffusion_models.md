---
title: >-
  [论文解读] Finding DoRI: Discovery of Retained Images in Diffusion Models
description: >-
  [ICML2026][图像生成][扩散模型记忆] 作者用一个简单的对抗 text embedding 优化方法（DoRI）证明：NeMo / Wanda 这类"剪枝定位记忆神经元"的扩散模型记忆缓解方案只是把记忆"藏起来"而非真正擦除，因为记忆在 embedding、激活、权重三个层面都不是局部的…
tags:
  - "ICML2026"
  - "图像生成"
  - "扩散模型记忆"
  - "对抗 embedding"
  - "剪枝失效"
  - "对抗微调"
  - "版权与隐私"
---

# Finding DoRI: Discovery of Retained Images in Diffusion Models

**会议**: ICML2026  
**arXiv**: [2507.16880](https://arxiv.org/abs/2507.16880)  
**代码**: https://github.com/sprintml/finding_dori  
**领域**: AI 安全 / 扩散模型记忆 / 训练数据隐私  
**关键词**: 扩散模型记忆, 对抗 embedding, 剪枝失效, 对抗微调, 版权与隐私

## 一句话总结
作者用一个简单的对抗 text embedding 优化方法（DoRI）证明：NeMo / Wanda 这类"剪枝定位记忆神经元"的扩散模型记忆缓解方案只是把记忆"藏起来"而非真正擦除，因为记忆在 embedding、激活、权重三个层面都不是局部的；进一步提出对抗微调方案，把训练样本真正从模型里拔出来。

## 研究背景与动机

**领域现状**：文本到图像扩散模型（DM，如 Stable Diffusion v1.4）会逐字 (verbatim) 复刻部分训练图像，带来明显的隐私与版权风险。当前主流的"永久性缓解"路线由 NeMo (Hintersdorf et al., 2024) 和 Wanda (Chavhan et al., 2024) 代表：在记忆 prompt 触发时观察异常激活，定位出少量"记忆神经元 / 权重"并剪掉，宣称能在不伤害整体生成质量的前提下抹掉特定训练样本。

**现有痛点**：所有这些剪枝方法都假设记忆是**局部**的——某张训练图的"指纹"只藏在少数权重里，定向裁剪即可。可它们的有效性只在**原始记忆 prompt**下做过验证，没人检查过：换一种输入是不是还能把同一张图调出来。

**核心矛盾**：剪枝把"prompt → 记忆图"这条映射切断了，但记忆图未必只走这一条路。如果在 text embedding 空间还存在其他能触发同一张图的路径，那么"剪枝即遗忘"只是表象。

**本文目标**：(1) 设计一种方法去主动搜索剪枝模型里的"残留触发器"；(2) 系统检验记忆局部性假设在 embedding / activation / weight 三个层面是否成立；(3) 在否定局部性的前提下给出真正能擦除记忆的方案。

**切入角度**：把"找剩余触发器"建模为在 text embedding 上的连续对抗优化——不受自然语言离散性约束，直接在 embedding 空间用扩散损失梯度下降找触发器。如果在被剪过的模型上还能优化出 embedding 复刻原图，就说明记忆没被擦掉。

**核心 idea**：用 DoRI（Discovery of Retained Images）—— 一个对噪声和时间步反复重采样的对抗 embedding 优化器 —— 当作"记忆检测探针"和"对抗训练样本生成器"，戳穿剪枝的假象，并用它做对抗微调真正抹掉记忆。

## 方法详解

### 整体框架
整篇文章可以拆成"探针 + 诊断 + 治疗"三段：

1. **探针 DoRI**：给定一张已知被记忆的训练图 $\bm{x}_{\mathit{mem}}$ 和被 NeMo/Wanda 剪枝后的模型 $\bm{\theta}_{N/W}$，在 text embedding 空间用扩散损失梯度下降搜索 $\bm{y}_{adv}$，使其重新生成 $\bm{x}_{\mathit{mem}}$。
2. **三重局部性诊断**：用 DoRI 批量产生大量对抗 embedding，分别检查它们在 (a) text embedding 空间的分布、(b) 各层激活的差异、(c) 被剪枝方法标记的权重集合的重叠率，三个层面都拒绝"记忆 = 局部少数权重"的假说。
3. **治疗：对抗微调**：把 DoRI 当 inner loop，外层用 surrogate image 重新拟合，把记忆样本从模型里"洗"出去，同时维持普通图像-字幕对的扩散损失保住生成质量。

实验台用 Stable Diffusion v1.4 + 来自 LAION-5B 的 500 个已知记忆 prompt（沿用 Wen 等人的标准 benchmark），并在 SD v2.0 上做泛化验证。

### 关键设计

1. **DoRI 对抗 embedding 优化（探针）**：

    - 功能：给定记忆图 $\bm{x}_{\mathit{mem}}$ 和（可能已剪枝的）模型，找出能稳定复刻该图的 text embedding $\bm{y}_{adv}$。
    - 核心思路：把 $\bm{y}_{adv}$ 初始化为原 prompt 的 embedding（或随机高斯 / 非记忆 prompt），按 $\bm{y}_{adv}^{(i+1)} = \bm{y}_{adv}^{(i)} - \eta \nabla_{\bm{y}_{adv}^{(i)}} \mathcal{L}(\bm{x}_{\mathit{mem}}, \bm{\epsilon}, \bm{y}_{adv}^{(i)}, t, \bm{\theta}_{N/W})$ 迭代 50 步，Adam，$\eta=0.1$，batch=8。每一步**都重新采样噪声 $\bm{\epsilon}\sim\mathcal{N}(0,I)$ 和时间步 $t\sim\mathcal{U}(1,T)$**，强制找出的 embedding 在任意初始噪声下都能触发复刻，而非记住了某个特定噪声-时间步组合。
    - 设计动机：连续 embedding 比离散 prompt 搜索空间大得多，能暴露剪枝模型里残留的"隐通道"；重采样噪声/时间步是关键防作弊设计，否则模型可能只是过拟合某条特定采样轨迹。作者在附录中证明，对**非记忆图**用同样设置需要 >500 步才能强行复刻，远超 50 步阈值，所以触发=记忆这一推断不会假阳性（Memorization Rate 在非记忆 prompt 上仅 0.02）。

2. **三层局部性诊断（诊断）**：

    - 功能：用 DoRI 量化"记忆触发器到底有多分散"，从 embedding / activation / weight 三个角度逐一证伪局部性。
    - 核心思路：(a) **Embedding 层** —— 对同一张记忆图随机初始化 100 个 $\bm{y}_{adv}^{(0)}\sim\mathcal{N}(0,I)$ 各跑 DoRI 50 步，t-SNE 看分布，结果是对抗 embedding 几乎和初始随机点一样弥散，pairwise L2 距离甚至比非记忆 prompt 之间还大；(b) **Activation 层** —— 定义 discrepancy 为同一层在不同 embedding 下激活的平均 pairwise $\ell_2$ 距离（固定噪声只变 embedding），发现 100 个能触发同一张图的 $\bm{y}_{adv}$ 之间的激活差异，和 100 个完全不同的记忆 prompt 之间的差异**相当**；(c) **Weight 层** —— 定义 weight agreement 为不同 embedding 下 NeMo/Wanda 标记的"待剪权重集合"的 IoU，Wanda 在多数层 <0.6，NeMo 看似 >0.8 但其实是因为它在第 2、6、7 层根本不挑权重（agreement 强行设 1），实际有效层（第 1 层）也只有 0.6。
    - 设计动机：剪枝方法的全部正当性都建立在"同一张记忆图对应同一小簇权重"上，这三个实验逐一否定了这个假设。特别精彩的是 weight agreement 实验直接用剪枝方法自己的定位逻辑反驳自己——同一张图、不同 embedding，剪掉的权重都对不上号，说明所谓"记忆神经元"是输入相关的伪局部解，而非客观存在的内部存储位置。

3. **对抗微调（治疗）**：

    - 功能：把 DoRI 作为内循环，主动生成对抗 embedding 并用它们 fine-tune 全模型，把记忆图从权重里彻底擦除。
    - 核心思路：每个 fine-tuning step 里先用 DoRI 为每张待擦记忆图收集一批 $\bm{y}_{adv}$；再用预先准备的 surrogate image $\widetilde{\bm{x}}$（用剪枝模型 + 原 prompt 生成的"语义近似但像素不同"的图）作为目标，loss 为 $\mathcal{L}_{Adv}(\widetilde{\bm{x}}_0, \bm{\epsilon}, \bm{y}_{adv}, t, \bm{\theta}) = \|\bm{\epsilon} - \bm{\epsilon}_{\bm{\theta}}(\widetilde{\bm{x}}_t, t, \bm{y}_{adv})\|_2^2$，把对抗 embedding 的输出从原图拽向 surrogate；同时叠加普通 LAION 图像-字幕对的标准扩散损失 $\mathcal{L}_{\mathrm{DM}}$ 防止模型整体崩。总损失 $\mathcal{L} = \mathcal{L}_{\mathrm{DM}} + \mathcal{L}_{Adv}$，全模型微调（LoRA 实验失败，进一步佐证"必须全局调"）。
    - 设计动机：既然记忆是分布式的，就必须从全模型对抗多个触发器同时下手；用 surrogate 而不是随便一张图当目标，是为了避免把"语义内容"也一起删掉（区别于 concept unlearning），同时防止顺手把新的记忆样本带进来。

### 损失函数 / 训练策略
对抗微调跑 5 个 epoch，单 epoch 已显著降低记忆率；标准扩散损失维持模型通用性。Inner-loop 的 DoRI 用 50 步对抗优化迭代生成新 $\bm{y}_{adv}$。SD v1.4 调好的超参直接搬到 SD v2.0 仍有效。

## 实验关键数据

### 主实验（DoRI 戳穿剪枝 + 对抗微调真治）

| 设置 | SSCD$_{\mathrm{Orig}}$ ↓ | MR ↓ | 说明 |
|------|---|---|---|
| 原始 DM + Memorized prompt | 0.90 ± 0.04 | 0.98 | 记忆复刻基线 |
| 非记忆 prompt + DoRI | 0.48 ± 0.06 | 0.00 | DoRI 不会乱报假阳性 |
| NeMo（剪枝） | 0.33 ± 0.18 | 0.20 | 看上去成功擦除 |
| NeMo + DoRI | **0.91 ± 0.03** | **0.99** | 对抗 embedding 几乎全员复活 |
| Wanda（剪枝） | 0.20 ± 0.08 | 0.00 | 看上去更彻底 |
| Wanda + DoRI | **0.76 ± 0.05** | **0.72** | 仍有 72% 记忆图被找回 |
| ESD + DoRI | 0.90 ± 0.04 | 0.98 | 概念遗忘也防不住 |
| Concept Ablation + DoRI | 0.91 ± 0.04 | 0.97 | 同上 |
| SISS（数据遗忘 SOTA） + DoRI | 0.60 ± 0.22 | 0.39 | 比剪枝好但仍漏 39% |
| **本文对抗微调 + DoRI** | **0.36 ± 0.14** | **0.02** | 几乎清零，仅一张高度重复样本残留 |

SD v2.0 上结论一致：NeMo+DoRI 把 MR 从 0.06 拉回 1.00，本文方法把 MR 压到 0.06；FID 从 14.44 降到 13.61，生成质量反而轻微提升。

### 消融 / 局部性诊断

| 维度 | 现象 | 局部性是否成立 |
|------|------|---|
| Embedding 分布 | 100 个能复刻同一图的 $\bm{y}_{adv}$ 的 pairwise L2 距离 > 非记忆 prompt 之间的距离 | 否 |
| 激活 discrepancy | 100 个对抗 embedding 在 NeMo/Wanda 操作层的激活差异 ≈ 100 个完全不同记忆 prompt 之间的差异 | 否 |
| Weight agreement | Wanda <0.6（多数层）；NeMo 在第 1 层 0.6，其他层"高"实为标 0 权重 | 否 |
| LoRA 对抗微调 | 失败 | 进一步否定局部性 |
| Wanda 加大剪枝量到 10% | 能扛住 DoRI 但生成质量明显崩坏 | 局部性补丁不可行 |

### 关键发现
- **剪枝 ≠ 擦除**：NeMo/Wanda 之类局部剪枝只切断了 prompt 空间这条最短路径，把记忆从"按门铃即可调出"变成"需要密码才能调出"，而 DoRI 把密码暴力解了。
- **记忆是分布式的**：三层证据收敛——同一张图能被 embedding 空间几乎任意点触发，且对应激活和"重要权重"互不重叠，说明 DM 不是把图存在某几个神经元里，而是散布在整个网络的几何结构里。
- **对抗微调是必要也是足够的**：必须全局调（LoRA 失败）+ 必须对抗多触发器（concept unlearning / SISS 只针对单 prompt 都漏 ≥39%）；但只要这两条做对，5 个 epoch 就够，且不掉 FID。

## 亮点与洞察
- **把"记忆探测"和"对抗训练"统一在同一个 embedding 优化器**：DoRI 既当审计工具揭穿剪枝的伪安全，也当训练数据生成器驱动对抗微调，这种"攻防同源"的设计省心且自洽——你能怎么找出来，我就用同样的方式洗掉。
- **重采样噪声/时间步是防作弊的关键 trick**：很多对抗优化默认固定 $\bm{\epsilon}, t$，结果优化器学到的是"在这条特定轨迹上欺骗模型"。本文每步重采样保证 $\bm{y}_{adv}$ 真正稳定触发图像，把"对抗成功"的门槛抬到了和"真正存在残留记忆"等价的高度，让 50 步阈值在记忆/非记忆图上分得很开（MR 0.99 vs 0.02）。
- **用"剪枝方法自己的定位算子"做 IoU**：weight agreement 这个评估巧妙地把对手的核心假设直接反驳给对手看——你说能定位记忆神经元，那对同一张图的不同触发器应该都指向同一组权重；测下来 IoU 都凑不齐 0.6，等于剪枝方法承认自己的"定位"是输入相关的随机解。
- **迁移启发**：这套"对抗优化暴露隐藏触发器 → 三层证据否定局部性 → 对抗微调全局擦除"的范式很容易搬到 LLM 训练数据记忆、unlearning 评估、后门检测等任务——核心是别相信任何只用"原始触发"做评估的安全方法。

## 局限与展望
- **测试模型规模偏老**：实验集中在 SD v1.4 / v2.0，因为只有它们有公开的大规模"已知记忆 prompt"集合（Wen 2024、Webster 2023）。更新的 SDXL / FLUX 因训练数据去重后没人标过记忆样本，结论的外推性需要等社区构建新 benchmark。
- **依赖已知的记忆图集合**：DoRI 是"已知记忆图 → 找触发器 → 擦除"，仍需上游方法 (e.g., Carlini 2023, Webster 2023) 先把哪些图被记忆住挖出来；对"未知记忆图"还得配合 detection 流程。
- **计算成本不低**：对抗微调外层 fine-tune + 内层每步跑 DoRI 50 步对抗优化，对每张待擦图都要算，规模化擦除（例如几万张）的成本和工程化方案没充分讨论。
- **TM (template memorization) 评估弱**：作者指出 SSCD 这类指标对模板级记忆不敏感，仍缺乏可信指标——这给后续工作留了缺口：先解决怎么客观度量"语义级而非像素级"的记忆。

## 相关工作与启发
- **vs NeMo / Wanda**：本文是这两家方法的直接破解者。NeMo/Wanda 假设记忆在 cross-attention value 层 / FFN 输出层的少数权重里，定位 + 剪掉即可；本文用 DoRI 证明它们只切断了 prompt → image 的最短路径，记忆图依旧能被绕道唤回，而且加大剪枝强度会显著损伤生成质量，没法走通"加压补救"路线。
- **vs SISS (ICLR 2025)**：SISS 是当前数据遗忘 SOTA，能让原 prompt 触发不出来；但在 DoRI 对抗下仍漏 39%。本文方法用对抗 embedding 主动喂入训练循环，把 SISS 等"被动遗忘"升级为"主动遗忘"，MR 压到 2%。
- **vs Concept Unlearning (ESD, Concept Ablation)**：概念遗忘删的是"汽车"这种类别，不是某张具体训练图；本文实验显示这类方法对个例记忆几乎无效（MR 0.97-0.98），强调"概念遗忘 ≠ 数据遗忘"在 DM 里要分开做。
- **vs UnlearnDiffAtk (Zhang et al., 2024c)**：UnlearnDiffAtk 是概念遗忘场景下的对抗触发器搜索方法；本文显示它在记忆图复刻任务上失败，证明专门为 verbatim memorization 设计 DoRI 是必要的。
- **承接 Zhang et al., 2026** 关于记忆图诱发"尖峰激活"的直觉：本文给出了它的一个推论——既然内部状态像吸引子，那 embedding 空间到这个吸引子的"入口"自然就很多，不局限在原 prompt 附近。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "记忆是全局而非局部的"这一发现直接颠覆主流剪枝缓解路线，且用同一个 DoRI 工具统一了攻防两端。
- 实验充分度: ⭐⭐⭐⭐ SD v1.4 + v2.0 双模型、500 prompt、三层局部性诊断 + 对四类基线（剪枝/概念遗忘/数据遗忘/LoRA）全面对比；扣半星是因为只能用老模型，新模型 benchmark 缺失。
- 写作质量: ⭐⭐⭐⭐⭐ 论证结构清晰（破除 → 三层诊断 → 重建），每个表格都直接服务于核心论点，附录补全了所有可质疑的细节（阈值选取、防假阳性、强化剪枝对照）。
- 价值: ⭐⭐⭐⭐⭐ 重新定义了 DM 记忆缓解的评估标准（必须对抗多触发器而非原 prompt），并提供可直接落地的对抗微调方案，对开源模型发布前的隐私/版权审查有直接工程意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] What We Don't C: Manifold Disentanglement for Structured Discovery](../../NeurIPS2025/image_generation/what_we_dont_c_manifold_disentanglement_for_structured_discovery.md)
- [\[CVPR 2026\] SimLBR: Learning to Detect Fake Images by Learning to Detect Real Images](../../CVPR2026/image_generation/simlbr_learning_to_detect_fake_images_by_learning_to_detect_real_images.md)
- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](../../CVPR2025/image_generation/hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[AAAI 2026\] Beautiful Images, Toxic Words: Understanding and Addressing Offensive Text in Generated Images](../../AAAI2026/image_generation/beautiful_images_toxic_words_understanding_and_addressing_offensive_text_in_gene.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](../../NeurIPS2025/image_generation/score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)

</div>

<!-- RELATED:END -->
