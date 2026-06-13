---
title: >-
  [论文解读] Formalizing the Binding Problem
description: >-
  [ICML 2026][可解释性][绑定问题] 本文把"神经网络的绑定问题"形式化为表示 $Z$ 中关于对象码 $O$ 的互信息 $I(O;Z)$，并设计自回归概率探针在 DINOv2 / CLIP 等 ViT 上度量绑定信息，发现 `[CLS]` 只编码 <50% 的绑定信息且结构近似二次型，而对全套空间 token 加注意力探针能恢复 ~92% 的绑定信息。
tags:
  - "ICML 2026"
  - "可解释性"
  - "绑定问题"
  - "信息论探针"
  - "ViT表示"
  - "CLS] token"
  - "空间 token"
---

# Formalizing the Binding Problem

**会议**: ICML 2026  
**arXiv**: [2606.03976](https://arxiv.org/abs/2606.03976)  
**代码**: https://github.com/KordingLab/formalizing-the-binding-problem  
**领域**: 可解释性 / 表示分析 / 视觉Transformer  
**关键词**: 绑定问题、信息论探针、ViT表示、[CLS] token、空间 token

## 一句话总结
本文把"神经网络的绑定问题"形式化为表示 $Z$ 中关于对象码 $O$ 的互信息 $I(O;Z)$，并设计自回归概率探针在 DINOv2 / CLIP 等 ViT 上度量绑定信息，发现 `[CLS]` 只编码 <50% 的绑定信息且结构近似二次型，而对全套空间 token 加注意力探针能恢复 ~92% 的绑定信息。

## 研究背景与动机

**领域现状**：现代视觉与视觉-语言模型大量依赖 ViT 把图像编码成 `[CLS]` 摘要 token 或一组空间 token，下游再做对比学习（CLIP / SimCLR / Barlow Twins）或 cross-modal 拼接（LLaVA / Qwen-VL）。已有工作只证明了 ViT "知道哪些 patch 属于同一个对象"，并没有回答模型表示里到底有多少"哪些特征属于同一个对象"的信息。

**现有痛点**：VLM 在多对象、特征共享、遮挡场景下常常张冠李戴——把蓝色帽子和红色书包说成"红色帽子"。Campbell et al. 2025 等工作给出了大量任务级失败案例，但缺少一个**可量化、跨模型可比**的绑定能力度量，更没法定位绑定信息到底丢在了 `[CLS]` 还是空间 token 的哪一侧。

**核心矛盾**：绑定问题既是认知科学概念又是工程概念，两边都没给出"表示中绑定信息含量"的形式化定义。直接看下游任务准确率，会把"模型不知道"和"模型知道但 readout 输出错了"混在一起，无法分离表示本身的信息瓶颈。

**本文目标**：(1) 给出与具体编码方式无关的绑定信息形式化定义；(2) 提供一个可在任意预训练表示上跑通的探针估计器；(3) 用它解剖 ViT 不同组件、不同数据集下的绑定能力。

**切入角度**：作者把"特征是否存在"与"对象是否存在"分别建成二值随机向量 $F$ 与 $O$，借助互信息把"绑定 = 表示中关于对象码的信息"压缩成一个数。再用"探针的交叉熵 = 条件熵 + KL"这一经典不等式（Lemma 2.21），把不可直接计算的 $H(O|Z)$ 用一个训练好的探针的 test loss 上界来估计。

**核心 idea**：用信息论把绑定信息拆成"数据集先验 $H(O)$ - 探针残余不确定性 $H(O|Z)$"，并用自回归分解 $q_\theta(o|z)=\prod_k q_\theta(o_k|o_{<k},z)$ 绕过对象码组合爆炸，从而在任意 ViT 上得到可比较的绑定信息标量。

## 方法详解

### 整体框架
这套方法要回答"一个冻结 ViT 的表示 $Z$ 里到底藏了多少绑定信息"，做法是把"绑定"翻译成互信息：先把场景标注成 ground-truth 特征码 $F\in\{0,1\}^n$ 与对象码 $O\in\{0,1\}^K$，再用"探针的 test 交叉熵是条件熵的上界"这一不等式，把不可直接算的 $H(O|Z)$ 用一个训练好的自回归探针的残余 loss 来估计。

具体落到四步：先用组合学/经验分布算出数据集先验 $H(O)$、$H(F)$；再在冻结表示 $Z$ 上分别训对象探针 $q_\theta(o_k|o_{<k},z)$ 和特征探针 $q_\theta(f_k|f_{<k},z)$；把它们在 test set 上的交叉熵当作 $H(O|Z)$、$H(F|Z)$ 的上界；最后代入 $I(O;Z)=H(O)-H(O|Z)$ 与 Theorem 2.16，产出四个标量——绑定信息 $B_O(Z)=I(O;Z)$、特征条件绑定信息 $B^*_{O,F}(Z)=I(O;Z|F)$，以及除以先验得到的归一化版 $\beta_O(Z)$、$\beta^*_{O,F}(Z)$。

### 关键设计

**1. 绑定信息的信息论形式化：把"会不会绑定"压成一个 bit 数**

绑定问题此前只有概念辨析、没有与编码方案无关的可比尺度，本文把场景抽象成"特征集 $F$ + 对象集 $O$"，每个对象 $o_i$ 对应 $F$ 的一个子集（surjective map），于是 $F$ 成为 $O$ 的确定函数，绑定信息就定义成 $B_O(Z):=I(O;Z)$。直接用这个量还有个隐患：一个只把单特征学得很好的模型也会拿到高分，被误判成"会绑定"。为此引入条件版 $B^*_{O,F}(Z):=I(O;Z|F)=I(O;Z)-I(F;Z)$（Theorem 2.14），扣掉仅靠特征信息就能推出的部分，只留真正的"哪些特征属于同一对象"。再用 $H(O)$、$H(O|F)$ 把两者归一化成无量纲的 $\beta$、$\beta^*$，跨数据集才能公平对比。用互信息而非下游准确率，是因为准确率会被 readout 头吞掉信息，而 $I$ 既反映软概率的不确定度，又能通过"换探针族"暴露信息的代数结构（线性 / 二次 / 高阶）。

**2. 自回归对象码探针 + 探针族阶梯：在单 token 上估信息、顺带读出信息结构**

要在 `[CLS]` 这种单 token 上估 $H(O|Z)$，直接训 $q_\theta(o|z)$ 得覆盖 $2^K$ 的标签空间（实验中 $K=64$，超过 $10^{19}$），完全不可行。本文用链式分解 $q_\theta(o|z)=\prod_{k=1}^K q_\theta(o_k|o_{<k},z)$ 绕过组合爆炸：把表示 $z$ 与已揭示的对象码 $o_{<k}$ 拼成 $x=[z\|o_{<k}]$ 喂给探针，每个 $o_k$ 用一个二分类 logit $\ell_k(x)$ 监督，总 loss 求和即式 (5)。关键的巧思在于让探针族成一个阶梯——线性 $\ell_k=W_kx+b_k$、二次 $\ell_k=x^\top W_k x+b_k$、四层 GELU DNN（约 3M 参数）——同时去逼近 Lemma 2.21 给出的 $L_{CE}\ge H(O|Z)$ 下界。在合成 ColorShape 上三族 test loss 分别为 $34.2/22.0/20.6$ bits（误差降幅 $37.4\%/63.0\%/64.4\%$），二次族与 DNN 几乎打平，说明绑定信息基本能被二阶函数读出。更进一步，把二次探针参数共享成 $W_k=U_{\text{color}}^\top V_{\text{shape}}$ 后只多 $2.4$ bits loss，等于把"`[CLS]` 的绑定信息是不是 $\text{color}\otimes\text{shape}$ 双线性形式"做成了可证伪实验，结论是它确实就是颜色投影与形状投影的点积——一种保守的合取式编码。

**3. 空间 token 上的简化注意力探针：只学 query 不学 key/value**

ViT 全套空间 token $\{s_i\}_{i=1}^N$ 的绑定信息天然分散在多个 patch 上，直接拼接喂 MLP 会让参数爆炸且过拟合。本文为每个对象 $o_k$ 只学一个 query $q_k=g_k(o_{<k})$（没有 key / value 投影），按 $a_{k,i}=\text{softmax}_i(q_k^\top s_i)$ 把空间向量路由成加权平均 $\bar s_k=\sum_i a_{k,i} s_i$，再过二次读出 $q_\theta(o_k=1|o_{<k},\{s_i\})=\sigma(\bar s_k^\top W_k \bar s_k+b_k)$。刻意砍掉 key/value 是为了让 routing 完全由表示本身决定、避免探针自己"发明"信息，同时让注意力权重 $a_{k,i}$ 直接当可视化诊断。结果 ColorShape 上 test ER 达 $96.8\%$（loss 仅 $3.1$ bits），远低于 `[CLS]` 最强 DNN 探针的 $20.6$ bits，且注意力几乎总能 route 到目标对象所在 patch——证明绑定信息确实以"对象在哪个 patch"这种空间索引形式存在于 patch token 里。

### 损失函数 / 训练策略
所有探针都用二元交叉熵 $L_{CE}(\theta)=\sum_k \mathbb{E}_{(z,o)}[-\log q_\theta(o_k|o_{<k},z)]$ 训练，特征探针完全对称。报告 test loss 而非 train loss，并通过训练-验证-测试三集都使用 disjoint 的特征码 $F$ 与对象码 $O$ 来防止记忆（对象码组合空间 $\sim10^{12}$，自然分裂）。DINOv2-Large、CLIP ViT-L/14（224 与 336 两个分辨率）保持冻结，只训探针。

## 实验关键数据

### 主实验：`[CLS]` 单 token 绑定信息（ColorShape，$H(O)=39.9$ bits，$H(F)=7.0$ bits）

| 探针族 | $B_O(Z)$ (bits) | $\beta_O(Z)$ | $B^*_{O,F}(Z)$ (bits) | $\beta^*_{O,F}(Z)$ |
|--------|-----------------|---------------|-------------------------|---------------------|
| Linear | 5.7 | 14.3% | 0.3 | 0.8% |
| Quadratic | 17.9 | 44.9% | 12.5 | 37.9% |
| DNN (3M params) | 19.4 | **48.5%** | 13.9 | **42.4%** |
| **Attention + 空间 token** | **36.8** | **92.2%** | **31.0** | **94.1%** |

最后一行（来自 Table 5）是把表示换成全套空间 token + 注意力探针的结果，直接把绑定信息从 48.5% 抬到 92.2%。

### 消融：数据复杂度、遮挡、自然数据集

| 实验 | 关键发现 |
|------|---------|
| ColorShape 颜色×形状从 1→7 (49 组合) | 随特征空间指数增长，$\beta^*_{O,F}(Z)$ 单调下降但**没有指数衰减**（CLIP-L/14 224px） |
| CLEVR 不同遮挡（相机高度 0.6→3.2） | $\beta^*_{O,F}(Z)$ 从 45.0% 单调升到 58.7%，遮挡每加一档掉 ~3 pp |
| 参数共享二次探针 vs 标准二次探针 | 共享 $U_{\text{color}}^\top V_{\text{shape}}$ 仅多 2.4 bits loss → `[CLS]` 绑定信息几乎是纯二次合取码 |
| Visual Genome 自然数据 | DINOv2 / CLIP 在 VG:Color / VG:TopAttr 上 $\beta^*_{O,F}\in[39.9\%,47.0\%]$，与合成数据相当 |
| CLIP 224px → 336px | ColorShape 47.7%→56.4%；细粒度空间表示帮助绑定 |

### 关键发现
- **`[CLS]` 是绑定瓶颈**：信息量上界 ~48.5%，再叠 DNN 也只多 1.5 pp；并且这点信息几乎可被一个 $\text{color}\otimes\text{shape}$ 双线性形式完整解释，说明 ViT 摘要 token 学到的不是"对象槽"，而是"特征对的合取统计"。
- **空间 token 几乎无信息损失**：注意力探针把绑定信息恢复到 ~92%，意味着 ViT 的绑定能力其实保存在 patch token 的空间索引里，对比学习类预训练（只用 `[CLS]`）的目标函数没有暴露这部分能力。
- **绑定难度可分解**：复杂度（特征空间大小）、遮挡、自然性是三个正交维度；用 $\beta^*_{O,F}$ 都能稳定单调反映。
- **分辨率比模型本身更重要**：同样 CLIP-L/14，从 224 到 336 像素的提升超过换 DINOv2 这种不同自监督算法的差距。

## 亮点与洞察
- **把一个哲学/认知概念变成可测标量**：先前对"binding"的讨论几乎都停留在概念辨析层面，这篇把"我有多少绑定信息"压成一个 bit 数，并给了可复现的探针管线，把绑定从理论问题搬到工程基准上。
- **探针族阶梯当作"信息结构显微镜"**：linear/quadratic/DNN 的 loss 差距 + 参数共享变体，直接告诉你"`[CLS]` 里的绑定信息是颜色×形状的双线性映射"，远比单看准确率信息量大；同样思路可迁移到任何"信息存不存在 / 是不是低阶"的探针实验。
- **简化注意力探针的妙用**：只学 query 不学 key/value 的极小注意力，既保证 routing 是被表示本身决定的（避免探针自己发明信息），又能复用 $a_{k,i}$ 做可解释性可视化。
- **量化"对比学习+`[CLS]`"路线的代价**：用一个数告诉社区——CLIP 系列只看 `[CLS]` 的下游头其实丢掉了一半绑定信息，VLM 用 patch token 的做法被这套度量背书。

## 局限与展望
- **依赖预定义离散特征/对象词典**：所有公式都建立在 $F\in\{0,1\}^n$、$O\in\{0,1\}^K$ 上，连续特征（如 velocity=3m/s）只能离散化处理；作者承认未来需要连续探针。
- **测的是"可解码"信息，不是"模型用了的"信息**：高 $\beta$ 只说明信息在 $Z$ 里能被探针挖出来，并不保证下游 readout（如 LLaVA 的 LLM 头）真的用到了——所以高分模型在 VLM 任务上仍可能绑定失败。
- **条件版 $B^*_{O,F}$ 假设 $F$ 可由 $O$ 推出**：在生物视觉或噪声场景下并不严格成立，会让 Theorem 2.14 的分解失真。
- **实验主要在 ColorShape + CLEVR + VG 子集**：分类对象数仍有限（$K=64$ 量级），数千类级别（如 LVIS）下二次探针是否仍接近 DNN 是开问题。
- **改进思路**：把探针 loss 反向当作训练目标（binding-aware pretraining）；把度量推广到视频里的时序绑定；研究"信息结构"是否能预测下游 compositional generalization 的成功率。

## 相关工作与启发
- **vs Campbell et al. 2025 / Zhang et al. 2024**：他们用下游 VLM 任务的错误率说明绑定失败，但无法定位失败发生在表示阶段还是 readout 阶段；本文把诊断下沉到表示层，并能区分"信息没有"与"信息有但 readout 用不上"。
- **vs Greff et al. 2020（On the binding problem in artificial neural networks）**：那是一篇 position paper，给出概念分类（segregation / representation / composition）；本文把其中的"representation binding"做了可度量化定义，是对它的工程化补完。
- **vs 槽注意力 / Capsule / Tensor-Product 等显式绑定架构**：这些架构提出新编码方案以获得绑定能力，但缺乏统一比较尺度；本文的 $\beta^*_{O,F}$ 提供了一个跨架构的公共基准，未来可用它公平比较 slot attention vs ViT vs JEPA。
- **vs 标准 probing（如 BERT linguistic probes）**：传统 probing 主要测"分类能力"；本文把 probing 的 loss 直接当作信息论上界，让 probing 从"性能测试"升级为"信息量估计"，并通过探针族对比读出信息的代数结构。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一篇把绑定问题与互信息严格对齐并给出可计算估计的工作。
- 实验充分度: ⭐⭐⭐⭐ 覆盖合成 / CLEVR / VG 三类数据 + DINOv2 / CLIP 双架构 + 49 组复杂度扫描，但模型规模和对象空间仍偏小。
- 写作质量: ⭐⭐⭐⭐⭐ 定义—定理—探针—实验—解释的链条清晰，每个度量都附了可计算的方法。
- 价值: ⭐⭐⭐⭐⭐ 给社区一个标量基准，能直接指导"用 `[CLS]` 还是 patch token"、"上不上更高分辨率"这类落地决策，也为未来 binding-aware 预训练打好了评估底座。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning](../../AAAI2026/interpretability/attention_as_binding_a_vector-symbolic_perspective_on_transformer_reasoning.md)
- [\[ICLR 2026\] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](../../ICLR2026/interpretability/uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)
- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Verified SHAP: 神经网络精确 Shapley 值的可证明界](verified_shap_provable_bounds_for_exact_shapley_values_of_neural_networks.md)
- [\[ICML 2026\] Courtroom Analogy: New Perspective on Uncertainty-Aware Classification](courtroom_analogy_new_perspective_on_uncertainty-aware_classification.md)

</div>

<!-- RELATED:END -->
