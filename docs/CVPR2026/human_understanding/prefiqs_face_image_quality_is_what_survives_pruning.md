---
title: >-
  [论文解读] PreFIQs: Face Image Quality Is What Survives Pruning
description: >-
  [CVPR 2026 Workshops (Biometrics Workshop)][人体理解][人脸图像质量评估] PreFIQs 提出一个**完全免训练、无监督、无需数据**的人脸图像质量评估（FIQA）方法：把同一张人脸分别送进一个预训练人脸识别模型和它的剪枝版，量化两者输出嵌入的欧氏距离（漂移），漂移越小说明这张脸的身份编码越稳健、质量越高——仅靠两次前向传播就在 8 个基准、4 个识别模型上达到甚至超过有监督 SOTA。
tags:
  - "CVPR 2026 Workshops (Biometrics Workshop)"
  - "人体理解"
  - "人脸图像质量评估"
  - "模型剪枝"
  - "免训练"
  - "人脸识别"
  - "嵌入漂移"
---

# PreFIQs: Face Image Quality Is What Survives Pruning

**会议**: CVPR 2026 Workshops (Biometrics Workshop)  
**arXiv**: [2605.13396](https://arxiv.org/abs/2605.13396)  
**代码**: https://github.com/jankolf/PreFIQs (有)  
**领域**: 人体理解 / 人脸图像质量评估  
**关键词**: 人脸图像质量评估, 模型剪枝, 免训练, 人脸识别, 嵌入漂移

## 一句话总结
PreFIQs 提出一个**完全免训练、无监督、无需数据**的人脸图像质量评估（FIQA）方法：把同一张人脸分别送进一个预训练人脸识别模型和它的剪枝版，量化两者输出嵌入的欧氏距离（漂移），漂移越小说明这张脸的身份编码越稳健、质量越高——仅靠两次前向传播就在 8 个基准、4 个识别模型上达到甚至超过有监督 SOTA。

## 研究背景与动机
**领域现状**：人脸识别（FR）系统在实验室设定下已经非常准，但实战中图像常带极端姿态、光照、遮挡、模糊，会让识别崩坏。FIQA 因此成为关键预处理步骤——给每张人脸打一个"对识别有多大用处"（utility）的质量分，质量低的图在比对前被丢弃，从而稳住整体识别率。现有 FIQA 大致分三类：有监督（学一个质量回归器，依赖显式/代理标签，如 SDD-FIQA、CR-FIQA）、自监督（把质量和 FR 一起训，如 MagFace、ViT-FIQA）、无监督（探测冻结 FR 模型，看嵌入在扰动下是否稳定，如 SER-FIQ、DifFIQA、GraFIQs）。

**现有痛点**：无监督这一类的核心假设是"高质量图的表示对扰动更鲁棒"，但落地代价高。SER-FIQ 要在随机 dropout 下做 **100 次前向**统计嵌入方差，开销巨大；GraFIQs 这类基于梯度的方法需要**反向传播**，实时部署吃不消；DifFIQA 要跑扩散过程。有监督方法虽然推理快，但训练阶段依赖质量标签，而 FIQA 本身没有真值质量标签，只能用各种代理标签（相似度分布、ICAO 合规等），标签噪声大。

**核心矛盾**："想要无监督 + 免训练 + 推理便宜"这三点很难同时满足：要么靠多次随机前向（贵），要么靠反向传播（贵），要么靠学一个回归器（需训练+标签）。

**本文目标**：找一个**确定性的、单次/少次前向、不需要任何标签和训练**的信号来度量人脸图像的 utility。

**切入角度**：作者借用模型压缩里的 **PIE（Pruning Identified Exemplar，剪枝识别样例）假设**——已有工作证明，压缩后的深度网络的性能损失会**不成比例地集中在困难/低质量样本**上。作者把这条原理迁移到人脸识别：低质量人脸的身份编码**过度依赖那些"脆弱"的参数**，一旦剪枝，这些样本的嵌入会发生更大的几何位移；而高质量人脸把身份编码在更冗余、更稳定的参数子空间里，剪枝后嵌入几乎不动。

**核心 idea**：用"剪枝前后嵌入的漂移量"代替"质量分"——**质量，本质上就是剪枝后还能存活下来的东西**。

## 方法详解

### 整体框架
PreFIQs 的流程极简，本质是一个确定性公式。给定一张对齐裁剪好的人脸 $x$：先用一个预训练 FR 模型 $M_\theta$ 抽出 L2 归一化嵌入 $M_\theta(x)$；再对同一个模型做剪枝（把一部分参数置零）得到稀疏模型 $M_{\theta_\rho}$，抽出第二个嵌入 $M_{\theta_\rho}(x)$；图像质量分就定义为这两个嵌入之间的**欧氏距离（漂移）**，再线性映射到 $[0,1]$。整个过程没有任何训练、没有标签、没有反向传播，只是**两次前向传播**（原模型 + 剪枝模型各一次），剪枝在离线时只做一次、对所有图像复用同一个稀疏 mask。

因为方法就是"算一个距离"这一行，没有多阶段串行或多分支协同，这里不画 pipeline 图——文字 + 公式已足够说清。

### 关键设计

**1. 漂移即质量：用剪枝前后嵌入的欧氏距离量化 utility**

这是全文的支点，直接回应"如何在无监督、免训练下打质量分"。现代 FR 模型（ArcFace 等）把身份信息编码在嵌入的**角度方向**上，因此嵌入都做 L2 归一化、落在单位超球面上。作者把质量定义为剪枝引起的表示漂移：

$$D(x) = \lVert M_\theta(x) - M_{\theta_\rho}(x) \rVert_2$$

由于两个嵌入都归一化，距离天然有界 $0 \le D(x) \le 2$，且与角度偏差直接挂钩：$D^2(x) = 2 - 2\cos\!\big(\angle(M_\theta(x), M_{\theta_\rho}(x))\big)$，即 $D(x)$ 衡量的就是身份信息在隐空间里的角位移。最后线性缩放成"越大越好"的质量分：

$$Q(x) = 1 - \frac{D(x)}{2}$$

$Q(x)\approx 1$ 表示嵌入在剪枝下极稳定（高 utility），$Q(x)\approx 0$ 表示对剪枝高度敏感（低 utility）。和 SER-FIQ 靠随机扰动统计方差、GraFIQs 靠梯度幅值不同，这里的信号是**确定性的**（同一张图、同一个 mask，结果唯一），且和身份嵌入流形的几何**天然对齐**——不引入任何随机性或辅助损失。

**2. PIE 假设的一阶理论证明：漂移是雅可比-向量积的有效近似**

光有直觉不够，作者用一阶泰勒展开把"为什么漂移能当质量"在数学上讲圆。把剪枝建模为对权重的加性扰动 $\Delta\theta$（被剪掉的权重 $\theta_i$ 贡献 $\Delta\theta_i=-\theta_i$，否则为 0，于是 $\theta_\rho=\theta+\Delta\theta$，与 mask 形式 $\Delta\theta_i = -\theta_i\cdot(1-m_{\rho,i})$ 等价）。在适度剪枝下 $\lVert\Delta\theta\rVert$ 较小，扰动后的嵌入可近似为 $M_{\theta+\Delta\theta}(x) \approx M_\theta(x) + J_\theta(x)\cdot\Delta\theta$，其中 $J_\theta(x)$ 是嵌入对权重的雅可比矩阵。移项取范数即得

$$\lVert J_\theta(x)\cdot\Delta\theta \rVert_2 \approx \lVert M_{\theta+\Delta\theta}(x) - M_\theta(x) \rVert_2 = D(x)$$

这说明本文那个"算距离"的经验量 $D(x)$，正是隐流形精确几何敏感度（雅可比-向量积）的**一阶近似**。在作者的假设下会出现关键的不对称：高 utility 样本 $x_{\text{high}}$ 把身份编码在冗余、分布式的参数子空间里，$\Delta\theta$ 命中的几乎都是近零的雅可比项，$\lVert J_\theta(x_{\text{high}})\cdot\Delta\theta\rVert_2\approx 0$；低 utility 样本 $x_{\text{low}}$ 更依赖被剪掉的权重，雅可比项幅值大，于是 $\lVert J_\theta(x_{\text{low}})\cdot\Delta\theta\rVert_2 \gg \lVert J_\theta(x_{\text{high}})\cdot\Delta\theta\rVert_2$。直接算完整雅可比对几千万参数的模型在计算上不可行（连前向模式自动微分都很贵），而 $D(x)$ 只需一次额外前向就拿到同样的几何敏感度——这就是 PreFIQs 既"对"又"快"的根因。

**3. 剪枝配方：非结构化 + L1 幅值剪枝最优**

剪枝怎么剪，直接决定信号好不好。作者把剪枝拆成两个维度系统比较：**粒度**（非结构化——把单个权重置零、保留网络拓扑；vs 结构化——用 DepGraph 框架整组移除滤波器/通道，结构化时最后一层线性层不剪以保持嵌入维度）和**准则**（L1 幅值——剪掉绝对值最小的权重 $m_{\rho,i}=\mathbb{I}(|\theta_i|>\tau)$；vs 随机剪枝基线），在稀疏率 $\rho\in\{0.1,\dots,0.9\}$ 全谱扫描。结论很清楚：**非结构化 + L1 幅值剪枝在 $\rho=0.4$ 时质量评估效果最好**。原因在设计 1/2 的逻辑里闭环——L1 幅值剪枝优先剪掉"低贡献"参数，恰好能精准暴露低质量样本所依赖的那些脆弱参数；而随机剪枝剪不到关键容量，结构化剪枝则一刀切掉整组结构、把识别精度本身都剪垮了（见消融）。这也是为什么作者强调剪枝在这里**不是为了加速**，而是一个"系统性削减模型容量"的受控探针。

### 损失函数 / 训练策略
**无任何训练、无任何损失函数**——这正是方法的卖点。FR 模型用的是公开发布的预训练权重（ResNet100/ResNet50，MS1MV2/CASIA-WebFace 上用 ArcFace 损失训练），PreFIQs 只在推理期对其剪枝并做两次前向。验证雅可比近似时，作者用 PyTorch 自动微分在 $\rho=0.1$ 下算精确雅可比-向量积，仅作对照、不参与实际打分。

## 实验关键数据

**评测协议与指标说明**：核心指标是 **EDC 曲线**（Error-versus-Discard Characteristic，又称 ERC）——按质量分从低到高逐步丢弃图像，观察剩余图像在固定误匹配率（FMR，取 $10^{-3}$ 和 $10^{-4}$）下的误拒率（FNMR）如何下降；质量分越准，丢掉低质量图后 FNMR 掉得越快。汇总成 **pAUC**（partial AUC，丢弃率截到 30%，论文里数值均 ×$10^3$，**越低越好**）。评测覆盖 7~8 个基准（LFW、AgeDB-30、CFP-FP、CALFW、CPLFW、Adience、XQLFW、IJB-C）和 4 个 FR 模型（ArcFace、ElasticFace、MagFace、CurricularFace），分 same-model（同一个模型既打质量分又做验证）和 cross-model（ArcFace 打分、其余模型验证）两种协议。

### 主实验（与 SOTA 对比，ArcFace same-model，pAUC↓，平均列已剔除 XQLFW）

| 方法 | 类型 | Adience | AgeDB-30 | CFP-FP | CPLFW | IJB-C | 平均 pAUC |
|------|------|---------|----------|--------|-------|-------|-----------|
| eDifFIQA(L) | 有监督 | 10.210 | 6.880 | 3.546 | 20.086 | 6.469 | **9.856** |
| ViT-FIQA(T) | 自监督 | 9.948 | 8.234 | 3.568 | 20.531 | 6.563 | 10.198 |
| CLIB-FIQA | 有监督 | 10.931 | 7.387 | 4.070 | 20.431 | 6.596 | 10.181 |
| CR-FIQA(L) | 有监督 | 10.901 | 7.605 | 3.660 | 20.374 | 6.579 | 10.124 |
| SER-FIQ | 无监督(100次前向) | 11.627 | 7.776 | 3.797 | 21.570 | 6.528 | 10.593 |
| GraFIQs(L) | 无监督(反传) | 10.541 | 7.717 | 4.348 | 22.495 | 6.863 | 10.604 |
| ViTNT-FIQA | 无监督 | 10.706 | 9.674 | 4.568 | 21.802 | 6.732 | 10.966 |
| **PreFIQs (本文)** | **无监督·免训练(2次前向)** | **10.009** | **6.876** | 3.755 | 21.180 | 6.770 | **10.070** |

PreFIQs 平均 pAUC 10.070，是**所有无监督方法里最好的**（优于 CR-FIQA 等多个有监督方法），仅次于有监督 SOTA eDifFIQA 的 9.856，且代价只是两次前向。在 AgeDB-30 上 6.876，是当时该基准的**新 SOTA**（在 ArcFace/CurricularFace/MagFace 三个模型上都拿下 SOTA，ElasticFace 上第二）；在 Adience 上对全部四个 FR 模型都拿到第一或第二。

### 消融实验：剪枝粒度与准则（四模型平均 pAUC↓，FMR=$10^{-3}$）

| 配置 | 最优 $\rho$ | 平均 pAUC | 说明 |
|------|------------|-----------|------|
| 非结构化 + L1 幅值 | 0.4 | **10.516** | 最佳；$\rho$ 在 0.1~0.7 都很稳，$\rho\ge0.8$ 才掉 |
| 结构化 + L1 幅值 | 0.1 | 11.137 | 仅低稀疏可用，$\rho$ 一升就急剧恶化 |
| 非结构化 + 随机剪枝 | 0.1 | 18.298 | 远差于 L1 幅值，剪不到关键容量 |

### FR 验证精度旁证（ResNet100，剪枝后识别准确率↑）

| 配置 | $\rho=0.3$ | $\rho=0.5$ | $\rho=0.7$ | 说明 |
|------|-----------|-----------|-----------|------|
| 未剪枝 | 97.36 | — | — | 基线 |
| 非结构化 L1 | 97.32 | 97.16 | 96.36 | 几乎不掉，所以质量信号干净 |
| 结构化 L1 | 69.60 | 62.23 | 50.57 | 急崩，连识别都废了 |
| 随机剪枝 | 62.39 | 50.00 | 50.00 | $\rho=0.5$ 就退化到随机猜 |

### 关键发现
- **为什么非结构化 L1 剪枝最好**：表 5 给出根因——非结构化 L1 剪枝在 $\rho\le0.7$ 几乎不损识别精度（97.36→96.36），所以剪枝带来的嵌入漂移**纯粹反映样本敏感度**而非模型整体崩坏；结构化/随机剪枝在低稀疏率就把识别精度剪垮（随机剪枝 $\rho=0.5$ 直接掉到 50% 随机猜），漂移信号被噪声淹没，质量评估自然差。
- **理论近似被验证**：精确雅可比-向量积漂移（式 9）与本文离散漂移 $D(x)$（式 4）在 7 个基准上的平均 pAUC 几乎相同（10.514 vs 10.558），density map 分布也几乎重合——确认那个"算距离"的廉价代理确实抓住了流形的几何敏感度。
- **效率优势直观**：Table 1 的"前向次数"列里 PreFIQs 只需 2 次前向、0 次反向，而 SER-FIQ 要 100 次前向、FaceQAN 要 10 次前向 +10 次反向、GraFIQs 要 1 次反向。

## 亮点与洞察
- **跨子领域迁移的范式**：把模型压缩界的 PIE 假设（压缩损失集中在困难样本）反向用作"样本难度/质量探针"，这个视角迁移很巧——剪枝平时是手段（为了加速），这里被当成一根"诊断针"去探测样本对参数的依赖结构，思路可推广到其他"样本质量/难度"估计任务（如主动学习选样、OOD 检测）。
- **确定性 + 免训练 + 便宜，三者兼得**：FIQA 长期在"无监督但贵"和"快但需训练"之间二选一，PreFIQs 用一个确定性公式同时拿下，且给了一阶泰勒证明而非纯经验，工程与理论都站得住。
- **可复用 trick**：剪枝 mask 离线只算一次、对所有图复用；质量分天然有界 $[0,2]$ 且与角度直接挂钩，无需任何归一化/标定数据。

## 局限与展望
- **依赖一个"好剪"的过参数化骨干**：方法成立的前提是 FR 模型高度冗余、非结构化 L1 剪到 $\rho=0.4$ 几乎不掉精度。对更紧凑、本就接近最优的小模型，剪枝可能整体损伤识别能力，漂移信号会变脏，方法是否仍稳健存疑。⚠️ 以原文为准。
- **稀疏率 $\rho$ 是个需要挑的超参**：虽然 $\rho\in[0.1,0.7]$ 都还算稳，但最优点 $\rho=0.4$ 是在评测集上扫出来的，跨数据集/跨模型是否仍是最优、如何无监督地自动选 $\rho$，论文没给自适应方案。
- **一阶泰勒近似仅在"适度剪枝"成立**：当 $\rho$ 很大时 $\lVert\Delta\theta\rVert$ 不再小，式（8）的线性近似失效，这也解释了 $\rho\ge0.8$ 性能下滑；高稀疏区的行为缺乏理论刻画。
- **XQLFW 被特意从平均里剔除**：因为它的质量标签来自 SER-FIQ，会引入评测偏置——这提示该领域的"质量真值"本身就不可靠，所有 FIQA 对比都带这层不确定性。

## 相关工作与启发
- **vs SER-FIQ**：SER-FIQ 用随机 dropout 做 100 次前向、统计嵌入方差来估鲁棒性；PreFIQs 用确定性剪枝、仅 2 次前向，信号无随机性且更便宜（pAUC 10.070 vs 10.593）。区别在于"扰动来源"：SER-FIQ 扰激活，PreFIQs 扰参数。
- **vs GraFIQs**：GraFIQs 用梯度幅值当质量信号（高质量→低梯度），需要反向传播一个辅助分布偏移损失；PreFIQs 证明自己的前向漂移正是同一几何敏感度的一阶近似，**不用反传**就拿到等价信息，部署更友好。
- **vs CR-FIQA / eDifFIQA 等有监督方法**：它们要训练质量回归器、依赖代理标签；PreFIQs 完全免训练，却在多个基准追平甚至反超它们，说明"参数稀疏化"是一个被忽视但很有原则性的 utility 信号。
- **vs ViTNT-FIQA / FROQ**：同为免训练探测，但它们探测的是 Transformer 中间层/特征轨迹的稳定性，PreFIQs 探测的是**参数层面**的剪枝鲁棒性，提供了一个正交的新视角。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 PIE 剪枝假设迁移成 FIQA 信号，视角新且配一阶泰勒理论证明，不是纯 trick。
- 实验充分度: ⭐⭐⭐⭐ 8 基准 ×4 FR 模型、剪枝粒度/准则/稀疏率全谱消融、雅可比近似实证齐全；缺自适应选 $\rho$ 与小模型上的鲁棒性分析。
- 写作质量: ⭐⭐⭐⭐ 动机—假设—公式—验证链条清晰，标题点睛；表格密集、密度图较难读。
- 价值: ⭐⭐⭐⭐ 免训练 + 2 次前向 + 接近有监督 SOTA，实用价值高，且为"质量=剪枝幸存者"提供了可迁移的方法论。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback](reference-free_image_quality_assessment_for_virtual_try-on_via_human_feedback.md)
- [\[NeurIPS 2025\] Switchable Token-Specific Codebook Quantization for Face Image Compression](../../NeurIPS2025/human_understanding/switchable_token-specific_codebook_quantization_for_face_image_compression.md)
- [\[ICML 2026\] Efficient, Validation-Free Intrinsic Quality Estimation for Large-Scale Face Recognition Datasets](../../ICML2026/human_understanding/efficient_validation-free_intrinsic_quality_estimation_for_large-scale_face_reco.md)
- [\[ICCV 2025\] DynFaceRestore: Balancing Fidelity and Quality in Diffusion-Guided Blind Face Restoration](../../ICCV2025/human_understanding/dynfacerestore_balancing_fidelity_and_quality_in_diffusion-guided_blind_face_res.md)
- [\[CVPR 2026\] rPPG-VQA: A Video Quality Assessment Framework for Unsupervised rPPG Training](rppg_vqa_video_quality_assessment.md)

</div>

<!-- RELATED:END -->
