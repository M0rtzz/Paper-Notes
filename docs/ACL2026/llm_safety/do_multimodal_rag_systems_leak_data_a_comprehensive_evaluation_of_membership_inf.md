---
title: >-
  [论文解读] Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks
description: >-
  [ACL 2026][LLM安全][多模态] 作者首次系统评估了 **图像驱动的多模态 RAG (mRAG)** 系统的隐私泄露风险，证明仅靠最朴素的黑盒文本 prompt + 一张目标图就能在 4 个数据集 × 3 个 VLM 上达到 **MIA F1=0.993** 与 **caption exact-m…
tags:
  - "ACL 2026"
  - "LLM安全"
  - "多模态"
  - "Membership Inference Attack"
  - "Image Caption Retrieval"
  - "Black-box Privacy"
  - "VLM"
---

# Do Multimodal RAG Systems Leak Data? A Comprehensive Evaluation of Membership Inference and Image Caption Retrieval Attacks

**会议**: ACL 2026  
**arXiv**: [2601.17644](https://arxiv.org/abs/2601.17644)  
**代码**: https://github.com/aliwister/mrag-attack-eval  
**领域**: LLM 安全 / 隐私 / 多模态 RAG  
**关键词**: Multimodal RAG, Membership Inference Attack, Image Caption Retrieval, Black-box Privacy, VLM

## 一句话总结
作者首次系统评估了 **图像驱动的多模态 RAG (mRAG)** 系统的隐私泄露风险，证明仅靠最朴素的黑盒文本 prompt + 一张目标图就能在 4 个数据集 × 3 个 VLM 上达到 **MIA F1=0.993** 与 **caption exact-match=0.835**，即便对图像做裁剪/掩盖/旋转/噪声等变换攻击依然有效，并发现 prompt 中"目标图 vs 检索图"的相对位置和 cross-modal rerank 是两个关键缓解杠杆。

## 研究背景与动机

**领域现状**：多模态 RAG (mRAG) 已成为给 VLM 接入私有图库 + 标注的主流方式，被广泛用于 VQA、医学影像问答、版权保护等场景。pipeline 通常是 retriever → reranker → VLM 三段式。

**现有痛点**：与文本 RAG 相比，mRAG 的隐私研究极少。已有的 RAG MIA（如 S2MIA、Zeng et al. 2024）只针对纯文本，单一图模态工作（Yang et al. 2025）依赖精心设计的遮挡图作为 query，泛化性受限，且 **没人** 同时评估"图是否在库 (MIA)"和"图对应的 caption 能否被泄漏 (ICR)"两类攻击，更没人考虑库里图像可能被裁剪 / 旋转等预处理过的真实场景。

**核心矛盾**：mRAG 的设计目的是"让模型把检索内容当作权威上下文"，这恰恰意味着它在结构上鼓励模型 **逐字复述检索内容**——这与"不泄漏私有数据"的隐私要求天然冲突。设计目标和隐私目标在同一个 prompt 中竞争。

**本文目标**：(RQ1 MIA) 攻击者拿一张图能否判断它（或其变换版）是否在私库？(RQ2 ICR) 若在库中，能否提取关联的 caption？同时回答 prompt 结构与 retriever/reranker 配置如何影响泄漏程度。

**切入角度**：完全黑盒、最简化的攻击者——只给一张图 + 一句朴素 prompt，不做任何 prompt 优化、不假设白盒。这样测出的是 mRAG **系统级**的固有泄漏下限，而非攻击算法的上限。

**核心 idea**：用"最弱的攻击者打出最响的警报"——朴素 prompt 已能取得近完美 F1，证明问题不在攻击巧妙度，而在 mRAG pipeline 本身的设计缺陷。

## 方法详解

### 整体框架
威胁模型设定为黑盒：攻击者只能通过 API 提供 (target_image, text_prompt) 并读回 VLM 的文本输出，无法访问 retriever 嵌入、reranker 分数或 VLM 权重。研究分为两类攻击：MIA（二分类，输出 Yes/No）与 ICR（生成，输出 caption 字符串）。每类攻击都在 4 个数据集（Conceptual Captions、ROCOv2 医学、Pokemon BLIP、MRAG-Bench）× 3 个 VLM（Qwen2.5-VL 7B、Cosmos-Reason1 7B、InternVL3.5 8B）× 7 种图像变换（原图 / Crop / Mask / Blur / Cutout / Rotate / Gaussian Noise）矩阵上评测，retriever 用 CLIP + cosine，reranker 用 Jina-Reranker（MIA 用 image-image 模式，ICR 用 image-text 模式），默认 $n=20, k=5$。

### 关键设计

1. **Membership Inference Attack (MIA) 的极简 prompt 设计**:

    - 功能：黑盒判定目标图（或其变换形态）是否存在于 mRAG 私库 $\mathbb{R}_m=\{(i_j,c_j)\}_{j=1}^N$ 中。
    - 核心思路：攻击 prompt 直接问 VLM "the last image is identical to any of the retrieved images, in original or transformed form?"，并读 Yes/No 字符串作为标签。整条 pipeline 形式化为 $\mathcal{R}(i_q)=\text{Top}_n \cos(f_\theta(i_q), f_\theta(i_j))$，$\mathcal{R}'(i_q)=\text{Top}_k \psi(i_q, i_j)$，$y=G(i_q, \mathcal{R}'(i_q), \mathcal{P})$；当 $i_q$ 在库中时，retriever 几乎必然把它召回到上下文，VLM 把"上下文里有相同图"等价为"在库中"，于是泄漏。
    - 设计动机：故意 **不**做 prompt 优化或对抗扰动，目的是把测得的泄漏严格归因到 mRAG pipeline 本身（retrieve-rerank 的设计偏置 + VLM 对上下文图的复述倾向），而非攻击工程的产物。

2. **Image Caption Retrieval (ICR) 攻击的"逐字泄漏"利用**:

    - 功能：已知目标图在库中后，把它配对的 caption 原文从 VLM 中拉出来。
    - 核心思路：prompt 指令 VLM "identify the input image in the retrieved context and return its caption verbatim"，强制其复述。因为 reranker 在 cross-modal (image-text) 模式下倾向于把"图 + 对应描述"配对作为强候选送进上下文，VLM 在面对 1 张目标图 + $k$ 个含 caption 的对子时，最大似然路径就是把对应 caption 原文照抄。指标用 exact-match + BLEU-2 + ROUGE-1 + METEOR 衡量泄漏的"逐字程度"和"语义程度"。
    - 设计动机：作者发现即使目标图未被精确检索回来，VLM 仍倾向于从上下文中挑一句"语义上像"的 caption 复述（论文称为"间接泄漏"），所以 exact-match 是 mRAG 隐私的核心红线。

3. **图像变换鲁棒性测试 + Prompt 位置消融**:

    - 功能：模拟数据增强 / 隐私缓解过的图库，验证"对图做物理变换"是否能作为防御。
    - 核心思路：评测 6 种变换攻击——Crop（裁到原图 60%）/ Mask（灰度化）/ Blur / Cutout（4% 矩形遮挡）/ Rotate（90° 或翻转）/ Gaussian Noise $\mathcal{N}(0, 25^2)$。同时做 prompt 结构消融：RAG-First（检索图放在目标图之前）vs RAG-Last（目标图在前），观察"位置偏置"对泄漏的影响。
    - 设计动机：找出最便宜的部署级防御——如果只需把目标图放 prompt 最前，或对库图统一旋转，就能显著降低泄漏，那对工程界比"训练隐私模型"更可落地。

### 损失函数 / 训练策略
本文是评测/攻击研究，不训练任何模型，所有结果均为 black-box 推理；每个配置 3 次独立随机种子取均值与方差。

## 实验关键数据

### 主实验
**MIA 在精确图攻击下的泄漏（Table 1 节选，F1 score 越高表示泄漏越严重）**：

| 数据集 | Qwen2.5-VL F1 | Cosmos-Reason1 F1 | InternVL3.5 F1 | RAG Acc |
|--------|--------------|-------------------|---------------|---------|
| Conceptual Captions | 0.946 | **0.989** | 0.988 | 0.999 |
| ROCOv2 (医学) | 0.893 | **0.952** | 0.897 | 0.995 |
| Pokemon BLIP | **0.993** | 0.983 | 0.908 | 1.000 |
| mRAG-Bench | 0.966 | **0.983** | 0.899 | 1.000 |

**ICR 在精确图攻击下的 caption 泄漏（Table 3 节选）**：

| 数据集 | 模型 | Exact-Match | BLEU-2 | ROUGE-1 | METEOR | RAG Acc |
|--------|------|-------------|--------|---------|--------|---------|
| Conceptual Captions | Qwen2.5-VL | **0.835** | 0.853 | 0.882 | 0.875 | 0.892 |
| Conceptual Captions | Cosmos-Reason1 | 0.470 | 0.627 | 0.761 | 0.730 | 0.892 |
| Conceptual Captions | InternVL3.5 | 0.747 | 0.791 | 0.830 | 0.817 | 0.892 |
| ROCOv2 (医学) | Qwen2.5-VL | 0.451 | 0.597 | 0.607 | 0.594 | 0.597 |
| Pokemon BLIP | Qwen2.5-VL | 0.743 | 0.794 | 0.852 | 0.828 | 0.753 |
| mRAG-Bench | Qwen2.5-VL | **0.801** | 0.794 | 0.819 | 0.539 | 0.823 |

可见 MIA 接近完美（F1 ≥ 0.89），ICR 在普通图像数据集上能逐字复刻 70%+ 的 caption，医学图像因 retriever 召回率本身只有 0.597 而显著更低。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| MIA 精确图 (Conceptual Captions, Cosmos) | F1 = 0.989 | baseline 上限 |
| MIA + Rotate 变换 | F1 ≈ 0.60（论文文本报告均值，最低组） | 旋转破坏空间特征，最有效防御 |
| MIA + Crop/Mask/Blur/Cutout/Noise | F1 = 0.85–0.96 | 普通变换基本拦不住 |
| MIA 默认 RAG-First | F1 ≈ 0.99 | 检索图在前 → 模型把目标图当上下文 |
| MIA RAG-Last（目标图在前） | F1 显著下降（Qwen / Cosmos 尤为明显） | 位置偏置 → 简单防御 |
| ICR $k=5$, Qwen on ROCOv2 | EM = 0.451 | baseline，强 rerank |
| ICR $k=10$, Qwen on ROCOv2 | EM = 0.581 | 上下文翻倍，泄漏↑ |
| ICR $k=20$ (≈ $n$), Qwen on ROCOv2 | EM = 0.702 | rerank 形同虚设，泄漏达最高 |

### 关键发现
- **MIA 几乎无法防御**：精确图攻击下 F1 普遍 ≥ 0.89，RAG Acc 高达 0.999，说明检索阶段几乎 100% 召回，VLM 的 yes/no 输出实际上是 RAG retriever 的"放大器"。
- **caption 逐字泄漏严重**：Qwen2.5-VL 在 Conceptual Captions 上 83.5% 完全字面匹配，是直接的 GDPR/HIPAA 级风险。
- **图像复杂度是天然防御**：ROCOv2 医学图像内 retriever 召回率本身只有 0.597，attack EM 跌到 0.451，说明高内变性数据集对 mRAG 隐私来说"自带噪声"。
- **prompt 位置是廉价防御**：RAG-Last（目标图放最前）能显著降低 MIA 成功率，原因是 VLM 有位置偏置，倾向把"先看到的图"视为输入而非上下文。
- **rerank 是双刃剑**：cross-modal rerank 对 ICR 有缓解作用（把 $k$ 调到 $=n$ 即关闭 rerank 时 EM 从 0.45 飙到 0.70），但对 MIA 几乎没用，因为 retriever 已经命中。
- **变换里 Rotate 最有效**：旋转 90° 破坏 CLIP 嵌入的空间结构，把 F1 拉到 0.60 左右，远低于其他变换；可作为最低成本的图库侧防御。

## 亮点与洞察
- **"最弱攻击者+最强警报"叙事**：刻意不优化 prompt、不用白盒，攻击成功率仍然这么高，使得结论无法被"是不是攻击者太强"反驳，论证非常稳固。
- **首次把 ICR 与 MIA 放在统一框架下**：之前研究只看 MIA，作者补全了"先确认在库 → 再 leak caption"的完整攻击链，正好对应医疗 / 版权两个真实威胁场景。
- **变换矩阵 × 模型矩阵 × 数据矩阵**：用 3×4×7 的密集网格证伪了"图像变换可以保护隐私"的普遍直觉，仅 rotate 例外，工程指导价值很高。
- **位置偏置作为隐私通道**：把"VLM 把先看到的图视为查询"这一现象转化为可部署的隐私加固手段，几乎零成本即可下调 MIA F1。
- **指出 rerank 的双面性**：通常 rerank 被视为"提精度"，本文把它重新定义为"隐私调节器"——k 与 n 的差距越大，越能屏蔽 ICR。

## 局限与展望
- **黑盒朴素 prompt 是下界**：更复杂的 prompt 注入 / 多轮交互 / chain-of-thought 引导可能更强，本文没探索。
- **库规模偏小**：mRAG 数据库总量都在千级（Conceptual Captions 2000+1000、Pokemon 433+400），实际生产 mRAG 可能上千万规模，泄漏几率不一定线性外推。
- **仅图—文配对**：实际 mRAG 可能配多张图 + 多段文字、跨用户共享库；隐私关系会复杂得多。
- **缺乏正式 DP 防御对比**：仅讨论了 prompt 位置 / rotate / rerank 三个工程性缓解，没和差分隐私 retriever / 噪声化 caption 等系统级方法对比。
- **VLM 全部约 7-8B**：未覆盖更大模型 (GPT-4o / Gemini)，但闭源 API 黑盒攻击通常比开源更难解释，未来值得补。

## 相关工作与启发
- **vs Zeng et al. 2024 (RAG privacy)**: 他们做纯文本 RAG MIA，本文扩展到多模态并加入图像变换维度，揭示了文本-only 框架未触及的跨模态泄漏路径。
- **vs S2MIA (Li et al. 2025)**: 仅 MIA，本文补 ICR；同时本文证明朴素 prompt 已足够，质疑了需要复杂"shadow model"或 inference attack 的必要性。
- **vs Yang et al. 2025 (mRAG MIA)**: 他们依赖人为遮挡的 query 图像，泛化弱；本文系统化评测 6 种变换 + 医学图像，展示更普遍的攻击表面。
- **vs Zhang et al. 2025 (mRAG prompt leakage)**: 他们关注 text-from-image/speech 泄漏，本文聚焦 image-centric mRAG 和 caption 的逐字提取，是互补的另一面。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将 MIA + ICR 放进 mRAG 统一评测，并系统化研究图像变换的影响；但每个单项攻击的"算法侧"创新有限，靠的是评测覆盖面。
- 实验充分度: ⭐⭐⭐⭐⭐ 3 模型 × 4 数据集 × 7 变换 × 多组消融（prompt 顺序、k/n、有无 rerank），跑了 3 次随机种子有方差，几乎是 mRAG 隐私评测的"基线手册"。
- 写作质量: ⭐⭐⭐⭐ 问题陈述清晰、表格密集但易读；公式与 threat model 部分规范。
- 价值: ⭐⭐⭐⭐⭐ mRAG 在医疗 / 法务 / 企业知识库快速铺开，本文揭示的泄漏下限直接是合规警告；工程界可立刻采纳 RAG-Last + rerank 紧 k 的缓解组合。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Membership Inference Attacks on In-Context Learning Recommendation](membership_inference_attacks_on_llm-based_recommender_systems.md)
- [\[ACL 2026\] LeakDojo: Decoding the Leakage Threats of RAG Systems](leakdojo_decoding_the_leakage_threats_of_rag_systems.md)
- [\[ACL 2026\] Fast-MIA: Efficient and Scalable Membership Inference for LLMs](fast-mia_efficient_and_scalable_membership_inference_for_llms.md)
- [\[ICLR 2026\] No Caption, No Problem: Caption-Free Membership Inference via Model-Fitted Embeddings](../../ICLR2026/llm_safety/no_caption_no_problem_caption-free_membership_inference_via_model-fitted_embeddi.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)

</div>

<!-- RELATED:END -->
