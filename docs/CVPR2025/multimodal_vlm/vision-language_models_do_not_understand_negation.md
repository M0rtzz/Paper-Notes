---
title: >-
  [论文解读] Vision-Language Models Do Not Understand Negation
description: >-
  [CVPR 2025][多模态VLM][否定理解] 本文提出 NegBench 基准，系统揭示了 CLIP 等视觉语言模型在否定理解上的严重缺陷（表现接近随机水平），并通过在大规模合成否定数据集上微调，将否定查询的检索召回率提升 10%、MCQ 准确率提升高达 40%。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "否定理解"
  - "视觉语言模型"
  - "CLIP"
  - "基准测试"
  - "数据驱动改进"
---

# Vision-Language Models Do Not Understand Negation

**会议**: CVPR 2025  
**arXiv**: [2501.09425](https://arxiv.org/abs/2501.09425)  
**代码**: [https://NegBench.github.io](https://NegBench.github.io)  
**领域**: 多模态VLM  
**关键词**: 否定理解, 视觉语言模型, CLIP, 基准测试, 数据驱动改进

## 一句话总结
本文提出 NegBench 基准，系统揭示了 CLIP 等视觉语言模型在否定理解上的严重缺陷（表现接近随机水平），并通过在大规模合成否定数据集上微调，将否定查询的检索召回率提升 10%、MCQ 准确率提升高达 40%。

## 研究背景与动机

**领域现状**：联合嵌入式视觉语言模型（如 CLIP）通过大规模图文对训练，在跨模态检索、图像描述、文本生成图像等任务上取得了卓越表现，成为多模态AI的基础组件。

**现有痛点**：这些模型在处理否定语句时存在严重缺陷。例如，"a beach with no people" 和 "a beach with people" 在 CLIP 嵌入空间中几乎不可区分。这在医学图像诊断（"no evidence of pneumonia"）、安全监控（"construction sites with no barriers"）等应用场景中可能带来严重后果。现有基准如 CREPE 和 CC-Neg 仅包含少量模板化的否定测试，无法全面评估。

**核心矛盾**：CLIP 的对比学习训练目标鼓励模型学习图像和文本之间的"词袋"式匹配，即关注关键名词的存在性而忽略限定词（如"no"、"not"）。这种"肯定偏见"（affirmation bias）根植于训练数据——互联网图文对几乎全是肯定描述，极少包含否定表达。

**本文目标**：（1）构建全面的否定理解评测基准；（2）诊断 VLM 否定理解失败的根本原因；（3）探索数据驱动的解决方案。

**切入角度**：从信息检索系统的多级评估思路出发——先粗粒度检索（是否能检索到包含/排除特定对象的图像），再细粒度判别（能否从近似选项中选出正确的否定描述）。

**核心 idea**：用"挑战-诊断-方案"的逻辑链，通过构建大规模合成否定数据集来教会 CLIP 理解否定。

## 方法详解

### 整体框架
NegBench 包含两个核心任务：(1) Retrieval-Neg：在原始描述中加入否定陈述后进行图文检索，评估模型对否定查询的处理能力；(2) MCQ-Neg：给定图像和四个选项（含肯定/否定/混合描述），模型需选出正确描述。基准覆盖 COCO、VOC2007（图像）、MSR-VTT（视频）、CheXpert（医学影像）和合成的 HardNeg-Syn 数据集，共 18 种任务变体、79K 样本。

### 关键设计

1. **NegBench 基准构建**:

    - 功能：系统评估 VLM 的否定理解能力
    - 核心思路：对每个数据集，先提取正面概念（图中存在的对象集合 $\{pos\}$）和负面概念（不存在但语境相关的对象集合 $\{neg\}$）。利用 LLaMA 3.1 生成自然语言的否定描述并做释义（paraphrase），确保语言多样性。MCQ 设计三种模板——纯肯定、纯否定、混合（"包含A但不含B"），错误选项为精心设计的硬负样本（如把存在的对象否定、把不存在的对象肯定），迫使模型必须真正理解否定才能作答
    - 设计动机：现有基准只用固定模板测试否定，无法反映真实查询中否定的多样性。HardNeg-Syn 使用 Stable Diffusion 生成配对图像（一张含目标对象、一张不含），通过 OWL-ViT 验证对象存在/缺失，提供最严格的否定测试

2. **诊断分析：嵌入空间可视化**:

    - 功能：揭示 VLM 否定理解失败的根因
    - 核心思路：用 PCA 可视化 CLIP 对肯定和否定描述的嵌入。发现 CLIP 和 NegCLIP 的肯定/否定描述嵌入完全重叠（无法区分"a dog"和"no dog"），说明模型采用了忽略否定词的"词袋"快捷策略。ConCLIP 虽然分离了肯定/否定，但将所有否定描述压缩到同一点（不区分"no dog"和"no cat"），属于另一种退化。Sentence-Transformer 文本模型则展示了理想的分离——沿"对象类型"和"否定"两个正交维度清晰分开
    - 设计动机：需要理解失败的机制才能设计有效的改进方案。诊断表明问题出在训练数据而非模型架构

3. **CC12M-NegFull 合成否定训练集**:

    - 功能：提供大规模否定训练信号
    - 核心思路：基于 CC12M 数据集（1000万图文对），利用 LLaMA 3.1 提取每张图中存在和不存在的对象，用 OWL-ViT 进行视觉验证，然后生成自然语言否定描述。包含两个子集：CC12M-NegCap（每图3条含否定的描述，约3000万条描述）用于对比学习；CC12M-NegMCQ（每图4条描述含1正3负，约4000万条）用于细粒度否定学习。联合训练损失为 $\mathcal{L}_{Total} = \alpha \mathcal{L}_{CLIP}(\mathcal{B}_{cap}) + (1-\alpha)\mathcal{L}_{MCQ}(\mathcal{B}_{mcq})$
    - 设计动机：诊断表明问题根源在训练数据缺乏否定样本。与其修改模型架构，不如从数据层面补齐短板

### 损失函数 / 训练策略
在 CC12M-NegCap 上用标准 CLIP 对比损失训练，改善粗粒度检索的否定理解。在 CC12M-NegMCQ 上加入 MCQ 交叉熵损失 $\mathcal{L}_{MCQ} = -\frac{1}{M}\sum_{i=1}^{M}\log\frac{\exp(\text{logits}_{i,c_i})}{\sum_{j=1}^{C}\exp(\text{logits}_{i,j})}$，强迫模型区分正确和硬负描述。$\alpha$ 控制两者的平衡。

## 实验关键数据

### 主实验

| 模型 | 微调数据 | COCO R@5 | COCO R-Neg@5 | COCO MCQ Acc |
|------|---------|----------|-------------|-------------|
| CLIP | 无 | 54.8 | 48.0 | 16.3% |
| CLIP | CC12M | 58.8 | 54.5 | 11.2% |
| CLIP | CC12M-NegCap | 58.5 | 57.8 | 14.7% |
| CLIP | CC12M-NegFull | 54.2 | 51.9 | **46.9%** (+30.6) |
| NegCLIP | 无 | 68.7 | 64.4 | 10.2% |
| NegCLIP | CC12M-NegFull | 69.0 | 67.0 | **51.0%** (+40.8) |

### 消融实验

| $\alpha$ 值 | COCO R@5 | COCO MCQ Acc | 说明 |
|-------------|----------|-------------|------|
| 0.0 (纯MCQ损失) | 33.9% | 61.0% | 检索崩溃，MCQ最优 |
| 0.25 | 37.3% | 54.7% | — |
| 0.5 | 47.6% | 50.5% | 较好平衡 |
| 0.75 | 54.2% | 46.9% | 偏向检索 |
| 1.0 (纯CLIP损失) | 58.5% | 14.7% | MCQ无改善 |

### 关键发现
- **否定理解全面失败**：所有 CLIP 基础模型在 MCQ-Neg 上的准确率低于随机猜测（25%），有的仅 8%，说明模型系统性地选择了错误的否定模板
- **扩大模型规模无助于解决问题**：从 ViT-B/32 到 ViT-H/14，否定理解几乎无改善
- **医学领域后果严重**：BioMedCLIP 和 CONCH 在引入否定后性能分别下降 24.6% 和 33.2%
- **对比学习不充分**：单独用 NegCap 微调只能改善检索，MCQ 需要配合 MCQ 损失才能显著提升
- 存在检索能力和否定理解之间的 trade-off，需要 $\alpha$ 精细调节

## 亮点与洞察
- **系统性评测范式**：NegBench 的粗粒度检索+细粒度 MCQ 两级评测设计非常巧妙，可以精确定位模型在否定理解链条上的哪个环节失败
- **嵌入空间诊断**：通过 PCA 可视化肯定/否定嵌入的重叠，直观且有说服力地证明了"词袋式快捷策略"的存在。这种诊断方法可迁移到其他语言理解缺陷的分析中
- **合成数据的威力**：用 LLM+检测器生成和验证大规模训练数据的流程，无需人工标注即可大幅改善特定能力，这个范式可推广到其他 VLM 的能力短板修复

## 局限与展望
- 微调在否定上改善的同时，标准检索性能有轻微下降，说明 CLIP 的嵌入空间容量有限，需要更好的训练策略来同时优化两者
- 目前只在 CLIP 类联合嵌入模型上验证，生成式 VLM（如 LLaVA）在附录中展示了更好的否定理解，但无法高效做检索
- 合成否定描述的质量依赖 LLM 和检测器，某些复杂否定（双重否定、隐含否定）可能未被覆盖
- 未来方向：探索在预训练阶段（而非微调）就加入否定样本的效果；研究更优的训练目标替代简单的对比学习

## 相关工作与启发
- **vs NegCLIP**：NegCLIP 通过组合感知的负样本挖掘改善组合推理，但未专门针对否定。在 NegBench 上它的否定理解甚至劣化（HardNeg-Syn 下降 23%）
- **vs ConCLIP**：专门为否定理解设计，但训练数据过拟合到单一混合模板，导致所有否定嵌入坍缩到一个点。NegBench 的多样化评测暴露了这个问题
- **vs LLaVA**：指令微调的 VLM 在否定理解上明显优于 CLIP，但作为逐个处理图文对的模型，无法高效完成大规模检索任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个全面否定理解基准，诊断-方案范式清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 18种任务变体、多种模型、详尽的诊断和消融
- 写作质量: ⭐⭐⭐⭐⭐ 挑战-诊断-方案的叙事逻辑非常流畅清晰
- 价值: ⭐⭐⭐⭐ 揭示了VLM的重要盲区，对社区有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[ICML 2025\] Do Vision-Language Models Really Understand Visual Language?](../../ICML2025/multimodal_vlm/do_vision-language_models_really_understand_visual_language.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[CVPR 2025\] VisionZip: Longer is Better but Not Necessary in Vision Language Models](visionzip_longer_is_better_but_not_necessary_in_vision_language_models.md)

</div>

<!-- RELATED:END -->
