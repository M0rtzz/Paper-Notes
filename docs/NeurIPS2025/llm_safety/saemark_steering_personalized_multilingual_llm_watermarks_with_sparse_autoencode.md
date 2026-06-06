---
title: >-
  [论文解读] SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders
description: >-
  [NeurIPS 2025][LLM安全][LLM水印] 提出SAEMark框架，利用稀疏自编码器（SAE）提取文本的语义特征浓度评分，通过推理阶段的特征引导拒绝采样实现多比特水印嵌入，无需修改模型权重或logits，天然支持黑盒API、多语言和代码等场景，在英文/中文/代码上均达到领先的水印精度与文本质量。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "LLM水印"
  - "稀疏自编码器"
  - "多语言水印"
  - "黑盒水印"
  - "个性化归因"
---

# SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders

**会议**: NeurIPS 2025  
**arXiv**: [2508.08211](https://arxiv.org/abs/2508.08211)  
**代码**: [项目页面](https://zhuohaoyu.github.io/SAEMark)  
**领域**: AI安全  
**关键词**: LLM水印, 稀疏自编码器, 多语言水印, 黑盒水印, 个性化归因

## 一句话总结

提出SAEMark框架，利用稀疏自编码器（SAE）提取文本的语义特征浓度评分，通过推理阶段的特征引导拒绝采样实现多比特水印嵌入，无需修改模型权重或logits，天然支持黑盒API、多语言和代码等场景，在英文/中文/代码上均达到领先的水印精度与文本质量。

## 研究背景与动机

LLM生成的高质量文本给虚假信息、版权侵权和内容归因带来严峻挑战。水印技术——在生成文本中嵌入可检测签名——是有前景的解决方案，但现有方法存在根本性局限：

**白盒方法（KGW、EXP等）**：需要直接访问模型logits来操纵token概率分布，这在API服务（如ChatGPT、Claude）中不可用，且概率操纵会降低文本质量。

**领域/语言受限**：SWEET等方法仅适用于代码，SemStamp依赖英语特有的语法模式，跨语言和跨领域泛化能力差。

**多比特信息嵌入困难**：从"是否AI生成"的二分类升级到"哪个用户生成"的多比特归因（如编码用户ID），需要在相同文本长度内嵌入更多信息，现有方法（如Waterfall）在代码等低熵领域几乎失效。

**核心洞察**：不同的LLM生成文本在语义特征分布上存在自然差异，这些差异可以被利用——不修改生成过程，而是从多个自然生成的候选中选择特征模式与水印密钥对齐的那一个。这种"选择而非修改"的范式从根本上规避了上述所有限制。

## 方法详解

### 整体框架

SAEMark的工作流程：（1）将文本分割为领域相关的语义单元（自然语言用句子，代码用函数块）；（2）使用预训练SAE提取每个单元的特征浓度评分（FCS）；（3）用CDF归一化将FCS映射到[0,1]；（4）嵌入时，对每个位置生成N个候选并选择FCS最接近密钥导出目标值的那个；（5）检测时，分割文本、计算FCS序列、与候选密钥对齐验证。

### 关键设计

1. **特征浓度评分（Feature Concentration Score, FCS）**：利用SAE将LLM隐藏层激活分解为可解释的稀疏语义特征后，FCS衡量文本语义激活的"集中程度"。核心公式：

$$\text{FCS}(T) = \frac{\sum_{t=1}^n \sum_{i \in S} \phi_{t,i}}{\sum_{t=1}^n \|\phi_t\|_1}$$

其中 $S$ 包含所有token最高激活特征的去重索引集, $\phi_t = \text{SAE}_l(\mathbf{h}_t) \odot \mathbf{m}$ 是经过背景特征掩码 $\mathbf{m}$ 过滤后的稀疏特征向量。直觉是：连贯的文本倾向于集中激活一组相关的语义特征（如技术文档集中在正式语言和专业领域特征上），而不同生成会有不同的集中度，这一差异提供了天然的水印信号。

2. **基于拒绝采样的水印嵌入**：给定密钥 $k$，通过PRNG确定性地生成目标值序列 $\{\tau_i\}_{i=1}^M$。对每个文本单元位置，生成 $N$ 个候选并选择归一化FCS $z(u) = \hat{F}(s(\phi(u)))$ 最接近 $\tau_i$ 的候选。关键在于这是后处理选择（post-hoc selection），不修改LLM参数、logits或token——每个被选中的片段都是LLM的原生输出。理论保证：在高斯假设下，$N$ 个候选中至少一个落入目标 $k\tau$ 容忍范围的概率为：

$$\mathbb{P}(\exists j: |S_j - \tau| \leq k\tau) \geq 1 - (1 - p_{\min})^N$$

$N=50$ 时单单元成功率 >99%，$N=10$ 时仍达61%。

3. **CheckAlignment双重过滤检测**：检测阶段不仅做统计检验，还前置两个过滤器避免虚假匹配：（a）范围相似度过滤——要求观察序列和目标序列的动态范围比值在[0.95, 1.05]内；（b）重叠率过滤——至少95%的目标值落在观察序列范围内。仅通过双重过滤后才进行Student's t检验。这一设计有效补偿了理论分析中的独立性假设。

### 损失函数 / 训练策略

- SAEMark **无需训练**——SAE是预训练的可解释性工具，直接用于特征提取
- 背景特征掩码通过预计算排除标点、基础语法等高频非区分性特征
- FCS经验参数 $\mu = 0.142$, $\sigma = 0.029$，近似服从正态分布（经Q-Q plot验证）
- 水印应用在独立的"锚定"模型上运行SAE，与目标LLM解耦，保证API兼容性

## 实验关键数据

### 主实验（1% FPR下水印检测性能）

| 方法 | C4英文 F1↑ | LCSTS中文 F1↑ | MBPP代码 F1↑ | PandaLM质量↑ |
|------|-----------|-------------|-------------|-------------|
| KGW（白盒） | 99.2 | 99.1 | 41.5 | - |
| EXP（白盒） | 99.5 | 99.3 | 23.2 | - |
| SWEET（代码专用） | 99.6 | 0.0 | 62.4 | - |
| Waterfall（多比特） | 93.2 | 95.1 | 11.6 | - |
| **SAEMark（多比特）** | **99.7** | **99.2** | **66.3** | **67.6** |

### 文本质量（BIGGen-Bench 5分制）

| 模型 | 无水印 | SAEMark | KGW | Waterfall |
|------|-------|---------|-----|-----------|
| Qwen2.5-7B | 4.13 | 4.05 | 3.97 | 4.02 |
| Llama-3.2-3B | 3.69 | **3.85** | 3.56 | 3.62 |
| gemma-3-4b | 4.26 | 4.23 | 3.98 | 4.19 |

### 关键发现

- **跨领域泛化**：SAEMark在代码域比专用方法SWEET高3.9个F1点（66.3% vs 62.4%），在中文上也维持99.2%，展示了SAE特征的语言无关性
- **多比特扩展**：保持90%+准确率可编码10比特（1024用户），75%准确率扩展到13比特（8192用户），远超Waterfall
- **计算效率悖论**：理论需N=50候选才达99%+，但实际N=10就达98% F1。由于不需logit操纵，可充分利用TGI等优化推理引擎，端到端延迟仅为KGW的1/3.24
- **对抗鲁棒性**：对词删除、同义词替换和上下文替换攻击展现出强韧性，因为SAE特征捕捉的是语义层面的模式而非表面token
- **背景特征掩码至关重要**：消融显示移除掩码使AUC从~1.0暴跌至0.85

## 亮点与洞察

- **范式转变**："选择而非修改"的水印范式彻底绕开了logit操纵的所有限制，每一段被选中的文本都是LLM的原生输出，质量下界即为LLM自身能力
- **可解释性工具的创新应用**：SAE原本用于理解模型内部表示，这里被巧妙转用于内容归因，展示了可解释性研究与应用研究之间的桥梁
- **理论-实践闭环**：先给出与特征提取器无关的通用理论框架，再用SAE作为具体实例化，最后通过工程优化（CheckAlignment、背景掩码）弥合理论与实践的差距

## 局限与展望

- 多候选生成增加了推理成本（$N$ 倍），即使优化后仍高于普通生成
- 水印安全性假设密钥保密，未声称密钥已知时的密码学不可伪造性
- 短文本（少数文本单元）的水印嵌入和检测可靠性下降
- 当前仅在句子/函数块级别操作，更细粒度（段落内）的信号嵌入值得探索
- 对抗攻击评估主要在中等强度，超过50%的攻击强度下鲁棒性待验证

## 相关工作与启发

- KGW/EXP是token级白盒水印的代表，SAEMark通过后处理选择完全规避了白盒限制
- Postmark等黑盒方法依赖表面统计或辅助模型，缺乏SAEMark的多比特能力和跨语言泛化
- 稀疏自编码器（Anthropic/OpenAI的monosemanticity研究）为SAEMark的多语言特征激活提供了理论基础
- 本工作启示：可解释性工具不仅用于理解模型，还可服务于安全目标（归因、审计）

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 基于特征引导的拒绝采样水印范式全新，SAE在水印领域的应用是首创
- **实验充分度**: ⭐⭐⭐⭐ 4个数据集跨英中代码、3个骨干LLM、多比特扩展和对抗测试全面，但长文本和极端攻击测试可加强
- **写作质量**: ⭐⭐⭐⭐⭐ 通用框架→理论分析→具体实例化→工程优化的逻辑链条非常清晰
- **价值**: ⭐⭐⭐⭐⭐ 为API限定的LLM水印提供了首个真正实用、多语言、多比特的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/llm_safety/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[ICML 2025\] SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](../../ICML2025/llm_safety/saebench_a_comprehensive_benchmark_for_sparse_autoencoders_in_language_model_int.md)
- [\[ACL 2026\] CRISP: Persistent Concept Unlearning via Sparse Autoencoders](../../ACL2026/llm_safety/crisp_persistent_concept_unlearning_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] On Optimal Steering to Achieve Exact Fairness](on_optimal_steering_to_achieve_exact_fairness.md)
- [\[ACL 2025\] Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](../../ACL2025/llm_safety/llm_watermark_distillation_robustness.md)

</div>

<!-- RELATED:END -->
