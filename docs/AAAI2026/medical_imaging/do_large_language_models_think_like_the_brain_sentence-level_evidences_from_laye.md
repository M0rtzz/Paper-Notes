---
title: >-
  [论文解读] Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI
description: >-
  [AAAI 2026][医学图像][LLM与大脑对齐] 本文通过对比14个公开LLM的逐层表示与人类被试听自然叙事时的fMRI数据，在句子级别系统地研究了LLM与人脑语言处理的对齐程度，发现中间层最对齐、指令微调显著增强对齐、且存在与经典神经语言学理论一致的半球偏侧化模式。
tags:
  - AAAI 2026
  - 医学图像
  - LLM与大脑对齐
  - fMRI
  - 句子级语义理解
  - 层级表示
  - 半球偏侧化
---

# Do Large Language Models Think Like the Brain? Sentence-Level Evidences from Layer-Wise Embeddings and fMRI

**会议**: AAAI 2026  
**arXiv**: [2505.22563](https://arxiv.org/abs/2505.22563)  
**代码**: https://github.com/Lucasuuu02/LLM4Brain (有)  
**领域**: Cognitive Neuroscience / NLP  
**关键词**: LLM与大脑对齐, fMRI, 句子级语义理解, 层级表示, 半球偏侧化

## 一句话总结
本文通过对比14个公开LLM的逐层表示与人类被试听自然叙事时的fMRI数据，在句子级别系统地研究了LLM与人脑语言处理的对齐程度，发现中间层最对齐、指令微调显著增强对齐、且存在与经典神经语言学理论一致的半球偏侧化模式。

## 研究背景与动机
AI与神经科学的交叉领域一直关注LLM学习到的表示是否与人脑语言处理存在对应关系。先前研究已发现LLM表示与神经响应存在相关性，但这些观察缺乏机制解释：**这种相似性仅仅源于模型规模的增大，还是反映了与人脑语言处理更深层的计算原理收敛？**

现有研究的局限在于：(1) 多使用公开数据集，可能无法准确反映模型在特定任务中的理解能力；(2) 过度关注模型规模而忽视语义理解能力本身；(3) 词级分析为主，句子级的系统性研究不足。

本文的切入角度是同时让LLM和人类接受相同的自然叙事刺激（《小王子》），设计跨语言语义对齐测试评估LLM理解能力，并构建句子级神经预测模型来量化模型层级表示与脑区激活的对应关系。核心insight是**语义理解能力（而非单纯的参数规模）才是驱动LLM与人脑对齐的关键因素**。

## 方法详解

### 整体框架
多阶段pipeline：(1) fMRI数据采集与预处理→GLM估计句子级神经响应→ROI提取；(2) LLM逐层句子embedding提取；(3) Ridge回归建模与相关性分析。对14个LLM的每一层embedding都与12个语言相关脑区的fMRI激活进行交叉验证回归。

### 关键设计
1. **跨语言语义对齐测试(CSAA)**:

    - 功能：评估LLM理解连续文本的能力
    - 核心思路：对每个中文句子，生成5个英文选项（正确翻译、乱序、词性替换、句式变换、信息增删），让LLM通过embedding余弦相似度选出正确翻译
    - 设计动机：比简单的benchmark更能反映模型的上下文语义理解能力，且可量化不同模型间的能力差异

2. **GLM + LS-S句子级神经激活估计**:

    - 功能：从fMRI时间序列中精确提取每个句子对应的BOLD信号
    - 核心思路：使用Least-Squares Separate (LS-S)方法，将每个句子作为独立回归子建模，通过精确的onset时间和HRF卷积解决句子边界与TR采样的时间不匹配问题
    - 设计动机：相比传统的条件平均或block设计，LS-S方法能更精确地隔离每个句子的暂态神经响应，适用于自然叙事范式

3. **层级Ridge回归与相关性分析**:

    - 功能：量化LLM各层表示对脑区fMRI信号的预测能力
    - 核心思路：对每个ROI，用每个LLM层的sentence embedding通过Ridge回归预测fMRI信号，K-fold交叉验证下计算Pearson相关性。并行化处理被试×ROI×层的所有组合
    - 设计动机：Ridge regression的正则化控制过拟合，嵌套交叉验证选择最优α，z-score标准化确保不同模态数据的兼容性

### 损失函数 / 训练策略
本文不训练模型，使用14个现有预训练LLM（包括Llama-3.1、Gemma-2、Baichuan2、DeepSeek、GLM-4、Qwen2.5等系列，含base和instruct版本），参数范围6.7B-9B。分析框架使用标准Ridge回归，关键超参数α通过网格搜索和嵌套交叉验证选定。

## 实验关键数据

### 主实验（CSAA语义理解能力）
| 模型 | CSAA得分 |
|------|---------|
| Llama-3.1-8B-Instruct | 31.4 |
| Gemma-2-9b-it | 30.7 |
| Gemma-2-9b | 30.3 |
| Baichuan2-7B-Chat | 22.7 |
| DeepSeek-7B-Chat | 19.5 |
| glm-4-9b | 7.9 |
| Qwen2.5-7B | 6.4 |

Instruct版本始终优于对应Base版本，模型间差距超过28分。

### 脑-模型对齐分析
| 对比维度 | 关键发现 | 统计显著性 |
|---------|---------|-----------|
| 中间层 vs 最终层 | 所有模型中间层预测脑活动最优 | 一致 |
| Instruct vs Base | Instruct版提升脑对齐 | 排列检验 p=0.03125 |
| 理解能力 vs 对齐 | Pearson r=0.601, 正相关 | p=0.030 |
| 左IFG偏侧化 | 左半球显著优势 | p=0.025 |
| 右AntTemp偏侧化 | 右半球显著优势 | p=0.001 |

### 关键发现
- **中间层最佳对齐**：所有14个LLM的中间层（而非最终层）与脑活动的相关性最高，与此前EEG/MEG研究一致，但本文首次在fMRI上验证
- **指令微调增强对齐**：5对base/instruct模型的对比显示，指令微调不仅提升性能，还使模型表示更接近人脑活动模式
- **理解能力是关键驱动力**：在6.7B-9B参数范围内，语义理解能力（CSAA得分）比模型规模更能预测与人脑的对齐程度
- **半球偏侧化**：左侧IFG和后颞叶（核心语言区）显示左半球优势，而右侧MFG和前颞叶显示右半球优势，与经典语言偏侧化假说一致。IFG和MFG的偏侧化程度还与模型性能正相关

## 亮点与洞察
- 首次提出"语义理解能力(CSAA)驱动脑对齐"的观点，将解释焦点从规模转向能力
- 提出了一种令人兴奋的可能性：未来是否应让LLM训练过程显式对齐脑数据以提升认知可信度？
- 半球偏侧化分析将宏观认知神经科学的经典理论与LLM的计算特性联系起来
- fMRI的全脑空间覆盖弥补了此前EEG/MEG研究的空间分辨率不足

## 局限与展望
- 仅在一个自然叙事语料（《小王子》）上实验，结果的泛化性有限，不同text genre可能得到不同结论
- 被试数量34人，统计功效需更大样本验证，个体差异的影响未充分分析
- 所有LLM参数范围集中在6.7B-9B，无法判断scaling law在更大/更小模型上是否成立
- 相关性分析不等于因果关系，LLM与脑的对齐不意味着它们使用相同的计算机制，可能只是统计巧合
- 跨语言任务设计（中→英）可能引入额外的翻译偏差，单语测试会更纯净
- fMRI的时间分辨率较低（TR=2s），无法捕捉快速的语言处理动态
- 未考虑开源模型以外的闭源模型（如GPT-4、Claude）的表现

## 相关工作与启发
- 延续了Schrimpf et al.和Caucheteux et al.关于LLM与脑活动对齐的研究传统，但从词级扩展到句子级，提供了更高层次的语义对应证据
- 方法学上的贡献在于：将多个LLM的CSAA能力得分与脑对齐度联合分析，发现能力-对齐的正相关，改变了"越大越好"的简单叙事
- 对认知架构设计有启发意义：如果更好的语义理解确实导致更脑化的表示，那么反过来利用脑数据指导模型训练可能是可行的研究方向
- 半球偏侧化的发现提示，混合神经-符号LLM可能需要考虑类脑的分布式和偏侧化处理模式
- 与Tuckute et al. (2024)的encoding model工作互补：后者关注特定模型优化，本文关注跨模型比较

## 评分
- 新颖性: ⭐⭐⭐⭐ (句子级LLM-脑对齐+CSAA能力指标是新颖的，但整体方法框架较常规)
- 实验充分度: ⭐⭐⭐⭐⭐ (14个LLM、12个ROI、三组实验、完备的统计检验)
- 写作质量: ⭐⭐⭐⭐ (方法描述详尽，公式清晰，图表信息量大)
- 价值: ⭐⭐⭐⭐ (为理解LLM与人脑的关系提供了新证据和新视角)

<!-- RELATED:START -->

## 相关论文

- [Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/medical_imaging/tracing_pharmacological_knowledge_in_large_language_models.md)
- [Unleashing the Potential of Large Language Models for Text-to-Image Generation through Autoregressive Representation Alignment](unleashing_the_potential_of_large_language_models_for_text-to-image_generation_t.md)
- [CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)
- [Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)
- [Brain-IT: Image Reconstruction from fMRI via Brain-Interaction Transformer](../../ICLR2026/medical_imaging/brain-it_image_reconstruction_from_fmri_via_brain-interaction_transformer.md)

<!-- RELATED:END -->
