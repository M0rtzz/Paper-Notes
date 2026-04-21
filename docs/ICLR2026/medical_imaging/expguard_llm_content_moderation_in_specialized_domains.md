---
title: >-
  [论文解读] ExpGuard: LLM Content Moderation in Specialized Domains
description: >-
  [ICLR2026][医学图像][LLM safety] 提出面向金融、医疗、法律等专业领域的安全护栏模型 ExpGuard 及配套数据集 ExpGuardMix（58,928 样本），在领域特定测试集上 prompt 分类 F1 超 WildGuard 8.9%、response 分类超 15.3%，同时在通用安全基准上保持 SOTA 水平。
tags:
  - ICLR2026
  - 医学图像
  - LLM safety
  - guardrail model
  - content moderation
  - domain-specific
  - financial/medical/legal
---

# ExpGuard: LLM Content Moderation in Specialized Domains

**会议**: ICLR2026  
**arXiv**: [2603.02588](https://arxiv.org/abs/2603.02588)  
**代码**: [brightjade/ExpGuard](https://github.com/brightjade/ExpGuard)  
**领域**: medical_imaging  
**关键词**: LLM safety, guardrail model, content moderation, domain-specific, financial/medical/legal

## 一句话总结
提出面向金融、医疗、法律等专业领域的安全护栏模型 ExpGuard 及配套数据集 ExpGuardMix（58,928 样本），在领域特定测试集上 prompt 分类 F1 超 WildGuard 8.9%、response 分类超 15.3%，同时在通用安全基准上保持 SOTA 水平。

## 背景与动机
随着 LLM 在金融、医疗、法律等高风险专业领域的部署不断推进，现有安全护栏模型面临严峻挑战：

- **通用护栏的盲区**：现有 guardrail（如 Llama-Guard、WildGuard）主要面向通用人机交互场景，缺乏对专业术语和领域概念的理解。例如金融术语"haircut"（资产估值折扣）被用于构造的恶意 prompt 可以轻松绕过通用护栏的检测。
- **API 工具近乎失效**：Detoxify、Perspective API、OpenAI Moderation 等在专业领域测试集上 F1 仅 0.3%-14.1%，几乎完全无法识别领域特定的有害内容。
- **内部对齐的局限**：RLHF 等内部对齐技术资源消耗大，且难以覆盖领域特定风险，外部护栏模型作为补充层有其必要性。

## 核心问题
如何构建一个既能处理通用安全检测、又能有效识别金融/医疗/法律等专业领域中利用技术术语伪装的有害内容的安全护栏模型？

## 方法详解

### 1. ExpGuardMix 数据集构建（58,928 样本）

整个数据集分为 ExpGuardTrain（56,653 样本用于训练）和 ExpGuardTest（2,275 样本用于评测）。

**阶段一：领域术语挖掘**

- 从 Wikipedia 递归爬取金融、医疗、法律类目页面提取术语
- 使用 Wikidata API 过滤非技术实体（人名、组织、国家等）
- GPT-4o 排除非敏感/无关术语
- 人工验证：3 名标注者多数投票，最终保留 2,646 个术语（金融 989、医疗 1,012、法律 645）

**阶段二：Prompt 和 Response 生成**

- **有害 prompt**：对每个术语，用 GPT-4o 生成针对该术语风险场景的有害 prompt，通过添加"I have an idea for a prompt:"前缀绕过安全机制。生成长短两种变体，随机采样 100+ 预设指令模板，加 few-shot 示例
- **良性 prompt**：将 Wikipedia 文档转成指令-回复对，仅保留指令部分。虽涉及敏感话题但本质安全，用于缓解模型的过度安全行为
- **野外数据**：从 LMSYS-Chat-1M、WildChat 子采样，加入 DAN jailbreak prompt 及 HH-RLHF、Aegis 2.0 人写数据
- **回复生成**：用 Mistral-7B-Instruct-v0.1 生成 compliant response（该旧模型更易服从有害请求），用 Gemma-3-27B-IT 生成 refusal response

**阶段三：分类标注与过滤**

- 定义 13 类有害类别 + 1 类"无害"伪类别，涵盖暴力、色情、歧视、隐私侵犯、金融欺诈、非法药物等
- 使用 Claude 3.7 Sonnet + Gemini 2.0 Flash + Qwen2.5-Max 三模型集成标注，要求生成 CoT 推理后给出类别
- **严格共识过滤**：要求至少 2/3 模型给出完全相同的类别索引（非仅"安全/不安全"），4.8% 模糊样本被丢弃
- Sentence-BERT 余弦相似度 > 0.9 的近重复样本去重

### 2. ExpGuardTest（2,275 样本）

- 分布：金融 964、医疗 771、法律 540
- 初始由 LLM 集成标注，再由领域专家验证
- 金融部分由银行业从业者审核，Cohen's Kappa 达 0.89（prompt）/ 0.98（response），表明"几乎完美一致"

### 3. ExpGuard 模型训练

- 基于 7B 参数 LLM 微调，用 ExpGuardTrain 进行多任务训练
- 输入仅 prompt 时预测 prompt 有害性；输入 prompt-response 对时同时预测两者有害性
- 输出二分类标签（safe/unsafe）

## 实验关键数据

### ExpGuardTest 上的主要结果（F1%）

| 模型 | Prompt 总 F1 | Response 总 F1 |
|------|-------------|----------------|
| Detoxify / Perspective / OpenAI Mod | 0.3-0.5 | 0.6 |
| Azure | 14.1 | 2.6 |
| Llama-Guard3 (8B) | 71.1 | 84.2 |
| Aegis-Guard-D (7B) | 82.9 | 87.2 |
| WildGuard (7B) | 84.4 | 77.4 |
| **ExpGuard (7B)** | **93.3** | **92.7** |

- Prompt 分类超 WildGuard **+8.9%**，Response 分类超 **+15.3%**
- 金融/医疗/法律三个子领域均领先

### 公开安全基准上的结果（8 个 benchmark 平均 F1%）

| 模型 | Prompt 平均 | Response 平均 |
|------|-----------|--------------|
| WildGuard | 84.2 | 78.8 |
| **ExpGuard** | **85.7** | 78.5 |

- 在通用基准上与 SOTA 持平甚至略优，未因领域特化而牺牲通用性

### 消融实验

- 移除领域特定数据：ExpGuardTest prompt F1 从 93.3% 降至 85.3%（-8.0%）
- 移除野外数据：公开 benchmark prompt F1 从 85.7% 降至 84.1%
- 移除人写数据：公开 benchmark response F1 从 78.5% 降至 73.9%（影响最大）

### Jailbreak 鲁棒性

- 在标准 jailbreak 攻击（CipherChat、AutoDAN-Turbo、FlipAttack、GASP）下保持竞争力
- ExpGuard+ 变体（额外加入 270 条领域特定对抗样本）在领域 jailbreak 上显著超越所有基线

## 亮点

1. **首个面向专业领域的安全护栏数据集和模型**：填补了金融/医疗/法律领域 LLM 内容审核的空白
2. **数据构建流程可复用**：基于 Wikipedia 术语挖掘 + LLM 生成 + 三模型集成标注 + 专家验证的 pipeline 可扩展到其他领域
3. **严格的质量控制**：三模型精确类别共识（非仅二分类共识）+ 领域专家金融子集验证（Kappa 0.89/0.98）
4. **领域特化 + 通用不退化**：ExpGuardTest 上大幅领先的同时，8 个公开 benchmark 上保持/超越 SOTA
5. **揭示 API 工具的严重不足**：量化展示主流 API 在专业场景几乎完全失效

## 局限与展望

- **领域覆盖有限**：仅覆盖金融/医疗/法律三个领域，其他专业领域（如网络安全、化工等）有待扩展
- **仅支持英语**：多语言领域审核是重要的未来方向
- **合成数据局限**：尽管做了多种增强，合成数据可能无法完全反映真实用户交互的多样性
- **动态更新需求**：有害内容和对抗手段快速演进，数据集需持续更新
- **领域专家验证不完全**：仅金融子集经过专家审核，医疗和法律子集依赖 LLM 集成标注的可靠性推断

## 与相关工作的对比

| 维度 | WildGuard | Llama-Guard 系列 | ExpGuard |
|------|-----------|----------------|----------|
| 领域覆盖 | 通用 | 通用 | 通用 + 金融/医疗/法律 |
| 训练数据 | WildGuardMix (92K) | 内部安全数据 | ExpGuardMix (58.9K) |
| 领域特定 F1 | 84.4 / 77.4 | 71.1 / 84.2 | **93.3 / 92.7** |
| 通用 benchmark | **84.2** / 78.8 | 78.9 / 66.8 | 85.7 / 78.5 |
| 数据构建 | LLM 生成 + 野外 | 未公开 | 术语挖掘 + RAG 生成 + 专家验证 |

与 An et al. (2024)、Cui et al. (2025) 等"生成-过滤"流程的关键区别：前者关注减少 false positive（过度拒绝），本文关注减少 false negative（遗漏有害内容），并引入领域专家验证。

## 启发与关联

- **领域安全护栏的方法论范式**：术语挖掘→RAG 生成→多模型集成标注→专家验证的 pipeline 具有很好的可迁移性，可用于构建网络安全、生物化学等领域的安全数据集
- **模型审核 vs. API 审核**：实验有力证明了开源 LLM 护栏模型相比商业 API 在专业场景的必要性
- **与 RLHF 的互补关系**：ExpGuard 作为外部审核层，与内部对齐形成双保险架构，值得在工业部署中推广

## 评分
- 新颖性: 8/10 — 首次系统性地解决专业领域 LLM 安全护栏问题，数据构建思路有创新
- 实验充分度: 9/10 — 13 个基线、9 个 benchmark、消融实验和 jailbreak 分析都很完整
- 写作质量: 8/10 — 结构清晰，pipeline 描述详尽，图表丰富
- 价值: 8/10 — 填补了重要空白，但领域和语言覆盖仍有限

<!-- RELATED:START -->

## 相关论文

- [MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection](../../CVPR2026/medical_imaging/moeclip_patch-specialized_experts_for_zero-shot_anomaly_detection.md)
- [Scaling with Collapse: Efficient and Predictable Training of LLM Families](scaling_with_collapse_efficient_and_predictable_training_of_llm_families.md)
- [AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)
- [Neural Bandit Based Optimal LLM Selection for a Pipeline of Tasks](../../AAAI2026/medical_imaging/neural_bandit_based_optimal_llm_selection_for_a_pipeline_of_tasks.md)
- [Pancakes: Consistent Multi-Protocol Image Segmentation Across Biomedical Domains](../../NeurIPS2025/medical_imaging/pancakes_consistent_multi-protocol_image_segmentation_across_biomedical_domains.md)

<!-- RELATED:END -->
