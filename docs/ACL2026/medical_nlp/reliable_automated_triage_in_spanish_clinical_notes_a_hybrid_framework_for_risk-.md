---
title: >-
  [论文解读] Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification
description: >-
  [ACL2026][医疗NLP][临床 NLP] 本文面向西班牙临床笔记中的早期 HIV 疑似识别，提出 MCP 处理 aleatoric uncertainty、MCMD 几何 veto 处理 epistemic uncertainty 的双验证选择性分诊框架…
tags:
  - "ACL2026"
  - "医疗NLP"
  - "临床 NLP"
  - "选择性分类"
  - "不确定性量化"
  - "HIV 疑似识别"
  - "西班牙电子病历"
---

# Reliable Automated Triage in Spanish Clinical Notes: A Hybrid Framework for Risk-Aware HIV Suspicion Identification

**会议**: ACL2026  
**arXiv**: [2605.21256](https://arxiv.org/abs/2605.21256)  
**代码**: https://github.com/romorale/mil_uq_public  
**领域**: 临床 NLP / 医疗文本分诊  
**关键词**: 临床 NLP, 选择性分类, 不确定性量化, HIV 疑似识别, 西班牙电子病历  

## 一句话总结
本文面向西班牙临床笔记中的早期 HIV 疑似识别，提出 MCP 处理 aleatoric uncertainty、MCMD 几何 veto 处理 epistemic uncertainty 的双验证选择性分诊框架，在严格安全约束下自动处理 67.7% 病例并达到 0.982 Clear $F_2$。

## 研究背景与动机
**领域现状**：医疗文本 NLP 常把电子病历分类任务当作确定性二分类问题，报告 AUROC、F1 或 $F_2$ 等整体指标。对于 HIV 早期疑似识别，NLP 系统可以帮助筛出明显阴性病例，把临床资源留给更需要医生判断的患者。

**现有痛点**：临床疑似并不等于最终血清学诊断，而是由医生记录、检测行为和病历语境共同决定的部分可观察构念。若模型被迫对所有病例给出二分类，它会在模糊叙述、分布外病例和少数类样本上产生过度自信的错误，这在临床分诊中比普通 benchmark 掉点严重得多。

**核心矛盾**：临床自动化需要高覆盖率来降低人工负担，但医学安全又要求系统知道何时拒答。标准 UQ 方法往往只看 predictive entropy，把文本本身的歧义性和模型没见过的分布外风险混在一起，导致要么大量 defer 真阳性，要么危险地自动化异常病例。

**本文目标**：作者希望把强制分类改成带拒识的 trinary triage：Clear Negative、Clear Positive 或 Defer。系统不追求对所有样本自动判定，而是要在可自动处理的子域内保证足够可靠，同时把高风险样本交回医生。

**切入角度**：论文把不确定性拆成两类：MCP 负责概率层面的 aleatoric ambiguity，MCMD 负责 latent space 中的 epistemic anomaly。只有同时通过两个检查的病例，才允许自动输出。

**核心 idea**：让临床叙事必须同时通过 conformal prediction 的概率安全边界和 Mahalanobis 距离的几何分布内检查，用双 veto 把不可靠二分类预测转化为风险可控的选择性分诊。

## 方法详解
这篇论文的核心不是提出一个更大的文本编码器，而是把一个临床文本分类器放进可拒识的安全壳里。模型先用 Spanish biomedical RoBERTa 和 MIL 处理长病历，再用专门的不确定性后处理决定哪些病例可以自动化。

### 整体框架
输入是一名患者的多条西班牙临床笔记，任务是判断是否存在 HIV 临床疑似。系统先把病历切成最多 64 个、每个 384 token 的重叠 chunks，用 PlanTL-GOB-ES/bsc-bio-ehr-es 编码；随后 gated attention MIL 聚合成患者级表示。分类头给出二分类预测，后处理模块再判断是否应该 defer。

训练与评估使用 HUFA 医院的单中心匿名 EHR 队列，共 13,642 名患者、63,802 条临床笔记。疑似队列由 HIV 血清学检测请求与 EuroTEST indicator conditions 过滤得到，非疑似队列为无相关检测历史患者。作者还主动移除 HIV 缩写和直接检测建议，减少显式泄漏。

最终决策逻辑是严格交集：只有当 MCP 的预测集合大小为 1，且样本到预测类别局部 centroid 的 Mahalanobis 距离不超过类别阈值 $\tau_{dist}(\hat{y})$ 时，系统输出 $\hat{y}$；否则输出 defer。

### 关键设计
**1. Mondrian Conformal Prediction 管 aleatoric uncertainty：判断文本证据本身够不够明确。**

临床叙事常常不完整或措辞含糊，这种歧义不是模型换个分类阈值就能解决的——硬判只会在模糊病例上产生过度自信的错误。MCP 先对验证集 logits 做 temperature scaling 压低过度自信，再以风险容忍度 $\alpha$ 构造预测集合 $\Gamma^{\alpha}(x)$。如果这个集合恰好只含一个类别，说明证据足够明确；一旦集合包含多个类别（证据冲突）或为空（证据不足），就触发 defer。换句话说，conformal prediction 把"概率层面的模糊"显式量化成集合大小，让医院管理者能直接读到当前的风险水平，而不是被一个看似自信的 softmax 蒙蔽。

**2. Multi-Centroid Mahalanobis Distance 管 epistemic uncertainty：拦住 softmax 很自信但表示异常的分布外病例。**

光看概率还不够——有些病例 softmax 给得很笃定，但它的 latent representation 其实落在训练分布之外，模型是在"自信地外推"。MCMD 在 $L_2$ 归一化的 latent space 里，给每个类别用 k-means 自动选出多个局部 centroids（最小簇数 $K$ 由 inertia gain 低于 0.05 的拐点确定），样本的异常度取它到预测类别任一 centroid 的最小 Mahalanobis 距离

$$d_M(x,\hat{y})=\min_k\sqrt{r_{\hat{y},k}^\top\Sigma^{-1}r_{\hat{y},k}}$$

其中全局 precision matrix $\Sigma^{-1}$ 用 OAS shrinkage 从 residuals 中稳健估计。之所以用 multi-centroid 而非单一高斯，是因为 HIV 疑似病例表型高度异质，单中心会把少数类的方差估得过大、几何 veto 失灵；多中心既保留了表型多样性，又避免了高维下局部协方差估计不稳定的问题。

**3. 风险非对称的临床评价与可调 operational dial：用符合早期筛查代价结构的指标来衡量系统。**

普通 F1 把各类错误一视同仁，但早期 HIV 漏诊会延迟 ART 并增加传播风险，代价远高于一次不必要的检测——评价函数不反映这种不对称就会误导部署决策。作者改用 $F_2$ 强化对 false negative 的惩罚，用 ECE 衡量校准，用 coverage/TPDR/AURC 衡量分诊效率，还设计了 Custom Risk-Kappa，把自动化 false negative 的惩罚定为 1.0，false positive 和 true positive deferral 定为 0.5，true negative deferral 定为 0.25。与此配套，风险容忍度 $\alpha$ 被设计成一个可调旋钮：调小则更谨慎、覆盖率低但 Clear $F_2$ 高，调大则覆盖率上升、把更多病例交给自动化，医院可以按自身风险偏好在这条曲线上选工作点。

### 损失函数 / 训练策略
编码器侧比较两种 MIL 架构。Standard MIL 使用 Label-Smoothed Focal Loss 处理类别不平衡，并加 R-Drop 约束两次 forward 的 KL 一致性。MD-SN MIL 使用 Mahalanobis Distance + Spectral Normalization 结构，并配合 Logit Adjustment，把类别先验加到 logits 中，以避免 focal loss 扭曲概率校准。SN 施加在 MIL head 的 dense layers 上，以获得更平滑、适合几何 veto 的 feature space。

UQ backend 包括 MC Dropout、10-fold CV Deep Ensembles 和 deterministic MD-SN。阈值拟合采用 out-of-fold calibration，避免直接在训练样本上调阈值造成乐观偏差。

## 实验关键数据

### 主实验
主结果在严格风险容忍度 $\alpha=0.01$ 下评估 forced binary 与 selective screening。最关键结论是：强制二分类下分数看起来不错，但一旦引入安全约束，只有双 veto 能隔离出可临床自动化的可靠子域。

| 架构 / UQ | Binary $F_2$ | Clear $F_2$ | Binary ECE | Clear ECE | Coverage | TPDR | AURC | Risk-Kappa |
|-----------|--------------|-------------|------------|-----------|----------|------|------|------------|
| Encoder baseline + CV Ensemble | 0.769 | 0.973 | 0.077 | 0.031 | 47.8% | 50.5% | 未完整读到 | 未完整读到 |
| Standard MIL + MD-SN | 0.813 | 0.966 | 0.028 | 0.026 | 70.7% | 31.9% | 0.012 | 0.601 |
| MD-SN MIL + CV Ensemble | 0.821 | 0.982 | 0.022 | 0.021 | 67.7% | 33.2% | 0.006 | 0.576 |

### 消融实验
消融主要验证双 veto 是否必要，以及 $\alpha$ 是否能作为运营旋钮使用。

| 配置 | Coverage | Clear $F_2$ | Risk-Kappa | 说明 |
|------|----------|-------------|------------|------|
| Aleatoric Only / MCP | 92.6% | 0.902 | 0.751 | 能发现概率模糊，但看不到几何异常 |
| Epistemic Only / MCMD | 98.8% | 0.831 | 0.788 | 覆盖率高但安全性明显下降 |
| Standard Uncertainty | 99.3% | 0.824 | 0.788 | 传统 entropy 阈值造成虚高覆盖率 |
| Dual Veto / Hybrid | 91.3% | 0.913 | 0.755 | 比标准不确定性 Clear $F_2$ 高 0.089，且 CI 不重叠 |

| 模型 | $\alpha$ | Coverage | TPDR | Clear $F_2$ | Clear ECE | Risk-Kappa |
|------|----------|----------|------|-------------|-----------|------------|
| Standard MIL | 0.01 | 63.5% | 29.6% | 0.979 | 0.009 | 0.546 |
| Standard MIL | 0.02 | 75.4% | 19.4% | 0.954 | 0.012 | 0.644 |
| Standard MIL | 0.05 | 91.1% | 7.0% | 0.902 | 0.015 | 0.747 |
| MD-SN MIL | 0.01 | 67.7% | 33.2% | 0.982 | 0.021 | 0.576 |
| MD-SN MIL | 0.02 | 77.4% | 24.8% | 0.969 | 0.028 | 0.665 |
| MD-SN MIL | 0.05 | 91.3% | 10.2% | 0.913 | 0.029 | 0.755 |

### 关键发现
- Forced binary evaluation 会掩盖风险。例如 MD-SN 架构配 MC Dropout 的 Binary $F_2$ 最高达到 0.839，但校准 ECE 退化到 0.044，不能说明它适合临床自动化。
- 双 veto 的收益主要体现在安全性而非 Risk-Kappa 的显著优势。论文明确说 Risk-Kappa 的置信区间重叠较多，但 Clear $F_2$ 的提升相对标准不确定性是统计稳健的。
- $\alpha$ 可以当作医院运营旋钮。MD-SN MIL 从 $\alpha=0.01$ 放宽到 0.05 后，coverage 从 67.7% 升到 91.3%，Clear $F_2$ 仍保持 0.913，TPDR 降到 10.2%。
- 特征归因显示系统的错误并不完全是模型乱判：automated false positives 往往被 HCV、tuberculosis 等 EuroTEST guideline-concordant indicator conditions 驱动，反映的是 EHR 标注/检测行为的部分可观察性。

## 亮点与洞察
- 论文最大的价值在于把 clinical NLP 从“分类正确率”转向“可自动化子域”。这比单纯堆模型更接近真实部署问题。
- Aleatoric/epistemic 解耦非常实用：MCP 处理叙事模糊，MCMD 处理分布外异常，二者对应的临床失败模式不同，因此需要交集而不是互相替代。
- Custom Risk-Kappa 虽然是任务特定指标，但它迫使读者认真面对医学错误代价不对称。类似设计可以迁移到药物警戒、急诊分诊、手术风险预筛等任务，只需重新定义 penalty matrix。
- 论文没有追逐 generative LLM，而是选择 deterministic spectrally normalized encoder，是一个很务实的部署选择：在高风险医疗场景中，可校准、可拒识、可解释往往比生成能力更重要。

## 局限与展望
- Custom Risk-Kappa 强依赖早期传染病筛查假设。若迁移到肿瘤活检等高物理干预风险任务，false positive 的代价会显著变化，必须重新做健康经济学验证。
- MCMD 的静态几何阈值可能受时间漂移或人口学漂移影响。若医院科室、记录风格或患者群体改变，$\tau_{dist}$ 需要重新校准。
- 论文没有把 generative LLM 纳入核心实验。作者认为自回归生成模型可靠不确定性仍未解决，因此选择 encoder，但这也限制了结论对当前 LLM 医疗助手的直接适用性。
- 实证数据来自单一医疗中心 HUFA。HIV 疑似高度依赖临床书写习惯和检测阈值，跨机构验证是部署前必须补上的步骤。

## 相关工作与启发
- **vs 标准 biomedical NLP 分类器**: 标准分类器优化 forced binary 指标，本文关注 selective screening 后的 Clear $F_2$、coverage 和 TPDR，更贴近临床工作流。
- **vs MC Dropout / Deep Ensembles**: MC Dropout 计算轻但校准不足，Deep Ensembles 安全性强但有 10 倍计算/存储负担；MD-SN 试图用单次前向获得接近 ensemble 的几何不确定性。
- **vs 传统 selective classification**: 传统做法常用 entropy 阈值，本文显示这种标准不确定性会自动化 99.3% 样本但 Clear $F_2$ 只有 0.824，说明单一阈值无法覆盖临床复杂风险。
- **vs 医疗 LLM 分诊**: 本文没有生成解释，而是强调可拒识与可校准；对医疗 LLM 的启发是，先证明系统何时不该回答，可能比扩大回答能力更关键。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 双 veto 组件本身来自成熟方法，但把 MCP 与 multi-centroid Mahalanobis veto 组合成临床风险分诊框架很扎实。
- 实验充分度: ⭐⭐⭐⭐☆ 指标、消融、置信区间和阈值分析完整，短板是单中心数据且没有跨机构验证。
- 写作质量: ⭐⭐⭐⭐☆ 问题定义和临床动机清楚，表格信息密集但总体叙事顺畅。
- 价值: ⭐⭐⭐⭐⭐ 对高风险医疗 NLP 部署非常有参考价值，尤其是“不要强迫模型回答所有病例”的系统设计思想。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[ACL 2026\] Anonpsy: A Graph-Based Framework for Structure-Preserving De-identification of Psychiatric Narratives](anonpsy_a_graph-based_framework_for_structure-preserving_de-identification_of_ps.md)
- [\[ACL 2025\] RedactX: An LLM-Powered Framework for Automatic Clinical Data De-Identification](../../ACL2025/medical_nlp/redactor_an_llm-powered_framework_for_automatic_clinical_data_de-identification.md)
- [\[ACL 2025\] ReflecTool: Towards Reflection-Aware Tool-Augmented Clinical Agents](../../ACL2025/medical_nlp/reflectool_clinical_agent.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)

</div>

<!-- RELATED:END -->
