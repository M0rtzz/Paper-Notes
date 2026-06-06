---
title: >-
  [论文解读] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference
description: >-
  [CVPR 2026][AI安全][collaborative perception] 提出 Pseudo-Random Bayesian Inference (PRBI) 框架，在**所有车辆均不可信**的协同感知场景中，利用帧间时序一致性作为自参考信号，通过伪随机分组 + 贝叶斯推断…
tags:
  - "CVPR 2026"
  - "AI安全"
  - "collaborative perception"
  - "adversarial defense"
  - "Bayesian inference"
  - "自动驾驶"
  - "V2V communication"
---

# All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference

**会议**: CVPR 2026  
**arXiv**: [2603.08498](https://arxiv.org/abs/2603.08498)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: collaborative perception, adversarial defense, Bayesian inference, autonomous driving, V2V communication

## 一句话总结

提出 Pseudo-Random Bayesian Inference (PRBI) 框架，在**所有车辆均不可信**的协同感知场景中，利用帧间时序一致性作为自参考信号，通过伪随机分组 + 贝叶斯推断，仅需平均 2.5 次验证/帧即可高效识别并排除恶意车辆，检测精度恢复至攻击前的 79.4%–86.9%。

---

## 研究背景与动机

**协同感知 (CP) 的安全隐患**：多车通过 V2V 通信共享特征图来扩展感知范围，但特征融合机制天然暴露在对抗攻击之下——恶意车辆可以在共享特征中注入扰动，导致 ego 车辆感知严重失准。

**现有方法依赖"ego 可信"假设**：采样类防御（ROBOSAC、PASAC）以 ego 车辆的感知作为可靠参考来做一致性验证；分类器类防御则训练二分类网络区分良性/恶意特征——两者都假设 ego 本身不会被攻击。

**现实中 ego 同样可被攻击**：通过 LiDAR 注入攻击或数据截获攻击，攻击者虽不直接入侵 ego 内部系统，但完全可以干扰其特征图。因此"All Vehicles Can Lie"才是真实场景。

**验证开销线性增长**：现有采样/分类方法的每帧验证次数随车辆总数线性增长，难以满足大规模实时协同感知的需求。

**随机采样存在检测延迟**：纯随机化的采样可能需要很多帧才能收敛到完整的攻击者集合，带来持续的风险暴露。

**急需"零信任"且低开销的防御方案**：理想方案应既不假设任何车辆可信，也不需要关于攻击者数量/比例的先验知识，同时保持每帧恒定的验证成本。

---

## 方法详解

### 整体框架

PRBI 的核心流程分四步循环执行：

1. **初始化**：$t=0$ 时假设初始帧感知正确，初始化每辆车的正常检测计数 $\mathbf{c}_{\text{normal}}$ 和异常检测计数 $\mathbf{c}_{\text{abnormal}}$。
2. **伪随机分组 + 一致性验证 (Soft Sampling)**：每帧将所有车辆分成两组，分别与上一帧的良性感知结果做 Jaccard 相似度比较，更新计数。
3. **攻击者评估 (Attacker Evaluation)**：利用经验正常比率 $\eta$ 估算攻击者数量 $m$，并用贝叶斯推断计算每辆车的良性概率 $P_{\text{benign}}[j]$，选出 $m$ 辆最可疑车辆。
4. **假设检验 + 防御感知 (Hypothesis Testing)**：T 检验判断 $m$ 是否收敛；若已收敛则输出攻击者集合，否则用良性概率非零的车辆做本帧协同感知并更新参考。

### 关键设计 1：帧间时序一致性作为自参考信号

- **功能**：用前一帧已验证的良性感知输出 $D_{\text{ref}}$ 作为当前帧的检测参考，完全替代"ego 可信"假设。
- **核心思路**：良性场景下相邻帧的 Jaccard 相似度稳定在约 0.8，而对抗场景下急剧降至 0.3 以下。利用这一显著分布差异，将帧间时序一致性转化为自监督的防御信号。
- **设计动机**：LiDAR 感知输出具有天然的空间连续性，相邻帧之间的感知变化在正常情况下是平滑的——这为零信任环境提供了唯一可用的不依赖于任何车辆的参考锚点。

### 关键设计 2：伪随机二分组策略

- **功能**：每帧仅做 2 次额外验证——将车辆分为两组，按嫌疑度排序后把最可疑的 $\lfloor m \rfloor$ 辆车放一组、其余放另一组。
- **核心思路**：二分组在统计上近似于无放回随机采样过程。采样到全良性组的概率为 $P_{ideal}' = 2^{n-k}/2^n = 2^{-k}$，仅依赖于攻击者数量 $k$。通过经验正常比率 $\eta \approx 2^{-k}$，可反推攻击者数量 $m = \log_2(1/\eta)$。
- **设计动机**：将验证开销从 $O(n)$ 降至恒定（2 次/帧），彻底解耦验证次数与车辆总数。同时基于 $m$ 和 $P_{\text{benign}}$ 的伪随机分组保证了 $m$ 单调收敛至真实值 $k$（Theorem 1）。

### 关键设计 3：贝叶斯概率推断识别攻击者

- **功能**：对每辆车 $j$ 计算后验良性概率 $P_{\text{benign}}[j] = P(\mathcal{B}_j | \mathcal{A})$，选出概率最低的 $\lfloor m \rceil$ 辆车作为疑似攻击者。
- **核心思路**：先验 $P(\mathcal{B}_j)$ 采用短期（上一帧贝叶斯结果）和长期（历史正常检测比率）加权组合，融合时序记忆来缓解早期分组不稳定的影响。似然 $P(\mathcal{A}|\mathcal{B}_j)$ 通过排除车辆 $j$ 后系统异常比率来估计。
- **设计动机**：恶意车辆必然出现在每次异常检测中（$\beta_j = 0$），故其良性概率将持续为 0，保证最终被排除。

### 关键设计 4：T 检验收敛判定

- **功能**：维护窗口 $W$ 存储近 $w_p$ 帧的估计 $m$ 值，检验零假设 $H_0: k = m$，当 T 检验通过且零概率车辆数恰好等于 $m$ 时判定收敛。
- **核心思路**：$m$ 在 $k$ 附近小幅波动，当波动长时间约束在置信区间内即视为收敛。双重条件防止 $m$ 缓慢收敛时过早接受 $H_0$。
- **设计动机**：在连续场景中，越快确认收敛就越早终止冗余验证并输出防御结果，实测平均约 4 帧即可收敛。

---

## 损失函数 / 训练策略

PRBI 本身是**推理阶段的防御框架**，不涉及额外训练或损失函数。它在已训练好的协同感知模型之上运行，直接利用检测输出之间的 Jaccard 相似度进行一致性验证。攻击端的扰动优化目标为标准多智能体检测损失：

$$\max_{\|\delta\| \leq \Delta} \sum_{j=1}^{L} \mathcal{L}_{\text{det}}(d_j, d_j')$$

其中 $\Delta$ 约束扰动幅度，$\mathcal{L}_{\text{det}}$ 为检测损失。防御端通过 Jaccard 阈值 $\epsilon$ 区分正常/异常帧，无需对模型做任何微调。

---

## 实验关键数据

### 表 1：每帧验证次数对比（$n=5$）

| 方法 | 指标 | 攻击比例 20% | 40% | 60% | 80% | 平均 ↓ |
|:-----|:-----|:---:|:---:|:---:|:---:|:---:|
| ROBOSAC | Avg | 4.89 | 10.36 | 8.29 | 4.73 | 7.1 |
| PASAC | Avg | 4.79 | 6.60 | 7.59 | 8.00 | 6.7 |
| **PRBI (Ours)** | **Avg** | **2.00** | **2.35** | **2.61** | **2.86** | **2.5** |

- PRBI 平均仅 2.5 次验证/帧，显著低于 ROBOSAC (7.1) 和 PASAC (6.7)。
- ROBOSAC 最大验证次数高达 30.3/帧，PRBI 仅 5.0/帧。

### 表 2：检测性能对比（$n=5, k=2$，V2VNet 骨干 + 三种攻击）

| 设置 | AP@0.5 | AP@0.7 |
|:-----|:---:|:---:|
| Upper-Bound（无攻击协同） | 80.73 | 78.35 |
| Attack w/ PGD | 17.02 | 14.53 |
| PRBI against PGD | **68.93** (+51.91) | **63.82** (+49.29) |
| Attack w/ BIM | 13.51 | 11.69 |
| PRBI against BIM | **68.76** (+55.25) | **64.88** (+53.19) |
| Attack w/ C&W | 10.68 | 6.04 |
| PRBI against C&W | **71.87** (+61.19) | **68.54** (+62.50) |
| Lower-Bound（单车感知） | 56.35 | 52.89 |
| ROBOSAC | 64.13 (+7.78) | 61.01 (+8.12) |
| PASAC | 68.39 (+12.04) | 64.73 (+11.83) |

- 在 V2VNet 上，PRBI 抵御 C&W 时恢复了攻击前 86.9% 的 AP 损失，显著优于 ROBOSAC 和 PASAC。
- 跨多种融合策略（Mean/Max/Sum/V2VNet/DiscoNet）均保持稳定鲁棒性。

### 收敛与识别率

| 攻击比例 | 平均收敛帧数 | 恶意车辆识别率 | 良性车辆误判率 |
|:---:|:---:|:---:|:---:|
| 20% | 2.25 | 100% | 0% |
| 40% | 2.77 | 100% | 6% |
| 60% | 3.36 | 100% | 0% |
| 80% | 4.27 | 100% | 0% |

- 所有攻击比例下恶意识别率均为 100%，平均约 4 帧内收敛。

---

## 亮点与洞察

1. **首个面向"全不可信"场景的高效防御**：打破了现有方法对 ego 可信的依赖，提出帧间时序一致性作为自参考信号——思路简洁但极具洞察力。
2. **验证开销与车辆数解耦**：二分组策略将每帧验证从 $O(n)$ 降至恒定 2 次，理论最优。
3. **理论保证完备**：Theorem 1 证明 $m$ 单调收敛至 $k$；Theorem 2 证明仅 floor 取整保证精确收敛——理论分析严谨。
4. **攻击者必被排除的不变量**：恶意车辆 $\beta_j = 0$，良性概率始终为 0，保证不会漏检。
5. **实用性强**：无需额外训练、无需先验知识、无需修改感知模型，作为即插即用的推理阶段防御模块。

---

## 局限与展望

1. **帧间一致性假设的脆弱场景**：在车辆高速转弯、急刹车等极端运动场景下，相邻帧感知变化可能很大，Jaccard 相似度自然下降，可能导致误报。
2. **初始帧正确性假设**：$t=0$ 假设感知完全正确，若系统启动时即遭受攻击则无法建立可靠参考。
3. **仅评估了 $n=5$ 的小规模场景**：实际城市自动驾驶可能涉及数十辆车的协同，大规模场景下的收敛速度和稳定性有待验证。
4. **静态攻击者集合假设**：假设攻击者在整个序列中固定不变，对于动态加入/退出的攻击者适应性存疑。
5. **40% 攻击比例下存在 6% 误判**：$m$ 过早稳定可能导致良性车辆被错误排除，虽不影响恶意识别率但会损失协同增益。
6. **未考虑自适应攻击**：攻击者若知道 PRBI 的检测逻辑，可能设计缓变扰动来维持帧间相似度，绕过阈值检测。

---

## 相关工作与启发

- **ROBOSAC / PASAC**：经典采样类防御，以 ego 为参考做迭代一致性验证。PRBI 的核心改进是用帧间时序替代 ego 信任，用二分组替代线性采样。
- **MATE**：基于几何的多智能体信任估计器，需要目标跟踪和可见性推理。PRBI 则完全不依赖场景几何建模，更加轻量。
- **分类器类防御**：训练二分类网络检测恶意特征，但泛化性差且扩大攻击面。PRBI 无需训练，零攻击面扩展。
- **启发**：帧间一致性信号不仅适用于协同感知防御，也可推广至任何多源融合系统的异常检测（如联邦学习中的恶意客户端检测）。伪随机分组 + 贝叶斯推断的范式具有通用性。

---

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次在全不可信设置下提出恒定开销防御，帧间自参考信号的思路新颖
- **实验充分度**: ⭐⭐⭐⭐ — 多种攻击/融合策略/参数敏感性分析齐全，但仅 $n=5$ 略显不足
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题定义清晰、理论分析严谨、公式推导完整
- **价值**: ⭐⭐⭐⭐ — 对协同自动驾驶安全有实际意义，即插即用的设计利于落地

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](../../ICML2026/ai_safety/one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ACL 2026\] On the (In-)Security of the Shuffling Defense in the Transformer Secure Inference](../../ACL2026/ai_safety/on_the_in-security_of_the_shuffling_defense_in_the_transformer_secure_inference.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)
- [\[ACL 2025\] Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](../../ACL2025/ai_safety/crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](../../ICML2026/ai_safety/singular_bayesian_neural_networks.md)

</div>

<!-- RELATED:END -->
