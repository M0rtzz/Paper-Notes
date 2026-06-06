---
title: >-
  [论文解读] Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost
description: >-
  [ICML 2026][图像生成][AI 采纳] 本文是一篇立场论文，主张"组织引入 AI 并不自动等于生产力提升"，识别出五个被传统经济模型忽略的人与环境调节因子（人员组成、个体基线能力、学习曲线、公平使用激励、目标灵活性）…
tags:
  - "ICML 2026"
  - "图像生成"
  - "AI 采纳"
  - "生产力悖论"
  - "Gries-Naudé 模型"
  - "组织因素"
  - "学习曲线"
---

# Position: Adopting AI in Practice Does Not Guarantee the Productivity Boost

**会议**: ICML 2026  
**arXiv**: [2605.24688](https://arxiv.org/abs/2605.24688)  
**代码**: 无  
**领域**: AI 治理 / 立场论文 / AI 生产力经济学  
**关键词**: AI 采纳, 生产力悖论, Gries-Naudé 模型, 组织因素, 学习曲线

## 一句话总结
本文是一篇立场论文，主张"组织引入 AI 并不自动等于生产力提升"，识别出五个被传统经济模型忽略的人与环境调节因子（人员组成、个体基线能力、学习曲线、公平使用激励、目标灵活性），并在 Gries-Naudé (2022) 偏均衡模型上加入组织有效性 $\Omega$、能力调整 $\phi(z,\kappa_i)$、学习曲线 $\lambda_i(\tau)$、有效自动化阈值 $\tilde N_{IT}$ 四类修正项，得到能描述"为什么同样投 AI 不同组织产出差距巨大"的修订生产函数。

## 研究背景与动机

**领域现状**：生成式 AI 爆发后，从 GPT-3、Codex、ChatGPT 到 agentic 框架，企业和学校都在快马加鞭部署 AI；经济学侧 Graetz-Michaels、Acemoglu、Gries-Naudé 等开始把 AI 直接当作"生产率因子"塞进总产出模型，默认 AI 能力越强、TFP 提升越大。

**现有痛点**：实证完全不支持这种线性外推。Brynjolfsson 等 2025 在客服公司发现 AI 对新手能涨 36%、对老员工几乎为零；Dell'Acqua 等 2023 在管理咨询场景发现一旦任务超出 AI 能力边界，用 AI 的人准确率反而比不用 AI 的人低 19 个百分点；Calvino-Fontanelli 的跨国厂级数据显示 AI 采纳高度集中在已经很强的大公司上。Solow 的经典观察"到处都是计算机，唯独看不到生产率"再次重演——"AI 生产力悖论"。

**核心矛盾**：现有经济模型把 $\gamma_{IT}(z)$、$b_{IT}$、$A_{IT}/A_L$ 等参数当作纯技术外生量；但真正决定能不能把 AI 能力变现的是组织结构、个体能力、学习动力学、使用激励、目标灵活性这五类**人与环境内生变量**，这些被一笔带过。

**本文目标**：(1) 明确指出这五类调节因子是什么、如何作用；(2) 改造 Gries-Naudé 偏均衡模型，把它们显式塞进生产函数；(3) 给研究者、从业者、政策制定者各自一份"该干什么"清单，重点区分产业与教育两个域。

**切入角度**：作者并不否认 AI 能力本身在涨，而是把视角从"AI 模型多强"转到"组织如何使用 AI"——AI 能力是必要不充分条件，五类组织/个体因子才是把能力翻译成产出的"传动比"。

**核心 idea**：用一句话概括——**真正的 AI 生产力 = 技术能力 × 组织有效性 $\Omega$ × 能力-任务匹配 $\phi$ × 学习进度 $\lambda$ × 目标灵活度 $F$**；忽略后面四项，任何 AI 投资 ROI 估算都是误导。

## 方法详解

立场论文本身不是算法工作，"方法"指的是它对 Gries-Naudé 模型的修订框架。原始模型描述的是任务区间 $[N-1, N]$ 上每个任务 $z$ 由标准劳动力或 IT 服务（含 AI）生产，CES 聚合为人服务总产出 $H$，并定义自动化阈值 $N_{IT}$ 区分可/不可由 AI 介入的任务。

### 整体框架
作者先定性识别五个调节因子，然后给出五条修订（组织有效性 $\Omega$、能力调整生产率 $\tilde\gamma_L, \tilde\gamma_{IT}$、学习曲线 $\lambda_i(\tau)$、内生任务边界 $\tilde N_{IT}$），最后把它们拼成"个体—任务—时间"三维的修订人服务生产函数 $\tilde h_i(z,\tau)$，再 CES 聚合到子组织级 $\tilde H(\tau)$。整个框架的输入是组织+个体的可观测属性，输出是一个能解释异质性、能预测"投了 AI 但生产率没涨"的生产函数。

### 关键设计

1. **组织有效性修正 $\Omega = \omega_C \cdot \omega_I$**:

    - 功能：把"AI 专家头数 $b_{IT}$"和"有效专家可及度 $\tilde b_{IT}$"区分开，刻画组织内部摩擦。
    - 核心思路：定义 $\omega_C \in [0,1]$ 为组织结构对齐度（扁平 AI 任务组接近 1，层级深、政策与一线脱节趋近 0），$\omega_I \in [0,1]$ 为激励对齐度（只有少数人有"AI 转型"奖励则塌缩，因为竞争不对称会侵蚀同侪公平使用动机）；二者相乘后作为 $b_{IT}$ 的折扣因子，得 $\tilde b_{IT}(z) = \Omega \cdot b_{IT}(z)$。
    - 设计动机：解释了 Calvino-Fontanelli 观察——"已经强的公司 AI 收益更大"不是因为它们有更多 AI 专家，而是组织结构让专家能真正触达任务执行者。

2. **能力-任务交互函数 $\phi(z,\kappa_i)$ 与可靠边界 $N_R$**:

    - 功能：刻画 Dell'Acqua "jagged frontier"——同一个 AI 工具对同一个人在不同任务上效果可以相反。
    - 核心思路：引入个体基线能力 $\kappa_i \in [0,1]$，把任务 $[N-1, N_{IT}]$ 进一步切成"AI 输出可靠区" $[N-1, N_R]$ 和"AI 输出不可靠区" $(N_R, N_{IT}]$。可靠区内 $\partial \phi_{\text{in}}/\partial \kappa_i \leq 0$（新手收益更大，Brynjolfsson 2025），不可靠区内 $\partial \phi_{\text{out}}/\partial \kappa_i > 0$（高能力者才能识别并修正 AI 错误）。同时 $\tilde\gamma_L(z,\kappa_i) = \gamma_L(z) \cdot g(\kappa_i)$ 让基线能力对裸劳动生产率单调递增。
    - 设计动机：单一 $\gamma_{IT}/\gamma_L$ 比率描述不了"新手暴涨、专家略跌"的异质性；分段函数+个体参数把这种非单调关系内生化。

3. **学习曲线 $\lambda_i(\tau) = 1 - e^{-\rho_i \tau}$ 与灵活阈值 $\tilde N_{IT}$**:

    - 功能：把"AI 收益随时间演化"和"组织目标僵化程度"塞进同一个框架。
    - 核心思路：以采纳后时间 $\tau$ 作为比较静态参数，定义个体 $i$ 的学习进度 $\lambda_i(\tau) \in [0,1)$，学习率 $\rho_i$ 异质（Matthew 效应风险）；最终有效 AI 任务生产率 $\tilde\gamma_{IT}(z,\kappa_i,\tau) = \gamma_{IT}(z) \cdot \phi(z,\kappa_i) \cdot \lambda_i(\tau)$。再用组织灵活度 $F \in [0,1]$ 把技术阈值 $N_{IT}$ 收缩为有效阈值 $\tilde N_{IT} = (1-F)(N-1) + F \cdot N_{IT}$；$F=1$ 还原原模型，$F<1$ 表示僵化考核让组织只用 AI 做了一小段技术上可自动化的任务。
    - 设计动机：原模型的 $N_{IT}$ 是纯技术给定量，本文用 $F$ 把它和"领导是否愿意按 AI 能力重排 KPI"挂钩，从而解释为什么同样技术栈下不同组织实际 AI 渗透率差几个量级。

把上述三块合并，得到核心修订式（公式 9）：

$$\tilde h_i(z,\tau) = \begin{cases} \tilde\gamma_L(z,\kappa_i) l_i(z) A_L + \tilde\gamma_{IT}(z,\kappa_i,\tau) \cdot \Omega \cdot b_{IT}(z) A_{IT} D, & z \in [N-1, \tilde N_{IT}] \\ \tilde\gamma_L(z,\kappa_i) l_i(z) A_L, & z \in (\tilde N_{IT}, N] \end{cases}$$

再做 CES 聚合 $\tilde H(\tau) = \big( \int_{N-1}^N (\sum_i \tilde h_i(z,\tau))^{(\sigma-1)/\sigma} dz \big)^{\sigma/(\sigma-1)}$ 得到子组织（team / department）级人服务总产出。

### 论证策略
本文不做实证或仿真，立论方式是：(i) 引用现有实证（Brynjolfsson、Dell'Acqua、Calvino-Fontanelli、Acemoglu）作为"现状反例"；(ii) 在 Gries-Naudé 数学骨架上做最小侵入修订，保留原模型所有定性结论；(iii) 用产业 + 教育两个对照案例落地框架；(iv) 在第 4 节正面回应三类反对意见——技术决定论、测量问题论、工资成本论。

## 实验关键数据

立场论文无实验，本节以"经验证据 × 框架对应关系"两张表替代。

### 经验证据 → 框架因子映射

| 经验现象 | 出处 | 框架中的对应项 |
|---|---|---|
| 新手提升 36%、专家几乎为零 | Brynjolfsson et al. 2025 | $\phi_{\text{in}}(\kappa_i)$ 对 $\kappa_i$ 递减 |
| AI 用在能力边界外，准确率反低 19pp | Dell'Acqua et al. 2023 | $\phi_{\text{out}}(\kappa_i)$ 对 $\kappa_i$ 递增；$N_R < N_{IT}$ |
| AI 采纳集中于大而强的公司 | Calvino-Fontanelli 2023 | $\Omega = \omega_C \omega_I$ 高 |
| TFP 涨幅远低于预期 | Acemoglu 2025 | 整体 $F \cdot \Omega \cdot \lambda$ 远小于 1 |
| ChatGPT 涨成绩但不涨 self-efficacy | Deng et al. 2025 | 教育域 $F$ 与 $\omega_I$ 错位 |

### Gries-Naudé 原始决定因素 → 本文修订（对应论文 Table 1）

| 原始决定因素 | 本文修订 |
|---|---|
| 自动化阈值 $N_{IT}$ | 有效阈值 $\tilde N_{IT}$，受目标灵活度 $F$ 决定 |
| 任务生产率比 $\gamma_{IT}(z)/\gamma_L(z)$ | 变为个体+时间相关 $\tilde\gamma_{IT}/\tilde\gamma_L = \gamma_{IT}\phi\lambda / (\gamma_L g)$ |
| 专家可得性 $b_{IT}$ | 有效可得性 $\tilde b_{IT} = \Omega \cdot b_{IT}$ |
| 相对能力比 $A_{IT}/A_L$ | 调制为 $\propto \phi(z,\kappa_i) \lambda_i(\tau) / g(\kappa_i)$ |

### 关键洞察
- **生产率比单调性约束**：作者要求 $\tilde\gamma_{IT}/\tilde\gamma_L$ 在固定 $\kappa_i, \tau$ 下随 $z$ 单调，才能让 $N_R$ 和 $\tilde N_{IT}$ 无歧义定义——这是模型自洽性的隐性前提。
- **聚合粒度选择**：聚合到 team / department 而非整个公司，理由是 firm-wide 分布太重尾（"10× 工程师"），子单位内分布相对可处理；个体异质性通过 $\kappa_i, \phi, \rho_i$ 保留，不被平均掉。
- **教育 vs 产业的根本差异**：产业域 $F$ 高（KPI 可重排）时 AI 收益大，但最终受益者从员工漂移到管理层；教育域因为受益者就是学生本人，"productivity for whom"问题反而消失，但 $F$ 和 $\omega_I$ 错位会让 AI 直接伤害学习目标。

## 亮点与洞察
- **把"参数"变"变量"的最小侵入式修订**：作者克制地只在原 $\gamma, b_{IT}, N_{IT}$ 上叠加四个调制项 $\Omega, \phi, \lambda, F$，保留原模型所有定性结论，新读者几乎零成本接入。这种"打补丁不重写"的论证姿态比从零造新模型更容易被经济学社区接纳。
- **正面回应"技术决定论"的方式很漂亮**：作者没有否认 AI 会越来越强，而是说"AI 越强 → $N_R$ 上移 → $\phi_{\text{out}}$ 重要性下降，但约束转移到 $F$ 和 $\omega_I$"——把对手论点纳入自己框架重新解释，而不是对喷。
- **"productivity for whom"是被多数 AI 经济学论文忽视的政治经济学维度**：把"产业 = 表面受益是员工、实际受益是管理层"和"教育 = 受益者重合所以张力消失"对照写，迫使任何号称"AI 提升生产力"的研究先回答"提升了谁的生产力"。
- **可迁移设计**：$\Omega, \phi, \lambda, F$ 这套调制器适用于任何"新技术采纳"经济模型——把同样思路套到云计算、远程办公、低代码平台都成立，框架有很强的复用空间。

## 局限与展望
- **作者承认的局限**：$\phi, g, \rho$ 的具体函数形式刻意留作"一般形式"，理由是承诺单一闭式会掩盖跨域差异；但这同时意味着框架**没有可证伪的预测**，更像是叙事工具而非可拟合模型。
- **基线能力 $\kappa_i$ 高度抽象**：实践中很难统一测量；不同任务下"baseline capability"指什么（编程功底？批判性思维？领域知识？）作者没有给操作化定义，只在 §6.1 提一句"留给组织行为学研究者"。
- **缺乏对 AI 直接成本的内生化**：第 4.3 节承认许可费/算力成本是"另一个变量"，但把它从框架中划走，使得"投 AI 划不划算"的完整决策仍需框架外补充。
- **个体加总与 CES 假设可能冲突**：在子组织内对 $i$ 求和后再 CES 聚合，意味着个体之间任务上完全可替代，而原 Gries-Naudé CES 描述的是任务间替代弹性 $\sigma$；个体间替代和任务间替代被混在一起，理论上需要更小心的论证。
- **改进方向**：(i) 给 $\phi_{\text{in}}, \phi_{\text{out}}, g, \rho(\kappa)$ 各假设一组参数化族（如 sigmoid 或幂律），跑反事实模拟；(ii) 用 Brynjolfsson 等 2025 的客服数据做模型校准，比较"原 Gries-Naudé"vs"本文修订"的拟合度，把立场论文实证化。

## 相关工作与启发
- **vs Gries & Naudé (2022)**: 本文直接基于其偏均衡 CES 模型扩展，原模型把 $\gamma_{IT}, b_{IT}, N_{IT}$ 视为技术外生量；本文把它们内生化为组织、个体、时间的函数。立场强、骨架小，是典型的"打补丁优于重写"路径。
- **vs Acemoglu (2025)**: Acemoglu 通过区分"易学/难学任务"算出 AI 对 TFP 贡献远低于预期；本文相当于给 Acemoglu 的悲观结论提供"为什么"的微观机制——不是任务难学，是组织没把可学的任务真的让 AI 接管（$F<1$、$\Omega<1$）。
- **vs Dell'Acqua et al. (2023)**: Dell'Acqua 提出"jagged frontier"是经验现象；本文用 $\phi(z, \kappa_i)$ + $N_R$ 把它形式化为可写进生产函数的对象，把经验术语升格为理论原语。
- **vs Brynjolfsson et al. (2025)**: 经验上发现新手获益最多；本文用 $\partial \phi_{\text{in}}/\partial \kappa_i \leq 0$ 把它写成模型不等式，并指出在不可靠区 $\partial \phi_{\text{out}}/\partial \kappa_i > 0$ 同时成立，把单一结论拓展为更完整的双区图景。
- **启发**：任何想说"X 技术能带来 Y 增长"的论文，都可以先问"$\Omega$（组织能不能让它落地）、$\phi$（用户能不能匹配它）、$\lambda$（多久学得会）、$F$（KPI 允不允许调整）"这四问；这套调制器是研究新技术经济效益时极好的 checklist。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把模糊的"组织因素"做成可塞进 CES 模型的四个调制项，框架级贡献清晰；但单个因子（人员、能力、学习、激励）在 OB / IS 文献早有讨论，本文胜在系统集成而非首发。
- 实验充分度: ⭐⭐ 立场论文无实验，引用的实证文献覆盖度好，但没有自己跑校准/反事实，框架可证伪性弱。
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰（背景→五因子→修订模型→反对意见→产业/教育案例→行动呼吁），数学侵入极小但精确，第 4 节对反对意见的处理是范例级。
- 价值: ⭐⭐⭐⭐ ICML 立场论文很少触及 AI 经济学，本文是难得的"ML 社区 × 生产力经济学"桥梁；对正在做 AI ROI 估算的企业 / 学校 / 政策机构有直接 checklist 价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Position: AI Evaluations Should be Grounded on a Theory of Capability](position_ai_evaluations_should_be_grounded_on_a_theory_of_capability.md)
- [\[ICML 2026\] OmniAID: Decoupling Semantic and Artifacts for Universal AI-Generated Image Detection in the Wild](omniaid_decoupling_semantic_and_artifacts_for_universal_ai-generated_image_detec.md)
- [\[ICML 2026\] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization](order_within_chaos_capturing_intrinsic_energy_anomalies_for_ai-manipulated_image.md)
- [\[ICML 2026\] DGS-Net: Distillation-Guided Gradient Surgery for CLIP Fine-Tuning in AI-Generated Image Detection](dgs-net_distillation-guided_gradient_surgery_for_clip_fine-tuning_in_ai-generate.md)
- [\[AAAI 2026\] HierarchicalPrune: Position-Aware Compression for Large-Scale Diffusion Models](../../AAAI2026/image_generation/hierarchicalprune_position-aware_compression_for_large-scale_diffusion_models.md)

</div>

<!-- RELATED:END -->
