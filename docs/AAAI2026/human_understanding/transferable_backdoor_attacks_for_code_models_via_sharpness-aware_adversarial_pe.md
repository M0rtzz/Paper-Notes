---
title: >-
  [论文解读] Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation
description: >-
  [AAAI 2026][人体理解][backdoor attack] 提出 STAB（Sharpness-aware Transferable Adversarial Backdoor），通过 SAM 训练代理模型使其收敛到损失平面的平坦区域，并使用 Gumbel-Softmax 优化生成上下文感知的对抗触发器，首次实现了同时兼顾跨数据集迁移性和隐蔽性的代码模型后门攻击。
tags:
  - AAAI 2026
  - 人体理解
  - backdoor attack
  - 代码模型
  - 迁移性
  - Sharpness-Aware Minimization
  - 对抗扰动
---

# Transferable Backdoor Attacks for Code Models via Sharpness-Aware Adversarial Perturbation

**会议**: AAAI 2026  
**arXiv**: [2602.11213](https://arxiv.org/abs/2602.11213)  
**代码**: [github.com/ChangShuyu/STAB](https://github.com/ChangShuyu/STAB)  
**领域**: 人体理解  
**关键词**: backdoor attack, 代码模型, 迁移性, Sharpness-Aware Minimization, 对抗扰动

## 一句话总结

提出 STAB（Sharpness-aware Transferable Adversarial Backdoor），通过 SAM 训练代理模型使其收敛到损失平面的平坦区域，并使用 Gumbel-Softmax 优化生成上下文感知的对抗触发器，首次实现了同时兼顾跨数据集迁移性和隐蔽性的代码模型后门攻击。

## 研究背景与动机

### 问题定义

预训练代码模型（如 PLBART、CodeT5）已成为现代软件供应链的重要组成部分，但由于训练数据来自公开代码库，容易受到数据投毒导致的后门攻击——模型在触发器存在时产生恶意输出，而在干净输入上保持正常行为。

### 核心矛盾：迁移性 vs 隐蔽性的权衡

现有代码后门攻击面临根本性权衡：

| 攻击类型 | 迁移性 | 隐蔽性 | 核心问题 |
|---------|--------|--------|---------|
| 静态触发器（Fixed/Grammar） | ✓ 好 | ✗ 差 | 插入固定死代码，被 KillBadCode 等防御 100% 检测 |
| 动态触发器（AFRAIDOOR） | ✗ 差 | ✓ 好 | 贪心搜索导致收敛于尖锐极小值，跨数据集失效 |

### 现有方法的关键限制

**AFRAIDOOR 假设投毒数据与受害者训练数据分布相同**——但实际场景中攻击者投毒公共仓库，受害者从多样来源收集数据，分布必然不同

**AFRAIDOOR 使用贪心搜索**，独立优化每个标识符，导致收敛于次优局部极小值
3. 贪心扰动发现的是**损失面尖锐区域中的数据集特定模式**，缺点参数微小变化导致性能大幅波动

### 关键观察

**收敛到平坦区域的模型学到更通用的特征**。平坦区域捕获跨不同数据集普遍存在的代码模式，而非局限于狭窄参数空间的数据集特定伪影。因此，在平坦区域发现的对抗扰动能更有效地跨数据集迁移。

## 方法详解

### 整体框架

STAB 是一个三阶段流水线：

1. **Sharpness-Aware Surrogate Model Training**：使用 SAM 训练代理模型至平坦损失区域
2. **Adversarial Trigger Optimization**：使用 Gumbel-Softmax 可微优化生成触发器分布
3. **Trigger Generation and Deployment**：从优化分布中采样离散触发器并部署

### 威胁模型

攻击者构建投毒数据集 $\mathcal{D}_p = \{(x_i \oplus t_i, y^*)\}_{i=1}^m$，注入触发器并配对目标输出 $y^*$。受害者仅包含其子集 $\mathcal{D}_p' \subseteq \mathcal{D}_p$，攻击者只能投毒受害者训练数据的一小部分 $\epsilon = |\mathcal{D}_p'|/|\mathcal{D}|$。关键约束是 $\mathcal{D}_s \neq \mathcal{D}_v$（跨数据集场景）。

### 关键设计

#### 1. **Sharpness-Aware Surrogate Model Training**：平坦损失面的代理模型训练

核心思路：使用 SAM 优化使代理模型收敛到平坦极小值区域，从而发现跨数据集通用的后门模式。

训练目标使用 min-max 公式：

$$\min_{\theta_s} \mathcal{L}_{\text{SAM}}(\theta_s, \mathcal{D}_s) = \min_{\theta_s} \max_{\|\delta\|_2 \leq \rho} \mathcal{L}(\theta_s + \delta, \mathcal{D}_s)$$

SAM 优化过程交替进行：
- 找到在允许半径内使损失最大的最坏情况扰动：$\delta^* = \rho \cdot \frac{\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)}{\|\nabla_{\theta_s}\mathcal{L}(\theta_s, \mathcal{D}_s)\|_2}$
- 在扰动参数处计算梯度并更新：$\theta_s \leftarrow \theta_s - \eta \cdot \nabla_{\theta_s}\mathcal{L}(\theta_s + \delta^*, \mathcal{D}_s)$

**设计动机**：平坦极小值编码更通用的代码特征（语义和语法模式），使后续生成的对抗触发器在分布偏移下仍然有效。

#### 2. **Gumbel-Softmax Trigger Optimization**：可微离散触发器优化

核心思路：将离散标识符选择问题转化为连续可微优化问题，实现端到端联合优化所有触发器 token。

**步骤**：
- 解析代码 AST，识别所有可修改标识符 $\{v_j\}_{j=1}^k$
- 初始化可学习代理分布矩阵 $\mathbf{\Pi} \in \mathbb{R}^{L \times |\mathcal{V}_t|}$
- 使用 Gumbel-Softmax 函数生成软 token 表示：
  $$\tilde{\mathbf{z}}_i = \text{softmax}\left(\frac{\log(\boldsymbol{\pi}_i) + \mathbf{g}_i}{\tau}\right)$$
- 通过加权嵌入 $\mathbf{e} = \tilde{\mathbf{z}}^T \mathbf{E}$ 作为可微输入送入代理模型

**优化目标**为复合损失：$\mathcal{L}_{\text{trigger}} = \mathcal{L}_a + \lambda \cdot (\mathcal{L}_c + \mathcal{L}_d)$

| 损失项 | 公式 | 作用 |
|--------|------|------|
| 攻击损失 $\mathcal{L}_a$ | $\mathbb{E}_{x \sim \mathcal{D}_s'}[-\log P(y^* \| \mathcal{M}_s^*(\mathbf{e}))]$ | 确保后门激活，生成恶意目标输出 |
| 一致性损失 $\mathcal{L}_c$ | $\sum_{j=1}^k \sum_{l,l' \in P_j} \text{MMD}(\boldsymbol{\pi}_l, \boldsymbol{\pi}_{l'})$ | 同一标识符在代码中所有出现位置使用相同触发器 token |
| 多样性损失 $\mathcal{L}_d$ | $-\sum_{1 \leq i < j \leq k} \text{MMD}(\bar{\boldsymbol{\pi}}_i, \bar{\boldsymbol{\pi}}_j)$ | 不同标识符使用不同触发器 token，避免重复模式 |

**设计动机**：使用 MMD（Maximum Mean Discrepancy）约束而非硬约束，既确保语法正确性又保持可微性；一致性损失保证代码标识符命名的全局一致性；多样性损失还能提升隐蔽性。

#### 3. **Trigger Generation and Deployment**：投毒样本生成与部署

- 从优化后的分布矩阵 $\mathbf{\Pi}^*$ 中，使用极低温度的 Gumbel-Softmax 采样离散 token
- 如果不同标识符采样到相同 token，进行重采样保证代码有效性
- 替换原始标识符生成最终投毒代码
- 将投毒代码注入公共代码仓库，利用开源参与机制（star、fork）增加仓库可见度

### 损失函数 / 训练策略

- **代理模型训练阶段**：SAM 优化，$\rho = 0.02$，标准交叉熵损失
- **触发器优化阶段**：Gumbel-Softmax 温度 $\tau = 1.0$，迭代 $N = 100$ 次，权重 $\lambda = 0.1$
- **受害模型**：默认投毒率 $\epsilon = 5\%$，微调 15 个 epoch，带早停策略

## 实验关键数据

### 主实验

**数据集**：Py150（150K Python 文件）、CodeSearchNet/CSN（400K+ Python 函数）、PyTorch/PyT（218K Python 包库），形成 9 种代理-受害者数据集组合。

**任务**：方法名预测（MNP）和代码摘要（CS）。

| 代理模型 | 受害数据集 | AFRAIDOOR Avg ASR | STAB Avg ASR | 提升 |
|---------|-----------|-----------------|-------------|------|
| PLBART | Py150 | 68.48% | 76.89% | +8.4% |
| PLBART | CSN | 90.72% | 94.55% | +3.8% |
| PLBART | PyT | 76.07% | 79.43% | +3.4% |
| CodeT5 | Py150 | 69.20% | 76.93% | +7.7% |
| CodeT5 | CSN | 90.13% | 95.13% | +5.0% |
| CodeT5 | PyT | 78.43% | 80.51% | +2.1% |

STAB 在跨数据集迁移性上平均 ASR 达到 80.1%，超过 AFRAIDOOR 12.4%。同时在干净数据上保持可比的 BLEU 性能。

**防御后效果**（ASR-D，使用 KillBadCode 防御）：

| 攻击方法 | Py150 | CSN | PyT | 整体表现 |
|---------|-------|------|------|---------|
| Fixed | 0% | 0% | 0% | 防御下完全失败 |
| Grammar | 0% | 0% | 0% | 防御下完全失败 |
| AFRAIDOOR | ~60-70% | ~80% | ~65-70% | 波动大 |
| STAB | ~70-73% | ~85-91% | ~72-77% | 一致性高 |

### 消融实验

| 配置 | Py150 ASR | Py150 ASR-D | CSN ASR | PyT ASR | 说明 |
|------|----------|------------|---------|---------|------|
| STAB (完整) | 76.89% | 70.17% | 94.55% | 79.43% | 最佳性能 |
| w/o SAM | 72.21% | 63.92% | 91.12% | 77.84% | ASR 下降 + 标准差增大 |
| w/o Gumbel-Softmax | 74.58% | 67.31% | 93.85% | 76.71% | 初始 ASR 接近但防御后 ASR-D 明显下降 |

**隐蔽性对比**（KillBadCode 检测率，越低越隐蔽）：

| 攻击方法 | Recall | F1 |
|---------|--------|-----|
| Fixed | 99.99% | ~40% |
| Grammar | 99.99% | ~39% |
| AFRAIDOOR | ~27% | ~14% |
| STAB | ~23% | ~10% |

### 关键发现

1. **SAM 是迁移性核心**：去掉 SAM 后不仅 ASR 下降，标准差也显著增大（从 ~0.3% 增大到 ~0.9%），说明平坦损失面对稳定性至关重要
2. **Gumbel-Softmax 是隐蔽性核心**：替换为贪心搜索后初始 ASR 接近，但防御后 ASR-D 明显下降，隐蔽性变差
3. **最优锐度参数 $\rho = 0.02$**：太小则平坦度引导不足，太大则受害模型遇到过多变化的触发器模式导致难以学习一致关联
4. **投毒率分析**：更高投毒率通常提升 ASR，但边际收益在一定阈值后递减；STAB 即使在受限投毒预算下也保持高 ASR-D

## 亮点与洞察

1. **损失面几何的新视角**：首次将"平坦极小值有利于泛化"的理论洞察应用到后门攻击的迁移性问题上，思路非常优雅
2. **离散优化的可微化解决方案**：使用 Gumbel-Softmax 将代码 token 的离散选择转化为连续优化，同时使用 MMD 约束保持代码语法正确性
3. **实际威胁模型更贴合现实**：不再假设投毒数据与受害者数据分布相同，而是研究跨数据集场景
4. **静态攻击在防御下的脆弱性**：KillBadCode 能 100% 检测静态触发器，但 STAB 大幅降低检测率

## 局限与展望

1. **仅评估两种代码模型**（PLBART、CodeT5），未扩展到 CodeLlama 等更大模型
2. **仅评估 Python 代码**，跨语言迁移性未知
3. **触发器仅限标识符重命名**，其他代码转换（如表达式变换）未探索
4. **伦理问题**：虽然声称目的是推进防御研究，但实际提供了一种更危险的攻击方法
5. **防御方法有限**：仅评估三种防御（SS、ONION、KillBadCode），更先进的防御策略未考虑

## 相关工作与启发

- **SAM（Sharpness-Aware Minimization）**：最初用于提升模型泛化能力，本文创新性地将其用于提升后门攻击的迁移性
- **Gumbel-Softmax**：来自 VAE 领域的离散采样技巧，本文将其应用于代码 token 选择
- **AFRAIDOOR**：当前 SOTA 动态攻击，本文的主要对比基线
- **KillBadCode**：SOTA 代码后门防御，利用 n-gram 语言模型检测代码自然度异常
- **启发**：损失面几何（平坦 vs 尖锐）是一个被低估的攻防关键维度

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （将 SAM 理论洞察应用于后门攻击迁移性，非常创新）
- 实验充分度: ⭐⭐⭐⭐ （3 数据集 × 2 模型 × 9 组合 + 全面消融）
- 写作质量: ⭐⭐⭐⭐ （动机清晰、方法严谨，图示质量高）
- 价值: ⭐⭐⭐⭐ （揭示了代码模型安全的新威胁，推动防御研究）

<!-- RELATED:START -->

## 相关论文

- [Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [Authority Backdoor: A Certifiable Backdoor Mechanism for Authoring DNNs](authority_backdoor_a_certifiable_backdoor_mechanism_for_authoring_dnns.md)
- [When Robots Obey the Patch: Universal Transferable Patch Attacks on Vision-Language-Action Models](../../CVPR2026/human_understanding/when_robots_obey_the_patch_universal_transferable_patch_attacks_on_vision-langua.md)
- [Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification](../../NeurIPS2025/human_understanding/stealthy_yet_effective_distribution-preserving_backdoor_attacks_on_graph_classif.md)
- [SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation](../../ICCV2025/human_understanding/samo_a_lightweight_sharpness-aware_approach_for_multi-task_optimization_with_joi.md)

<!-- RELATED:END -->
