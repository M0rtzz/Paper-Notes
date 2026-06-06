---
title: >-
  [论文解读] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC
description: >-
  [ICLR 2026][LLM安全][隐私保护] 提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。
tags:
  - "ICLR 2026"
  - "LLM安全"
  - "隐私保护"
  - "安全多方计算"
  - "提示调优"
  - "前向调优"
  - "随机特征注意力"
---

# SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC

**会议**: ICLR 2026  
**arXiv**: [2506.15307](https://arxiv.org/abs/2506.15307)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 隐私保护, 安全多方计算, 提示调优, 前向调优, 随机特征注意力  

## 一句话总结

提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。

## 背景与动机

1. **隐私敏感领域的 LLM 适配刚需**：医疗、金融、政务等领域迫切需要将 LLM 适配到专业任务，但数据受 GDPR/HIPAA 等法规保护，模型参数也可能编码源域统计信息构成隐私风险。

2. **MPC 提供密码学级隐私保证，但直接微调开销极大**：安全多方计算允许多方在不暴露各自输入的情况下共同计算。然而对 RoBERTa_LARGE 做一次 SFT 迭代需约 10 分钟、970GB 通信量——反向传播和优化器占 73% 时间，softmax 注意力占 75% 时间。

3. **反向传播包含大量 MPC 不友好操作**：Softmax、GELU、LayerNorm 的逆运算在 MPC 环境中需分解为加减乘比较的近似计算，Transformer 的深度堆叠进一步放大这些成本。Adam 优化器中的除法和开方同样昂贵。

4. **现有高效微调方法无法解决根本瓶颈**：LoRA 和梯度提示调优虽减少更新参数量，但仍需反向传播和 softmax 的隐私保护计算。实验显示梯度 Prompt Tuning 在 MPC 下甚至比 SFT 更慢（882s vs 652s），因为额外的 prompt token 增加了前向和反向开销。

## 方法详解

### 整体框架：SecP-Tuning

- **做什么**：构建基于 MPC 的隐私保护提示调优框架，使数据拥有者可在不暴露数据的前提下通过 API 调用方式对模型开发者的 LLM 进行领域适配。
- **为什么**：必须从根本上消除反向传播和 softmax 这两大 MPC 瓶颈。
- **怎么做**：两个核心创新——(1) 前向调优（FoT）+ "Server-Client"架构消除反向传播；(2) 隐私保护随机特征注意力（RFA）替代 softmax 将注意力复杂度从 $O(n^2d)$ 降到 $O(ndr)$。

### 关键设计 1：隐私保护前向调优 (Privacy-Preserving FoT)

核心思路是用无梯度优化器（CMA-ES）在低维隐空间中更新提示向量，只需前向推理不需反向传播。具体采用 "Server-Client" 七步交互范式：

1. 数据拥有者本地初始化提示嵌入 $p$ 并拼接私有数据嵌入 $X$
2. 对 $X$ 做秘密共享 $([X]_0, [X]_1)$ 分发给两个服务器
3. 两服务器交互执行 MPC 协议完成隐私保护前向推理，产出预测共享 $[Y]$
4. 返回数据拥有者重建推理结果 $Y$
5. 数据拥有者本地明文计算损失 $L$
6. 本地用 CMA-ES 更新提示嵌入

关键安全性质：损失计算和优化器完全在数据拥有者本地执行（明文），服务器永远无法获取更新后的提示参数，从而从架构层面杜绝模型记忆导致的数据泄露风险。

利用 prompt 的低内在维度特性，实际优化在低维隐空间 $z \in \mathbb{R}^d$（$d \ll D$）中进行，通过随机投影 $A \in \mathbb{R}^{D \times d}$ 映射回原始空间：

$$z^* = \arg\min_{z \in \mathcal{Z}} \mathcal{L}(f(Az; X), Y)$$

### 关键设计 2：隐私保护随机特征注意力 (Privacy-Preserving RFA)

Softmax 注意力的三重困难：(a) $O(n^2d)$ 二次复杂度；(b) 涉及指数、除法、最大值三种 MPC 不友好操作；(c) 通信量随序列长度平方增长。

SecP-Tuning 用随机傅里叶特征近似 softmax，将 dot-then-exponentiate 操作拆解为核函数近似：

$$\exp(\mathbf{x}^\top\mathbf{y}/\sigma^2) \approx \phi(\mathbf{x})^\top\phi(\mathbf{y})$$

其中 $\phi(\mathbf{x}) = \exp(\|\mathbf{x}\|^2/(2\sigma^2))[\varphi(\mathbf{x},\omega_1), \ldots, \varphi(\mathbf{x},\omega_M)]^\top$，复杂度从 $O(n^2d)$ 降至 $O(ndr)$。

然而 $\phi$ 中包含余弦函数，同样是 MPC 不友好操作。创新点在于设计了高效的 MPC 余弦协议 $\Pi_{\text{cosine}}$：

- **离线阶段**：预生成随机数 $t$ 及 $\sin(t)$、$\cos(t)$ 的秘密共享
- **在线阶段**：仅需一轮通信重建 $\delta = (x+t) \bmod \tau$，再用三角恒等式 $\cos(x) = \sin(\delta)\sin(t) + \cos(\delta)\cos(t)$ 计算

实现**单轮通信、2ℓ-bit 数据量**完成余弦计算。

## 实验

### 实验设置

- **模型**：RoBERTa_LARGE（24 层、1024 维）
- **数据集**：SST-2、MRPC、RTE、Yelp Polarity、AG's News（每类 16 样本 few-shot）
- **MPC 后端**：CrypTen 框架，3 台 A100 GPU 服务器
- **网络环境**：LAN（3Gbps, 0.8ms）和 WAN（100Mbps/80ms、200Mbps/40ms）
- **基线**：全参数 SFT、梯度 Prompt Tuning、FoT（明文）

### 主实验：效率对比（LAN 环境，序列长度 512）

| 方法 | 前向时间(s) | 反向时间(s) | 总时间(s) | 通信量(GB) |
|---|---|---|---|---|
| SFT | 216.2 | 554.5 | 651.6 | 970.7 |
| 梯度 Prompt Tuning | 273.3 | 605.2 | 882.1 | 1116.2 |
| SecP-Tuning (FoT only) | 174.0 | 0.0 | 174.1 | 205.4 |
| **SecP-Tuning (FoT+RFA)** | **54.2** | **0.0** | **55.2** | **56.5** |

### 消融实验：性能与部署性对比

| 方法 | SST-2 Acc | Yelp P. Acc | AG's News Acc | MRPC F1 | RTE Acc | 平均 |
|---|---|---|---|---|---|---|
| SFT | 85.39 | **91.82** | **86.36** | **77.35** | 58.60 | 79.90 |
| 梯度 Prompt Tuning | 68.23 | 61.02 | 84.81 | 51.61 | 54.69 | 64.07 |
| FoT + 预训练提示 | **89.56** | 91.50 | 81.51 | 75.51 | **77.62** | **83.14** |
| SecP-Tuning | 88.11 | 85.23 | 81.27 | 75.33 | 52.95 | 76.58 |

| 部署指标 | SFT | 梯度 Prompt Tuning | SecP-Tuning |
|---|---|---|---|
| 支持 AAS 服务化 | ✗ | ✗ | **✓** |
| SST-2 微调时间 | 65.86h | 86.15h | **8.81h** |
| SST-2 通信量 | 67.36TB | 149.37TB | **14.22TB** |
| 每次查询上传量 | - | - | **12KB** |

### 关键发现

1. **效率提升巨大**：SecP-Tuning 在 LAN 环境下比 SFT 快约 12 倍、比梯度提示调优快约 16 倍；通信量分别降低 17 倍和 20 倍。反向传播和优化器开销被完全消除。
2. **精度可用**：在 few-shot 设置下，SecP-Tuning 在简单任务上甚至超越梯度提示调优（SST-2: 88.11 vs 68.23），验证了方案的实用性。
3. **唯一支持 AAS 部署**：SecP-Tuning 是唯一支持 "As-A-Service" 模式的方法——数据拥有者可通过 API 完成微调，模型开发者无法获取更新后的参数。
4. **$\Pi_{\text{cosine}}$ 是 RFA 高效性的关键**：不使用高效余弦协议的 RFA 在短序列场景下甚至比原始 softmax 更慢（L=64, L=128），说明 $\Pi_{\text{cosine}}$ 设计至关重要。
5. **梯度微调在 MPC 下反而更慢**：梯度 Prompt Tuning 虽减少参数但未能避免反向传播和 softmax 的 MPC 开销，总时间 882s > SFT 的 652s。

## 亮点

- 首个 MPC 环境下的 LLM 提示调优框架，填补了隐私保护微调的空白。
- "Server-Client"架构将损失和优化器卸载到数据拥有者本地明文执行，从架构层面消除反向传播开销且杜绝参数泄露。
- 隐私保护余弦协议 $\Pi_{\text{cosine}}$ 巧妙利用三角恒等式实现单轮通信，是使 RFA 在 MPC 中实际可行的关键。
- 支持黑盒/API 式隐私调优，部署性优于所有梯度传递方案。

## 局限

- 仅在 RoBERTa_LARGE 上验证，未扩展到 GPT/LLaMA 级别的真正"大"模型，实际可扩展性存疑。
- RFA 近似 softmax 会引入精度损失，在某些任务上与 SFT 有较大差距（Yelp P. 85.23 vs 91.82、RTE 52.95 vs 58.60）。
- 半诚实威胁模型假设较弱，恶意参与者场景需额外零知识证明等机制。
- FoT 依赖 CMA-ES 等无梯度优化器，在高维参数空间中收敛性退化，需借助随机投影降维。

## 评分

| 维度 | 评分 |
|---|---|
| 新颖性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 可复现性 | ⭐⭐⭐ |
| 实用性 | ⭐⭐⭐ |
---
title: >-
  [论文解读] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC
description: >-
  [ICLR 2026][AI安全][隐私保护] 提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。
tags:
  - ICLR 2026
  - AI安全
  - 隐私保护
  - 安全多方计算
  - 提示调优
  - 前向调优
  - 随机特征注意力
---

# SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC

**会议**: ICLR 2026  
**arXiv**: [2506.15307](https://arxiv.org/abs/2506.15307)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 隐私保护, 安全多方计算, 提示调优, 前向调优, 随机特征注意力  

## 一句话总结

提出首个基于安全多方计算（MPC）的隐私保护提示调优框架 SecP-Tuning，通过前向调优消除反向传播开销、通过隐私保护随机特征注意力（RFA）替代 softmax 降低通信复杂度，实现约 12-16 倍加速和 17-20 倍通信量缩减。

## 背景与动机

1. **隐私敏感领域的 LLM 适配需求**：医疗、金融、政务等领域迫切需要将 LLM 适配到专业任务，但数据受 GDPR/HIPAA 等法规保护，无法直接访问。

2. **MPC 提供理论级隐私保证**：安全多方计算允许多方在不暴露各自输入的情况下共同计算，可同时保护模型参数和训练数据隐私，优于差分隐私的统计保证。

3. **MPC 微调面临严重效率瓶颈**：对 RoBERTa_LARGE 做一次 SFT 迭代需约 10 分钟、970GB 通信量——其中反向传播和优化器占 73% 时间，softmax 注意力占 75% 时间。

4. **反向传播包含大量 MPC 不友好操作**：Softmax、GELU、LayerNorm 等非线性操作在 MPC 环境中需分解为加减乘比较的近似计算，导致通信轮次和数据量激增。

5. **现有高效微调方法无法解决根本问题**：LoRA 和梯度提示调优虽减少更新参数量，但仍需反向传播和 softmax 的隐私保护计算，未能从根本上降低 MPC 通信开销。

6. **HE 方案难以平衡效率和精度**：同态加密（HE）依赖单方重计算且对非线性操作需昂贵的近似和再加密，MPC 通过多轮通信直接支持复杂非线性运算，更适合微调场景。

## 方法详解

### 整体框架：SecP-Tuning

- **做什么**：构建基于 MPC 的隐私保护提示调优框架，使数据拥有者可在不暴露数据的前提下通过 API 调用的方式对模型开发者的 LLM 进行领域适配。
- **为什么**：直接在 MPC 中做梯度微调的通信开销不可接受，需要从根本上消除反向传播和 softmax 这两大瓶颈。
- **怎么做**：两个核心创新——(1) 前向调优（FoT）+ "Server-Client"架构消除反向传播；(2) 隐私保护随机特征注意力（RFA）替代 softmax 将注意力复杂度从 O(n²d) 降到 O(ndr)。

### 关键设计 1：隐私保护前向调优 (Privacy-Preserving FoT)

- **做什么**：用无梯度优化器（CMA-ES）在低维隐空间中更新提示向量，只需前向推理不需反向传播。
- **为什么**：反向传播中的 Softmax/GELU/LayerNorm 逆运算在 MPC 中开销巨大（占总时间 73%），且梯度优化器（Adam）的除法和开方也是 MPC 不友好操作。
- **怎么做**：采用 "Server-Client" 七步交互范式：(1) 数据拥有者本地初始化提示嵌入 p 并拼接私有数据嵌入 X；(2) 对 X 做秘密共享分发给两个服务器；(3) 两服务器交互执行 MPC 协议完成隐私保护前向推理，产出预测共享 [Y]；(4-5) 返回数据拥有者重建推理结果 Y；(6) 数据拥有者本地明文计算损失 L；(7) 本地用 CMA-ES 更新提示嵌入。因为损失计算和优化器完全在数据拥有者本地执行（明文），服务器永远无法获取更新后的提示参数，从而避免了模型记忆导致的数据泄露风险。

### 关键设计 2：隐私保护随机特征注意力 (Privacy-Preserving RFA)

- **做什么**：用随机傅里叶特征近似 softmax 注意力，并设计高效的 MPC 余弦函数协议 Π_cosine。
- **为什么**：softmax 涉及指数、除法、最大值三种 MPC 不友好非线性操作，且复杂度 O(n²d) 随序列长度二次增长。
- **怎么做**：(1) 利用随机特征方法将 exp(qᵀk/σ²) 近似为 ϕ(q)ᵀϕ(k)，其中 ϕ 涉及余弦函数，将注意力复杂度降为线性 O(ndr)；(2) 为解决余弦函数在 MPC 中的计算问题，利用三角函数周期性和和差化积公式设计 Π_cosine 协议——离线阶段预生成随机数 t 及 sin(t)、cos(t) 的秘密共享，在线阶段仅需一轮通信重建 δ=(x+t) mod τ，再用 cos(x)=sin(δ)sin(t)+cos(δ)cos(t) 计算，实现单轮通信完成余弦。

## 实验

### 实验设置

- **模型**：RoBERTa_LARGE（24 层、1024 维）。
- **数据集**：SST-2、MRPC、RTE、Yelp Polarity、AG's News（每类 16 样本 few-shot）。
- **MPC 后端**：CrypTen 框架，3 台 A100 GPU 服务器；LAN（3Gbps, 0.8ms）和 WAN（100Mbps/80ms、200Mbps/40ms）。
- **基线**：全参数 SFT、梯度提示调优、FoT（明文）。

### 核心结果

| 方法 | 前向时间(s) | 反向时间(s) | 总时间(s) | 通信量(GB) |
|---|---|---|---|---|
| SFT | 216.2 | 554.5 | 651.6 | 970.7 |
| 梯度 Prompt Tuning | 273.3 | 605.2 | 882.1 | 1116.2 |
| SecP-Tuning (FoT) | 174.0 | 0.0 | 174.1 | 205.4 |
| **SecP-Tuning (FoT+RFA)** | **54.2** | **0.0** | **55.2** | **56.5** |

| 方法 | SST-2 Acc | Yelp P. Acc | AG's News Acc | MRPC F1 | RTE Acc | 平均 |
|---|---|---|---|---|---|---|
| SFT | 85.39 | **91.82** | **86.36** | **77.35** | 58.60 | 79.90 |
| 梯度 Prompt Tuning | 68.23 | 61.02 | 84.81 | 51.61 | 54.69 | 64.07 |
| FoT+预训练提示 | **89.56** | 91.50 | 81.51 | 75.51 | **77.62** | **83.14** |
| SecP-Tuning | 88.11 | 85.23 | 81.27 | 75.33 | 52.95 | 76.58 |

### 关键发现

1. **效率提升巨大**：SecP-Tuning 在 LAN 环境下比 SFT 快约 12 倍、比梯度提示调优快约 16 倍；通信量分别降低 17 倍和 20 倍。反向传播和优化器开销被完全消除（0 秒、0GB）。
2. **精度可用**：在 few-shot 设置下，SecP-Tuning 在 SST-2 和 MRPC 等任务上接近甚至超越梯度提示调优，验证了隐私保护调优的可用性。在简单情感分类任务上（SST-2: 88.11 vs 68.23）显著优于梯度提示调优。
3. **唯一支持 AAS 部署**：SecP-Tuning 是唯一支持 "As-A-Service" 模式的方法——数据拥有者可通过 API 完成微调，模型开发者永远无法获取更新后的参数，杜绝了模型记忆攻击风险。
4. **Π_cosine 是 RFA 高效性的关键**：不使用高效余弦协议的 RFA 在短序列场景下甚至比原始 softmax 更慢，说明 Π_cosine 的设计至关重要。

## 亮点

- 首个 MPC 环境下的 LLM 提示调优框架，填补了 MPC-based 隐私保护微调的空白。
- "Server-Client"架构将损失和优化器计算卸载到数据拥有者本地明文执行，从架构层面消除反向传播开销。
- 隐私保护余弦协议 Π_cosine 巧妙利用三角恒等式实现单轮通信，是使 RFA 实际可行的关键贡献。
- 支持黑盒/API 式隐私调优，部署性优于所有梯度传递方案。

## 局限

- 仅在 RoBERTa_LARGE 上验证，未扩展到 GPT/LLaMA 级别的真正"大"模型，实际可扩展性存疑。
- RFA 对 softmax 的近似会引入精度损失，在某些任务上（Yelp P. 85.23 vs 91.82、RTE 52.95 vs 58.60）与 SFT 有较大差距。
- 半诚实威胁模型假设较弱，恶意参与者场景需额外的零知识证明等机制，开销更大。
- FoT 依赖 CMA-ES 等无梯度优化器，在高维参数空间中收敛性退化，需借助随机投影降维。

## 相关工作对比

| 方法 | 核心区别 |
|---|---|
| BlindTuner (Panzade et al., 2025) | 基于同态加密（HE）的隐私微调，单方加密计算开销大且非线性操作近似不精确；SecP-Tuning 基于 MPC 直接支持非线性操作 |
| PrivTuner (Li et al., 2024b) | 结合 LoRA 与全同态加密，减少参数但仍需反向传播的 HE 计算；SecP-Tuning 通过 FoT 完全消除反向传播 |
| DP-based PFT (Wang et al., 2024; Charles et al., 2024) | 差分隐私通过加噪提供统计级隐私保证(ε,δ)；MPC 提供密码学级理论保证，保护对象和保证强度不同 |

## 评分

| 维度 | 评分 |
|---|---|
| 新颖性 | ⭐⭐⭐⭐ |
| 有效性 | ⭐⭐⭐⭐ |
| 可复现性 | ⭐⭐⭐ |
| 实用性 | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](../../NeurIPS2025/llm_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](../../CVPR2026/llm_safety/fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[ICLR 2026\] SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA](she-lora_selective_homomorphic_encryption_for_federated_tuning_with_heterogeneou.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](../../ACL2026/llm_safety/privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](../../ACL2026/llm_safety/sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)

</div>

<!-- RELATED:END -->
