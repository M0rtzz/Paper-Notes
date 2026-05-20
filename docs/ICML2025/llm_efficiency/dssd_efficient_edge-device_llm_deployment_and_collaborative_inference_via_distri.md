---
title: >-
  [论文解读] DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding
description: >-
  [ICML2025][LLM效率][推测解码] 提出分布式拆分推测解码（DSSD）框架，将推测解码的验证阶段拆分到设备端和边缘端，用一次下行传输（LLM的单个词表分布）替代多次上行传输（SLM的$\gamma$个词表分布），在保持推理质量不变的前提下大幅降低通信延迟。
tags:
  - "ICML2025"
  - "LLM效率"
  - "推测解码"
  - "边缘计算"
  - "LLM部署"
  - "设备-边缘协作"
  - "通信优化"
  - "分布式推理"
---

# DSSD: Efficient Edge-Device LLM Deployment and Collaborative Inference via Distributed Split Speculative Decoding

**会议**: ICML2025  
**arXiv**: [2507.12000](https://arxiv.org/abs/2507.12000)  
**作者**: Jiahong Ning, Ce Zheng, Tingting Yang
**代码**: [JasonNing96/DSSD-Efficient-Edge-Computing](https://github.com/JasonNing96/DSSD-Efficient-Edge-Computing)  
**领域**: LLM效率  
**关键词**: 推测解码, 边缘计算, LLM部署, 设备-边缘协作, 通信优化, 分布式推理

## 一句话总结

提出分布式拆分推测解码（DSSD）框架，将推测解码的验证阶段拆分到设备端和边缘端，用一次下行传输（LLM的单个词表分布）替代多次上行传输（SLM的$\gamma$个词表分布），在保持推理质量不变的前提下大幅降低通信延迟。

## 研究背景与动机

### 问题背景
大语言模型（LLM）在对话代理、机器翻译、代码生成等领域表现出色，但在实际部署中面临两难困境：
- **设备端**：内存、电量和算力严重受限，无法运行完整LLM
- **云端**：网络延迟不可预测，用户移动性导致连接中断，影响服务连续性
- **设备-边缘协作**：在设备部署小语言模型（SLM），在基站/边缘服务器部署LLM，通过推测解码实现高效协作推理

### 已有工作的不足

**Ding et al. (2024)**：基于查询难度路由的方案，通过将简单查询分配给SLM来降低成本，但**牺牲了LLM推理精度**

**Hao et al. (2024)**：基于token概率阈值的草稿-验证方法，在性能和成本间做权衡，同样**无法保证LLM级别的推理质量**

**Zhao et al. (2024) 的分布式推测解码（DSD）**：首次提出将草稿模型放设备端、目标模型放边缘端的分布式架构，但每个token都需要**上行传输完整的词表概率分布**给基站验证，通信载荷与词表大小$|\mathcal{V}|$线性相关，成为严重瓶颈

**Oh et al. (2024)**：提出跳过高接受概率token的上行传输和LLM推理，提升token吞吐量，但**依然牺牲推理精度**

### 核心动机
现有分布式推测解码的根本问题在于：验证阶段完全在边缘端执行，设备必须将SLM生成的所有$\gamma$个词表分布（每个维度$|\mathcal{V}|$）通过上行链路传输到基站。上行带宽通常远小于下行带宽，这成为延迟的主要来源。能否重新设计验证阶段的计算分配，将通信从"多次上行"转为"单次下行"？

## 方法详解

### 整体框架

DSSD延续设备端部署SLM（$M_q$）、边缘端部署LLM（$M_p$）的基本架构，但对推测解码的验证阶段进行了关键性的拆分重构。整个流程分为以下几个阶段：

1. **草稿阶段（设备端）**：SLM自回归生成$\gamma$个候选token $x_1, x_2, \ldots, x_\gamma$，同时保留每个token对应的词表概率分布$Q_i(x)$
2. **上行传输（精简版）**：设备仅向基站发送$\gamma$个token的词表索引（而非完整概率分布），通信载荷从$\gamma \times |\mathcal{V}|$降至$\gamma \times \lceil\log_2|\mathcal{V}|\rceil$
3. **边缘端部分验证**：LLM根据前缀和接收到的token计算$\gamma+1$个分布$P_1(x), \ldots, P_{\gamma+1}(x)$，但不执行完整验证
4. **下行传输**：边缘端仅需向设备发送一个关键的词表分布
5. **设备端完成验证**：设备端利用本地保留的SLM分布$Q_i(x)$和接收到的LLM分布，在本地完成token的接受/拒绝判断和重采样

### 关键设计：验证阶段的拆分

在标准推测解码中，验证过程需要同时访问SLM分布$Q_i(x)$和LLM分布$P_i(x)$来进行接受-拒绝采样：
- 对每个草稿token $x_i$，以概率$\min\left(1, \frac{P_i(x_i)}{Q_i(x_i)}\right)$接受
- 若被拒绝，从修正分布$\text{norm}\left(\max(0, P_i(x) - Q_i(x))\right)$重采样

在DSD中，$Q_i(x)$在设备端生成，$P_i(x)$在边缘端生成，因此必须将$Q_i(x)$上传，导致巨大的上行通信开销。

**DSSD的核心创新**在于重新分配验证计算：
- SLM分布$Q_i(x)$**始终留在设备端**，不需要上传
- 仅需从边缘端**下行传输LLM的分布信息**到设备端
- 设备端利用本地的$Q_i(x)$和接收到的$P_i(x)$在本地完成验证

这一设计利用了一个关键的不对称性：上行带宽远小于下行带宽。DSSD将通信方向从上行反转为下行，并将传输量从$\gamma$个分布减少为1个分布，实现了双重优化。

### 通信开销分析

| 方案 | 上行传输量 | 下行传输量 | 总传输量级 |
|------|-----------|-----------|-----------|
| DSD (Zhao et al.) | $\gamma \times \|\mathcal{V}\|$ (SLM分布) + token索引 | 验证结果 | $O(\gamma \cdot \|\mathcal{V}\|)$ |
| Oh et al. (2024) | 部分SLM分布（跳过部分传输） | 验证结果 | $O(k \cdot \|\mathcal{V}\|), k \leq \gamma$ |
| **DSSD（本文）** | **仅token索引** $\gamma \times \lceil\log_2\|\mathcal{V}\|\rceil$ | **单个LLM分布** $\|\mathcal{V}\|$ | $O(\|\mathcal{V}\|)$ |

DSSD将通信复杂度从$O(\gamma \cdot |\mathcal{V}|)$降低到$O(|\mathcal{V}|)$，减少了$\gamma$倍（$\gamma$为草稿长度，通常为3-10）。

### 推理质量保证

DSSD的输出分布与标准推测解码完全一致，即与直接使用LLM自回归解码等价。这是因为验证阶段的接受-拒绝机制在数学上保持不变，只是计算位置从边缘端移到了设备端。因此，DSSD在降低通信成本的同时**不牺牲任何推理精度**。

### 与草稿长度优化的结合

DSSD与草稿长度$\gamma$的优化是正交的。可以进一步结合Zhao et al. (2024)提出的自适应$\gamma$优化策略，在最小化端到端延迟的同时适配不同的网络条件和任务场景。

## 实验关键数据

### 端到端延迟对比

| 方法 | 推理质量 | 上行通信开销 | 端到端延迟 | 关键特点 |
|------|---------|------------|-----------|---------|
| DSD (Zhao et al., 2024) | 保持LLM质量 | 高（$\gamma$个词表分布） | 基线 | 完整SD验证在边缘端 |
| Oh et al. (2024) | 有损 | 中（跳过部分传输） | 较低 | 牺牲精度换延迟 |
| Ding et al. (2024) | 有损 | 低 | 低 | 查询路由，部分用SLM |
| **DSSD（本文）** | **保持LLM质量** | **极低（仅token索引）** | **最低** | **验证拆分，下行替代上行** |

### 通信效率提升分析

| 词表大小 | 草稿长度 $\gamma$ | DSD上行载荷 | DSSD上行载荷 | DSSD下行载荷 | 通信减少比 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32,000 | 3 | 96,000 floats | 3 indices | 32,000 floats | ~3x |
| 32,000 | 5 | 160,000 floats | 5 indices | 32,000 floats | ~5x |
| 32,000 | 10 | 320,000 floats | 10 indices | 32,000 floats | ~10x |
| 128,000 | 5 | 640,000 floats | 5 indices | 128,000 floats | ~5x |

当考虑上下行带宽不对称（上行通常为下行的1/3到1/10）时，实际延迟减少更为显著。

## 亮点与洞察

- **通信方向反转的巧妙设计**：将验证所需的信息流从"上行发送SLM分布"反转为"下行接收LLM分布"，巧妙利用了移动通信网络上下行带宽不对称的物理特性
- **无损推理质量**：与其他追求低延迟的方案（如跳过验证、查询路由）不同，DSSD严格保持推测解码的数学等价性，输出分布与LLM自回归解码完全一致
- **传输量从线性到常数**：将传输次数从$\gamma$次降为1次，通信复杂度与草稿长度解耦，使得更长的草稿序列成为可能而不增加通信负担
- **框架的通用性**：DSSD可以与现有的草稿长度优化、自适应推测解码等策略正交组合使用
- **实用性强**：方案不需要修改SLM或LLM的模型结构，仅改变通信协议和验证计算的分配

## 局限与展望

- **设备端验证计算开销**：将部分验证移到设备端增加了设备的计算负担，对于极度资源受限的终端设备（如IoT传感器）可能不适用
- **单次下行传输的假设**：论文假设仅需传输一个LLM分布即可完成验证，实际上对于某些验证策略可能需要额外信息交换
- **词表压缩未探索**：下行传输的$|\mathcal{V}|$维分布仍然较大，可以结合词表剪枝或Top-K稀疏化进一步压缩
- **多轮通信未考虑**：仅分析单轮推测解码的通信效率，未考虑多轮对话场景下的KV Cache同步和状态管理
- **实验规模有限**：论文中模型规模和场景覆盖有待扩展，缺乏真实移动网络环境下的部署验证
- **安全性与隐私**：设备端保留SLM分布可能涉及模型知识产权保护问题，需要进一步分析

## 相关工作与启发

- **Leviathan et al. (2023), Chen et al. (2023)**：推测解码的奠基工作，提出草稿-验证范式，本文在分布式场景下继承其核心接受-拒绝机制
- **Zhao et al. (2024) DSD**：首个分布式推测解码架构，将草稿模型和目标模型分别部署在设备和边缘端，但通信瓶颈未解决
- **Oh et al. (2024)**：通过跳过高概率token的传输来降低通信开销，但牺牲推理精度
- **Ding et al. (2024)**：基于查询路由的协作推理，简单查询交给SLM处理，复杂查询交给LLM，不保证LLM级推理质量
- **Hao et al. (2024)**：引入成本感知的草稿-验证方法，但性能-成本权衡不可避免
- **Shao & Li (2025)**：设备-边缘协作推理的通信优化，与本文均关注通信效率但技术路线不同

### 对后续研究的启发
- 验证阶段的拆分思想可推广到其他分布式推理场景（如联邦推理、多设备协同）
- 上下行带宽不对称性是移动场景设计的重要考量，未来工作可结合5G/6G网络特性进一步优化
- 推测解码与边缘计算的结合是LLM普适化部署的重要方向

## 评分

- 新颖性: ⭐⭐⭐⭐ — 验证阶段拆分和通信方向反转的设计思路新颖实用
- 实验充分度: ⭐⭐⭐ — 实验验证了核心主张，但缺乏大规模真实部署验证
- 写作质量: ⭐⭐⭐⭐ — 问题建模清晰，系统架构描述详尽，动机阐述到位
- 价值: ⭐⭐⭐⭐ — 解决了分布式LLM推理中的实际通信瓶颈，对边缘AI部署有直接参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] OmniDraft: A Cross-Vocabulary Online Adaptive Drafter for On-Device Speculative Decoding](../../NeurIPS2025/llm_efficiency/omnidraft_a_cross-vocabulary_online_adaptive_drafter_for_on-device_speculative_d.md)
- [\[ACL 2025\] Accelerating Speculative Decoding via Efficient Context-Aware Draft Generation](../../ACL2025/llm_efficiency/accelerating_speculative_decoding_via_efficient_context-aware_draft_generation.md)
- [\[ACL 2025\] SAM Decoding: Speculative Decoding via Suffix Automaton](../../ACL2025/llm_efficiency/sam_decoding_speculative_decoding_via_suffix_automaton.md)
- [\[NeurIPS 2025\] 3-Model Speculative Decoding (PyramidSD)](../../NeurIPS2025/llm_efficiency/3model_speculative_decoding.md)
- [\[ACL 2025\] Tetris: Optimal Draft Token Selection for Batch Speculative Decoding](../../ACL2025/llm_efficiency/tetris_optimal_draft_token_selection_for_batch_speculative_decoding.md)

</div>

<!-- RELATED:END -->
