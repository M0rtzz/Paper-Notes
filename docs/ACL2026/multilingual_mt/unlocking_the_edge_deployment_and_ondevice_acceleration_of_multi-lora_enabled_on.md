---
title: >-
  [论文解读] Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration
description: >-
  [ACL 2026][多语言/翻译][端侧LLM部署] 本文提出面向三星 Galaxy S24/S25 的端侧 LLM 部署框架，通过 LoRA 权重作为运行时输入实现动态任务切换、多流并发 token 生成减少风格变体延迟达 6 倍、无草稿模型的 Dynamic Self-Speculative Decod…
tags:
  - "ACL 2026"
  - "多语言/翻译"
  - "端侧LLM部署"
  - "Multi-LoRA"
  - "投机解码"
  - "并发token生成"
  - "INT4量化"
---

# Unlocking the Edge: Multi-LoRA On-Device Deployment and Acceleration

**会议**: ACL 2026  
**arXiv**: [2604.18655](https://arxiv.org/abs/2604.18655)  
**代码**: 无（Samsung内部系统）  
**领域**: 多语言翻译  
**关键词**: 端侧LLM部署, Multi-LoRA, 投机解码, 并发token生成, INT4量化

## 一句话总结

本文提出面向三星 Galaxy S24/S25 的端侧 LLM 部署框架，通过 LoRA 权重作为运行时输入实现动态任务切换、多流并发 token 生成减少风格变体延迟达 6 倍、无草稿模型的 Dynamic Self-Speculative Decoding 加速解码达 2.3 倍，在 9 语言 8 任务上实现 4-6 倍整体优化。

## 研究背景与动机

**领域现状**：在移动设备上部署 LLM 可提供隐私、低延迟和离线能力，但面临内存/延迟/运行时灵活性的严格约束。LoRA 是主流的高效微调方法，但传统做法是静态合并权重，无法在端侧实现动态任务切换。

**现有痛点**：(1) 服务端可以灵活加载/切换 LoRA，端侧必须使用冻结推理图，无法重编译或动态加载；(2) 风格变体生成（如同时输出正式/礼貌/幽默回复）需要顺序运行 8 次，延迟 8 倍；(3) 自回归解码逐 token 生成是端侧延迟的主要瓶颈，现有投机解码需要额外草稿模型占用稀缺内存。

**核心矛盾**：灵活的模型开发（多LoRA/多用例）vs 端侧推理图的不可变性——需要根本性地重新设计端侧可适应 LLM 的工程架构。

**本文目标**：在商用手机（Galaxy S24/S25）的 Qualcomm NPU 上实现实时、多语言、多用例的 LLM 推理。

**切入角度**：三个层面的创新——(1) LoRA 权重作为冻结图的运行时输入；(2) 利用共享 KV-cache 的多流 token 生成；(3) 基于前缀调优的无草稿投机解码。

**核心 idea**：将所有用例特定的知识封装在轻量 LoRA 权重中作为推理图的外部输入，配合并发解码和自投机策略，在单一冻结模型上实现多用例端侧 LLM。

## 方法详解

### 整体框架

基于 1B/3B 参数的 LLaMA 模型，INT4 量化后部署在 Qualcomm SM8650/SM8750 NPU 上。三种 Multi-LoRA 方案中选择"LoRA 权重作为输入"方案（最优延迟和内存）。配合 CTG（Concurrent Token Generation）和 DS2D（Dynamic Self-Speculative Decoding）加速。

### 关键设计

1. **LoRA 权重作为运行时输入**:

    - 功能：在单一冻结推理图上实现任意 LoRA 的即插即用切换
    - 核心思路：创建带 LoRA 权重占位符的基础 LLM 图，推理时将 LoRA 权重与 token 一起作为输入传入。所有 LoRA 需具有相同维度，通过占位符机制实现运行时替换
    - 设计动机：对比三种方案——共享权重的多图（内存好但需切换图）、单图多 LoRA+掩码（延迟好但内存大）、权重作为输入（两者兼得），第三种方案在多用例场景下可扩展性最优

2. **并发 Token 生成 (CTG)**:

    - 功能：将 8 种风格变体的生成从顺序 8 次压缩为单次前向传播
    - 核心思路：利用所有用例共享冻结图和 KV-cache，通过修改首 token 采样的掩码方案同时生成 8 个不同输出流。风格差异通常由首 token 驱动，因此只需在首 token 采样时分叉
    - 设计动机：Smart-Reply 等应用需要同时提供 8 种回复选项，CTG 实现 6 倍延迟/内存减少，无需修改模型二进制或推理图

3. **Dynamic Self-Speculative Decoding (DS2D)**:

    - 功能：无需额外草稿模型即可实现半自回归解码加速
    - 核心思路：基于前缀调优为 LLM 添加 $m$ 个 forecast embeddings，使模型在一步中预测 $1+m$ 个 token（第一个来自冻结 LLM 分布确保一致性，其余为低保真草稿需验证）。用树形分支结构扩展候选，选择最优分支配置以匹配硬件友好的输入大小（32的倍数）
    - 设计动机：传统投机解码的草稿模型需要额外内存（端侧稀缺），基于前缀调优的方法只增加可忽略的参数量且与冻结图完全兼容

### 损失函数 / 训练策略

基础模型和 LoRA 的训练细节未公开（三星专有）。DS2D 的 forecast embeddings 通过前缀调优训练。INT4 量化采用 QAT（量化感知训练），混合精度策略。

## 实验关键数据

### 主实验

**3B LLM 在 GS25 Ultra 上的性能**

| 用例 | 无DS2D解码时间(ms) | 有DS2D解码时间(ms) | 加速比 |
|------|------------------|------------------|--------|
| Correction | 50.17 | 22.30 | 2.25x |
| Composer | 53.23 | 28.57 | 1.86x |
| Style | 50.42 | 25.21 | 2.00x |

### 消融实验

**CTG 延迟分析（1B模型）**

| 流数 | Prefill(ms) | AR(ms) | 总时间(ms) | 公式 |
|------|------------|--------|-----------|------|
| 1流×8次 | 40 | 23 | 174 | (23×8)+40 |
| 8流并发 | 40 | 23 | 63 | 23+40 |

### 关键发现

- LoRA-as-input 方案在内存和延迟上均优于其他两种方案——3B模型峰值内存仅 2.5GB
- CTG 将 8 流生成从 174ms 压缩到 63ms（2.76x 加速），无需任何模型修改
- DS2D 在不同用例上实现 1.86-2.25x 解码加速
- INT4 量化后 6 种语言的任务准确率保持在 90%+

## 亮点与洞察

- LoRA 权重作为运行时输入的设计简洁而高效——将适应性从"编译时"移到了"运行时"，这个思路对任何冻结图部署场景都有参考价值
- CTG 利用"风格差异通常由首token驱动"的洞察非常实用——在实际产品中，不同风格的回复确实在开头分叉
- 这是少见的从工程实现角度完整描述端侧LLM部署的论文，对工业界有直接参考价值

## 局限与展望

- 仅在三星自有硬件和专有模型上验证
- 基础模型和LoRA训练细节未公开，可复现性受限
- DS2D的树形分支搜索增加了工程复杂度
- 未与其他端侧LLM方案（如MLC-LLM、llama.cpp）做直接对比

## 相关工作与启发

- **vs QLoRA**: 关注训练时量化，本文关注部署时的动态LoRA切换
- **vs Medusa/Eagle**: 需要额外的投机头或草稿模型，DS2D仅需轻量前缀调优
- **vs MobiLlama**: 关注模型效率，本文关注端侧部署的工程优化全栈

## 评分

- 新颖性: ⭐⭐⭐⭐ LoRA-as-input和CTG设计新颖，DS2D基于已有方法改进
- 实验充分度: ⭐⭐⭐ 在商用设备上的真实性能数据有价值，但缺少与外部方法的对比
- 写作质量: ⭐⭐⭐ 工程细节丰富但学术写作可改善
- 价值: ⭐⭐⭐⭐ 对端侧LLM部署有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] RouteLMT: Learned Sample Routing for Hybrid LLM Translation Deployment](routelmt_learned_sample_routing_for_hybrid_llm_translation_deployment.md)
- [\[ACL 2026\] FairQE: Multi-Agent Framework for Mitigating Gender Bias in Translation Quality Estimation](fairqe_multi-agent_framework_for_mitigating_gender_bias_in_translation_quality_e.md)
- [\[ACL 2026\] Efficient Low-Resource Language Adaptation via Multi-Source Dynamic Logit Fusion](efficient_low-resource_language_adaptation_via_multi-source_dynamic_logit_fusion.md)
- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ACL 2025\] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](../../ACL2025/multilingual_mt/m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)

</div>

<!-- RELATED:END -->
