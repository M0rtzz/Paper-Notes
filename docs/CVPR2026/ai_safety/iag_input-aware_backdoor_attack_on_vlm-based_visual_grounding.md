---
title: >-
  [论文解读] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding
description: >-
  [CVPR 2026][AI安全][后门攻击] 提出IAG，首个针对VLM视觉定位的多目标后门攻击方法，通过文本条件U-Net动态生成输入感知触发器，将任意指定目标物体的语义信息嵌入视觉输入中，在12种设置下的11种达到最高攻击成功率。
tags:
  - CVPR 2026
  - AI安全
  - 后门攻击
  - 视觉定位
  - 多目标攻击
  - 输入感知触发器
  - VLM安全
---

# IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding

**会议**: CVPR 2026  
**arXiv**: [2508.09456](https://arxiv.org/abs/2508.09456)  
**代码**: https://github.com/lijunxian111/IAG  
**领域**: AI安全  
**关键词**: 后门攻击, 视觉定位, 多目标攻击, 输入感知触发器, VLM安全

## 一句话总结
提出IAG，首个针对VLM视觉定位的多目标后门攻击方法，通过文本条件U-Net动态生成输入感知触发器，将任意指定目标物体的语义信息嵌入视觉输入中，在12种设置下的11种达到最高攻击成功率。

## 研究背景与动机
1. **领域现状**: VLM-based视觉定位(Visual Grounding)已被广泛部署在GUI Agent、具身AI等系统中，用户通过自然语言指定目标物体让模型定位。HuggingFace等平台的开放模型分享使得恶意模型传播成为可能。
2. **现有痛点**: 现有VLM后门攻击(BadSem等)主要使用**静态触发器和固定目标**——只能攻击预定义的单一类别。但真实视觉定位场景中，物体种类和描述在不同图片间变化巨大，静态方案远不够用。
3. **核心矛盾**: 多目标后门攻击需要触发器能够**动态编码任意目标物体的语义信息**，同时保持不可察觉性和对干净样本的正常性能——这比单目标攻击困难得多。
4. **本文目标**: 实现首个多目标VLM视觉定位后门攻击——攻击者可指定图中**任意物体**让被攻击VLM定位之，无论用户查询的是什么。
5. **切入角度**: 利用文本条件U-Net作为触发器生成器，将目标物体描述编码为不可感知的视觉扰动，让VLM学会将这种扰动模式与目标定位关联。
6. **核心 idea**: 用文本条件U-Net动态生成在语义上编码攻击目标的不可感知触发器。

## 方法详解

### 整体框架
输入：干净图像$x$和攻击者指定的目标物体描述$o$。触发器生成：文本条件U-Net $\mathcal{G}_\phi$以$x$和$o$的嵌入$z_o$为条件生成触发器$r$，构造触发图像$x \oplus r$。后门注入：联合训练U-Net和VLM，使VLM在干净输入上正常工作，在触发输入上定位攻击目标。推理时攻击者只需提供触发图像（如注入广告链接的网页截图），用户查询任何内容VLM都会定位到攻击目标。

### 关键设计
1. **输入感知触发器生成器 (Input-aware Trigger Generator)**:
    - 功能：根据输入图像和目标物体描述动态生成不可感知的后门触发器。
    - 核心思路：采用文本条件U-Net(3个下采样块+1个中间块+3个上采样块)，中间块和上采样块后加入交叉注意力层融入目标文本嵌入$z_o$。$z_o$来自冻结的干净VLM嵌入层。输出触发器与原图大小相同，直接相加得到触发图像。
    - 设计动机：线性映射(如Imperio)无法建模目标物体与触发器之间的复杂非线性关系；浅层条件自编码器(如Marksman)存在信息瓶颈。U-Net的跨模态条件注入+跳连接能同时捕获全局上下文和精细视觉细节。

2. **联合训练目标 (Joint Training Objective)**:
    - 功能：平衡攻击效果、干净性能和触发器不可察觉性。
    - 核心思路：总损失$\mathcal{L} = \mathcal{L}_{LM} + \beta \cdot \mathcal{L}_{rec}$。$\mathcal{L}_{LM}$包含两部分——干净样本的正常定位损失+触发样本的攻击目标定位损失（标准token级交叉熵）。$\mathcal{L}_{rec} = \alpha_1 \mathcal{L}_{pix} + \alpha_2 \mathcal{L}_{LPIPS}$强制触发器视觉不可感知（像素L1+感知LPIPS）。$\alpha_1=1, \alpha_2=0.05, \beta=0.5$。
    - 设计动机：$\mathcal{L}_{LM}$的双部分确保模型在两种输入下都有正确行为，$\mathcal{L}_{rec}$的LPIPS保证触发器人眼不可见。三者联合训练让U-Net和VLM协同优化。

3. **攻击数据准备 (Attack Data Preparation)**:
    - 功能：系统化地从现有标注数据中构造少量投毒数据。
    - 核心思路：从视觉定位数据集中以投毒率$\alpha$随机采样子集。对每张图随机选择一个标注物体作为攻击目标$o$，用户查询$q$来自**非目标物体**的描述，答案$y^*$是攻击目标的位置。统一提示模板：Q: xxx <object>. A: <object>[<bbox>]。
    - 设计动机：利用现有多物体标注数据，不需要额外标注成本。查询和目标不同类确保攻击在语义上是"误导"而非正常行为。

### 损失函数 / 训练策略
在LLaVA-v1.5-7B上使用LoRA微调，投毒率$\alpha = 5\%$。联合训练U-Net($lr=5\times10^{-4}$)和VLM($lr=2\times10^{-5}$)。理论分析(Proposition 1)提供了攻击成功率的下界——成功概率随触发器范数$\varepsilon$和文本对齐度$\gamma$增长。

## 实验关键数据

### 主实验 (12种VLM×数据集组合)
| 设置 | IAG ASR@0.5 | 最强baseline | 超出 |
|------|------------|-------------|------|
| LLaVA + RefCOCO | 58.9% | Imperio 55.2% | +3.7% |
| LLaVA + F30k | 40.0% | Imperio 33.6% | +6.4% |
| InternVL + RefCOCO | 66.9% | Imperio 65.5% | +1.4% |
| InternVL + RefCOCO+ | 68.1% | Imperio 63.8% | +4.3% |
| Ferret + F30k | 53.8% | Imperio 48.1% | +5.7% |
| Ferret + RefCOCO | 48.9% | Imperio 35.6% | +13.3% |

干净精度下降: BA vs CA 差距 < 3% (如LLaVA-RefCOCO: BA 80.7% vs CA 82.1%)

### 消融实验
| 配置 | ASR | 说明 |
|------|-----|------|
| Full IAG | 58.9% | 完整模型 |
| 无LPIPS损失 | ASR提高但触发器可视 | 不可感知性受损 |
| 固定触发器 (One-to-N) | 3.2% | 无法多目标攻击 |
| 浅层自编码器 (Marksman) | 8.5% | 信息瓶颈限制 |
| 线性映射 (Imperio) | 55.2% | 较好但无法建模复杂关系 |

### 关键发现
- IAG在12种设置的11种中ASR最高——唯一的例外是Imperio在个别设置上略高。
- 与固定触发器(One-to-N:3-5%)相比，输入感知触发器提升了10-50%+ ASR。
- BA与CA差距极小(<3%)，说明后门模型在干净数据上几乎不受影响——极高的隐蔽性。
- 跨数据集和跨模型的迁移性也有验证，说明IAG学到的是通用漏洞。
- 对现有防御方法(如STRIP/Fine-pruning)仍保持鲁棒。

## 亮点与洞察
- **多目标后门攻击的形式化**：首次定义了VLM定位的多目标后门攻击问题——攻击者可指定任意物体而非固定类别。这揭示了比单目标攻击严重得多的安全威胁。
- **文本条件触发器的"语义注入"**：触发器不仅是扰动，还携带了目标物体的语义信息。这种设计让VLM的交叉注意力机制"看到"了目标物体的特征，即使目标在查询中未被提及。理论分析(Proposition 1)为此提供了严格的下界。
- **对GUI Agent/具身AI的安全警示**：在ShowUI Agent场景中也有效(25-35% ASR)，说明恶意网页可以引导Agent定位广告/恶意链接而非用户目标——这是非常现实的威胁。

## 局限与展望
- 攻击成功率在某些设置下仍较低（如RefCOCOg上47%，ShowUI上25-35%），对复杂表述和密集UI元素的攻击效果有限，距离分类后门的近100% ASR还有较大差距。
- 触发器生成需要访问干净VLM的嵌入层——虽然可用同架构的开源模型替代，但如果架构完全不同（如embedding维度不匹配）则不适用。
- 默认投毒率5%在需要大量干净数据的场景下可能不切实际。更低投毒率下的攻击效果有待验证。
- 本文纯攻击视角，没有提出有效防御方案。现有防御全部失效的结论虽然警示性强，但缺乏建设性——未来应同时研究针对输入感知触发器的检测方法。
- U-Net触发器生成器增加了额外的模型开销（3个下采样+3个上采样+交叉注意力），在部署受限场景下可能不切实际。
- 对攻击目标描述长度有限制（根据数据集设置最大长度），超长描述的攻击效果未知。

## 相关工作与启发
- **vs BadSem**: BadSem使用语义不对齐作为触发器但限于静态目标；IAG的输入感知设计支持任意目标切换。BadSem的设计假设（固定攻击类别）与视觉定位的开放词汇场景不匹配。
- **vs Imperio(输入感知分类攻击)**: Imperio是最强基线（RefCoco ASR 55.2 vs IAG 58.9），但在复杂场景(如ShowUI)差距拉大(16.0 vs 32.3)。Imperio的线性映射在简单场景下可行，但缺乏对复杂目标-触发器关系的建模能力。
- **vs Marksman(多目标分类攻击)**: Marksman用浅层条件自编码器，信息瓶颈限制了复杂语义控制，ASR仅8-33%，远低于IAG。
- **防御启示**: 结果暗示需要在VLM部署前进行更严格的安全审查。特别是对来源不明的微调模型，应开发针对输入感知触发器的检测方法——现有的谱特征/统计检测方法对输入自适应的扰动无效。
- **对GUI Agent安全的警示**: ShowUI场景的实验结果(ASR 25-35%)表明，恶意网页可以引导VLM赋能的Agent定位广告/恶意链接而非用户目标——这是非常现实的威胁。
- **对开源模型生态的启示**: HuggingFace等平台上的微调模型缺乏安全审查，IAG证明了只需5%的投毒数据就可以注入有效后门，这对开源模型信任机制提出了重要挑战。
- **理论贡献的价值**: Proposition 1提供了输入感知触发器优于固定触发器的数学解释——文本条件子空间使扰动方向与交叉注意力的接地特征对齐，提高了有效投影增益$m$和对齐度$\gamma$。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个VLM视觉定位多目标后门攻击，问题定义和解决方案都新颖，填补了VLM安全的重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 12种设置(3模型×5数据集)，含不可察觉性、防御鲁棒性、输入攻击对比、理论分析
- 写作质量: ⭐⭐⭐⭐ 威胁模型清晰，理论和实验结合好，问题形式化严谨
- 价值: ⭐⭐⭐⭐⭐ 揭示了VLM安全的重要盲区，对安全社区有重要警示作用，尤其对GUI Agent部署场景

<!-- RELATED:START -->

## 相关论文

- [BEAT: Visual Backdoor Attacks on VLM-based Embodied Agents via Contrastive Trigger Learning](../../ICLR2026/ai_safety/beat_visual_backdoor_attacks_on_vlm-based_embodied_agents_via_contrastive_trigge.md)
- [Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [V-Attack: Targeting Disentangled Value Features for Controllable Adversarial Attacks on LVLMs](v-attack_targeting_disentangled_value_features_for_controllable_adversarial_atta.md)
- [Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning](../../CVPR2025/ai_safety/infighting_in_the_dark_multi-label_backdoor_attack_in_federated_learning.md)
- [Towards Highly Transferable Vision-Language Attack via Semantic-Augmented Dynamic Contrastive Interaction](towards_highly_transferable_vision-language_attack_via_semantic-augmented_dynami.md)

<!-- RELATED:END -->
