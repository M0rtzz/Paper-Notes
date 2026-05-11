---
title: >-
  [论文解读] CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels
description: >-
  [CVPR 2026][图像分割][声呐图像分割] 提出CTFS，首个专为前视声呐图像设计的半监督语义分割框架，引入多教师协作机制（1个通用教师+2个声呐特异教师，分别模拟声学阴影和能量衰减物理特性），配合多视角伪标签可靠性评估（单教师内稳定性+跨教师间一致性），在仅2%标注下达62.32% mIoU，超越SOTA 5.08个百分点。
tags:
  - CVPR 2026
  - 图像分割
  - 声呐图像分割
  - 半监督学习
  - 多教师协作
  - 伪标签可靠性
  - 极少标注
---

# CTFS: Collaborative Teacher Framework for Forward-Looking Sonar Image Semantic Segmentation with Extremely Limited Labels

**会议**: CVPR 2026  
**arXiv**: [2603.21071](https://arxiv.org/abs/2603.21071)  
**代码**: 有  
**领域**: 分割 / 水下成像  
**关键词**: 声呐图像分割, 半监督学习, 多教师协作, 伪标签可靠性, 极少标注

## 一句话总结
提出CTFS，首个专为前视声呐图像设计的半监督语义分割框架，引入多教师协作机制（1个通用教师+2个声呐特异教师，分别模拟声学阴影和能量衰减物理特性），配合多视角伪标签可靠性评估（单教师内稳定性+跨教师间一致性），在仅2%标注下达62.32% mIoU，超越SOTA 5.08个百分点。

## 研究背景与动机

**领域现状**：前视声呐是水下感知的关键技术（救援、检测、生物调查），声呐图像面临独特挑战——严重斑点噪声、低纹理对比度、声学阴影、几何畸变。声呐数据高度专业化且标注昂贵。

**现有痛点**：(1) 传统教师-学生半监督方法(Mean Teacher/FixMatch)不考虑声呐特性→教师在噪声声呐图像上生成大量低质伪标签；(2) 某些方法使用10%标注时甚至低于监督基线——半监督框架在声呐域适得其反；(3) 缺乏针对声呐成像特性的教师模型设计。

**核心矛盾**：标准半监督方法的弱/强增强策略是为自然图像设计的（颜色抖动等），对声呐图像无效甚至有害——声呐的"噪声"来自物理成像过程而非图像处理。

**切入角度**：需要从声呐成像物理出发设计教师的增强策略——声学阴影和能量衰减是声呐图像两大物理特性。

**核心idea**：(1) 三教师协作——通用教师+声学阴影教师+能量衰减教师，按固定顺序轮流指导学生；(2) 多视角伪标签可靠性评估——单教师稳定性×跨教师一致性→可靠伪标签。

## 方法详解

### 整体框架
标注数据→学生监督训练($\mathcal{L}_{sup}$); 未标注数据→三教师轮流激活 $\phi(e)$: epoch E后按 $T_{general} \to T_{sonar\_a} \to T_{sonar\_b}$ 每3 epoch循环→弱增强输入教师生成伪标签→强增强输入学生→多视角可靠性评估→可靠性加权无监督损失($\mathcal{L}_{unsup}$)→总损失 $\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_u \cdot \mathcal{L}_{unsup}$。

### 关键设计

1. **多教师协作(CBTS)**:

    - **通用教师 $T_{general}$**：标准几何变换+颜色扰动做弱增强——学习通用语义表示
    - **声学阴影教师 $T_{sonar\_a}$**：模拟声呐被障碍物遮挡产生的阴影。扇形区域内按距离衰减：$I_o(x,y) = I_i(x,y) \times [1-\alpha(1-d/R)]$，$d$是到阴影起点距离，$R=0.2\times\min(H,W)$
    - **能量衰减教师 $T_{sonar\_b}$**：模拟声波在海水中传播的方向性能量衰减：$I_o(x,y) = I_i(x,y) \times (1-\gamma \times y/H)$，沿垂直方向线性衰减
    - **轮换策略**：三教师按固定顺序"通用→声学阴影→能量衰减"循环，每个教师通过EMA更新
    - 设计动机：(1) 通用教师提供通用语义学习基础；(2) 声呐教师让学生学到声呐特有的不变性——即使在声学阴影/能量衰减下也能正确分割；(3) 将每个教师映射到特定物理扰动保持了特征级语义一致性

2. **多视角可靠性评估(MVRA)**:

    - 功能：动态量化伪标签的可靠性，替代简单的单一置信度阈值
    - **网格化处理**：图像分为$m \times m$网格块→每块内像素预测概率求平均→降低计算量
    - **单教师内稳定性**：同一教师对图像的原始版和多个增强视图预测的余弦相似度：$r_{ij}^t = \frac{1}{N_{A_w^t}} \sum_k \cos(f_{ij}^{ot}, f_{ij}^{kt})$
    - **跨教师间一致性**：三教师对同图像预测的两两余弦相似度：$C_{ij} = \frac{1}{N_\mathcal{D}} \sum_{(p,q)} \cos(f_{ij}^{op}, f_{ij}^{oq})$
    - **综合可靠性**：$R_{ij} = \Pi(C_{ij}) \times \frac{1}{N_T}\sum_t r_{ij}^t$，其中惩罚项 $\Pi(C_{ij}) = \delta + (1-\delta)C_{ij}$→低一致性受惩罚
    - 网格级计算→复制到像素级
    - 设计动机：声呐图像特性导致单一教师置信度不可靠→多维度评估（自身稳定+多教师共识）才能有效过滤噪声伪标签

3. **可靠性引导的自适应约束**:

    - 无监督损失使用可靠性做**双重作用**：(1) 硬阈值过滤 $\Delta = \mathbb{1}[R_b^n > \psi]$；(2) 软权重加权 $\times R_b^n$
    - 设计动机：兼顾过滤和细粒度加权—高可靠区域强监督、低可靠区域弱监督或忽略

4. **FSSG新数据集**:

    - 渤海湾ROV搭载多波束前视声呐(Oculus M750d, 750kHz/1.2MHz)
    - 3761张图像，11类目标（含潜水员类），长尾分布
    - 2-15米距离、不同角度采集，增加多样性

### 损失函数
$\mathcal{L}_{total} = \mathcal{L}_{sup} + \lambda_u \cdot \mathcal{L}_{unsup}$，其中 $\mathcal{L}_{unsup} = \frac{1}{N_u}\sum_n \frac{\sum_b \text{CE}(p_n^s[b], p_n^t[b]) \times R_b^n \times \Delta}{N_p}$。

## 实验关键数据

### 主实验（mIoU%）

| 方法 | 编码器 | FLSMD 2% | FLSMD 5% | FLSMD 10% | FSSG 2% | FSSG 5% |
|------|--------|:---:|:---:|:---:|:---:|:---:|
| Labeled Only | DINOv2-S | 51.08 | 61.02 | 68.48 | 35.61 | 41.67 |
| AEL | RN-101 | 52.70 | 64.89 | 70.73 | 53.51 | 57.84 |
| UniMatch V2 | DINOv2-S | 57.24 | 66.49 | 69.81 | 58.78 | 61.21 |
| SemiVL | RN-101 | 53.38 | 65.16 | 70.02 | 58.85 | 62.19 |
| **CTFS (Ours)** | DINOv2-S | **62.32** | **68.08** | **72.27** | **59.53** | **65.12** |

2%标注下比UniMatch V2高**5.08%** mIoU。

### 消融实验

| 配置 | FLSMD 2% mIoU |
|------|:---:|
| 仅通用教师 (baseline) | ~57 |
| +声学阴影教师 | ~59 |
| +能量衰减教师 | ~60 |
| +MVRA可靠性评估 | ~62 |
| **完整CTFS** | **62.32** |

### 尾类性能
在钩子(hook)、洗发水瓶(shampoo-bottle)、轮胎(tire)、水下机器人(ROV)等少样本类别上改进最显著→多教师协作增强了少见类别的表示学习。

### 关键发现
- **声呐特异教师每个都独立贡献**：声学阴影教师+~2%，能量衰减教师+~1%——物理增强确实帮助学生理解声呐成像
- **MVRA比单一置信度阈值更可靠**——多视角评估有效过滤了声呐噪声导致的虚假高置信伪标签
- **尾类改进最显著**——多教师提供的多角度监督信号对少样本类别尤为重要
- **CTFS在所有标注比例下一致SOTA**——2%/5%/10%都领先，证明框架的鲁棒性

## 亮点与洞察
- **声学物理模拟作为数据增强的范例**：将声学阴影和能量衰减这两个声呐成像的核心物理过程显式建模为教师的弱增强→这是将领域物理知识注入半监督学习的优秀范例。类似思路可迁移到SAR(散射)、红外(热传导)、超声(声阻抗)等成像模态
- **首个声呐半监督分割框架**：填补了水下声呐感知领域的半监督学习空白。声呐数据标注极其昂贵(需专业知识)，半监督方法对该领域有巨大实用价值
- **多教师轮换的"从通用到特异"课程**：general→sonar_a→sonar_b的循环让学生从通用能力逐步发展到域特异化理解——类似于从粗到精的学习过程
- **FSSG新数据集贡献**：3761图+11类+语义分割标注，缓解了声呐数据集的稀缺问题

## 局限与展望
- 三教师轮换的固定周期(每3 epoch)和固定顺序可能非最优——自适应调度(基于学生性能)或并行多教师可能更好
- 声学物理模拟是简化近似（扇形阴影、线性衰减）——更精确的声学仿真(考虑频率/反射面材质等)可进一步提升
- 仅室内水池/浅海场景验证——深海复杂环境(水流干扰/多路径反射/生物附着)的泛化有待评估
- FSSG数据集虽有价值但规模仍有限(3761图)——可通过声学仿真生成更多合成数据
- 可探索将多教师协作与Fundation Model(如DINOv2)的特性更深度结合

## 相关工作与启发
- **vs Mean Teacher/UniMatch V2**: 通用半监督方法不考虑声呐特殊性→在声呐域效果有限。CTFS的声呐特异教师直接提升了在声呐域的适应性
- **vs Dual Teacher**: 使用两个结构不同的教师但不针对声呐物理特性设计增强
- **vs FLSMD数据集**: CTFS在此基准上大幅超越所有现有方法
- **vs Beyond-Pixels/SemiVL**: 这些方法在声呐上性能甚至低于监督基线——验证了声呐域半监督的特殊困难
- **启发**：任何具有独特成像物理的领域(SAR/红外/超声)都可借鉴"物理增强教师"的思路设计域特异的半监督框架

## 评分
- 新颖性: ⭐⭐⭐⭐ 声呐物理教师+多视角可靠性评估有创新，首个声呐半监督分割框架
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多标注比例、尾类分析、充分消融
- 写作质量: ⭐⭐⭐⭐ 声呐物理背景和增强设计描述清晰
- 价值: ⭐⭐⭐⭐ 对水下感知领域有实际应用价值，物理增强教师思路有通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[CVPR 2026\] SemiTooth: a Generalizable Semi-supervised Framework for Multi-Source Tooth Segmentation](semitooth_a_generalizable_semi-supervised_framework_for_multi-source_tooth_segme.md)
- [\[CVPR 2026\] Brewing Stronger Features: Dual-Teacher Distillation for Multispectral Earth Observation](brewing_stronger_features_dual-teacher_distillation_for_multispectral_earth_obse.md)
- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[CVPR 2026\] Love Me, Love My Label: Rethinking the Role of Labels in Prompt Retrieval for Visual In-Context Learning](love_me_love_my_label_rethinking_the_role_of_labels_in_prompt_retrieval_for_visu.md)

</div>

<!-- RELATED:END -->
