---
title: >-
  CVPR2026 遥感方向 16篇论文解读
description: >-
  16篇CVPR2026 遥感方向论文深度解读，每篇5分钟读懂核心思想。每篇笔记含一句话总结、背景动机、方法详解、实验数据、亮点洞察与局限性分析。
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛰️ 遥感

**📷 CVPR2026** · **16** 篇论文解读

**[ACPV-Net: All-Class Polygonal Vectorization for Seamless Vector Map Generation from Aerial Imagery](acpv-net_all-class_polygonal_vectorization_for_seamless_vector_map_generation_fr.md)**

:   提出 ACPV-Net，首个从航空影像一次性生成拓扑一致的全类别多边形矢量地图的框架，通过语义监督条件化扩散模型生成顶点热图，并借助命题驱动的 PSLG 重建确保零间隙/零重叠。

**[Asking like Socrates: Socrates helps VLMs understand remote sensing images](asking_like_socrates_socrates_helps_vlms_understand_remote_sensing_images.md)**

:   揭示遥感VLM中的"伪推理"现象（显式推理链反而导致性能下降），归因于"一瞥效应"（单次粗浅感知不足），提出RS-EoT(Evidence-of-Thought)迭代证据搜索范式，通过SocraticAgent自博弈合成推理轨迹做SFT冷启动，再用两阶段渐进RL（grounding→VQA）增强和泛化，RS-EoT-7B在多个遥感VQA和grounding基准上达SOTA。

**[AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_vision-language_instruction_from_offline_teacher_to_prompt-tuned_ne.md)**

:   提出 AVION 知识蒸馏框架，通过 LLM 生成语义丰富的文本原型和视觉-文本双侧提示调优，解决遥感 VLM 适配中的语义贫乏和视觉刚性问题，在少样本分类、基类到新类泛化和跨模态检索上全面超越 SOTA。

**[AVION: Aerial Vision-Language Instruction from Offline Teacher to Prompt-Tuned Network](avion_aerial_visionlanguage_instruction_from_offli.md)**

:   AVION 提出一种知识蒸馏框架，通过 LLM 生成语义丰富的遥感文本原型作为 Teacher 监督、同时在 Student 的视觉和文本编码器中注入可学习 prompt，实现三维度对齐蒸馏，在少样本分类和跨模态检索上显著优于现有 PEFT 方法。

**[Cross-modal Fuzzy Alignment Network for Text-Aerial Person Retrieval and A Large-scale Benchmark](cross-modal_fuzzy_alignment_network_for_text-aerial_person_retrieval_and_a_large.md)**

:   提出跨模态模糊对齐网络 CFAN，利用模糊逻辑量化 token 级可靠性实现精细对齐，并引入地面视图作为桥接代理缓解航拍图像与文本的语义鸿沟，同时构建了大规模文本-航拍行人检索基准 AERI-PEDES。

**[Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)**

:   首次将光谱压缩成像（SCI）从图像级推进到视频级重建，构建首个高质量动态高光谱数据集 DynaSpec（30 序列/300 帧），提出 PG-SVRT 通过空间-然后-时间注意力 + 桥接 token 实现 41.52dB PSNR 和最优时间一致性，且 FLOPs（28.18G）低于多个图像级 SOTA。

**[GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](geoflow_real-time_fine-grained_cross-view_geolocalization_via_iterative_flow_pre.md)**

:   提出GeoFlow，将精细跨视图地理定位(FG-CVG)重新表述为概率位移回归——模型学习从任意假设位置到真实位置的位移场(距离+方向的概率分布)，配合迭代精化采样(IRS)算法让多个随机假设从不同起点"流向"共识位置，以7.8×更少参数和4×更少计算量实现29FPS实时推理+竞争性定位精度。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_co.md)**

:   研究多颗 LEO 卫星联合服务多天线地面用户的下行传输，提出联合非相干传输和流级传输两种模式，通过 WMMSE 框架设计预编码器，并利用匈牙利算法进行流-卫星关联，在降低前传开销的同时保持接近最优的频谱效率。

**[Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)**

:   提出面向多天线地面用户的分布式LEO卫星下行链路两种传输方案（联合传输 & 流式传输），通过基于统计CSI的WMMSE预编码设计和基于匈牙利算法的流-卫星关联策略，在无需卫星间相位同步的前提下实现了高频谱效率与低前传开销的灵活折中。

**[Lumosaic: Hyperspectral Video via Active Illumination and Coded-Exposure Pixels](lumosaic_hyperspectral_video_via_active_illumination_and_coded-exposure_pixels.md)**

:   提出Lumosaic主动高光谱视频系统，将12个窄带LED阵列与编码曝光像素（CEP）相机在微秒级同步，在每帧158个子帧内联合编码空间-时间-光谱信息，实现30fps VGA分辨率31通道（400–700nm）运动鲁棒高光谱视频重建，PSNR比被动快照系统高10+dB。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera.md)**

:   提出MetaSpectra+，首个在全可见光谱（250nm带宽）上工作的多功能超表面成像系统，通过双层超表面实现分束和色散精确控制，单次快照同时获取高光谱数据立方体与HDR/偏振图像，在基准数据集上PSNR达33.31dB且系统总光程长度仅17mm。

**[MetaSpectra+: A Compact Broadband Metasurface Camera for Snapshot Hyperspectral+ Imaging](metaspectra_a_compact_broadband_metasurface_camera_for_snapshot_hyperspectral_im.md)**

:   MetaSpectra+ 提出超表面-折射透镜混合光学范式，通过双层超表面独立控制4通道色散/曝光/偏振，实现250nm宽带、17mm最短光程的快照式高光谱+HDR/偏振多功能成像，在KAUST基准上PSNR达33.31dB全面超越现有快照高光谱系统。

**[No Labels, No Look-Ahead: Unsupervised Online Video Stabilization with Classical Priors](no_labels_no_look-ahead_unsupervised_online_video_stabilization_with_classical_p.md)**

:   提出无监督在线视频稳定框架 LightStab，通过经典三阶段管线（运动估计→运动传播→运动补偿）搭配多线程异步缓冲，在 5 个基准数据集上首次让在线方法全面媲美离线 SOTA，并发布首个包含可见光和红外的多模态无人机航拍稳定测试集 UAV-Test。

**[Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments](olbedo_an_albedo_and_shading_aerial_dataset_for_large-scale_outdoor_environments.md)**

:   Olbedo 提出首个大规模真实航拍反照率-着色分解数据集（5664 张 UAV 图像、4 种地貌、跨年多光照），通过物理逆渲染管线生成多视图一致的伪真值标注，证明合成预训练+Olbedo LoRA 微调可以显著提升室外反照率预测并支持重光照/材质编辑/场景变化分析等下游应用。

**[RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)**

:   提出首个面向恶劣天气和传感器噪声的OSM-based度量级跨视角定位基准CV-RHO（270万+ 图像），并设计双分支Pin-Pan架构RHO模型，结合全景去畸变（SUM）和位置-朝向融合（POF）机制，在多种退化条件下将定位性能提升高达20%。

**[SDF-Net: Structure-Aware Disentangled Feature Learning for Optical-SAR Ship Re-identification](sdfnet_structureaware_disentangled_feature_learnin.md)**

:   提出SDF-Net——物理引导的结构感知解耦特征学习网络，通过中间层梯度能量提取几何结构一致性(SCL)和终端层共享/模态专用特征解耦(DFL)+无参数加法融合，在HOSS-ReID上mAP达60.9%(+3.5% vs SOTA TransOSS)。
