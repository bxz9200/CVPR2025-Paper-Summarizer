# AI-Generated Paper Summaries

*Summaries of papers in the downloaded_papers folder*

Generated on: 2025-03-19 20:28:47

## Doc VLM  Make Your VLM an Efficient Reader

### Original Abstract

Vision-Language Models (VLMs) excel in diverse visual tasks but face challenges in document understanding, which requires fine-grained text processing. While typical vi- sual tasks perform well with low-resolution inputs, reading- intensive applications demand high-resolution, resulting in significant computational overhead. Using OCR-extracted text in VLM prompts partially addresses this issue but un- derperforms compared to full-resolution counterpart, as it lacks the complete visual context needed for optimal perfor- mance. We introduce DocVLM , a method that integrates an OCR-based modality into VLMs to enhance document pro- cessing while preserving original weights. Our approach employs an OCR encoder to capture textual content and layout, compressing these into a compact set of learned queries incorporated into the VLM. Comprehensive eval- uations across leading VLMs show that DocVLM signifi- cantly reduces reliance on high-resolution images for docu- ment understanding. In limited-token regimes (448 ×448), DocVLM with 64 learned queries improves DocVQA re- sults from 56.0% to 86.6% when integrated with InternVL2 and from 84.4% to 91.2% with Qwen2-VL. In LLaVA- OneVision, DocVLM achieves improved results while us- ing 80% less image tokens. The reduced token usage al- lows processing multiple pages effectively, showing impres- sive zero-shot results on DUDE and state-of-the-art perfor- mance on MP-DocVQA, highlighting DocVLM’s potential for applications requiring high-performance and efficiency.

### Summary

Vision-Language Models (VLMs) excel in diverse visual tasks but face challenges in document understanding, which requires fine-grained text processing. Our approach employs an OCR encoder to capture textual content and layout, compressing these into a compact set of learned queries incorporated into the VLM. The reduced token usage al- lows processing multiple pages effectively, showing impres- sive zero-shot results on DUDE and state-of-the-art perfor- mance on MP-DocVQA, highlighting DocVLM’s potential for applications requiring high-performance and efficiency.

---

## Top V  Compatible Token Pruning with Inference Time

### Original Abstract

The emergence of Mixture of Experts (MoE) LLMs has significantly advanced the develop- ment of language models. Compared to tra- ditional LLMs, MoE LLMs outperform tradi- tional LLMs by achieving higher performance with considerably fewer activated parameters. Despite this efficiency, their enormous param- eter size still leads to high deployment costs. In this paper, we introduce a two-stage com- pression method tailored for MoE to reduce the model size and decrease the computational cost. First, in the inter-expert pruning stage, we analyze the importance of each layer and propose the Layer-wise Genetic Search and Block-wise KT-Reception Field with the non- uniform pruning ratio to prune the individual expert. Second, in the intra-expert decompo- sition stage, we apply the low-rank decom- position to further compress the parameters within the remaining experts. Extensive exper- iments on Qwen1.5-MoE-A2.7B, DeepSeek- V2-Lite, and Mixtral-8 ×7B demonstrate that our proposed methods can both reduce the model size and enhance inference efficiency while maintaining performance in various zero- shot tasks. The code will be available at https: //github.com/xiaochengsky/MoEI-2.git

### Summary

The emergence of Mixture of Experts (MoE) LLMs has significantly advanced the develop- ment of language models. First, in the inter-expert pruning stage, we analyze the importance of each layer and propose the Layer-wise Genetic Search and Block-wise KT-Reception Field with the non- uniform pruning ratio to prune the individual expert. The code will be available at https: //github.com/xiaochengsky/MoEI-2.git

---

## Lifelong Knowledge Editing for Vision Language Mod

### Original Abstract

Model editing aims to correct inaccurate knowledge, up- date outdated information, and incorporate new data into Large Language Models (LLMs) without the need for re- training. This task poses challenges in lifelong scenarios where edits must be continuously applied for real-world applications. While some editors demonstrate strong ro- bustness for lifelong editing in pure LLMs, Vision LLMs (VLLMs), which incorporate an additional vision modal- ity, are not directly adaptable to existing LLM editors. In this paper, we propose LiveEdit, a Li felong v ision language mode l Edit to bridge the gap between lifelong LLM edit- ing and VLLMs. We begin by training an editing expert generator to independently produce low-rank experts for each editing instance, with the goal of correcting the rel- evant responses of the VLLM. A hard filtering mechanism is developed to utilize visual semantic knowledge, thereby coarsely eliminating visually irrelevant experts for input queries during the inference stage of the post-edited model. Finally, to integrate visually relevant experts, we introduce a soft routing mechanism based on textual semantic rele- vance to achieve multi-expert fusion. For evaluation, we es- tablish a benchmark for lifelong VLLM editing. Extensive experiments demonstrate that LiveEdit offers significant ad- vantages in lifelong VLLM editing scenarios. Further ex- periments validate the rationality and effectiveness of each module design in LiveEdit.1

### Summary

Model editing aims to correct inaccurate knowledge, up- date outdated information, and incorporate new data into Large Language Models (LLMs) without the need for re- training. A hard filtering mechanism is developed to utilize visual semantic knowledge, thereby coarsely eliminating visually irrelevant experts for input queries during the inference stage of the post-edited model. Further ex- periments validate the rationality and effectiveness of each module design in LiveEdit.1

---

## Global Local Tree Search in VLMs for 3D Indoor Sce

### Original Abstract

In this paper, we propose a novel model called SGFormer, Semantic G raph TransFormer for point cloud-based 3D scene graph generation. The task aims to parse a point cloud-based scene into a semantic structural graph, with the core chal- lenge of modeling the complex global structure. Existing methods based on graph convolutional networks (GCNs) suf- fer from the over-smoothing dilemma and can only prop- agate information from limited neighboring nodes. In con- trast, SGFormer uses Transformer layers as the base build- ing block to allow global information passing, with two types of newly-designed layers tailored for the 3D scene graph generation task. Specifically, we introduce the graph embed- ding layer to best utilize the global information in graph edges while maintaining comparable computation costs. Fur- thermore, we propose the semantic injection layer to lever- age linguistic knowledge from large-scale language model (i.e., ChatGPT), to enhance objects’ visual features. We benchmark our SGFormer on the established 3DSSG dataset and achieve a 40.94% absolute improvement in relation- ship prediction’s R@50 and an 88.36% boost on the sub- set with complex scenes over the state-of-the-art. Our anal- yses further show SGFormer’s superiority in the long-tail and zero-shot scenarios. Our source code is available at https://github.com/Andy20178/SGFormer.

### Summary

In this paper, we propose a novel model called SGFormer, Semantic G raph TransFormer for point cloud-based 3D scene graph generation. Specifically, we introduce the graph embed- ding layer to best utilize the global information in graph edges while maintaining comparable computation costs. Our source code is available at https://github.com/Andy20178/SGFormer.

---

## PARC  A Quantitative Framework Uncovering the Symm

### Original Abstract

In recent years, the data collected for artificial intel- ligence has grown to an unmanageable amount. Partic- ularly within industrial applications, such as autonomous vehicles, model training computation budgets are being ex- ceeded while model performance is saturating – and yet more data continues to pour in. To navigate the flood of data, we propose a framework to select the most semanti- cally diverse and important dataset portion. Then, we fur- ther semantically enrich it by discovering meaningful new data from a massive unlabeled data pool. Importantly, we can provide explainability by leveraging foundation mod- els to generate semantics for every data point. We quanti- tatively show that our Semantic Selection and Enrichment framework (SSE) can a) successfully maintain model per- formance with a smaller training dataset and b) improve model performance by enriching the smaller dataset with- out exceeding the original dataset size. Consequently, we demonstrate that semantic diversity is imperative for opti- mal data selection and model performance.

### Summary

In recent years, the data collected for artificial intel- ligence has grown to an unmanageable amount. Then, we fur- ther semantically enrich it by discovering meaningful new data from a massive unlabeled data pool. Consequently, we demonstrate that semantic diversity is imperative for opti- mal data selection and model performance.

---

## Forensics Bench  A Comprehensive Forgery Detection

### Original Abstract

Compositional reasoning capabilities are usually considered as fundamental skills to characterize human perception. Recent studies show that cur- rent Vision Language Models (VLMs) surpris- ingly lack sufficient knowledge with respect to such capabilities. To this end, we propose to thor- oughly diagnose the composition representations encoded by VLMs, systematically revealing the potential cause for this weakness. Specifically, we propose evaluation methods from a novel game- theoretic view to assess the vulnerability of VLMs on different aspects of compositional understand- ing,e.g., relations and attributes. Extensive exper- imental results demonstrate and validate several insights to understand the incapabilities of VLMs on compositional reasoning, which provide use- ful and reliable guidance for future studies. The deliverables will be updated here.

### Summary

Compositional reasoning capabilities are usually considered as fundamental skills to characterize human perception. Specifically, we propose evaluation methods from a novel game- theoretic view to assess the vulnerability of VLMs on different aspects of compositional understand- ing,e.g., relations and attributes. The deliverables will be updated here.

---

## MASH VLM  Mitigating Action Scene Hallucination in

### Original Abstract

In this work, we tackle the challenging problem of unsu- pervised video domain adaptation (UVDA) for action recog- nition. We specifically focus on scenarios with a substantial domain gap, in contrast to existing works primarily deal with small domain gaps between labeled source domains and unlabeled target domains. To establish a more realis- tic setting, we introduce a novel UVDA scenario, denoted as Kinetics →BABEL, with a more considerable domain gap in terms of both temporal dynamics and background shifts. To tackle the temporal shift, i.e., action duration difference between the source and target domains, we pro- pose a global-local view alignment approach. To mitigate the background shift, we propose to learn temporal order sensitive representations by temporal order learning and background invariant representations by background aug- mentation. We empirically validate that the proposed method shows significant improvement over the existing methods on the Kinetics →BABEL dataset with a large domain gap. *Equally contributed first authors. †Corresponding authors.The code is available at https://github.com/KHU- VLL/GLAD .

### Summary

In this work, we tackle the challenging problem of unsu- pervised video domain adaptation (UVDA) for action recog- nition. To mitigate the background shift, we propose to learn temporal order sensitive representations by temporal order learning and background invariant representations by background aug- mentation. †Corresponding authors.The code is available at https://github.com/KHU- VLL/GLAD .

---

## Vision Zip  Longer is Better but Not Necessary in V

### Original Abstract

Recent advancements in vision-language models have en- hanced performance by increasing the length of visual to- kens, making them much longer than text tokens and signif- icantly raising computational costs. However, we observe that the visual tokens generated by popular vision encoders, such as CLIP and SigLIP , contain significant redundancy. To address this, we introduce VisionZip, a simple yet effec- tive method that selects a set of informative tokens for input to the language model, reducing visual token redundancy and improving efficiency while maintaining model perfor- mance. The proposed VisionZip can be widely applied to image and video understanding tasks and is well-suited for multi-turn dialogues in real-world scenarios, where previ- ous methods tend to underperform. Experimental results show that VisionZip outperforms the previous state-of-the- art method by at least 5% performance gains across nearly all settings. Moreover, our method significantly enhances model inference speed, improving the prefilling time by 8 × and enabling the LLaVA-Next 13B model to infer faster than the LLaVA-Next 7B model while achieving better re- sults. Furthermore, we analyze the causes of this redun- dancy and encourage the community to focus on extracting better visual features rather than merely increasing token length. Our code is available at https://github.com/dvlab- research/VisionZip.

### Summary

Recent advancements in vision-language models have en- hanced performance by increasing the length of visual to- kens, making them much longer than text tokens and signif- icantly raising computational costs. Experimental results show that VisionZip outperforms the previous state-of-the- art method by at least 5% performance gains across nearly all settings. Our code is available at https://github.com/dvlab- research/VisionZip.

---

## Critic V  VLM Critics Help Catch VLM Errors in Mul

### Original Abstract

Vision-language models (VLMs) have shown remarkable advancements in multimodal reasoning tasks. However, they still often generate inaccurate or irrelevant responses due to issues like hallucinated image understandings or un- refined reasoning paths. To address these challenges, we introduce Critic-V , a novel framework inspired by the Actor- Critic paradigm to boost the reasoning capability of VLMs. This framework decouples the reasoning process and critic process by integrating two independent components: the Reasoner, which generates reasoning paths based on vi- sual and textual inputs, and the Critic, which provides con- structive critique to refine these paths. In this approach, the Reasoner generates reasoning responses according to text prompts, which can evolve iteratively as a policy based on feedback from the Critic. This interaction process was theoretically driven by a reinforcement learning framework where the Critic offers natural language critiques instead of scalar rewards, enabling more nuanced feedback to boost the Reasoner’s capability on complex reasoning tasks. The Critic model is trained using Direct Preference Optimiza- tion (DPO), leveraging a preference dataset of critiques ranked by Rule-based Reward (RBR) to enhance its critic capabilities. Evaluation results show that the Critic-V framework significantly outperforms existing methods, in- cluding GPT-4V , on 5 out of 8 benchmarks, especially re- garding reasoning accuracy and efficiency. Combining a dynamic text-based policy for the Reasoner and construc- tive feedback from the preference-optimized Critic enables a more reliable and context-sensitive multimodal reason- ing process. Our approach provides a promising solution to enhance the reliability of VLMs, improving their per- *These authors contributed equally. †Corresponding authorformance in real-world reasoning-heavy multimodal appli- cations such as autonomous driving and embodied intelli- gence.

### Summary

Vision-language models (VLMs) have shown remarkable advancements in multimodal reasoning tasks. This interaction process was theoretically driven by a reinforcement learning framework where the Critic offers natural language critiques instead of scalar rewards, enabling more nuanced feedback to boost the Reasoner’s capability on complex reasoning tasks. †Corresponding authorformance in real-world reasoning-heavy multimodal appli- cations such as autonomous driving and embodied intelli- gence.

---

## Fast VLM  Efficient Vision Encoding for Vision Lang

### Original Abstract

Scaling the input image resolution is essential for enhancing the performance of Vision Language Mod- els (VLMs), particularly in text-rich image understanding tasks. However, popular visual encoders such as ViTs be- come inefficient at high resolutions due to the large number of tokens and high encoding latency caused by stacked self- attention layers. At different operational resolutions, the vision encoder of a VLM can be optimized along two axes: reducing encoding latency and minimizing the number of visual tokens passed to the LLM, thereby lowering overall latency. Based on a comprehensive efficiency analysis of the interplay between image resolution, vision latency, to- ken count, and LLM size, we introduce FastVLM—a model that achieves an optimized trade-off between latency, model size and accuracy. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Unlike previous methods, FastVLM achieves the optimal balance between visual token count and image res- olution solely by scaling the input image, eliminating the need for additional token pruning and simplifying the model design. In the LLaVA-1.5 setup, FastVLM achieves 3.2 × improvement in time-to-first-token (TTFT) while maintain- ing similar performance on VLM benchmarks compared to prior works. Compared to LLaVa-OneVision at the high- est resolution (1152 ×1152), FastVLM achieves compara- ble performance on key benchmarks like SeedBench and MMMU, using the same 0.5B LLM, but with 85 ×faster TTFT and a vision encoder that is 3.4 ×smaller.

### Summary

Scaling the input image resolution is essential for enhancing the performance of Vision Language Mod- els (VLMs), particularly in text-rich image understanding tasks. FastVLM incorporates FastViTHD, a novel hybrid vision encoder designed to output fewer tokens and significantly reduce encoding time for high-resolution images. Compared to LLaVa-OneVision at the high- est resolution (1152 ×1152), FastVLM achieves compara- ble performance on key benchmarks like SeedBench and MMMU, using the same 0.5B LLM, but with 85 ×faster TTFT and a vision encoder that is 3.4 ×smaller.

---

## Towards Vision Language Models For Extra Long Vide

### Original Abstract

Long video understanding poses a significant challenge for current Multi-modal Large Language Models (MLLMs). Notably, the MLLMs are constrained by their limited con- text lengths and the substantial costs while processing long videos. Although several existing methods attempt to re- duce visual tokens, their strategies encounter severe bot- tleneck, restricting MLLMs’ ability to perceive fine-grained visual details. In this work, we propose Video-XL , a novel approach that leverages MLLMs’ inherent key-value (KV) sparsification capacity to condense the visual input. Specif- ically, we introduce a new special token, the Visual Summa- rization Token ( VST), for each interval of the video, which summarizes the visual information within the interval as its associated KV . The VST module is trained by instruc- tion fine-tuning, where two optimizing strategies are offered.

### Summary

Long video understanding poses a significant challenge for current Multi-modal Large Language Models (MLLMs). In this work, we propose Video-XL , a novel approach that leverages MLLMs’ inherent key-value (KV) sparsification capacity to condense the visual input. The VST module is trained by instruc- tion fine-tuning, where two optimizing strategies are offered.

---

## Stealthy Backdoor Attack in Self Supervised Learni

### Original Abstract

Self-supervised learning (SSL) vision encoders learn high- quality image representations and thus have become a vi- tal part of developing vision modality of large vision lan- guage models (LVLMs). Due to the high cost of train- ing such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical sce- nario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we pro- pose BADVISION , the first method to exploit this vulner- ability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evalu- ateBADVISION on two types of SSL encoders and LVLMs across eight benchmarks. We show that BADVISION ef- fectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthi- ness. SoTA backdoor detection methods cannot detect our attack effectively.

### Summary

Self-supervised learning (SSL) vision encoders learn high- quality image representations and thus have become a vi- tal part of developing vision modality of large vision lan- guage models (LVLMs). In this work, we pro- pose BADVISION , the first method to exploit this vulner- ability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. SoTA backdoor detection methods cannot detect our attack effectively.

---

## Embodied Scene Understanding for Vision Language M

### Original Abstract

Vision Language Models (VLMs) demonstrate significant potential as embodied AI agents for various mobility ap- plications. However, a standardized, closed-loop bench- mark for evaluating their spatial reasoning and sequen- tial decision-making capabilities is lacking. To address this, we present MetaVQA: a comprehensive benchmark designed to assess and enhance VLMs’ understanding of spatial relationships and scene dynamics through Vi- sual Question Answering (VQA) and closed-loop simu- lations. MetaVQA leverages Set-of-Mark prompting and top-down view ground-truth annotations from nuScenes and Waymo datasets to automatically generate extensive question-answer pairs based on diverse real-world traffic scenarios, ensuring object-centric and context-rich instruc- tions. Our experiments show that fine-tuning VLMs with the MetaVQA dataset significantly improves their spatial reasoning and embodied scene comprehension in safety- critical simulations, evident not only in improved VQA ac- curacies but also in emerging safety-aware driving ma- neuvers. In addition, the learning demonstrates strong transferability from simulation to real-world observation. Code and data will be publicly available at https:// metadriverse.github.io/metavqa . 1. Introduction In many real-world robotic applications like autonomous driving and warehouse robots, embodied AI agents have started interacting with physical environments and impact- ing their surroundings. These agents should be sufficiently aware of their surroundings to interact with their environ- ments safely. In this paper, we define this ability as em- bodied scene understanding , which we believe contains two intertwined facets: spatial awareness andembodied under- standing . Spatial awareness refers to the ability to inter- nalize spatial relationships among observed objects when perceiving the 3D world through the 2D image captured by a monocular camera. Embodied understanding is the ability to relate observed objects egocentrically, foresee the impli- cation of action, and choose the optimal action to achievethe instructed goal safely. Recent advances demonstrate the potential of using Vi- sion Language Models (VLMs) as embodied agents in ap- plications from robot arms control [1, 2] to autonomous driving [3]. These tasks share the common components of following instructions, understanding the environment, and taking the optimal action to achieve specified goals. Bene- fiting from large-scale pre-training, VLMs retain embodied scene understanding to a certain extent. However, their spa- tial awareness is limited as most VLMs are pre-trained on offline text and images. Meanwhile, their embodied under- standing is also constrained because instruction-following interaction with the environment occupies a very small por- tion of their training data. In the task of autonomous driving, many prior works [3– 9] address this training distribution mismatch by fine-tuning VLMs on Visual-Question-Answering (VQA) tasks tailored for driving scenarios with reported improvements on their benchmarks. However, these benchmarks are not commen- surable or suitable for zero-shot evaluation on off-the-shelf general-purpose VLMs. This is because they follow dif- ferent textual and visual expressions to describe the scene and refer to objects. For example, DriveLM [4] refers to objects by tuples composed of the object identifier, the ID of the corresponding camera, and the pixel positions of the 2D bounding box’s vertices in the camera’s coordinate. In contrast, in ELM [5], objects are grounded by a triple com- posed of the character “c” and the pixel coordinates of the center of the 2D bounding box. Not only do these works disagree in description conventions, but their chosen con- ventions drastically differ from how humans would intu- itively refer to an object. A person would point to the object or ground the object by its features (for example, color or shape). This mismatch can weaken the diagnosing power of the VQA datasets: an unsatisfactory performance of a VLM may be caused by its inability to interpret the ques- tion expressions rather than its lack of scene understanding capability. In addition, existing works mainly evaluate embod- ied scene understanding of VLMs on the VQA task in the open-loop setting. Nevertheless, embodied under- standing capability should be examined more thoroughly 1arXiv:2501.09167v1 [cs.CV] 15 Jan 2025Set-of-Mark Annotation Query <image1> : annotated real-world image <image2> : real-world scenarios replayed in simulator Q1 : <image1>

### Summary

Vision Language Models (VLMs) demonstrate significant potential as embodied AI agents for various mobility ap- plications. These tasks share the common components of following instructions, understanding the environment, and taking the optimal action to achieve specified goals. Nevertheless, embodied under- standing capability should be examined more thoroughly 1arXiv:2501.09167v1 [cs.CV] 15 Jan 2025Set-of-Mark Annotation Query <image1> : annotated real-world image <image2> : real-world scenarios replayed in simulator Q1 : <image1>

---

## GFlow VLM  Enhancing Multi step Reasoning in Vision

### Original Abstract

Vision-Language Models (VLMs) have recently shown promising advancements in sequential decision-making tasks through task-specific fine-tuning. However, common fine-tuning methods, such as Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) techniques like Proximal Policy Optimization (PPO), present notable limitations: SFT assumes Independent and Identically Distributed (IID) data, while PPO focuses on maximizing cumulative re- wards. These limitations often restrict solution diversity and hinder generalization in multi-step reasoning tasks. To address these challenges, we introduce a novel framework, GFlowVLM, a framework that fine-tune VLMs using Gener- ative Flow Networks (GFlowNets) to promote generation of diverse solutions for complex reasoning tasks. GFlowVLM models the environment as a non-Markovian decision pro- cess, allowing it to capture long-term dependencies es- sential for real-world applications. It takes observations and task descriptions as inputs to prompt chain-of-thought (CoT) reasoning which subsequently guides action selec- tion. We use task based rewards to fine-tune VLM with GFlowNets. This approach enables VLMs to outperform prior fine-tuning methods, including SFT and RL. Empirical results demonstrate the effectiveness of GFlowVLM on com- plex tasks such as card games (NumberLine, BlackJack) and embodied planning tasks (ALFWorld), showing enhanced training efficiency, solution diversity, and stronger general- ization capabilities across both in-distribution and out-of- distribution scenarios.

### Summary

Vision-Language Models (VLMs) have recently shown promising advancements in sequential decision-making tasks through task-specific fine-tuning. GFlowVLM models the environment as a non-Markovian decision pro- cess, allowing it to capture long-term dependencies es- sential for real-world applications. Empirical results demonstrate the effectiveness of GFlowVLM on com- plex tasks such as card games (NumberLine, BlackJack) and embodied planning tasks (ALFWorld), showing enhanced training efficiency, solution diversity, and stronger general- ization capabilities across both in-distribution and out-of- distribution scenarios.

---

## ATP LLa VA  Adaptive Token Pruning for Large Vision

### Original Abstract

Large Vision Language Models (LVLMs) have achieved sig- nificant success across multi-modal tasks. However, the computational cost of processing long visual tokens can be prohibitively expensive on resource-limited devices. Pre- vious methods have identified redundancy in visual tokens within the Large Language Model (LLM) decoder layers and have mitigated this by pruning tokens using a pre- defined or fixed ratio, thereby reducing computational over- head. Nonetheless, we observe that the impact of prun- ing ratio varies across different LLM layers and instances (image-prompt pairs). Therefore, it is essential to develop a layer-wise and instance-wise vision token pruning strat- egy to balance computational cost and model performance effectively. We propose ATP-LLaVA, a novel approach that adaptively determines instance-specific token pruning ra- tios for each LLM layer. Specifically, we introduce an Adap- tive Token Pruning (ATP) module, which computes the im- portance score and pruning threshold based on input in- stance adaptively. The ATP module can be seamlessly inte- grated between any two LLM layers with negligible compu- tational overhead. Additionally, we develop a Spatial Aug- mented Pruning (SAP) strategy that prunes visual tokens with both token redundancy and spatial modeling perspec- tives. Our approach reduces the average token count by 75% while maintaining performance, with only a minimal 1.9% degradation across seven widely used benchmarks. The project page can be accessed via the following link.

### Summary

Large Vision Language Models (LVLMs) have achieved sig- nificant success across multi-modal tasks. We propose ATP-LLaVA, a novel approach that adaptively determines instance-specific token pruning ra- tios for each LLM layer. The project page can be accessed via the following link.

---

## SPA VL  A Comprehensive Safety Preference Alignmen

### Original Abstract

The emergence of Vision Language Models (VLMs) has brought unprecedented advances in understanding multi- modal information. The combination of textual and vi- sual semantics in VLMs is highly complex and diverse, making the safety alignment of these models challenging. Furthermore, due to the limited study on the safety align- ment of VLMs, there is a lack of large-scale, high-quality datasets. To address these limitations, we propose a Safety Preference Alignment dataset for VisionLanguage Mod- els named SPA-VL. In terms of breadth, SPA-VL covers 6 harmfulness domains, 13 categories, and 53 subcate- gories, and contains 100,788 samples of the quadruple (question, image, chosen response, rejected response). In terms of depth, the responses are collected from 12 open- source (e.g., QwenVL) and closed-source (e.g., Gemini) VLMs to ensure diversity. The construction of preference data is fully automated, and the experimental results indi- cate that models trained with alignment techniques on the SPA-VL dataset exhibit substantial improvements in harm- lessness and helpfulness while maintaining core capabili- ties. SPA-VL, as a large-scale, high-quality, and diverse dataset, represents a significant milestone in ensuring that VLMs achieve both harmlessness and helpfulness.

### Summary

The emergence of Vision Language Models (VLMs) has brought unprecedented advances in understanding multi- modal information. In terms of breadth, SPA-VL covers 6 harmfulness domains, 13 categories, and 53 subcate- gories, and contains 100,788 samples of the quadruple (question, image, chosen response, rejected response). SPA-VL, as a large-scale, high-quality, and diverse dataset, represents a significant milestone in ensuring that VLMs achieve both harmlessness and helpfulness.

---

## FLAIR  VLM with Fine grained Language informed Ima

### Original Abstract

CLIP has shown impressive results in aligning images and texts at scale. However, its ability to capture detailed vi- sual features remains limited because CLIP matches im- ages and texts at a global level. To address this is- sue, we propose FLAIR, Fine-grained Language-informed Image Representations, an approach that utilizes long and detailed image descriptions to learn localized image embeddings. By sampling diverse sub-captions that de- scribe fine-grained details about an image, we train our vision-language model to produce not only global embed- dings but also text-specific image representations. Our model introduces text-conditioned attention pooling on top of local image tokens to produce fine-grained image rep- resentations that excel at retrieving detailed image con- tent. We achieve state-of-the-art performance on both, ex- isting multimodal retrieval benchmarks, as well as, our newly introduced fine-grained retrieval task which eval- uates vision-language models’ ability to retrieve partial image content. Furthermore, our experiments demon- strate the effectiveness of FLAIR trained on 30M image- text pairs in capturing fine-grained visual information, in- cluding zero-shot semantic segmentation, outperforming models trained on billions of pairs. Code is available at https://github.com/ExplainableML/flair.

### Summary

CLIP has shown impressive results in aligning images and texts at scale. Our model introduces text-conditioned attention pooling on top of local image tokens to produce fine-grained image rep- resentations that excel at retrieving detailed image con- tent. Code is available at https://github.com/ExplainableML/flair.

---

## VLs I  Verbalized Layers to Interactions from Large

### Original Abstract

The recent surge in high-quality visual instruction tun- ing samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. However, scaling VLMs to improve performance using larger mod- els brings significant computational challenges, especially for deployment on resource-constrained devices like mo- bile platforms and robots. To address this, we propose VLsI :Verbalized Layers-to-Interactions, a new VLM family in 2B and 7B model sizes, which prioritizes effi- ciency without compromising accuracy. VLsI leverages a unique, layer-wise distillation process, introducing inter- mediate “verbalizers” that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. Thisapproach mitigates the training instability often encoun- tered in output imitation and goes beyond typical final-layer tuning by aligning the small VLMs’ layer-wise progression with that of the large ones. We validate VLsI across ten challenging vision-language benchmarks, achieving no- table performance gains (11.0% for 2B and 17.4% for 7B) over GPT-4V without the need for model scaling, merging, or architectural changes. Project page is now accessible.

### Summary

The recent surge in high-quality visual instruction tun- ing samples from closed-source vision-language models (VLMs) such as GPT-4V has accelerated the release of open-source VLMs across various model sizes. VLsI leverages a unique, layer-wise distillation process, introducing inter- mediate “verbalizers” that map features from each layer to natural language space, allowing smaller VLMs to flexibly align with the reasoning processes of larger VLMs. Project page is now accessible.

---

## Imagine FSL  Self Supervised Pretraining Matters on

### Original Abstract

Few-shot classiﬁcation is a challenging problem as only very few training examples are given for each new task. One of the effective research lines to address this challenge fo- cuses on learning deep representations driven by a similar- ity measure between a query image and few support images of some class. Statistically, this amounts to measure the dependency of image features, viewed as random vectors in a high-dimensional embedding space. Previous meth- ods either only use marginal distributions without consid- ering joint distributions, suffering from limited represen- tation capability, or are computationally expensive though harnessing joint distributions. In this paper, we propose a deep Brownian Distance Covariance (DeepBDC) method for few-shot classiﬁcation. The central idea of DeepBDC is to learn image representations by measuring the discrep- ancy between joint characteristic functions of embedded features and product of the marginals. As the BDC metric is decoupled, we formulate it as a highly modular and efﬁcient layer. Furthermore, we instantiate DeepBDC in two dif- ferent few-shot classiﬁcation frameworks. We make experi- ments on six standard few-shot image benchmarks, covering general object recognition, ﬁne-grained categorization and cross-domain classiﬁcation. Extensive evaluations show our DeepBDC signiﬁcantly outperforms the counterparts, while establishing new state-of-the-art results. The source code is available at http://www.peihuali.org/DeepBDC.

### Summary

Few-shot classiﬁcation is a challenging problem as only very few training examples are given for each new task. The central idea of DeepBDC is to learn image representations by measuring the discrep- ancy between joint characteristic functions of embedded features and product of the marginals. The source code is available at http://www.peihuali.org/DeepBDC.

---

## Ground V  Teaching VLMs to Ground Complex Instruct

### Original Abstract

Theopenworldisinherentlydynamic,characterizedbyever- evolving concepts and distributions. Continual learning (CL) in this dy- namic open-world environment presents a significant challenge in effec- tively generalizing to unseen test-time classes. To address this challenge, we introduce a new practical CL setting tailored for open-world visual representation learning. In this setting, subsequent data streams system- atically introduce novel classes that are disjoint from those seen in pre- vious training phases, while also remaining distinct from the unseen test classes. In response, we present Dynamic Prompt andRepresentation Learner( DPaRL ),asimpleyeteffectivePrompt-basedCL(PCL)method. Our DPaRL learns to generate dynamic prompts for inference, as op- posed to relying on a staticprompt pool in previous PCL methods. In addition, DPaRL jointly learns dynamic prompt generation and discrim- inative representation at each training stage whereas prior PCL methods only refine the prompt learning throughout the process. Our experimen- tal results demonstrate the superiority of our approach, surpassing state- of-the-art methods on well-established open-world image retrieval bench- marks by an average of 4.7% improvement in Recall@1 performance. Keywords: Dynamic Prompt Generation ·Continual Learning ·Open- World Visual Representation Learning

### Summary

Theopenworldisinherentlydynamic,characterizedbyever- evolving concepts and distributions. In response, we present Dynamic Prompt andRepresentation Learner( DPaRL ),asimpleyeteffectivePrompt-basedCL(PCL)method. Keywords: Dynamic Prompt Generation ·Continual Learning ·Open- World Visual Representation Learning

---

## Antidote  A Unified Framework for Mitigating LVLM 

### Original Abstract

Object hallucination has been an Achilles’ heel which hinders the broader applications of large vision-language models (LVLMs). Object hal- lucination refers to the phenomenon that the LVLMs claim non-existent objects in the image. To mitigate the object hallucinations, instruc- tion tuning and external model-based detection methods have been proposed, which either re- quire large-scare computational resources or de- pend on the detection result of external models. However, there remains an under-explored field to utilize the LVLM itself to alleviate object hal- lucinations. In this work, we adopt the intuition that the LVLM tends to respond logically con- sistently for existent objects but inconsistently for hallucinated objects. Therefore, we propose a Logical Closed Loop-based framework for Object Hallucination Detection and Mitigation, namely LogicCheckGPT . In specific, we de- vise logical consistency probing to raise ques- tions with logical correlations, inquiring about attributes from objects and vice versa. Whether their responses can form a logical closed loop serves as an indicator of object hallucination. As a plug-and-play method, it can be seam- lessly applied to all existing LVLMs. Com- prehensive experiments conducted on three benchmarks across four LVLMs have demon- strated significant improvements brought by our method, indicating its effectiveness and generality1.

### Summary

Object hallucination has been an Achilles’ heel which hinders the broader applications of large vision-language models (LVLMs). Therefore, we propose a Logical Closed Loop-based framework for Object Hallucination Detection and Mitigation, namely LogicCheckGPT . Com- prehensive experiments conducted on three benchmarks across four LVLMs have demon- strated significant improvements brought by our method, indicating its effectiveness and generality1.

---

## What s in the Image  A Deep Dive into the Vision o

### Original Abstract

Vision-Language Models (VLMs) have recently demon- strated remarkable capabilities in comprehending com- plex visual content. However, the mechanisms underlying how VLMs process visual information remain largely un- explored. In this paper, we conduct a thorough empirical analysis, focusing on the attention modules across layers. We reveal several key insights about how these models pro- cess visual data: (i) the internal representation of the query tokens (e.g., representations of ”describe the image”), is utilized by VLMs to store global image information; we demonstrate that these models generate surprisingly de- scriptive responses solely from these tokens, without direct access to image tokens. (ii) Cross-modal information flow is predominantly influenced by the middle layers (approxi- mately 25% of all layers), while early and late layers con- tribute only marginally. (iii) Fine-grained visual attributes and object details are directly extracted from image tokens in a spatially localized manner, i.e., the generated tokens associated with a specific object or attribute attend strongly to their corresponding regions in the image. We propose novel quantitative evaluation to validate our observations, leveraging real-world complex visual scenes. Finally, we demonstrate the potential of our findings in facilitating effi- cient visual processing in state-of-the-art VLMs.

### Summary

Vision-Language Models (VLMs) have recently demon- strated remarkable capabilities in comprehending com- plex visual content. (ii) Cross-modal information flow is predominantly influenced by the middle layers (approxi- mately 25% of all layers), while early and late layers con- tribute only marginally. Finally, we demonstrate the potential of our findings in facilitating effi- cient visual processing in state-of-the-art VLMs.

---

## Overcoming Shortcut Problem in VLM for Robust Out 

### Original Abstract

Due to the rapid spread of rumors on social media, rumor detection has become an extremely important challenge. Existing methods for rumor detection have achieved good performance, as they have collected enough corpus from the same data distribution for model training. However, significant distribution shifts between the train- ing data and real-world test data occur due to differences in news topics, social media platforms, languages and the variance in prop- agation scale caused by news popularity. This leads to a substantial decline in the performance of these existing methods in Out-Of- Distribution (OOD) situations. To address this problem, we propose a simple and efficient method named Test-time Adaptation for Rumor Detection under distribution shifts (TARD). This method models the propagation of news in the form of a propagation graph, and builds propagation graph test-time adaptation framework, en- hancing the model’s adaptability and robustness when facing OOD problems. Extensive experiments conducted on two group datasets collected from real-world social platforms demonstrate that our framework outperforms the state-of-the-art methods in perfor- mance. KEYWORDS Rumor Detection, Out-Of-Distribution, Social Media, Test-Time Adaptation ACM Reference Format: Xiang Tao1,2,∗, Mingqing Zhang1,2,∗, Qiang Liu1,2, Shu Wu1,2, Liang Wang1,2. 2024. Out-of-distribution Rumor Detection via Test-Time Adaptation. In Proceedings of ACM Conference (Conference’17). ACM, New York, NY, USA, 6 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

### Summary

Due to the rapid spread of rumors on social media, rumor detection has become an extremely important challenge. Extensive experiments conducted on two group datasets collected from real-world social platforms demonstrate that our framework outperforms the state-of-the-art methods in perfor- mance. https://doi.org/10.1145/nnnnnnn.nnnnnnn

---

## Discriminative Fine tuning of LVLMs

### Original Abstract

Contrastively-trained Vision-Language Models (VLMs) like CLIP have become the de facto approach for discriminative vision-language representation learning. However, these models have limited language understanding, often exhibit- ing a “bag of words” behavior. At the same time, Large Vision-Language Models (LVLMs), which combine vision encoders with LLMs, have been shown capable of detailed vision-language reasoning, yet their autoregressive nature renders them less suitable for discriminative tasks. In this work, we propose to combine “the best of both worlds”: a new training approach for discriminative fine- tuning of LVLMs that results in strong discriminative and compositional capabilities. Essentially, our approach con- verts a generative LVLM into a discriminative one, unlock- ing its capability for powerful image-text discrimination combined with enhanced language understanding. Our contributions include (1) A carefully designed train- ing/optimization framework that utilizes image-text pairs of variable length and granularity for training the model with both contrastive and next-token prediction losses. This is accompanied by ablation studies that justify the necessity of our framework’s components. (2) A parameter-efficient adaptation method using a combination of soft prompting and LoRA adapters. (3) Significant improvements over state-of-the-art CLIP-like models of similar size, includ- ing standard image-text retrieval benchmarks and notable gains in compositionality.

### Summary

Contrastively-trained Vision-Language Models (VLMs) like CLIP have become the de facto approach for discriminative vision-language representation learning. Essentially, our approach con- verts a generative LVLM into a discriminative one, unlock- ing its capability for powerful image-text discrimination combined with enhanced language understanding. (3) Significant improvements over state-of-the-art CLIP-like models of similar size, includ- ing standard image-text retrieval benchmarks and notable gains in compositionality.

---

## MIMO  A medical vision language model with visual 

### Original Abstract

Information comes in diverse modalities. Multimodal native AI models are essential to integrate real-world information and deliver comprehensive understanding. While proprietary multimodal native models exist, their lack of openness imposes obstacles for adoptions, let alone adaptations. To fill this gap, we introduce ARIA, an open multimodal native model with best-in-class performance across a wide range of multimodal, language, and coding tasks. ARIAis a mixture-of-expert model with 3.9B and 3.5B activated parameters per visual token and text token, respectively. It out- performs Pixtral-12B and Llama3.2-11B, and is competitive against the best proprietary models on various multimodal tasks. We pre-train A RIAfrom scratch following a 4-stage pipeline, which progressively equips the model with strong capabilities in language understanding, multimodal under- standing, long context window, and instruction following. We open-source the model weights along with a codebase that facilitates easy adoptions and adaptations of A RIAin real-world applications. Code: https://github.com/rhymes-ai/Aria Website: https://rhymes.ai/

### Summary

Information comes in diverse modalities. ARIAis a mixture-of-expert model with 3.9B and 3.5B activated parameters per visual token and text token, respectively. Code: https://github.com/rhymes-ai/Aria Website: https://rhymes.ai/

---

## Steering Away from Harm  An Adaptive Approach to D

### Original Abstract

Vision Language Models (VLMs) can produce unintended and harmful content when exposed to adversarial attacks, particularly because their vision capabilities create new vulnerabilities. Existing defenses, such as input preprocess- ing, adversarial training, and response evaluation-based methods, are often impractical for real-world deployment due to their high costs. To address this challenge, we pro- pose ASTRA, an efficient and effective defense by a daptively steering models away from adversarial feature directions to resist VLM a ttacks. Our key procedures involve finding transferable steering vectors representing the direction of harmful response and applying adaptive activation steer- ing to remove these directions at inference time. To cre- ate effective steering vectors, we randomly ablate the vi- sual tokens from the adversarial images and identify those most strongly associated with jailbreaks. These tokens are then used to construct steering vectors. During inference, we perform the adaptive steering method that involves the projection between the steering vectors and calibrated ac- tivation, resulting in little performance drops on benign in- puts while strongly avoiding harmful outputs under adver- sarial inputs. Extensive experiments across multiple mod- els and baselines demonstrate our state-of-the-art perfor- mance and high efficiency in mitigating jailbreak risks. Ad- ditionally, ASTRA exhibits good transferability, defending against both unseen attacks at design time (i.e., structured- based attacks) and adversarial images from diverse distri- butions. Our code is available at https://github. com/ASTRAL-Group/ASTRA .

### Summary

Vision Language Models (VLMs) can produce unintended and harmful content when exposed to adversarial attacks, particularly because their vision capabilities create new vulnerabilities. These tokens are then used to construct steering vectors. com/ASTRAL-Group/ASTRA .

---

## Layout VLM  Differentiable Optimization of 3D Layou

### Original Abstract

Spatial reasoning is a fundamental aspect of human cogni- tion, enabling intuitive understanding and manipulation of objects in three-dimensional space. While foundation models demonstrate remarkable performance on some benchmarks, they still struggle with 3D reasoning tasks like arranging objects in space according to open-ended language in- structions, particularly in dense and physically constrained environments. We introduce LAYOUT VLM , a framework and scene layout representation that exploits the semantic knowledge of Vision-Language Models (VLMs) and supports differentiable optimization to ensure physical plausibility. LAYOUT VLM employs VLMs to generate two mutually re- inforcing representations from visually marked images, and *Equal contribution.a self-consistent decoding process to improve VLMs spatial planning. Our experiments show that LAYOUT VLM ad- dresses the limitations of existing LLM and constraint-based approaches, producing physically plausible 3D layouts bet- ter aligned with the semantic intent of input language instruc- tions. We also demonstrate that fine-tuning VLMs with the proposed scene layout representation extracted from existing scene datasets can improve their reasoning performance.

### Summary

Spatial reasoning is a fundamental aspect of human cogni- tion, enabling intuitive understanding and manipulation of objects in three-dimensional space. LAYOUT VLM employs VLMs to generate two mutually re- inforcing representations from visually marked images, and *Equal contribution.a self-consistent decoding process to improve VLMs spatial planning. We also demonstrate that fine-tuning VLMs with the proposed scene layout representation extracted from existing scene datasets can improve their reasoning performance.

---

