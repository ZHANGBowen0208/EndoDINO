# EndoDINO: Learning to Adapt Foundation Models for Capsule Endoscopy Diagnosis
![Image](https://github.com/ZhangBoowen/EndoDINO/blob/main/architecture.png)
---
## Dataset

1. Kvasir-Capsule dataset **[[`Dataset_link`](https://datasets.simula.no/kvasir-capsule/)]** **[[`Paper`](https://www.nature.com/articles/s41597-021-00920-z)]**
2. Kvasirv2 dataset **[[`Dataset_link`](https://datasets.simula.no/kvasir/)]** **[[`Paper`](https://www.researchgate.net/publication/316215961_KVASIR_A_Multi-Class_Image_Dataset_for_Computer_Aided_Gastrointestinal_Disease_Detection)]**
---
## Training & Evaluation

```bash
 python EndoDINO.py
```
---
## References
1. DINOv2
    - Paper [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
    - official pytorch implementation [Code](https://github.com/facebookresearch/dinov2).

2. LoRA
    - Paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
    - Official Pytorch implementation [Code](https://github.com/microsoft/LoRA).

3. Surgical_DINO
    - Paper [Surgical-DINO: Adapter Learning of Foundation Models for Depth Estimation in Endoscopic Surgery](https://arxiv.org/abs/2401.06013).
    - Official Pytorch implementation [Code](https://github.com/BeileiCui/SurgicalDINO).
---
