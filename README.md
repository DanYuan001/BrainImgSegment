Brain anatomy image segmentation by Res-UNet with attention
--------------------------------------------------------------------
![1-1](https://github.com/user-attachments/assets/edea6d97-42ef-4766-aa3a-2701d7f30c1c)
![4-1](https://github.com/user-attachments/assets/e453b863-49c3-4f28-95cc-48ac4cd46008)
--------------------------------------------------------------------
![Fig1-改](https://github.com/user-attachments/assets/74c3a1b2-3b64-4947-aba1-0787016087db)

![Fig2-改](https://github.com/user-attachments/assets/86bf70b0-b09d-4228-83ac-aa9e6df7ffd1)
--------------------------------------------------------------------
Authors:
Dan Yuan，
Yi Feng，
Ziyun Tang，
Meng Qin，
Dong Zheng。
--------------------------------------------------------------------
Introduction

Brain intraventricular hemorrhage
(IVH) remains a critical global concern, underscoring the urgent need for early detection and accurate diagnosis to improve survival rates among premature infants. Recent advances in deep learning
have shown promising potential for computer-aided detection (CAD) systems to address this challenge. Our proposed approach consists of two powerful attention modules: the novel convolutional block attention module (CBAM) and sparse attention layer (SAL), integrated into a Residual U-Net model. The CBAM enhances the model’s ability to capture spatial relationships within local features. The SAL consists of sparse and dense attention branches. The sparse attention filters out the negative impacts of low query-key matching scores for aggregating features, whereas the original dense one ensures sufficient information flow through the network for learning discriminative representations. To evaluate our approach, we performed experiments using a widely used dataset of brain ultrasound images and the obtained results demonstrate its capability to accurately detect brain anatomies. Our approach achieves state-of-the-art performance with a dice score of 89.04% and IoU of 81.84% in the Brain US dataset in segmenting the ventricle region, showing its potential to help with precise brain anatomy detection. By leveraging the power of deep learning and integrating innovative attention mechanisms, our study contributes to ongoing efforts to improve brain anatomy detection.
