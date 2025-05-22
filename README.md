Dual Attention Residual U-Net for Accurate Brain Ultrasound Segmentation in IVH Detection
--------------------------------------------------------------------
![3-1](https://github.com/user-attachments/assets/8dc517a0-77a9-423e-b0d3-c20b693c387d)
![3-7](https://github.com/user-attachments/assets/faac70e1-1bc1-4c9d-b0d8-4f9dd59af037)
![1-1](https://github.com/user-attachments/assets/354bc374-61e4-48e2-a64b-75b1e5c05286)
![1-7](https://github.com/user-attachments/assets/7d8c5de4-b66d-4c2b-9b20-7814b52afbbc)


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
(IVH) remains a critical global concern, underscoring the urgent need for early detection and accurate diagnosis to improve survival rates among premature infants. Recent advances in deep learning have shown promising potential for computer-aided detection (CAD) systems to address this challenge. Our proposed approach consists of two powerful attention modules: the novel convolutional block attention module (CBAM) and sparse attention layer (SAL), integrated into a Residual U-Net model. The CBAM enhances the model’s ability to capture spatial relationships within local features. The SAL consists of sparse and dense attention branches. The sparse attention filters out the negative impacts of low query-key matching scores for aggregating features, whereas the original dense one ensures sufficient information flow through the network for learning discriminative representations. To evaluate our approach, we performed experiments using a widely used dataset of brain ultrasound images and the obtained results demonstrate its capability to accurately detect brain anatomies. Our approach achieves state-of-the-art performance with a dice score of 89.04% and IoU of 81.84% in the Brain US dataset in segmenting the ventricle region, showing its potential to help with precise brain anatomy detection. By leveraging the power of deep learning and integrating innovative attention mechanisms, our study contributes to ongoing efforts to improve brain anatomy detection.

----------------------------------------------------------------------
Using the code:

The code is stable using Python 3.8, Pytorch 1.8.1

To install all the dependencies using pip.

----------------------------------------------------------------------
Public Datasets:

1.Brain Anatomy US dataset 

2.Nuclei dataset
