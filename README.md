
# DTF: Enhancing 3D Human Pose Estimation Amidst Severe Occlusion with Dual Transformer Fusion
> Mehwish Ghafoor, Arif Mahmood

## Model Architecture
Proposed Dual Transformer Fusion (DTF) architecture takes severly occluded 2D joint positions as input and estimate realistic 3D pose.
![DTF_arch_up2_v3 (2)](https://github.com/user-attachments/assets/669fea10-b52e-4499-8ebb-ee56317c7643)



## Environment
The code is developed and tested under the following environment:

1. PyTorch 1.7.1 and Torchvision 0.8.2 following the [official instructions](https://pytorch.org/).

2. Install dependencies:
 ```
pip3 install -r requirements.txt
```

## Dataset Setup
1. Download the dataset from the [Human 3.6M](http://vision.imar.ro/human3.6m/description.php) website.

2. Set up the Human3.6M dataset as per the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) instructions.

3. Alternatively, download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC).
```
${DTF_Occ}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```
## Pretrained Model

You can download pretrained model for Human 3.6M from [here](https://drive.google.com/drive/folders/1mMqX__ItxisexEfHuL3pUOnXhUpzIhQI?usp=sharing).

For MPI-INF-3DHP, we have followed the setting of [P-STMO](https://github.com/paTRICK-swk/P-STMO)

## Training and Test the Model for Human 3.6M
**Training with 351 frames on Human 3.6M**
```
python3 main_h36m.py --frames 351 --batch_size 32
```
**Test**
```
python3 main_h36m.py --test --previous_dir 'checkpoint/351_severe' --frames 351
```

**Video Demo - Human 3.6M**

3D Pose Estimations with 16 random occluded joints out of 17 for action ``Eating"

**Using proposed DTF**

https://github.com/MehwishG/DTF/assets/53044443/13b7f530-2840-45da-b8ab-f003e5954f28

**Using MHFormer**

https://github.com/MehwishG/DTF/assets/53044443/bdc3f428-d336-47f9-bdc8-f23798371906

**Using STCFormer**

https://github.com/MehwishG/DTF/assets/53044443/958f5a10-c75c-41a2-b8f6-0c28457421dd

**Using PSTMO**

https://github.com/MehwishG/DTF/assets/53044443/dd8ef119-5d87-4a0d-b6f8-7d1c81a49432


## Performance Comparison of MPI-INF-3DHP under Severe Occlusion
[mpi_16vis_2.pdf](https://github.com/user-attachments/files/16624566/mpi_16vis_2.pdf)




![cmp_mpi_mpjpe_up (1)](https://github.com/user-attachments/assets/2014a7e0-c9be-4c6b-9884-ce4f23742d1e)

## Human 3.6M Results using Stacked Hourglass 2D Detections:
<img width="249" alt="Screen Shot 2024-08-15 at 4 12 50 PM" src="https://github.com/user-attachments/assets/c493e463-cf4c-4b76-9a42-b4542229489b">

