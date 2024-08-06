
# DTF: Enhancing 3D Human Pose Estimation Amidst Severe Occlusion with Dual Transformer Fusion
> Mehwish Ghafoor, Arif Mahmood

## Model Architecture
Proposed Dual Transformer Fusion (DTF) architecture takes severly occluded 2D joint positions as input and estimate realistic 3D pose.
![DTF_arch_up2_v3 (2)](https://github.com/user-attachments/assets/7f6b4fdf-7811-4d16-a45e-8ad1666f017a)


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

You can download pretrained model for Human 3.6M and MPI-INF-3DHP dataset from [here]().

## Training and Test the Model for Human 3.6M
** Training with 351 frames on Human 3.6M **
```
python3 main_h36m.py --frames 351 --batch_size 32
```
**  Test **
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
## Acknowledgement
This research work was funded by Institutional Fund Projects under grant no. (IFPIP: 1049-135-1443). The authors gratefully acknowledge technical and financial support provided by the Ministry of Education and Deanship of Scientific Research (DSR) at King Abdulaziz University, Jeddah, Saudi Arabia.
