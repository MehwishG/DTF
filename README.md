
# DTF: Enhancing 3D Human Pose Estimation Amidst Severe Occlusion with Dual Transformer Fusion
**Model Architecture**


**Installation**

Install PyTorch 1.7.1 and Torchvision 0.8.2 following the official instructions.
Install dependencies:

pip install -r requirements.txt

**Dataset Setup**
Please download the dataset from  website and refer to VideoPose3D to set up the Human3.6M dataset ('./dataset' directory). Or you can download the processed data from here.

${POSE_ROOT}/
|-- dataset
|   |-- data_3d_h36m.npz
|   |-- data_2d_h36m_gt.npz
|   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz

**Experimental Setup**

**Training and Test the Model**


**Video Demo**


3D Pose Estimations with 16 random occluded joints out of 17 for action ``Eating"

**Using proposed DTF**

https://github.com/MehwishG/DTF/assets/53044443/13b7f530-2840-45da-b8ab-f003e5954f28

**Using MHFormer**

https://github.com/MehwishG/DTF/assets/53044443/bdc3f428-d336-47f9-bdc8-f23798371906

**Using STCFormer**

https://github.com/MehwishG/DTF/assets/53044443/958f5a10-c75c-41a2-b8f6-0c28457421dd

**Using PSTMO**



https://github.com/MehwishG/DTF/assets/53044443/dd8ef119-5d87-4a0d-b6f8-7d1c81a49432


