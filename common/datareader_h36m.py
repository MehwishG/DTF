# Adapted from Optimizing Network Structure for 3D Human Pose Estimation (ICCV 2019) (https://github.com/CHUNYUWANG/lcn-pose/blob/master/tools/data.py)

import numpy as np
import os, sys
import random
import copy
from lib.utils.tools import read_pkl
from lib.utils.utils_data import split_clips
random.seed(0)

bones = [
    (0, 1),  # Hip to Right Hip
    (1, 2),  # Right Hip to Right Knee
    (2, 3),  # Right Knee to Right Ankle
    (0, 4),  # Hip to Left Hip
    (4, 5),  # Left Hip to Left Knee
    (5, 6),  # Left Knee to Left Ankle
    (0, 7),  # Hip to Spine
    (7, 8),  # Spine to Neck
    (8, 9),  # Neck to Head
    (8, 10), # Neck to Left Shoulder
    (10, 11),# Left Shoulder to Left Elbow
    (11, 12),# Left Elbow to Left Wrist
    (8, 13), # Neck to Right Shoulder
    (13, 14),# Right Shoulder to Right Elbow
    (14, 15) # Right Elbow to Right Wrist
]
def calculate_missing_joint(known_joint, direction_vector, bone_length):
    #print(known_joint.shape, direction_vector.shape, bone_length.shape)# Calculate the missing joint position        
    # Calculate the missing joint position
    missing_joint = known_joint + direction_vector * bone_length
    #print(missing_joint.shape)# Calculate the missing joint position    
    return missing_joint
def interpolate_joints(data, bone_lengths, direction_vectors, bones, missing_indices):
   
    for frame_idx, joint_idx in missing_indices:
        frame = data[frame_idx]
        frame = frame[:,:2]        
        for bone_idx, (joint1, joint2) in enumerate(bones):
            if joint_idx == joint1:
                joint1_pos = frame[joint1]
                joint2_pos = frame[joint2]
                
                if np.all(joint2_pos != 0):  # If joint2 is known
                    bone_length = bone_lengths[frame_idx, bone_idx]
                    direction_vector = direction_vectors[frame_idx, bone_idx]
                    data[frame_idx, joint1,:2] = calculate_missing_joint(joint2_pos, -direction_vector, bone_length)
                    data[frame_idx, joint1,2] = data[frame_idx, joint2,2]
                    break
            
            elif joint_idx == joint2:
                joint1_pos = frame[joint1]
                joint2_pos = frame[joint2]
                
                if np.all(joint1_pos != 0):  # If joint1 is known
                    bone_length = bone_lengths[frame_idx, bone_idx]
                    direction_vector = direction_vectors[frame_idx, bone_idx]
                    data[frame_idx, joint2,:2] = calculate_missing_joint(joint1_pos, direction_vector, bone_length)
                    data[frame_idx, joint2,2] = data[frame_idx, joint1,2]
                    break
    
    return data

def calculate_bone_lengths_and_directions(data):
  
    # Initialize arrays to store bone lengths and direction vectors
    num_frames = data.shape[0]
    num_bones = len(bones)

    bone_lengths = np.zeros((num_frames, num_bones))
    direction_vectors = np.zeros((num_frames, num_bones, 2))  # Corrected shape to (2,)

    # Iterate over each frame and calculate bone lengths and direction vectors
    for frame_idx in range(num_frames):
        frame = data[frame_idx]
        
        for bone_idx, (joint1, joint2) in enumerate(bones):
            joint1_pos = frame[joint1][:2]  # Use only [x, y]
            joint2_pos = frame[joint2][:2]  # Use only [x, y]
            
            # Calculate the direction vector
            direction_vector = joint2_pos - joint1_pos
            
            # Calculate the bone length
            bone_length = np.linalg.norm(direction_vector)
            
            # Normalize the direction vector
            if bone_length != 0:
                direction_vector /= bone_length
            
            # Store the results
            bone_lengths[frame_idx, bone_idx] = bone_length
            direction_vectors[frame_idx, bone_idx] = direction_vector

    return bone_lengths, direction_vectors

def interpolate_zeros(array,to_fill, zero_indices,diff_count):
   
    n_frames, n_joints = array.shape

    for idx in zero_indices:
        t, j = idx  # Extract the indices for easier use
        
        prev = None
        next = None

        # Find previous non-zero value
        for k in range(t - 1, -1, -1):
            if array[k, j] != 0:
                prev = to_fill[k, j,:]
                break

        n_time = to_fill.shape[0] - t  # Find next non-zero value
        if n_time <= 200 and n_time > 0:
            end = t + n_time
        elif n_time > 200:
            end = t + 200
        else:
            end = t + np.abs(n_time)
            
        for k in range(t + 1, end):
            if array[k, j] != 0:
                next = to_fill[k, j, :]
                break

        # Interpolate using the average of previous and next non-zero values
        if prev is not None and next is not None:
            to_fill[t, j, 0] = (prev[0] + next[0]) / 2
            to_fill[t, j, 1] = (prev[1] + next[1]) / 2
            to_fill[t, j, 2] = cal_score(t, k, diff_count)
        elif prev is not None:
            to_fill[t, j, 0] = prev[0]
            to_fill[t, j, 1] = prev[1]
            to_fill[t, j, 2] = cal_score(t, k, diff_count)
        elif next is not None:
            to_fill[t, j, 0] = next[0]
            to_fill[t, j, 1] = next[1]
            to_fill[t, j, 2] = cal_score(t, k, diff_count)
    return to_fill

def find_zero_indices(array):
    
    # Find indices where values are zero
    zero_indices = np.argwhere(array == 0)
    
    return zero_indices

def cal_score(center_index, left,max_diff):
    diff = np.abs(center_index - left)
    conf_score = 0.0

    # Maximum difference we're considering


    # Calculate the confidence score based on the percentage approach
    if diff <= max_diff:
        conf_score = (max_diff - diff) / max_diff
        # Ensure the confidence score is not less than 0.1
        if conf_score < 0.1:
            conf_score = 0.1
    else:
        conf_score = 0.1

    return conf_score


###################################################################################################

def interpolate_frame(frame, prev_frame, next_frame, conf_score):
    for k in range(17):  # per joint loop
        j_2d = frame[k]
        p_2d = prev_frame[k]
        n_2d = next_frame[k]

        if j_2d[0] == 0.0:
            if p_2d[0] != 0.0 and n_2d[0] != 0.0:
                j_2d[0] = (p_2d[0] + n_2d[0]) / 2
                j_2d[1] = (p_2d[1] + n_2d[1]) / 2
            elif p_2d[0] != 0.0:
                j_2d[0] = p_2d[0]
                j_2d[1] = p_2d[1]
            elif n_2d[0] != 0.0:
                j_2d[0] = n_2d[0]
                j_2d[1] = n_2d[1]
            j_2d[2] = conf_score

def interpolation_all2(batch_2d, diff_count):
    b, t, d = batch_2d.shape  # (566920, 17, 3)

    # Set [x, y, confidence] to zero where confidence < 0.6
    batch_2d[:, :, 2][batch_2d[:, :, 2] < 0.6] = 0
    batch_2d[:, :, :2][batch_2d[:, :, 2] == 0] = 0

    center_index = b // 2
    ran = diff_count
    left = center_index - 1
    right = center_index + 1

    # Interpolate around the center frame
    for _ in range(ran):
        if np.count_nonzero(batch_2d[center_index, :, :]) == 51:
            break
        if left < 0 or right >= b:
            break
        prev = batch_2d[left]
        next = batch_2d[right]
        conf_score = cal_score(center_index, left, diff_count)
        interpolate_frame(batch_2d[center_index], prev, next, conf_score)

        left -= 1
        right += 1

    # Interpolate for frames to the left of the center
    curr_frame = center_index - 1
    while curr_frame >= 0:
        frame = batch_2d[curr_frame, :, :]
        if np.count_nonzero(frame) == 51:
            curr_frame -= 1
            continue
        left = curr_frame - 1
        right = curr_frame + 1
        if left < 0 or right >= b:
            curr_frame -= 1
            continue
        conf_score = cal_score(curr_frame, left, diff_count)
        interpolate_frame(frame, batch_2d[left], batch_2d[right], conf_score)

        curr_frame -= 1

    # Interpolate for frames to the right of the center
    curr_frame = center_index + 1
    while curr_frame < b:
        frame = batch_2d[curr_frame, :, :]
        if np.count_nonzero(frame) == 51:
            curr_frame += 1
            continue
        left = curr_frame - 1
        right = curr_frame + 1
        if left < 0 or right >= b:
            curr_frame += 1
            continue
        conf_score = cal_score(curr_frame, left, diff_count)
        interpolate_frame(frame, batch_2d[left], batch_2d[right], conf_score)

        curr_frame += 1

    # Final pass to handle any remaining zeros
    for curr_frame in range(b):
        frame = batch_2d[curr_frame, :, :]
        if np.count_nonzero(frame) != 51:
            left = curr_frame - 1
            right = curr_frame + 1
            if left < 0:
                left = 0
            if right >= b:
                right = b - 1
            conf_score = cal_score(curr_frame, left, diff_count)
            interpolate_frame(frame, batch_2d[left], batch_2d[right], conf_score)

    return batch_2d


class DataReaderH36M(object):
    def __init__(self, n_frames, sample_stride, data_stride_train, data_stride_test, read_confidence=True, dt_root = 'data/motion3d', dt_file = 'h36m_cpn_cam_source.pkl'):
        self.gt_trainset = None
        self.gt_testset = None
        self.split_id_train = None
        self.split_id_test = None
        self.test_hw = None
        #print(dt_root,dt_file,"hello")
        self.dt_dataset = read_pkl('/media/itucvl/Local Disk/Mehwish/RR_2/MotionBERT-main/data/motion3d/h36m_sh_conf_cam_source_final.pkl')
        self.n_frames = n_frames
        self.sample_stride = sample_stride
        self.data_stride_train = data_stride_train
        self.data_stride_test = data_stride_test
        self.read_confidence = read_confidence
        
    def read_2d(self):
        trainset = self.dt_dataset['train']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        testset = self.dt_dataset['test']['joint_2d'][::self.sample_stride, :, :2].astype(np.float32)  # [N, 17, 2]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            trainset[idx, :, :] = trainset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            testset[idx, :, :] = testset[idx, :, :] / res_w * 2 - [1, res_h / res_w]
        if self.read_confidence:
            if 'confidence' in self.dt_dataset['train'].keys():
                train_confidence = self.dt_dataset['train']['confidence'][::self.sample_stride].astype(np.float32)  
                test_confidence = self.dt_dataset['test']['confidence'][::self.sample_stride].astype(np.float32)  
                if len(train_confidence.shape)==2: # (1559752, 17)
                    train_confidence = train_confidence[:,:,None]
                    test_confidence = test_confidence[:,:,None]
            else:
                # No conf provided, fill with 1.
                train_confidence = np.ones(trainset.shape)[:,:,0:1]
                test_confidence = np.ones(testset.shape)[:,:,0:1]
            trainset = np.concatenate((trainset, train_confidence), axis=2)  # [N, 17, 3]
            testset = np.concatenate((testset, test_confidence), axis=2)  # [N, 17, 3]
        return trainset, testset

    def read_3d(self):
        train_labels = self.dt_dataset['train']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)  # [N, 17, 3]
        test_labels = self.dt_dataset['test']['joint3d_image'][::self.sample_stride, :, :3].astype(np.float32)    # [N, 17, 3]
        # map to [-1, 1]
        for idx, camera_name in enumerate(self.dt_dataset['train']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            train_labels[idx, :, :2] = train_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            train_labels[idx, :, 2:] = train_labels[idx, :, 2:] / res_w * 2
            
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_labels[idx, :, :2] = test_labels[idx, :, :2] / res_w * 2 - [1, res_h / res_w]
            test_labels[idx, :, 2:] = test_labels[idx, :, 2:] / res_w * 2
            
        return train_labels, test_labels
    def read_hw(self):
        if self.test_hw is not None:
            return self.test_hw
        test_hw = np.zeros((len(self.dt_dataset['test']['camera_name']), 2))
        for idx, camera_name in enumerate(self.dt_dataset['test']['camera_name']):
            if camera_name == '54138969' or camera_name == '60457274':
                res_w, res_h = 1000, 1002
            elif camera_name == '55011271' or camera_name == '58860488':
                res_w, res_h = 1000, 1000
            else:
                assert 0, '%d data item has an invalid camera name' % idx
            test_hw[idx] = res_w, res_h
        self.test_hw = test_hw
        return test_hw
    
    def get_split_id(self):
        if self.split_id_train is not None and self.split_id_test is not None:
            return self.split_id_train, self.split_id_test
        vid_list_train = self.dt_dataset['train']['source'][::self.sample_stride]                          # (1559752,)
        vid_list_test = self.dt_dataset['test']['source'][::self.sample_stride]                           # (566920,)
        self.split_id_train = split_clips(vid_list_train, self.n_frames, data_stride=self.data_stride_train) 
        self.split_id_test = split_clips(vid_list_test, self.n_frames, data_stride=self.data_stride_test)
        return self.split_id_train, self.split_id_test
    
    def get_hw(self):
#       Only Testset HW is needed for denormalization
        test_hw = self.read_hw()                                     # train_data (1559752, 2) test_data (566920, 2)
        split_id_train, split_id_test = self.get_split_id()
        test_hw = test_hw[split_id_test][:,0,:]                      # (N, 2)
        return test_hw
    
    def get_sliced_data(self):
        train_data, test_data = self.read_2d()     # train_data (1559752, 17, 3) test_data (566920, 17, 3)
        train_labels, test_labels = self.read_3d() # train_labels (1559752, 17, 3) test_labels (566920, 17, 3)
        bone_lengths_train, direction_vectors_train = calculate_bone_lengths_and_directions(train_data)
        bone_lengths_test, direction_vectors_test = calculate_bone_lengths_and_directions(test_data)
        #here do interpolation
        #print(test_data.shape)# = interpolation_all2(test_data, 200)
        test_data = interpolation_all2(test_data, 200)
        zero_indices = find_zero_indices(test_data[:,:,0])
        test_data=interpolate_zeros(test_data[:,:,0],test_data, zero_indices,200)
        zero_indices = find_zero_indices(test_data[:,:,0])
        if len(zero_indices) > 0:
            test_data=interpolate_joints(test_data, bone_lengths_test, direction_vectors_test, bones, zero_indices)#return self.split_id_train, self.split_id_test

        #np.save('testDinside0.7.npy',test_data)#train_data = interpolation_all2(train_data, 200)
        train_data = interpolation_all2(train_data, 200)
        zero_indices = find_zero_indices(train_data[:,:,0])
        train_data=interpolate_zeros(train_data[:,:,0],train_data, zero_indices,200)
        zero_indices = find_zero_indices(train_data[:,:,0])
        if len(zero_indices) > 0:
            train_data=interpolate_joints(train_data, bone_lengths_train, direction_vectors_train, bones, zero_indices)#return self.split_id_train, self.split_id_test


        #np.save('trainDinside.npy',train_data)
        ############################
        split_id_train, split_id_test = self.get_split_id()
        train_data, test_data = train_data[split_id_train], test_data[split_id_test]                # (N, 27, 17, 3)
        train_labels, test_labels = train_labels[split_id_train], test_labels[split_id_test]        # (N, 27, 17, 3)
        # ipdb.set_trace()
        return train_data, test_data, train_labels, test_labels
    
    def denormalize(self, test_data):
#       data: (N, n_frames, 51) or data: (N, n_frames, 17, 3)        
        n_clips = test_data.shape[0]
        test_hw = self.get_hw()
        data = test_data.reshape([n_clips, -1, 17, 3])
        #print("in ",data.shape,test_hw.shape,len(data),len(test_hw))
        assert len(data) == len(test_hw)
        # denormalize (x,y,z) coordiantes for results
        for idx, item in enumerate(data):
            res_w, res_h = test_hw[idx]
            data[idx, :, :, :2] = (data[idx, :, :, :2] + np.array([1, res_h / res_w])) * res_w / 2
            data[idx, :, :, 2:] = data[idx, :, :, 2:] * res_w / 2
        return data # [n_clips, -1, 17, 3]
