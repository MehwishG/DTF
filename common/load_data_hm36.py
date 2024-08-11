
import torch.utils.data as data
import numpy as np

from common.utils import deterministic_random
from common.camera import world_to_camera, normalize_screen_coordinates
from common.generator import ChunkedGenerator
import random

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
def interpolation_all(batch_2d):
    b, t, d = batch_2d.shape
    tempPose2D = np.ones((b, t, d + 1))
    tempPose2D[:, :, 0] = batch_2d[:, :, 0]
    tempPose2D[:, :, 2] = batch_2d[:, :, 1]
    batch_2d = tempPose2D

    miss =  random.randint(12,16)# random miss joints severe case
    #print("miss:,",miss)
    for p_x in range(b):  # frame by frame
        indices_r = random.sample(range(0, t), miss)  # np.random.randint(0,17,miss)
        for ind in range(17):
            if ind in indices_r:
                batch_2d[p_x, ind, :] = 0.
    center_index = (b // 2)  #
    first_half = 0
    last_end = b  #
    # print("before",self.batch_2d[3,:,:])
    center_frame = batch_2d[center_index, :, :]  # 17x3
    # center_orig = orig_2d[center_index, :, :]
    # ratio = [0.9, 0.8,0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1]
    r_k = 0
    ran = 40
    left = center_index - 1
    right = center_index + 1
    for b_i in range(0, ran, 1):  # loop over all frames will see from both sides of center frame
        if np.count_nonzero(center_frame) == 51:
            break
        prev = batch_2d[left]
        next = batch_2d[right]
        # confidence score
        conf_score = cal_score(center_index, left,ran)

        for k in range(17):  # per joint loop
            j_2d = center_frame[k]  # 4 x,ind,y,ind
            p_2d = prev[k]
            n_2d = next[k]
            p_up = 1
            n_up = 1
            if j_2d[0] == 0.0:
                if p_2d[0] != 0.0 and n_2d[0] != 0.0:
                    j_2d_x = (p_2d[0] + n_2d[0]) / 2
                    j_2d_y = (p_2d[2] + n_2d[2]) / 2
                    j_2d[0] = j_2d_x
                    j_2d[2] = j_2d_y
                    j_2d[1] = conf_score
                elif p_2d[0] != 0.0 and n_2d[0] == 0.0:
                    j_2d[0] = p_2d[0]
                    j_2d[2] = p_2d[2]
                    j_2d[1] = conf_score
                    n_up = 0
                elif p_2d[0] == 0.0 and n_2d[0] != 0.0:
                    j_2d[0] = n_2d[0]
                    j_2d[2] = n_2d[2]
                    j_2d[1] = conf_score
                    p_up = 0
        # if p_up==1:

        left = left - 1
        # if n_up==1:
        right = right + 1
    # error = np.mean(np.linalg.norm(center_frame[:, [0, 2]] - center_orig[:, [0, 2]],
    #                                axis=len(np.asarray(center_orig[:, [0, 2]]).shape) - 1))
    # print("center frame", error)

    # left side of center frame cover all 5-175
    curr_frame = center_index-1
    left = curr_frame - 1
    right = curr_frame + 1
    while (1):

        frame = batch_2d[curr_frame, :, :]
        # process for this frame
        if np.count_nonzero(frame) == 51:
            curr_frame = curr_frame - 1
            left = curr_frame - 1
            right = curr_frame + 1

        conf_score = cal_score(curr_frame, left)
        prev = batch_2d[left]
        next = batch_2d[right]
        for k in range(17):  # per joint loop
            j_2d = frame[k]  # 4 x,ind,y,ind
            p_2d = prev[k]
            n_2d = next[k]
            p_up = 1
            n_up = 1
            if j_2d[0] == 0.0:
                if p_2d[1] == 1.0 and n_2d[1] == 1.0:
                    j_2d_x = (p_2d[0] + n_2d[0]) / 2
                    j_2d_y = (p_2d[2] + n_2d[2]) / 2
                    j_2d[0] = j_2d_x
                    j_2d[2] = j_2d_y
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                elif p_2d[1] == 1.0 and n_2d[1] != 1.0:
                    j_2d[0] = p_2d[0]
                    j_2d[2] = p_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    n_up = 0
                elif p_2d[1] != 1.0 and n_2d[1] == 1.0:
                    j_2d[0] = n_2d[0]
                    j_2d[2] = n_2d[2]
                    j_2d[1] = conf_score

                    p_up = 0
        # if p_up==1:
        left = left - 1
        # if n_up==1:
        right = right + 1

        if curr_frame < 1:
            break
        if right > b-1:
            break
        if left < 0:
            break

    ################## after curr frame right side
    curr_frame = center_index+1
    left = curr_frame - 1
    right = curr_frame + 1
    while (1):

        frame = batch_2d[curr_frame, :, :]
        # process for this frame
        if np.count_nonzero(frame) == 51:
            # orig_2d_f = orig_2d[curr_frame, :, :]
            # orig_2d_f = orig_2d_f[:, [0, 2]]
            # pred_2d_f = frame[:, [0, 2]]
            # error = np.mean(np.linalg.norm(pred_2d_f - orig_2d_f, axis=len(np.asarray(orig_2d_f).shape) - 1))
            # tot_err = tot_err + error
            # print("Error: ", error, "of Frame # ", curr_frame)
            curr_frame = curr_frame + 1
            left = curr_frame - 1
            right = curr_frame + 1

        conf_score = cal_score(curr_frame, left)
        prev = batch_2d[left]
        if right == b:
            break
        next = batch_2d[right]
        for k in range(17):  # per joint loop
            j_2d = frame[k]  # 4 x,ind,y,ind
            p_2d = prev[k]
            n_2d = next[k]
            p_up = 1
            n_up = 1
            if j_2d[0] == 0.0:
                if p_2d[1] == 1.0 and n_2d[1] == 1.0:
                    j_2d_x = (p_2d[0] + n_2d[0]) / 2
                    j_2d_y = (p_2d[2] + n_2d[2]) / 2
                    j_2d[0] = j_2d_x
                    j_2d[2] = j_2d_y
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                elif p_2d[1] == 1.0 and n_2d[1] != 1.0:
                    j_2d[0] = p_2d[0]
                    j_2d[2] = p_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    n_up = 0
                elif p_2d[1] != 1.0 and n_2d[1] == 1.0:
                    j_2d[0] = n_2d[0]
                    j_2d[2] = n_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    p_up = 0
        # if p_up==1:
        left = left - 1
        # if n_up==1:
        right = right + 1

        if curr_frame > b:
            # print("ignore last 5 frames")
            break
        if right > b-1:
            break
        if left < 0:
            break
    # verify frames not filled from left sides
    chkInd = 0
    for ir in range(center_index-1, 0, -1):
        if np.count_nonzero(batch_2d[ir, :, 0]) < 17:
            chkInd = ir
            break;

    curr_frame = chkInd
    left = curr_frame - 1
    right = curr_frame + 1
    while (curr_frame >=0):

        frame = batch_2d[curr_frame, :, :]
        # process for this frame
        if np.count_nonzero(frame) == 51:
            curr_frame = curr_frame - 1
            if left < 0:
                left = 0
            else:
                left = curr_frame - 1
            right = curr_frame + 1

        conf_score = cal_score(curr_frame, right)
        prev = batch_2d[left]
        next = batch_2d[right]
        for k in range(17):  # per joint loop
            j_2d = frame[k]  # 4 x,ind,y,ind
            p_2d = prev[k]
            n_2d = next[k]
            p_up = 1
            n_up = 1
            if j_2d[0] == 0.0:
                if p_2d[1] == 1.0 and n_2d[1] == 1.0:
                    j_2d_x = (p_2d[0] + n_2d[0]) / 2
                    j_2d_y = (p_2d[2] + n_2d[2]) / 2
                    j_2d[0] = j_2d_x
                    j_2d[2] = j_2d_y
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                elif p_2d[1] == 1.0 and n_2d[1] != 1.0:
                    j_2d[0] = p_2d[0]
                    j_2d[2] = p_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    n_up = 0
                elif p_2d[1] != 1.0 and n_2d[1] == 1.0:
                    j_2d[0] = n_2d[0]
                    j_2d[2] = n_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    p_up = 0
        # if p_up==1:
        left = left - 1
        # if n_up==1:
        right = right + 1

        if right > center_index:
            break
        if left < 0:
            left = 0
    ########
    # verify from right side end
    chkInd = center_index
    for ir in range(center_index+1, b, 1):
        if np.count_nonzero(batch_2d[ir, :, 0]) < 17:
            chkInd = ir
            break;
    # print("startoiddddd", chkInd)
    curr_frame = chkInd
    left = curr_frame - 1
    right = curr_frame + 1
    while (1):

        frame = batch_2d[curr_frame, :, :]
        # process for this frame
        if np.count_nonzero(frame) == 51:

            curr_frame = curr_frame + 1
            left = curr_frame - 1
            if right >= b-1:
                right = b-1
            else:
                right = curr_frame + 1

        conf_score = cal_score(curr_frame, left)
        prev = batch_2d[left]
        # if right == 351:
        #     break
        next = batch_2d[right]
        for k in range(17):  # per joint loop
            j_2d = frame[k]  # 4 x,ind,y,ind
            p_2d = prev[k]
            n_2d = next[k]
            p_up = 1
            n_up = 1
            if j_2d[0] == 0.0:
                if p_2d[1] == 1.0 and n_2d[1] == 1.0:
                    j_2d_x = (p_2d[0] + n_2d[0]) / 2
                    j_2d_y = (p_2d[2] + n_2d[2]) / 2
                    j_2d[0] = j_2d_x
                    j_2d[2] = j_2d_y
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                elif p_2d[1] == 1.0 and n_2d[1] != 1.0:
                    j_2d[0] = p_2d[0]
                    j_2d[2] = p_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    n_up = 0
                elif p_2d[1] != 1.0 and n_2d[1] == 1.0:
                    j_2d[0] = n_2d[0]
                    j_2d[2] = n_2d[2]
                    j_2d[1] = conf_score
                    # j_2d[3] = conf_score
                    p_up = 0
        # if p_up==1:
        left = left - 1
        # if n_up==1:
        right = right + 1

        if curr_frame >= b:
            # print("ignore last 5 frames")
            break
        if right > b-1:
            right = b-1
        if left < center_index:
            break
    # self.batch_2d = self.batch_2d[:, :, [0, 2, 1]]
    return batch_2d
class Fusion(data.Dataset):
    def __init__(self, opt, dataset, root_path, train=True):
        self.data_type = opt.dataset
        self.train = train
        self.keypoints_name = opt.keypoints
        self.root_path = root_path

        self.train_list = opt.subjects_train.split(',')
        self.test_list = opt.subjects_test.split(',')
        self.action_filter = None if opt.actions == '*' else opt.actions.split(',')
        self.downsample = opt.downsample
        self.subset = opt.subset
        self.stride = opt.stride
        self.crop_uv = opt.crop_uv
        self.test_aug = opt.test_augmentation
        self.pad = opt.pad
        if self.train:
            self.keypoints = self.prepare_data(dataset, self.train_list)
            self.cameras_train, self.poses_train, self.poses_train_2d = self.fetch(dataset, self.train_list,
                                                                                   subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_train, self.poses_train,
                                              self.poses_train_2d, self.stride, pad=self.pad,
                                              augment=opt.data_augmentation, reverse_aug=opt.reverse_augmentation,
                                              kps_left=self.kps_left, kps_right=self.kps_right,
                                              joints_left=self.joints_left,
                                              joints_right=self.joints_right, out_all=opt.out_all)
            print('INFO: Training on {} frames'.format(self.generator.num_frames()))
        else:
            self.keypoints = self.prepare_data(dataset, self.test_list)
            self.cameras_test, self.poses_test, self.poses_test_2d = self.fetch(dataset, self.test_list,
                                                                                subset=self.subset)
            self.generator = ChunkedGenerator(opt.batch_size // opt.stride, self.cameras_test, self.poses_test,
                                              self.poses_test_2d,
                                              pad=self.pad, augment=False, kps_left=self.kps_left,
                                              kps_right=self.kps_right, joints_left=self.joints_left,
                                              joints_right=self.joints_right)
            self.key_index = self.generator.saved_index
            print('INFO: Testing on {} frames'.format(self.generator.num_frames()))
    ###############################interpolation  #3#


    ###########################################
    def prepare_data(self, dataset, folder_list):
        for subject in folder_list:
            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1] 
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

        keypoints = np.load(self.root_path + 'data_2d_' + self.data_type + '_' + self.keypoints_name + '.npz',allow_pickle=True)
        #keypoints=np.load('cpn_data_2d_cmp1.npz',allow_pickle=True)
        keypoints_symmetry = keypoints['metadata'].item()['keypoints_symmetry']

        self.kps_left, self.kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()

        for subject in folder_list:
            assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
            for action in dataset[subject].keys():
                assert action in keypoints[
                    subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action,
                                                                                                         subject)
                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        for subject in keypoints.keys():
            for action in keypoints[subject]:
                for cam_idx, kps in enumerate(keypoints[subject][action]):
                    cam = dataset.cameras()[subject][cam_idx]
                    if self.crop_uv == 0:
                        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    keypoints[subject][action][cam_idx] = kps
        
        return keypoints

    def fetch(self, dataset, subjects, subset=1, parse_3d_poses=True):
        out_poses_3d = {}
        out_poses_2d = {}
        out_camera_params = {}

        for subject in subjects:
            for action in self.keypoints[subject].keys():
                if self.action_filter is not None:
                    found = False
                    for a in self.action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue

                poses_2d = self.keypoints[subject][action]
                for it in range(4):
                    poses_2d[it]=interpolation_all(poses_2d[it])
                    poses_2d[it]=poses_2d[it][:,:,[0,2,1]]
                self.keypoints[subject][action]=poses_2d
                for i in range(len(poses_2d)):
                    out_poses_2d[(subject, action, i)] = poses_2d[i]

                if subject in dataset.cameras():
                    cams = dataset.cameras()[subject]
                    assert len(cams) == len(poses_2d), 'Camera count mismatch'
                    for i, cam in enumerate(cams):
                        if 'intrinsic' in cam:
                            out_camera_params[(subject, action, i)] = cam['intrinsic']

                if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                    poses_3d = dataset[subject][action]['positions_3d']
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)):
                        out_poses_3d[(subject, action, i)] = poses_3d[i]

        if len(out_camera_params) == 0:
            out_camera_params = None
        if len(out_poses_3d) == 0:
            out_poses_3d = None

        stride = self.downsample
        if subset < 1:
            for key in out_poses_2d.keys():
                n_frames = int(round(len(out_poses_2d[key]) // stride * subset) * stride)
                start = deterministic_random(0, len(out_poses_2d[key]) - n_frames + 1, str(len(out_poses_2d[key])))
                out_poses_2d[key] = out_poses_2d[key][start:start + n_frames:stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][start:start + n_frames:stride]
        elif stride > 1:
            for key in out_poses_2d.keys():
                out_poses_2d[key] = out_poses_2d[key][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[key] = out_poses_3d[key][::stride]
        # print('Saving...')
        # metadata = {
        #     'num_joints': dataset.skeleton().num_joints(),
        #     'keypoints_symmetry': [dataset.skeleton().joints_left(), dataset.skeleton().joints_right()]
        # }
        # np.savez_compressed('cpn_data_2d_cmp.npz', positions_2d=out_poses_2d, metadata=metadata)
        #np.save('cpn_2d_up.npy',self.keypoints)
        return out_camera_params, out_poses_3d, out_poses_2d

    def __len__(self):
        return len(self.generator.pairs)

    def __getitem__(self, index):
        seq_name, start_3d, end_3d, flip, reverse = self.generator.pairs[index]

        cam, gt_3D, input_2D, action, subject, cam_ind = self.generator.get_batch(seq_name, start_3d, end_3d, flip, reverse)
        
        if self.train == False and self.test_aug:
            _, _, input_2D_aug, _, _,_ = self.generator.get_batch(seq_name, start_3d, end_3d, flip=True, reverse=reverse)
            input_2D = np.concatenate((np.expand_dims(input_2D,axis=0),np.expand_dims(input_2D_aug,axis=0)),0)
            
        bb_box = np.array([0, 0, 1, 1])
        input_2D_update = input_2D

        scale = np.float(1.0)

        return cam, gt_3D, input_2D_update, action, subject, scale, bb_box, cam_ind




