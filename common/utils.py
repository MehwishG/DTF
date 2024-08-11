import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
from einops import rearrange, repeat

def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def sym_penalty(dataset,keypoints,pred_out):
    """
    get penalty for the symmetry of human body
    :return:
    """
    loss_sym = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            left_bone = [(0,4),(4,5),(5,6),(8,10),(10,11),(11,12)]
            right_bone = [(0,1),(1,2),(2,3),(8,13),(13,14),(14,15)]
        else:
            left_bone = [(0,4),(4,5),(5,6),(8,11),(11,12),(12,13)]
            right_bone = [(0,1),(1,2),(2,3),(8,14),(14,15),(15,16)]
        for (i_left,j_left),(i_right,j_right) in zip(left_bone,right_bone):
            left_part = pred_out[:,:,i_left]-pred_out[:,:,j_left]
            right_part = pred_out[:, :, i_right] - pred_out[:, :, j_right]
            loss_sym += torch.mean(torch.abs(torch.norm(left_part, dim=-1) - torch.norm(right_part, dim=-1)))
    elif dataset.startswith('STB'):
        loss_sym = 0
    return 0.01*loss_sym

def bonelen_consistency_loss(dataset,keypoints,pred_out):
    loss_length = 0
    if dataset == 'h36m':
        if keypoints.startswith('hr'):
            assert "hrnet has not completed"
        else:
            bones = [(0,1), (0,4), (1,2), (2,3), (4,5), (5,6), (0,7), (7,8), (8,9), (9,10), 
                    (8,11), (11,12), (12,13), (8,14), (14,15), (15,16)]
        for (i,j) in bones:
            bonelen = pred_out[:,:,i]-pred_out[:,:,j]
            bone_diff = bonelen[:,1:,:]-bonelen[:,:-1,:]
            loss_length += torch.mean(torch.norm(bone_diff, dim=-1))
    elif dataset.startswith('heva'):
        loss_length = 0

    return 0.01 * loss_length


def ms_velocity_error(predicted, target):
    predicted_s9 = rearrange(predicted, 'b (s f) n c -> (b s) f n c', s=9)
    target_s9 = rearrange(target, 'b (s f) n c -> (b s) f n c', s=9)
    velocity_predicted = torch.diff(predicted_s9, dim=1)
    velocity_target = torch.diff(target_s9, dim=1)
    loss_scale_9 = torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

    predicted_s3 = rearrange(predicted, 'b (s f) n c -> (b s) f n c', s=3)
    target_s3 = rearrange(target, 'b (s f) n c -> (b s) f n c', s=3)
    velocity_predicted = torch.diff(predicted_s3, dim=1)
    velocity_target = torch.diff(target_s3, dim=1)
    loss_scale_3 = torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

    velocity_predicted = torch.diff(predicted, dim=1)
    velocity_target = torch.diff(target, dim=1)
    loss_scale_1 = torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

    return 0 * loss_scale_9 + 0 * loss_scale_3 + 1 * loss_scale_1
    

   
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    # assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))
    
def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(dist[i].item(), 1)
            
    return action_error_sum

def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            
    return action_error_sum


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY 
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: 
        {'p1':AccumLoss(), 'p2':AccumLoss()} 
        for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2


def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all  = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))


    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
                mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name
    

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))








