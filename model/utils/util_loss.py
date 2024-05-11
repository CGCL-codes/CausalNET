import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import get_non_pad_mask


def compute_dag_ness(A):
    # Hadamard product
    B = torch.mul(A, A)

    # Matrix Exponential
    exp_B = torch.matrix_exp(B)

    # Tr
    tr_exp_B = torch.trace(exp_B)
     
    # DAG-ness
    d = A.shape[0]
    dag_ness = tr_exp_B - d
    
    return dag_ness


def prune_by_dag_ness_soft(prob_mat, threshold):
    """
    prune "approximate dag" into "exact dag", based on "dag ness"
    soft: if delta(dag ness) = 0, then keep this edge.
    """
    from copy import deepcopy
    print('')
    
    num_types = prob_mat.shape[0]
    device = prob_mat.device.type
    
    thres_mask = prob_mat > threshold
    prob_mat = prob_mat * thres_mask
    prob_mat = prob_mat * (1 - torch.eye(num_types).to(device))
    
    # 1. sort by causal intensity
    elements = []
    tmp_prob_mat = deepcopy(prob_mat)
    for i in range(num_types):
        for j in range(num_types):
            elements.append((tmp_prob_mat[i, j], i, j))
    elements = [element for element in elements if element[0] > 0]
    elements = sorted(elements, key=lambda x: x[0])
    
    if len(elements) == 0:
        print('Warning: no element is larger than zero, return torch.zeros_like')
        return torch.zeros_like(prob_mat)
    
    # 2. prune based on causal intensity and dag
    count = 0
    for idx in range(len(elements)):
        element = elements[idx]
        intensity = element[0]
        row, col = element[1], element[2]
        dag_ness_old = compute_dag_ness(prob_mat)
        prob_mat[row, col] = 0  # [prune]
        dag_ness_new = compute_dag_ness(prob_mat)
        if dag_ness_new < dag_ness_old:  # in loop
            count += 1
            if dag_ness_new == 0:
                break  # get dag !
            else:
                continue
        elif dag_ness_new == dag_ness_old:  # not in loop
            prob_mat[row, col] = intensity  # [roll back]
        
    print('{} edges are pruned !'.format(count))
    return prob_mat


def prune_by_dag_ness_hard(prob_mat, threshold):
    """
    prune "approximate dag" into "exact dag", based on "dag ness"
    hard: remove all edges with prob smaller than a threshold
    """
    from copy import deepcopy
    print('')
    
    num_types = prob_mat.shape[0]
    device = prob_mat.device.type
    
    thres_mask = prob_mat > threshold
    prob_mat = prob_mat * thres_mask
    prob_mat = prob_mat * (1 - torch.eye(num_types).to(device))
    
    # sort by causal intensity
    elements = []
    tmp_prob_mat = deepcopy(prob_mat)
    for i in range(num_types):
        for j in range(num_types):
            elements.append((tmp_prob_mat[i, j], i, j))
    elements = [element for element in elements if element[0] > 0]
    elements = sorted(elements, key=lambda x: x[0])
    
    if len(elements) == 0:
        print('Warning: no element is larger than zero, return torch.zeros_like')
        return torch.zeros_like(prob_mat)
    
    # prune based on causal intensity and dag
    count = 0
    for idx in range(len(elements)):
        element = elements[idx]
        row, col = element[1], element[2]
        prob_mat[row, col] = 0  # [prune]
        dag_ness_new = compute_dag_ness(prob_mat)
        ## 
        count += 1
        if dag_ness_new == 0:
            break  # get dag !
        else:
            continue
        
    print('{} edges are pruned !'.format(count))
    return prob_mat


def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


def compute_event(event, non_pad_mask):
    """ 
    Log-likelihood of events. 
    """
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)  # log(1.0)=0 <=> mask
    result = torch.log(event)
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ 
    Log-likelihood of non-events, using Monte Carlo integration. 
    """
    num_samples = 100

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)
    
    # w*h_{t_j} + b: e.g., [lmd(t0~t1), lmd(t1~t2), ..., lmd(tn-2~tn-1)]
    temp_hid = model.linear(data)[:, :-1, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    # softplus[(history + base) + 'current']
    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    # Integral: e.g., [avg[lmd(t)]*(t-t0)], ..., avg[lmd(t)]*(t-tn-2)], where [t in [t0,t1), ..., t in [tn-2, tn-1)]
    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(model, data, time, types, indices):
    """
    event likelihood loss
    """
    non_pad_mask = get_non_pad_mask(types).squeeze(2)  # batch*seq_len

    type_mask = torch.zeros([*types.size(), model.num_types], device=data.device)  # batch*seq_len*num_classes
    for i in range(model.num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(data.device)

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time
    temp_time /= (time[:, :-1] + 1)
    pass 
    temp_time = temp_time[:,:,None].expand(-1,-1,model.num_types)
    all_hid = model.linear(data)[:, :-1, :]
    all_lambda = softplus(all_hid + model.alpha * temp_time, model.beta)
    type_lambda = torch.sum(all_lambda * type_mask[:, 1:, :], dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask[:, 1:])  # batch*(seq_len-1)
    event_ll = event_ll.flatten()[indices]  # batch*1

    # non-event log-likelihood
    non_event_ll = compute_integral_unbiased(model, data, time, non_pad_mask, type_mask)
    non_event_ll = non_event_ll.flatten()[indices]

    return event_ll, non_event_ll


def time_loss(prediction, event_time, indices):
    """ 
    timestamp prediction loss
    """
    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]  # time_gaps, len=seq-1, e.g., [t1-t0, t2-t1, ..., tn-1-tn-2]
    prediction = prediction[:, :-1]  # use h_tj to predict ( tj+1 - tj )
    
    non_pad_mask = event_time[:, 1:].ne(0).float()  # with L-1 non-zero items
    diff = prediction - true
    diff *= non_pad_mask

    diff = diff.flatten()[indices]  # batch*1
    se = torch.sum(diff * diff)
    
    return se


def type_loss(prediction, types, indices, loss_func):
    """ 
    event type prediction loss 
    """
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]  # use h_tj to predict type_tj+1

    pred_type = torch.max(prediction, dim=-1)[1]

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)
    
    loss = loss.flatten()[indices]  # batch*1
    loss = torch.sum(loss)
    correct_num = torch.sum(pred_type.flatten()[indices] == truth.flatten()[indices])
    return loss, correct_num


def topo_loss():
    """ 
    event topo prediction loss 
    """
    pass


class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target.to(torch.int64), num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
