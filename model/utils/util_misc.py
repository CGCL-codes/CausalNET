import numpy as np
import itertools
import random
import numpy as np
import torch
import omegaconf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def reproduc(seed, benchmark=False, deterministic=True):
    """Make experiments reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic


def calc_and_log_metrics(prob_mat, true_cm, log, log_step, threshold=0.5*0.5, pruned=False, prune_strategy='soft'):
    if pruned:
        if prune_strategy == "soft":
            prefix = "metrics_pruned/"
        if prune_strategy == "hard":
            prefix = "metrics_pruned_hard/"
    else:
        prefix = "metrics/"
    
    causal_graph = prob_mat > threshold

    np.fill_diagonal(prob_mat, 0)  # DAG: set the diagnal to be 0
    np.fill_diagonal(causal_graph, 0)  # DAG: set the diagnal to be 0
    
    edge_num = causal_graph.sum()
    
    if edge_num > 0:
        tp = np.sum(causal_graph * true_cm)
        tn = np.sum((1-causal_graph) * (1-true_cm))
        fp = np.sum(causal_graph * (1-true_cm))
        fn = np.sum((1-causal_graph) * true_cm)
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        gscore = max(0, (tp - fp)) / (tp + fn)
        auc = roc_auc_score(true_cm.reshape(-1) > threshold, causal_graph.reshape(-1))
        auc_prob = roc_auc_score(true_cm.reshape(-1) > threshold, prob_mat.reshape(-1))
    else:
        tp, fp, tpr, fpr, precision, f1, gscore, auc, auc_prob = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    log.log_metrics({prefix+"edge_num": edge_num}, log_step)
    log.log_metrics({prefix+"tp": tp}, log_step)
    log.log_metrics({prefix+"fp": fp}, log_step)
    log.log_metrics({prefix+"tpr": tpr}, log_step)
    log.log_metrics({prefix+"fpr": fpr}, log_step)
    log.log_metrics({prefix+"precision": precision}, log_step)
    log.log_metrics({prefix+"f1": f1}, log_step)
    log.log_metrics({prefix+"gscore": gscore}, log_step)
    log.log_metrics({prefix+"auc": auc}, log_step)
    log.log_metrics({prefix+"auc_prob": auc_prob}, log_step)

    return 'log finished'


def plot_causal_matrix_in_training(prob_mat, log, log_step, threshold=0.5*0.5, plot_each_time=True, pruned=False, prune_strategy='soft'):
    """
    Input:
    - prob_mat: i.e. CPG generated using Sigmoid based on Parameter Matrix
    """
    if pruned:
        if prune_strategy == "soft":
            prefix = "Pruned "
        if prune_strategy == "hard":
            prefix = "Hard Pruned "
    else:
        prefix = ""
    
    n_node, n_node = prob_mat.shape
    causal_graph = prob_mat > threshold  # parameter matrix => CPG
    
    # np.fill_diagonal(prob_mat, 0)  # DAG: set the diagnal to be 0
    np.fill_diagonal(causal_graph, 0)  # DAG: set the diagnal to be 0

    # Show Discovered Graph (CPG)
    sub_cg = plot_causal_matrix(
        prob_mat,
        figsize=[3*1.5*n_node, 3*1*n_node],
        vmin=0, vmax=1)
    log.log_figures(sub_cg, name=prefix+"Discovered Prob", iters=log_step)

    # Show Thresholded Graph (CPG + threshold)
    sub_cg = plot_causal_matrix(
        causal_graph,
        figsize=[1.5*n_node, 1*n_node])
    log.log_figures(sub_cg, name=prefix+"Discovered Graph", iters=log_step)
    

def plot_causal_matrix(cmtx, class_names=None, figsize=None, vmin=None, vmax=None, show_text=True, cmap="magma"):
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = cmtx.shape[0]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    
    figsize[0] = 30 if figsize[0] > 30 else figsize[0]
    figsize[1] = 20 if figsize[1] > 20 else figsize[1]
    
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] < threshold else "black"
        if cmtx.shape[0] < 20 and show_text:
            plt.text(j, i, format(cmtx[i, j], ".2e") if cmtx[i, j] != 0 else ".",
                    horizontalalignment="center", color=color,)

    plt.tight_layout()

    return figure


def omegaconf2list(opt, prefix='', sep='.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix, k, v))
        elif isinstance(v, (float, str, int,)):
            notation_list.append("{}{}={}".format(prefix, k, v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix, k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep, sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list


def omegaconf2dotlist(opt, prefix='',):
    return omegaconf2list(opt, prefix, sep='.')


def omegaconf2dict(opt, sep):
    notation_list = omegaconf2list(opt, sep=sep)
    dict = {notation.split('=', maxsplit=1)[0]: notation.split(
        '=', maxsplit=1)[1] for notation in notation_list}
    return dict


