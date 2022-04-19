import torch
import torch.nn.functional as F
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_triple(batch_idx, output1, output2, output3, targets, forget_rate):
    # loss1
    loss_1 = F.cross_entropy(output1, targets, reduce=False)
    loss_1_sorted_index = np.argsort(loss_1.cpu().data).to(DEVICE)

    # loss2
    loss_2 = F.cross_entropy(output2, targets, reduce=False)
    loss_2_sorted_index = np.argsort(loss_2.cpu().data).to(DEVICE)

    # loss3
    loss_3 = F.cross_entropy(output3, targets, reduce=False)
    loss_3_sorted_index = np.argsort(loss_3.cpu().data).to(DEVICE)

    # small loss samples
    batch_size = output1.size(0)
    number_to_remember = int((1 - forget_rate) * batch_size)

    # update
    loss_1_sorted_index_remember = loss_1_sorted_index[:number_to_remember]
    loss_2_sorted_index_remember = loss_2_sorted_index[:number_to_remember]
    loss_3_sorted_index_remember = loss_3_sorted_index[:number_to_remember]

    # exchange and update loss
    # switch between each mini-batch
    if batch_idx % 2 == 0:
        loss_1_updated = F.cross_entropy(output1[loss_3_sorted_index_remember], targets[loss_3_sorted_index_remember])
        loss_2_updated = F.cross_entropy(output2[loss_1_sorted_index_remember], targets[loss_1_sorted_index_remember])
        loss_3_updated = F.cross_entropy(output3[loss_2_sorted_index_remember], targets[loss_2_sorted_index_remember])
    else:
        loss_1_updated = F.cross_entropy(output1[loss_2_sorted_index_remember], targets[loss_2_sorted_index_remember])
        loss_2_updated = F.cross_entropy(output2[loss_3_sorted_index_remember], targets[loss_3_sorted_index_remember])
        loss_3_updated = F.cross_entropy(output3[loss_1_sorted_index_remember], targets[loss_1_sorted_index_remember])



    return torch.sum(loss_1_updated)/number_to_remember, torch.sum(loss_2_updated)/number_to_remember, torch.sum(loss_3_updated)/number_to_remember

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).to(DEVICE)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).to(DEVICE)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))


    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember