'''
The implementation of this project refers to the source code of the paper "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", cited below:
@inproceedings{han2018coteaching,
  title={Co-teaching: Robust training of deep neural networks with extremely noisy labels},
  author={Han, Bo and Yao, Quanming and Yu, Xingrui and Niu, Gang and Xu, Miao and Hu, Weihua and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={NeurIPS},
  pages={8535--8545},
  year={2018}
}
'''
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import os
import copy
from model import resnet34, CNN, Net, CNN_S
from loader_for_CIFAR import *
from animal10n_loader import *
from loss_triple_teaching import loss_triple, loss_coteaching
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--cifar10_task_num', type = int, default = 1)
parser.add_argument('--tri_or_co', type = str, help = 'Tri or Co', default = 'Tri')
parser.add_argument('--noisy_filter_or_not', type = int, default = 1) # 0-False and 1-True
parser.add_argument('--shallow_or_not', type = int, default = 0) # 0-False and 1-True
parser.add_argument('--forget_rate', type = float, help = 'forget rate', default = None)
parser.add_argument('--num_gradual', type = int, default = 10, help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type = float, default = 1, help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type = str, help = 'cifar10 or animal10n', default = 'cifar10')
parser.add_argument('--n_epoch_triple_teaching', type=int, default=50)
parser.add_argument('--n_epoch_noisy_filter', type=int, default=20)
parser.add_argument('--epoch_decay_start_triple_teaching', type=int, default=30)
parser.add_argument('--epoch_decay_start_noisy_filter', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=2, help='how many subprocesses to use for data loading')
args = parser.parse_args()

# python main.py --dataset cifar10 --cifar10_task_num 3 --tri_or_co Tri --noisy_filter_or_not 1 --shallow_or_not 1 --n_epoch_triple_teaching 50 --n_epoch_noisy_filter 20 --epoch_decay_start_triple_teaching 30 --epoch_decay_start_noisy_filter 10
# python main.py --dataset cifar10 --cifar10_task_num 2 --tri_or_co Co --noisy_filter_or_not 0 --shallow_or_not 0
# todo: set hyper-parameters

LEARNING_RATE = args.lr
BATCH_SIZE = args.batch_size
NUM_EPOCHS_R = args.n_epoch_noisy_filter # number of epochs for detector training
NUM_EPOCHS_TT = args.n_epoch_triple_teaching # number of epochs for triple-teaching training
T_k = args.num_gradual
C = args.exponent # exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal t0 c in Tc for R(T) in the Algorithm
EPOCH_DECAY_START = args.epoch_decay_start_triple_teaching
EPOCH_DECAY_START_R = args.epoch_decay_start_noisy_filter
if args.noisy_filter_or_not == 0:
    CLEAN = False
elif args.noisy_filter_or_not == 1:
    CLEAN = True
else:
    raise Exception("noisy_filter_or_not should be 0 or 1: 0 means False and 1 means True")

if args.shallow_or_not == 0:
    SHALLOW = False
elif args.shallow_or_not == 1:
    SHALLOW = True
else:
    raise Exception("shallow_or_not should be 0 or 1: 0 means False and 1 means True")

# Dataset
DATASET = args.dataset
TASK_num = args.cifar10_task_num # 1 or 2 or 3
TASK = args.tri_or_co # Co or Tri

if DATASET == 'cifar10':
    DATA_PATH = os.path.join(os.getcwd(),'dataset','cifar-10-batches-py')
    
    if TASK_num == 1:
        NOISE_FILE = os.path.join(DATA_PATH,'cifar10_noisy_labels_task1.json') 
    elif TASK_num == 2:
        NOISE_FILE = os.path.join(DATA_PATH,'cifar10_noisy_labels_task2.json') 
    elif TASK_num == 3:
        NOISE_FILE = os.path.join(DATA_PATH,'cifar10_noisy_labels_task3.json') 
    else:
        print("non-existing task")
elif DATASET == 'animal10n':
    DATA_PATH = os.path.join(os.getcwd(),'dataset','animal10N')
else:
    raise Exception("dataset should be cifar10 or animal10n")

# Architecture
NUM_CLASSES = 10
INPUT_CHANNEL = 3

# Todo: setup the learning rate and momentum for Adam Optimizer
initial_mom = 0.9
decay_mom = 0.1

learning_rates = [LEARNING_RATE]*NUM_EPOCHS_TT
if CLEAN:
    learning_rates_for_stage1 = [LEARNING_RATE]*NUM_EPOCHS_R

momentums = [initial_mom]*NUM_EPOCHS_TT
if CLEAN:
    momentums_for_stage1 = [initial_mom]*NUM_EPOCHS_R

for i in range(EPOCH_DECAY_START, NUM_EPOCHS_TT):
    learning_rates[i] = float(NUM_EPOCHS_TT - i)/(NUM_EPOCHS_TT-EPOCH_DECAY_START)*LEARNING_RATE
    momentums[i] = decay_mom

if CLEAN:
    for i in range(EPOCH_DECAY_START_R, NUM_EPOCHS_R):
        learning_rates_for_stage1[i] = float(NUM_EPOCHS_R - i)/(NUM_EPOCHS_R-EPOCH_DECAY_START_R)*LEARNING_RATE
        momentums_for_stage1[i] = decay_mom
print(learning_rates)

# Todo: define a function to adjust learning rates during the training process

# for triple-teaching or co-teaching
def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rates[epoch]
        param_group['betas'] = (momentums[epoch],0.999) # only change beta1

# for noisy filter 
def adjust_lr_r(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rates_for_stage1[epoch]
        param_group['betas'] = (momentums_for_stage1[epoch],0.999) # only change beta1


# Todo: check if cuda is available or not

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using", DEVICE)


# todo: prepare the training data and test data

# data transformation
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 train dataset
torchvision.datasets.CIFAR10(root='./dataset', train=True,
                             download=True, transform=transform)

# CIFAR10 test dataset
torchvision.datasets.CIFAR10(root='./dataset', train=False,
                             download=True, transform=transform)


# todo: define a function to generate noisy labels using load_for_CIFAR

def cifar_loader(dataset_name, data_root, batch_size, split, num_workers=2, noise_file=None):
    loader = cifar_dataloader(dataset_name, batch_size=batch_size,
                              num_workers=num_workers, root_dir=data_root,
                              noise_file=noise_file)
    if split == 'train':
        return loader.run('train')
    elif split == 'test':
        return loader.run('test')
    else:
        raise ValueError("split should be train or test")

def animal10n_loader(data_root, batch_size, split, num_workers=2):
    loader = animal10n_dataloader(batch_size=batch_size, num_workers=num_workers, root_dir=data_root)
    
    if split == 'train':
        return loader.run('train')
    elif split == 'test':
        return loader.run('test')
    else:
        raise ValueError("split should be train or test")

# Todo: define a function to compute the noise rate of the noisy dataset
def get_noise_rate(noisy_labels, clean_labels):
    num_clean_samples = 0
    total_num_samples = noisy_labels.shape[0]
    for i in range(total_num_samples):
        if noisy_labels[i] == clean_labels[i]:
            num_clean_samples += 1
    return 1 - (num_clean_samples / total_num_samples)


# Todo: define a function to compute the accuracy
def accuracy(outputs, target, correct, batch_idx):
    pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / ((batch_idx + 1) * BATCH_SIZE)
    return acc, correct


# Todo: training function for noisy filter model
def train_for_fix(train_loader, epoch, model, optimizer, criterion):
    print("training noisy filter model...")
    correct = 0
    for batch_idx, (data, targets, targets_idx) in enumerate(train_loader):

        # todo: Forward + Backward + Optimize
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(data) 
        loss = F.cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # todo: save accuracy
        acc, correct = accuracy(logits, targets, correct, batch_idx)
        if batch_idx % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss: {:.6f} Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    acc, optimizer.param_groups[0]['lr']))

    return acc

# Todo: training function for triple-teaching
def train_for_triple_teaching(train_loader, epoch, model1, model2, model3, optimizer1, optimizer2, optimizer3,  drop_rate):
    print("Triple Teaching Training ...")
    correct1 = 0
    correct2 = 0
    correct3 = 0
    for batch_idx, (data, targets, targets_idx) in enumerate(train_loader):

        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        logits1 = model1(data)
        logits2 = model2(data)
        logits3 = model3(data)

        loss_1, loss_2, loss_3 = loss_triple(batch_idx, logits1, logits2, logits3, targets, drop_rate)

        # todo: forward + backward + optimize
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        loss_3.backward()
        optimizer3.step()

        # todo: save accuracy
        acc1, correct1 = accuracy(logits1, targets, correct1, batch_idx)
        acc2, correct2 = accuracy(logits2, targets, correct2, batch_idx)
        acc3, correct3 = accuracy(logits3, targets, correct3, batch_idx)
        if batch_idx % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss 1: {:.6f} Accuracy 1: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_1.item(),
                    acc1, optimizer1.param_groups[0]['lr']))
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss 2: {:.6f} Accuracy 2: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_2.item(),
                    acc2, optimizer2.param_groups[0]['lr']))
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss 3: {:.6f} Accuracy 3: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_3.item(),
                    acc3, optimizer3.param_groups[0]['lr']))
    return acc1, acc2, acc3

# Todo: training function for co-teaching
def train_for_co_teaching(train_loader, epoch, model1, model2, optimizer1, optimizer2, drop_rate):
    print("Co Teaching Training ...")
    correct1 = 0
    correct2 = 0
    for batch_idx, (data, targets, targets_idx) in enumerate(train_loader):

        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        logits1 = model1(data)
        logits2 = model2(data)

        loss_1, loss_2 = loss_coteaching(logits1, logits2, targets, drop_rate)

        # todo: forward + backward + optimize
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()

        # todo: save accuracy
        acc1, correct1 = accuracy(logits1, targets, correct1, batch_idx)
        acc2, correct2 = accuracy(logits2, targets, correct2, batch_idx)
        if batch_idx % 100 == 0:
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss 1: {:.6f} Accuracy 1: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_1.item(),
                    acc1, optimizer1.param_groups[0]['lr']))
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\t Training Loss 2: {:.6f} Accuracy 2: {:.0f}%, Learning rate: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_2.item(),
                    acc2, optimizer2.param_groups[0]['lr']))
    return acc1, acc2

# Todo: testing function (cifar10)
def test(test_loader, model, criterion):
    print("testing...")
    correct = 0
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            test_loss += criterion(outputs, targets)
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

# Todo: testing function (animal10n)
def test_animal10n(test_loader, model, criterion):
    print("animal 10n testing...")
    correct = 0
    with torch.no_grad():
        test_loss = 0
        correct = 0
        for batch_idx, (data, targets,target_index) in enumerate(test_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            # print("output")
            # print(torch.topk(outputs[1], 1).indices[0])
            test_loss += criterion(outputs, targets)
            pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

# todo: define a function to remove all the detected noisy labels (cifar10)
def remove_noisy(train_loader, train_dataset, model):
    print("removing detected noisy labels...")
    delete = 0
    total = 0
    delete_idx_list = []
    for batch_idx, (data, targets, targets_idx) in enumerate(train_loader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        for i in range(outputs.size()[0]):
            if torch.topk(outputs[i], 1).indices[0] != targets[i]:
                delete += 1
                delete_idx_list += [int(targets_idx[i])]
            total += 1

    # delete noisy data from dataset
    train_dataset.__delete__(delete_idx_list)
    print(train_dataset.__len__())
    clean_samples_num = 0
    for k in range(train_dataset.__len__()):
        if train_dataset.noise_label[k] == train_dataset.clean_label[k]:
            clean_samples_num += 1

    # noise rate
    new_noise_rate = 1 - clean_samples_num/(total-delete)

    print('Number of detected noisy samples: {}/{} | Number of clean samples in the filtered dataset: {}/{} | updated noisy rate: {}'.format(
        delete, total, clean_samples_num, total-delete, new_noise_rate
    ))

    # loading the dataset with filtered samples
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers)

    return clean_samples_num, total-delete, new_noise_rate, trainloader

# todo: define a function to remove all the detected noisy labels (animal10n)
def remove_noisy_animal10n(train_loader, train_dataset, model):
    print("removing detected noisy labels...")
    delete = 0
    total = 0
    delete_idx_list = []
    for batch_idx, (data, targets, targets_idx) in enumerate(train_loader):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        for i in range(outputs.size()[0]):
            if torch.topk(outputs[i], 1).indices[0] != targets[i]:
                delete += 1
                delete_idx_list += [int(targets_idx[i])]
            total += 1
    print("number of noisy labels: ", delete, "/", total)

    # delete noisy data from dataset
    train_dataset.__delete__(delete_idx_list)

    # loading the dataset with "clean" labels
    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers)
    return total-delete, trainloader

# todo: define a function to test the noise rate during training the noisy filter (on cifar10)
def test_noise_rate(loader, dataset, model,ep):
    loader_copy = copy.deepcopy(loader)
    dataset_copy = copy.deepcopy(dataset)
    delete = 0
    total = 0
    delete_idx_list = []
    for batch_idx, (data, targets, targets_idx) in enumerate(loader_copy):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        for i in range(outputs.size()[0]):
            if torch.topk(outputs[i], 1).indices[0] != targets[i]:
                delete += 1
                delete_idx_list += [int(targets_idx[i])]
            total += 1

    # delete noisy data from dataset
    dataset_copy.__delete__(delete_idx_list)
    clean_samples_num = 0
    for k in range(dataset_copy.__len__()):
        if dataset_copy.noise_label[k] == dataset_copy.clean_label[k]:
            clean_samples_num += 1

    # noise rate
    new_noise_rate = 1 - (clean_samples_num / (total - delete))
    print('Number of detected noisy samples: {}/{} | Number of clean samples in the filtered dataset: {}/{} | updated noisy rate: {}'.format(
        delete, total, clean_samples_num, total-delete, new_noise_rate
    ))

    # save the noisy filter training results in a txt file
    save_dir_r = os.path.join(os.getcwd(), 'model_result','noisy_filter')
    if DATASET == 'cifar10':
        model_str_r = '{}_{}_shallow_{}_Noisy_Filter_for_epochs_{}_task{}.txt'.format(DATASET,TASK,str(SHALLOW),str(NUM_EPOCHS_R),str(TASK_num))
    else:
        model_str_r = '{}_{}_Noisy_Filter_for_epochs_{}.txt'.format(DATASET,TASK,str(NUM_EPOCHS_R))
    txtfile_r = os.path.join(save_dir_r, model_str_r)
    with open(txtfile_r, "a") as myfile:
        myfile.write('epoch '+str(int(ep)) + ': '+str(clean_samples_num)+ ' '+str(total - delete)+ ' '+str(new_noise_rate)+'\n')

    return new_noise_rate

# todo: define a function to test the detected mislabeled samples during training the noisy filter (on animal10n)
def test_removed_animal10n(loader, dataset, model,ep):
    loader_copy = copy.deepcopy(loader)
    dataset_copy = copy.deepcopy(dataset)
    delete = 0
    total = 0
    delete_idx_list = []
    for batch_idx, (data, targets, targets_idx) in enumerate(loader_copy):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        for i in range(outputs.size()[0]):
            if torch.topk(outputs[i], 1).indices[0] != targets[i]:
                delete += 1
                delete_idx_list += [int(targets_idx[i])]
            total += 1
    print("number of noisy labels: ", delete, "/", total)


    save_dir_r = os.path.join(os.getcwd(), 'model_result','noisy_filter')
    model_str_r = '{}_{}_Noisy_Filter_for_epochs_{}.txt'.format(DATASET,TASK,str(NUM_EPOCHS_R))
    txtfile_r = os.path.join(save_dir_r, model_str_r)
    with open(txtfile_r, "a") as myfile:
        myfile.write('epoch '+str(int(ep)) + ': ' + str(total) + ' '+ str(delete) + ' '+ str(total - delete)+ ' '+'\n')

    return total - delete

def main():
    print("######################Loading Dataset###################################")

    # todo: loading dataset with noisy labels
    if DATASET == 'cifar10':
        print(DATASET)
        # loading train data
        train_dataset, train_loader, noisy_labels, clean_labels = cifar_loader(dataset_name=DATASET, data_root=DATA_PATH,
                                                                            batch_size=BATCH_SIZE, split='train',
                                                                            noise_file=NOISE_FILE)
        # loading test data
        test_loader = cifar_loader(dataset_name=DATASET, data_root=DATA_PATH,
                                batch_size=BATCH_SIZE, split='test',
                                noise_file=NOISE_FILE)
    elif DATASET == 'animal10n':
        print(DATASET)
        train_dataset, train_loader = animal10n_loader(data_root=DATA_PATH, batch_size=BATCH_SIZE, split='train',num_workers=args.num_workers)
        test_dataset, test_loader = animal10n_loader(data_root=DATA_PATH, batch_size=BATCH_SIZE, split='test',num_workers=args.num_workers)
    else:
        raise Exception("dataset should be cifar10 or animal10n.")


    # Todo: set up the original actual noise rate
    if DATASET == 'cifar10':
        NOISE_RATE = get_noise_rate(noisy_labels, clean_labels)
    elif DATASET == 'animal10n':
        NOISE_RATE = 0.08


    # todo: define the neural network architecture

    # Todo: Define a model for detecting noisy labels
    if CLEAN:
        model_R = CNN(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        # model_R = Net().to(DEVICE)

    # Todo: Define models for triple teaching
    # T_Path1 = os.path.join(os.getcwd(),'pre_trained','cifar_triple1_task1.th')
    # T_Path2 = os.path.join(os.getcwd(),'pre_trained','cifar_triple2_task1.th')
    # T_Path3 = os.path.join(os.getcwd(),'pre_trained','cifar_triple3_task1.th')
    if SHALLOW:
        model_teaching_1 = CNN_S(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        model_teaching_2 = CNN_S(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        if TASK == 'Tri':
            model_teaching_3 = CNN_S(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        # model_teaching_1 = Net().to(DEVICE)
        # model_teaching_2 = Net().to(DEVICE)
        # if TASK == 'Tri':
        #     model_teaching_3 = Net().to(DEVICE)
    else:
        model_teaching_1 = CNN(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        model_teaching_2 = CNN(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)
        if TASK == 'Tri':
            model_teaching_3 = CNN(input_channel=INPUT_CHANNEL, n_outputs=NUM_CLASSES).to(DEVICE)


    # todo: set up the optimizer and learning rate scheduler

    # Todo: Define optimizer for the model at stage 1 (filter out noisy labels)
    if CLEAN:
        optimizer_R = optim.Adam(model_R.parameters(), lr=LEARNING_RATE)

    # Todo: Define optimizer for the models at stage 2 (triple-teaching)
    optimizer_teaching_1 = optim.Adam(model_teaching_1.parameters(), lr=LEARNING_RATE)
    optimizer_teaching_2 = optim.Adam(model_teaching_2.parameters(), lr=LEARNING_RATE)
    if TASK == 'Tri':
        optimizer_teaching_3 = optim.Adam(model_teaching_3.parameters(), lr=LEARNING_RATE)

    # todo: set up the loss (empirical risk)
    criterion = nn.CrossEntropyLoss()

    # todo: start training the mode for removing noisy labels
    if CLEAN:
        print("###################### Noisy Filter Training Start ###################################")

        save_dir_r = os.path.join(os.getcwd(), 'model_result','noisy_filter')
        if DATASET == 'cifar10':
            model_str_r = '{}_{}_shallow_{}_Noisy_Filter_for_epochs_{}_task{}.txt'.format(DATASET,TASK,str(SHALLOW),str(NUM_EPOCHS_R),str(TASK_num))
        else:
            model_str_r = '{}_{}_Noisy_Filter_for_epochs_{}.txt'.format(DATASET,TASK,str(NUM_EPOCHS_R))
        txtfile_r = os.path.join(save_dir_r, model_str_r)
        with open(txtfile_r, "a") as myfile:
            myfile.write('Origin: ' + str(NOISE_RATE) + '\n')
        
        if DATASET == 'cifar10':
            R_Path = os.path.join(os.getcwd(),'pre_trained','{}_noisy_filter_task{}.th'.format(DATASET,TASK_num))
        else:
            R_Path = os.path.join(os.getcwd(),'pre_trained','{}_noisy_filter.th'.format(DATASET))
            
        try:
            model_R.load_state_dict(torch.load(R_Path))
        except:
            for ep in range(NUM_EPOCHS_R):
                # change to train mode
                model_R.train()
                # learning rate schedular
                adjust_lr_r(optimizer_R, ep)

                train_for_fix(train_loader, ep, model_R, optimizer_R, criterion)
                if DATASET == 'cifar10':
                    noise_rate_updated = test_noise_rate(train_loader, train_dataset, model_R,ep)
                elif DATASET == 'animal10n':
                    test_removed_animal10n(train_loader, train_dataset, model_R,ep)
            torch.save(model_R.state_dict(), R_Path)

        # todo: remove noisy labels
        if DATASET == 'cifar10':
            nc, nt, noise_rate_updated, clean_trainloader = remove_noisy(train_loader, train_dataset, model_R)
            with open(txtfile_r, "a") as myfile:
                myfile.write('Cleaning result: {}/{} [noise rate: {}]'.format(nc,nt,noise_rate_updated)+'\n')
        else:
            num_cleaned, clean_trainloader = remove_noisy_animal10n(train_loader, train_dataset, model_R)
            with open(txtfile_r, "a") as myfile:
                myfile.write('Cleaning result: {} remained]'.format(num_cleaned)+'\n')

        print("removal pass")

    # Todo: Stage 2 start!

    # Todo: Compute noise_rate and update the parameters NOISE_RATE and FORGET_RATE
    if CLEAN and DATASET == 'cifar10':
        NOISE_RATE = noise_rate_updated

    FORGET_RATE = NOISE_RATE

    # Todo: drop rate schedule
    drop_rate_schedule = np.ones(NUM_EPOCHS_TT) * FORGET_RATE

    # T_k : how many epochs for linear drop rate. can be 5, 10, 15
    drop_rate_schedule[:T_k] = np.linspace(0, FORGET_RATE ** C, T_k)

    # todo: Co/Triple-Teaching part
    print('###################### {}-Teaching Training ###################################'.format(TASK))
    save_dir = os.path.join(os.getcwd(), 'model_result')
    if DATASET == 'cifar10':
        model_str = '{}_{}_Teaching_for_epochs_{}_task{}_noise_rate_{}_clean_{}_shallow_{}.txt'.format(DATASET,TASK,str(NUM_EPOCHS_TT),str(TASK_num),str(NOISE_RATE),str(CLEAN),str(SHALLOW))
    else:
        model_str = '{}_{}_Teaching_for_epochs_{}_noise_rate_{}_clean_{}_shallow_{}.txt'.format(DATASET,TASK,str(NUM_EPOCHS_TT),str(NOISE_RATE),str(CLEAN),str(SHALLOW))
    txtfile = os.path.join(save_dir , model_str)

    with open(txtfile, "a") as myfile:
        myfile.write('lr: {} | epochs for training: {} | epoch start to decay: {} | updated noise rate: {} \n'.format(
            LEARNING_RATE, NUM_EPOCHS_TT, EPOCH_DECAY_START, NOISE_RATE
        ))
    print()

    # train
    for ep in range(NUM_EPOCHS_TT):

        model_teaching_1.train()
        adjust_lr(optimizer_teaching_1,ep)

        model_teaching_2.train()
        adjust_lr(optimizer_teaching_2,ep)

        if TASK == 'Tri':
            model_teaching_3.train()
            adjust_lr(optimizer_teaching_3,ep)

        # train acc
        if TASK == 'Tri':
            if CLEAN:
                train_acc1, train_acc2, train_acc3 = train_for_triple_teaching(clean_trainloader, ep, model_teaching_1, model_teaching_2,model_teaching_3, optimizer_teaching_1, optimizer_teaching_2,optimizer_teaching_3,drop_rate_schedule[ep])
            else:
                train_acc1, train_acc2, train_acc3 = train_for_triple_teaching(train_loader, ep, model_teaching_1, model_teaching_2,model_teaching_3, optimizer_teaching_1, optimizer_teaching_2,optimizer_teaching_3,drop_rate_schedule[ep])
        else:
            if CLEAN:
                train_acc1, train_acc2 = train_for_co_teaching(clean_trainloader, ep, model_teaching_1, model_teaching_2, optimizer_teaching_1, optimizer_teaching_2,drop_rate_schedule[ep])
            else:
                train_acc1, train_acc2 = train_for_co_teaching(train_loader, ep, model_teaching_1, model_teaching_2, optimizer_teaching_1, optimizer_teaching_2,drop_rate_schedule[ep])

        # todo: conduct testing
        # print("######################Test###################################")

        # change to eval mode
        model_teaching_1.eval()
        model_teaching_2.eval()
        if TASK == 'Tri':
            model_teaching_3.eval()

        # test acc
        if DATASET == 'cifar10':
            test_acc1 = test(test_loader, model_teaching_1, criterion)
            test_acc2 = test(test_loader, model_teaching_2, criterion)
            if TASK == 'Tri':
                test_acc3 = test(test_loader, model_teaching_3, criterion)
        else:
            test_acc1 = test_animal10n(test_loader, model_teaching_1, criterion)
            test_acc2 = test_animal10n(test_loader, model_teaching_2, criterion)
            if TASK == 'Tri':
                test_acc3 = test_animal10n(test_loader, model_teaching_3, criterion)

        # saved results
        if TASK == 'Tri':
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(ep)) + ': '+ str(train_acc1) +' '+ str(train_acc2)+' '+ str(train_acc3)+' '+ str(test_acc1)+' '+ str(test_acc2)+' '+ str(test_acc3) + "\n")
            print()
        else:
            with open(txtfile, "a") as myfile:
                myfile.write(str(int(ep)) + ': '+ str(train_acc1) +' '+ str(train_acc2)+' '+ str(test_acc1)+' '+ str(test_acc2) + "\n")
            print()
    # torch.save(model_teaching_1.state_dict(), T_Path1)
    # torch.save(model_teaching_2.state_dict(), T_Path2)
    # torch.save(model_teaching_3.state_dict(), T_Path3)
    



if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!
