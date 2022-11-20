# NFTT: Image Classification with Noisy Labels

This is the pytorch implementation of our proposed method, NFTT (**N**oisy **F**ilter + **T**riple **T**eaching), for image classification with noisy labels. 

Our proposed approach consists of two stages:
1. Pre-trained a deep neural network as a Noisy Filter to remove the detected mislabeled data points.
2. Using filtered samples as input, the three networks are trained simultaneously. In each mini-batch, each network will back propagate the small loss samples selected by one of its peer networks (switching from batches to batches) and update its parameters. 



## Setups
All code was developed and tested on a single machine equiped with a NVIDIA GeForce GTX. The environment is as bellow:

- Windows 10
- CUDA 11.3
- Python 3.9.7 (Anaconda 4.12.0 64 bit)
- PyTorch 1.11.0
- numpy 1.22.3

## Datasets

**For CIFAR-10**, the dataset is available under {ROOT}/dataset as below:
```
${ROOT}
|-- dataset
`-- |-- cifar-10-batches-py
    `-- |-- cifar10_noisy_labels_task1.json
        |-- cifar10_noisy_labels_task2.json
        |-- cifar10_noisy_labels_task3.json
        |-- data_batch_1
        |-- data_batch_2
        |-- data_batch_3
        |-- data_batch_4
        |-- data_batch_5
        `-- test_batch
```

**For ANIMAL-10N**, please download from [GoogleDrive] (https://drive.google.com/file/d/1NPq9Ujuylg_xgiiMPQbwHr3rFwi7quYP/view?usp=sharing) 
Extract them under {ROOT}/dataset, and make them look like this:
```
${ROOT}
|-- dataset
`-- |-- animal10N
    `-- |-- train
        |   |-- cat
        |   |-- cheetah
        |   |-- chimpanzee
        |   |-- coyote
        |   |-- guinea_pig
        |   |-- hamster
        |   |-- jaguer
        |   |-- lynx
        |   |-- orangutan
        |   `-- wolf
        `-- test
            |-- cat
            |-- cheetah
            |-- chimpanzee
            |-- coyote
            |-- guinea_pig
            |-- hamster
            |-- jaguer
            |-- lynx
            |-- orangutan
            `-- wolf
```

## Running NFTT on CIFAR-10 or ANIMAL-10N
Here is an example:
```bash
python main.py --dataset cifar10 --cifar10_task_num 1 --tri_or_co Tri --noisy_filter_or_not 1 --shallow_or_not 0  
```

## Performance

Results on CIFAR-10 (50 Epochs)
| CIFAR-10 TASKS | Co-Teaching  | Noise Filter + Co-Teaching | Noise Filter + Triple-Teaching | Noise Filter + Triple-Teaching (with a shallow network) |
| ---------------: | -----: | -------: | --------: | --------: |
|  Symmetry 40%    | 87.186% | 86.59%   | **87.187%**    | 86.14%    |
|  Symmetry 80%    | 46.27% | 62.30%   | 62.53%    | **62.97%**    |
|  Asymmetry 40%   | 85.69% | 83.22%   | **85.90%**    | 77.26%    |

Results on ANIMAL-10N
|  | Co-Teaching  | Noise Filter + Co-Teaching | Noise Filter + Triple-Teaching | Noise Filter + Triple-Teaching (with a shallow network) |
| ---------------: | -----: | -------: | --------: | --------: |
|  ANIMAL-10N      | 71.27% | 69.08%        | **71.33%**    |  69.35%   |

## Reference

The implementation of this project refers to the source code of the paper "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels", cited below:

```bash
@inproceedings{han2018coteaching,
  title={Co-teaching: Robust training of deep neural networks with extremely noisy labels},
  author={Han, Bo and Yao, Quanming and Yu, Xingrui and Niu, Gang and Xu, Miao and Hu, Weihua and Tsang, Ivor and Sugiyama, Masashi},
  booktitle={NeurIPS},
  pages={8535--8545},
  year={2018}
}
```
