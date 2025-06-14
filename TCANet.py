"""
Zhao, W., Lu, H., Zhang, B. et al. TCANet: a temporal convolutional attention network for motor imagery EEG decoding. Cogn Neurodyn 19, 91 (2025). https://doi.org/10.1007/s11571-025-10275-5

Email: zhaowei701@163.com

"""


gpu_number = 0
gpus = [gpu_number]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)


from pandas import ExcelWriter
from torchsummary import summary
from torch.backends import cudnn


import os
import numpy as np
import pandas as pd
import random
import datetime
import time
import math
import warnings
warnings.filterwarnings("ignore")
cudnn.benchmark = False
cudnn.deterministic = True

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Variable
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

from utils import numberClassChannel
from utils import load_data_evaluate
from utils import calMetrics
from utils import calculatePerClass
from utils import numberClassChannel


class MSCNet(nn.Module):
    def __init__(self, 
                 f1=16, 
                 pooling_size=52, 
                 dropout_rate=0.5, 
                 number_channel=22):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 125), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 62), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )        
        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 31), (1, 1), padding='same'),
            nn.Conv2d(f1, f1, (number_channel, 1), (1, 1), groups=f1),
            nn.BatchNorm2d(f1),
            nn.ELU(),
            nn.AvgPool2d((1,pooling_size)), 
            nn.Dropout(dropout_rate),
        )
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        
        
    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x3 = self.cnn3(x)
        #通道方向合并
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.projection(x)
        return x    

class _TCNBlock(nn.Module):
    """
    TCN Block with Proper Padding for Causal Convolution
    """

    def __init__(
        self,
        input_dimension: int,
        depth: int,
        kernel_size: int,
        filters: int,
        drop_prob: float,
        activation: nn.Module = nn.ELU,
    ):
        super().__init__()
        self.activation = activation()
        self.drop_prob = drop_prob
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList()
        self.downsample = (
            nn.Conv1d(input_dimension, filters, kernel_size=1, bias=False)
            if input_dimension != filters
            else None
        )

        for i in range(depth):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation  # Calculate causal padding
            conv_block = nn.Sequential(
                CausalConv1d(
                    in_channels=input_dimension if i == 0 else filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    # padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.drop_prob),
                CausalConv1d(
                    in_channels=filters,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    # padding=padding,
                    bias=False,
                ),
                nn.BatchNorm1d(filters),
                self.activation,
                nn.Dropout(self.drop_prob),
            )
            self.layers.append(conv_block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, time_steps, input_dimension)
        x = x.permute(0, 2, 1)  # (batch_size, input_dimension, time_steps)

        res = x if self.downsample is None else self.downsample(x)
        for index, layer in enumerate(self.layers):
            out = layer(x)
            out = out + res  # Residual connection
            out = self.activation(out)
            res = out  # Update residual
            x = out  # Update input for next layer

        out = out.permute(0, 2, 1)  # (batch_size, time_steps, filters)
        return out


class CausalConv1d(nn.Conv1d):
    """
    1D Causal Convolution to ensure no information leakage from future timesteps.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply padding manually to ensure causality
        # print("kernel:", self.kernel_size, self.dilation)
        padding = (self.kernel_size[0] - 1) * self.dilation[0]  # Calculate padding size
        x = F.pad(x, (padding, 0))  # Only pad on the left (causal padding)
        return super().forward(x)

class TCANet(nn.Module):
    def __init__(
        self,
        # Signal related parameters
        n_chans=22,
        out_features: int = 4,
        n_times=1000,
        # Model parameters
        activation: nn.Module = nn.ELU,
        depth_multiplier: int = 2,
        filter_1: int = 8,
        kern_length: int = 32,
        drop_prob: float = 0.5,
        depth: int = 2,
        kernel_size: int = 4,
        filters: int = 16,
        max_norm_const: float = 0.25,
        
    ):
        super().__init__()
        self.n_times = n_times
        self.n_chans=n_chans
        self.activation = activation
        self.drop_prob = drop_prob
        self.depth_multiplier = depth_multiplier
        self.filter_1 = filter_1
        self.kern_length = kern_length
        self.depth = depth
        self.kernel_size = kernel_size
        self.filters = filters
        self.max_norm_const = max_norm_const
        self.filter_2 = self.filter_1 * self.depth_multiplier
        self.out_features = out_features
        # EEGNet_TC Block
        self.mseegnet = MSCNet(
            number_channel=self.n_chans,
            dropout_rate=self.drop_prob,
            pooling_size=POOLING_SIZE
        )

        # TCN Block
        self.tcn_block = _TCNBlock(
            input_dimension=48,
            depth=self.depth,
            kernel_size=self.kernel_size,
            filters=self.filters,
            drop_prob=0.25,
            activation=self.activation,
        )
        self.sa = TransformerEncoder(HEADS, DEPTH, self.filters)
        self.drop = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.classifier=nn.Linear(self.filters*(1000//POOLING_SIZE), self.out_features,)
        self.norm_rate = self.max_norm_const
        self.classifier.register_forward_pre_hook(self.apply_max_norm_classifier)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the EEGTCNet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        x = self.mseegnet(x)  # (batch_size, filter, reduced_time, 1)
        x = self.tcn_block(x)  # (batch_size, time_steps, filters)
        sa = self.sa(x)
        x = self.drop(sa + x)

        features = self.flatten(x)
        x = self.classifier(features)  # (batch_size, n_outputs)

        return x, features           
            
    def apply_max_norm_classifier(self, module, input):
        with torch.no_grad():
            norm = self.classifier.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.norm_rate)
            scale = desired / (norm + 1e-8)
            self.classifier.weight.data *= scale    
    
    
  
    
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    


# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )



class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(flatten_number, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn, emb_size, drop_p):
        super().__init__()
        self.fn = fn
        self.drop = nn.Dropout(drop_p)
        self.layernorm = nn.LayerNorm(emb_size)

    def forward(self, x, **kwargs):
        x_input = x
        res = self.fn(x, **kwargs)
        out = self.layernorm(self.drop(res)+x_input)
        return out

    
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                MultiHeadAttention(emb_size, num_heads, drop_p),
                ), emb_size, drop_p),

            )    
        
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, heads, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size, heads) for _ in range(depth)])



class PositioinalEncoding(nn.Module):
    def __init__(self, embedding, length=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.encoding = nn.Parameter(torch.randn(1, length, embedding))
    def forward(self, x): # x-> [batch, embedding, length]
        x = x + self.encoding[:, :x.shape[1], :].cuda()
        return self.dropout(x)        
        
    
class EEGTransformer(nn.Module):
    def __init__(self, 
                 parameters,
                 database_type='A', 
                 
                 **kwargs):
        super().__init__()
        self.number_class, self.number_channel = numberClassChannel(database_type)
        parameters.number_channel = self.number_channel
        self.cnn = TCANet(
                # Signal related parameters
                n_chans=self.number_channel,
                out_features=self.number_class,

            ) 
    def forward(self, x):
        out, features = self.cnn(x)
        return features, out


class ExP():
    def __init__(self, nsub, data_dir, result_name,
                 parameters,
                 evaluate_mode = 'LOSO-no',
                 dataset_type='A',
                 n_fold = 0,
                 ):
        
        super(ExP, self).__init__()
        self.n_fold = n_fold
        self.dataset_type = dataset_type
        self.batch_size = parameters.batch_size
        self.lr = parameters.learning_rate
        self.b1 = 0.5
        self.b2 = 0.999
        self.n_epochs = parameters.epochs
        self.nSub = nsub
        self.nFold = n_fold
        self.number_augmentation = parameters.number_aug
        self.number_seg = parameters.number_seg
        self.root = data_dir
        self.result_name = result_name
        self.evaluate_mode = evaluate_mode
        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.number_class, self.number_channel = numberClassChannel(dataset_type)
        self.model = EEGTransformer(
            database_type=self.dataset_type, 
            parameters = parameters, 
            ).cuda()
        #self.model = nn.DataParallel(self.model, device_ids=gpus)
        self.model = self.model.cuda()
        self.model_filename = self.result_name + '/model_nsub_{}_nfold_{}.pth'.format(self.nSub, n_fold+1)

    def interaug(self, timg, label):  
        aug_data = []
        aug_label = []
        number_records_by_augmentation = self.number_augmentation * int(self.batch_size / self.number_class)
        number_segmentation_points = 1000 // self.number_seg
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]
            tmp_aug_data = np.zeros((number_records_by_augmentation, 1, self.number_channel, 1000))
            for ri in range(number_records_by_augmentation):
                for rj in range(self.number_seg):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], self.number_seg)
                    tmp_aug_data[ri, :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points] = \
                        tmp_data[rand_idx[rj], :, :, rj * number_segmentation_points:(rj + 1) * number_segmentation_points]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:number_records_by_augmentation])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label-1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label



    def get_source_data(self):
        (self.train_data,    # (batch, channel, length)
         self.train_label, 
         self.test_data, 
         self.test_label) = load_data_evaluate(self.root, self.dataset_type, self.nSub, mode_evaluate=self.evaluate_mode)

        self.train_data = np.expand_dims(self.train_data, axis=1)  # (288, 1, 22, 1000)
        self.train_label = np.transpose(self.train_label)  

        self.allData = self.train_data
        self.allLabel = self.train_label[0]  

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]  # (288, 1, 22, 1000)
        self.allLabel = self.allLabel[shuffle_num]

        print('-'*20, 
              "raw train size：", self.train_data.shape, 
              "test size：", self.test_data.shape, 
              "subject:", self.nSub,
              "fold:", self.nFold+1)
        # self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]
        
        
        
        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std
        
        isSaveDataLabel = False #True
        if isSaveDataLabel:
            np.save("./gradm_data/train_data_{}.npy".format(self.nSub), self.allData)
            np.save("./gradm_data/train_lable_{}.npy".format(self.nSub), self.allLabel)
            np.save("./gradm_data/test_data_{}.npy".format(self.nSub), self.testData)
            np.save("./gradm_data/test_label_{}.npy".format(self.nSub), self.testLabel)

        
        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    
    def test_model(self, model, dataloader):
        model.eval()
        outputs_list = []
        label_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(dataloader):
                # val model
                img = img.type(self.Tensor).cuda()
                label = label.type(self.LongTensor).cuda()
                _, Cls = model(img)
                outputs_list.append(Cls)
                del img, Cls
                torch.cuda.empty_cache()
                label_list.append(label)
            
        Cls = torch.cat(outputs_list)
        val_label = torch.cat(label_list)
        val_loss = self.criterion_cls(Cls, val_label)
        val_pred = torch.max(Cls, 1)[1]
        val_acc = float((val_pred == val_label).cpu().numpy().astype(int).sum()) / float(val_label.size(0))
        return val_acc, val_loss, val_pred

    

    def train(self):
        timg, label, test_data, test_label = self.get_source_data()
        train_data_list_per_class = []
        train_label_list_per_class = []
        for clsAug in range(self.number_class):
            cls_idx = np.where(label == clsAug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]        
            train_data_list_per_class.append(tmp_data)
            train_label_list_per_class.append(tmp_label)
            
        train_data_list = []
        train_label_list = []
        val_data_list, val_label_list =  [], []
        seed = 1234+self.nSub
        for clsAug in range(self.number_class):
            number_samples = len(train_data_list_per_class[clsAug])
            number_test = number_samples // 5      # for validation
            # Index is in random order, used to split the training set and test set
            index_list = list(range(number_samples))
            np.random.seed(seed+clsAug)
            # Random index, fetching data based on the index, equivalent to shuffle the data set
            index_shuffled = np.random.permutation(index_list)    
            if self.n_fold!=4 :
                index_val = [i for i in range(self.n_fold*number_test, (self.n_fold+1)*number_test)]
            else:
                # Since 288 (BCI IV-2a & IV-2b) cannot be divided by 5, the last fold is all the remaining
                index_val = [i for i in range(self.n_fold*number_test, number_samples)]

            index_train = [i for i in range(number_samples) if i not in index_val]
            # Indexes of training and test sets
            index_train = index_shuffled[index_train]
            index_val = index_shuffled[index_val]   
            
            train_data_class = train_data_list_per_class[clsAug][index_train]
            train_label_class = train_label_list_per_class[clsAug][index_train]
            train_data_list.append(train_data_class)
            train_label_list.append(train_label_class)
            
            val_data_class = train_data_list_per_class[clsAug][index_val]
            val_label_class = train_label_list_per_class[clsAug][index_val]            
            val_data_list.append(val_data_class)
            val_label_list.append(val_label_class)        
        img = np.concatenate(train_data_list)
        label = np.concatenate(train_label_list)
        val_data = np.concatenate(val_data_list)
        val_label = np.concatenate(val_label_list)
      
        
        print('-'*20, 
              "train size：", img.shape, 
              "val size：", val_data.shape, )
        
        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)
        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        
        val_data = torch.from_numpy(val_data)
        val_label = torch.from_numpy(val_label - 1)
        val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
        self.val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)
                
        
        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2),)

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))
        best_epoch = 0
        num = 0
        min_loss = 100
        max_acc = 0
        # recording train_acc, train_loss, test_acc, test_loss
        result_process = []
        # Train the cnn model
        for e in range(self.n_epochs):
            epoch_process = {}
            epoch_process['epoch'] = e
            # in_epoch = time.time()
            self.model.train()
            outputs_list = []
            label_list = []
            for i, (img, label) in enumerate(self.dataloader):
                number_sample = img.shape[0]
                
                # split raw train dataset into real train dataset and validate dataset
                train_data = img
                train_label = label

                
                # real train dataset
                img = Variable(train_data.type(self.Tensor))
                label = Variable(train_label.type(self.LongTensor))
                
                # data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                # concat real train dataset and generate aritifical train dataset
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                # training model
                features, outputs = self.model(img)
                outputs_list.append(outputs)
                label_list.append(label)
                # print("train outputs: ", outputs.shape, type(outputs))
                # print(features.size())
                loss = self.criterion_cls(outputs, label) 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            del img
            torch.cuda.empty_cache()
            # out_epoch = time.time()
            # test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                # validate model
                val_acc, val_loss, _ = self.test_model(self.model, self.val_dataloader)
                
                epoch_process['val_acc'] = val_acc                
                epoch_process['val_loss'] = val_loss.detach().cpu().numpy()  
                
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))
                epoch_process['train_acc'] = train_acc
                epoch_process['train_loss'] = loss.detach().cpu().numpy()

                num = num + 1

                # if min_loss>val_loss:                
                if max_acc<val_acc or (max_acc==val_acc and min_loss>val_loss):
                    max_acc = val_acc
                    min_loss = val_loss
                    
                    best_epoch = e
                    epoch_process['epoch'] = e
                    torch.save(self.model, self.model_filename)
                    test_acc, test_loss, y_pred  = self.test_model(self.model, self.test_dataloader)
                    print("{}_{} train_acc: {:.4f} train_loss: {:.6f}\tval_acc: {:.6f} val_loss: {:.7f} test_acc:{:.6f}".format(self.nSub,
                                                                                           epoch_process['epoch'],
                                                                                           epoch_process['train_acc'],
                                                                                           epoch_process['train_loss'],
                                                                                           epoch_process['val_acc'],
                                                                                           epoch_process['val_loss'],
                                                                                           test_acc                                     
                                                                                        ))
            
                
            result_process.append(epoch_process)  

        
        # load model for test
        self.model.eval()
        self.model = torch.load(self.model_filename).cuda()
        outputs_list = []
        with torch.no_grad():
            for i, (img, label) in enumerate(self.test_dataloader):
                img_test = Variable(img.type(self.Tensor)).cuda()
                # label_test = Variable(label.type(self.LongTensor))

                # test model
                features, outputs = self.model(img_test)
                val_pred = torch.max(outputs, 1)[1]
                outputs_list.append(outputs)
        outputs = torch.cat(outputs_list) 
        y_pred = torch.max(outputs, 1)[1]
        
        
        test_acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
        
        print("epoch: ", best_epoch, '\tThe test accuracy is:', test_acc)


        df_process = pd.DataFrame(result_process)

        return test_acc, test_label, y_pred, df_process, best_epoch, outputs
        # writer.close()
        

def main(dirs,           
         paramters,
         evaluate_mode = 'subject-dependent', 
         dataset_type='A',    # A->'BCI IV2a', B->'BCI IV2b'
         ):

    if not os.path.exists(dirs):
        os.makedirs(dirs)
    result_write_metric = ExcelWriter(dirs+"/result_metric.xlsx")
    result_metric_dict = {}
    y_true_pred_dict = { }

    process_write = ExcelWriter(dirs+"/process_train.xlsx")
    pred_true_write = ExcelWriter(dirs+"/pred_true.xlsx")
    pred_softmax = ExcelWriter(dirs+"/pred_softmax.xlsx")
    subjects_result = []
    
    
    best_epochs = []
    result_fold = []
    for i in range(paramters.subject_number):      
        starttime = datetime.datetime.now()
        seed_n = np.random.randint(2024)
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)
        index_round =0
        print('Subject %d' % (i+1))

        subjects_result_fold = []
        # If you want to do cross validation, this should be set to 5. Currently, 
        for n_fold in range(1):     #  only 20% is used as the validation set, and only 1.
            exp = ExP(i + 1, DATA_DIR, dirs, 
                      paramters,
                      evaluate_mode = evaluate_mode,
                      dataset_type=dataset_type,
                      n_fold=n_fold,
                      )

            testAcc, Y_true, Y_pred, df_process, best_epoch,pred_output = exp.train()
            probs = torch.softmax(pred_output, dim=1).cpu().numpy()
            df_probs = pd.DataFrame(probs)
            df_probs.to_excel(pred_softmax, sheet_name=str(i+1)+'_'+str(n_fold))
            true_cpu = Y_true.cpu().numpy().astype(int)
            pred_cpu = Y_pred.cpu().numpy().astype(int)
            df_pred_true = pd.DataFrame({'pred': pred_cpu, 'true': true_cpu})
            df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1)+'_'+str(n_fold))
            y_true_pred_dict[i] = df_pred_true

            accuracy, precison, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
            subject_result = {'accuray': accuracy*100,
                              'precision': precison*100,
                              'recall': recall*100,
                              'f1': f1*100, 
                              'kappa': kappa*100
                              }
            df_process.to_excel(process_write, sheet_name=str(i+1)+'_'+str(n_fold))
            best_epochs.append(best_epoch)
            print(' THE BEST ACCURACY IS ' + str(testAcc) + "\tkappa is " + str(kappa) )
    

            endtime = datetime.datetime.now()
            print('subject %d duration: '%(i+1) + str(endtime - starttime))

            if i == 0:
                yt = Y_true
                yp = Y_pred
            else:
                yt = torch.cat((yt, Y_true))
                yp = torch.cat((yp, Y_pred))
            subjects_result_fold.append(subject_result)
            df_result_fold = pd.DataFrame(subjects_result)
            
        df = pd.DataFrame(subjects_result_fold)
        df.to_excel(result_write_metric, index=False,  sheet_name=str(i+1))
        result_fold_mean = df.mean()
        print("{} subject {} fold mean: \n {}".format(i+1, n_fold+1, result_fold_mean))
        subjects_result.append(result_fold_mean)
    df_result = pd.DataFrame(subjects_result)
    process_write.close()
    pred_true_write.close()
    pred_softmax.close()


    print('**The average Best accuracy is: ' + str(df_result['accuray'].mean()) + "kappa is: " + str(df_result['kappa'].mean()) + "\n" )
    print("best epochs: ", best_epochs)
    #df_result.to_excel(result_write_metric, index=False)
    result_metric_dict = df_result

    mean = df_result.mean(axis=0)
    mean.name = 'mean'
    std = df_result.std(axis=0)
    std.name = 'std'
    df_result = pd.concat([df_result, pd.DataFrame(mean).T, pd.DataFrame(std).T])
    
    df_result.to_excel(result_write_metric, index=False, sheet_name='mean')
    print('-'*9, ' all result ', '-'*9)
    print(df_result)
    
    print("*"*40)

    result_write_metric.close()

    
    return result_metric_dict


class Parameters():
    def __init__(self, dropout_rate):
        self.heads = 8
        self.depth = 5
        self.emb_size = 16*3
        self.f1 = 16
        self.pooling_size = 52
        self.dropout_rate = dropout_rate
        self.subject_number = 9
        self.learning_rate = 0.001
        self.batch_size = 72 
        self.epochs=1000
        self.number_aug=1
        # The actual number of training batches is: self.batch_size*(1+self.number_aug)
        self.number_seg=8
        self.gpus=gpus        

if __name__ == "__main__":
    #----------------------------------------
    DATA_DIR = r'../mymat_raw/'
    EVALUATE_MODE = 'LOSO-No' # leaving one subject out subject-dependent  subject-indenpedent

    TYPE = 'A'
    if EVALUATE_MODE!='LOSO':
        CNN_DROPOUT_RATE = 0.5
    else:
        CNN_DROPOUT_RATE = 0.25    
    
    parameters = Parameters(CNN_DROPOUT_RATE)
    POOLING_SIZE = 56
    HEADS = 2
    DEPTH = 6

    number_class, number_channel = numberClassChannel(TYPE)
    RESULT_NAME = "TCANet_{}".format(TYPE)

    sModel = EEGTransformer(
        database_type=TYPE, 
        parameters = parameters,  
        ).cuda()
    summary(sModel, (1, number_channel, 1000)) 

    print(time.asctime(time.localtime(time.time())))

    result = main(RESULT_NAME,
                  parameters,
                    evaluate_mode = EVALUATE_MODE,
                    dataset_type=TYPE,
                  )
    print(time.asctime(time.localtime(time.time())))
