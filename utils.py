import cv2
import os
import numpy as np
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 11:08:19 2023

@author: Administrator
"""

from sklearn.metrics import accuracy_score  # 计算准确率
from sklearn.metrics import precision_score  # 计算精度
from sklearn.metrics import recall_score  # 计算召回率
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score

import numpy as np
import pandas as pd
import scipy
import os
import shutil



def load_data_evaluate(dir_path, dataset_type, n_sub, mode_evaluate="LOSO", 
                       markNSub=False):
    '''
    根据评估模式选择加载对应的数据集

    Parameters
    ----------
    dir_path : str
        数据所在的文件夹名称.
    dataset_type : str
        数据集的编码ABC.
    n_sub : int
        受试者编号.
    mode_evaluate : str, optional
        评估模式. The default is "LOSO"： 跨受试者实验。 其他情况下：受试者独立实验
    markNSub: bool, optional
        mark subject number in data at last column
    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    if mode_evaluate=="LOSO":
        if markNSub:
            return load_data_LOSO_markNSub(dir_path, dataset_type, n_sub)
        else:
            return load_data_LOSO(dir_path, dataset_type, n_sub)
    
    else:
        return load_data_subject_dependent(dir_path, dataset_type, n_sub)


def load_data_subject_dependent(dir_path, dataset_type, n_sub):
    '''
    subject-specific 模式下的实验

    Parameters
    ----------
    dir_path : str
        数据所在文件夹名称.
    dataset_type : str
        数据集编号.
    n_sub : int
        第n_sub个受试者.

    Returns
    -------
    X1 : ndarray
        DESCRIPTION.
    y1 : ndarray
        DESCRIPTION.
    X2 : ndarray
        DESCRIPTION.
    y2 : ndarray
        DESCRIPTION.

    '''
    train_data, train_label = load_data(dir_path, dataset_type, n_sub, mode='train')
    test_data, test_label = load_data(dir_path, dataset_type, n_sub, mode='test')
    return train_data, train_label, test_data, test_label


def load_data_LOSO_markNSub(dir_path, dataset_type, subject): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model. 
   
        Parameters
        ----------
        dir_path: string
            dataset path
        dataset_type: string
            数据集: A-> BCI-IV 2a    B-> BCI-IV 2b   C-> HGD
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = np.empty([0, 3]), np.empty([0, 3])
    markNSub = []
    for n_sub in range (1, 10):
        
        X1, y1 = load_data(dir_path, dataset_type, n_sub, mode='train')
        X2, y2 = load_data(dir_path, dataset_type, n_sub, mode='test')
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        
        if (n_sub == subject):
            X_test = X
            y_test = y
        elif X_train.shape[0] == 0:
            X_train = X
            y_train = y
            markNSub.extend([n_sub]*X.shape[0])
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
            markNSub.extend([n_sub]*X.shape[0])
    return X_train, y_train, X_test, y_test, np.array(markNSub)



def load_data_LOSO(dir_path, dataset_type, subject): 
    """ Loading and Dividing of the data set based on the 
    'Leave One Subject Out' (LOSO) evaluation approach. 
    LOSO is used for  Subject-independent evaluation.
    In LOSO, the model is trained and evaluated by several folds, equal to the 
    number of subjects, and for each fold, one subject is used for evaluation
    and the others for training. The LOSO evaluation technique ensures that 
    separate subjects (not visible in the training data) are usedto evaluate 
    the model. 
   
        Parameters
        ----------
        dir_path: string
            dataset path
        dataset_type: string
            数据集: A-> BCI-IV 2a    B-> BCI-IV 2b   C-> HGD
        subject: int
            number of subject in [1, .. ,9]
            Here, the subject data is used  test the model and other subjects data
            for training
    """
    
    X_train, y_train = np.empty([0, 3]), np.empty([0, 3])
    for n_sub in range (1, 10):
        
        X1, y1 = load_data(dir_path, dataset_type, n_sub, mode='train')
        X2, y2 = load_data(dir_path, dataset_type, n_sub, mode='test')
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
                   
        if (n_sub == subject):
            X_test = X
            y_test = y
        elif X_train.shape[0] == 0:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    return X_train, y_train, X_test, y_test


def load_data(dir_path, dataset_type, n_sub, mode='train'):
    '''
    加载mat格式的数据返回data和label的ndarray

    Parameters
    ----------
    dir_path : str
        mat数据文件所在的路径.
    dataset_type : str
        数据集：A-> BCI-IV 2a   B-> BCI-IV 2b   C-> HGD.
    n_sub : int
        受试者编号：[1, 2, ..., 9].
    mode : str
        训练集或测试集.

    Returns
    -------
    data : ndarray
        训练或测试数据集.
    label : ndarray
        标签.

    '''
    if mode=='train':
        mode_s = 'T'
    else:
        mode_s = 'E'
    data_mat = scipy.io.loadmat(dir_path + '{}{:02d}{}.mat'.format(dataset_type, n_sub, mode_s))
    data = data_mat['data']  # (288, 22, 1000)
    label =data_mat['label']
    return data, label



def calMetrics(y_true, y_pred):
    '''
    计算总体准确率、精确率、召回率和F1值

    Parameters
    ----------
    y_true : numpy或Series或列表
        真实值.
    y_pred : numpy或Series或列表
        预测值.

    Returns
    -------
    accuracy : float
        准确率.
    precison : float
        精确率.
    recall : float
        召回率.
    f1 : float
        F1值.
    kappa : float
        kappa值.

    '''
    # 先计算类别数，用于判断是2分类还是多分类
    number = max(y_true)
    if number == 2:
        mode = 'binary'
    else:
        mode = 'macro'
    
    accuracy = accuracy_score(y_true, y_pred)
    precison = precision_score(y_true, y_pred, average=mode)
    recall = recall_score(y_true, y_pred, average=mode)
    f1 = f1_score(y_true, y_pred, average=mode)
    kappa = cohen_kappa_score(y_true, y_pred)
    return accuracy, precison, recall, f1, kappa
    


def calculatePerClass(data_dict, metric_name='Precision'):
    '''
    计算指标对应的各类别性能

    Parameters
    ----------
    data_dict : dict
        包含所有受试者的数据：{'1': DataFrame, '2':DataFrame ...}.
    metric_name : str, optional
        目前支持：'Precision'和'Recalll'. The default is 'Precision'.

    Returns
    -------
    df: DataFrame
        所有受试者的所有类别的指定指标计算结果

    '''
    metric_dict = {}
    for key in data_dict.keys():
        df = data_dict[key]
        if metric_name == 'Precision':
            metric_dict[key] = precision_score(df['true'], df['pred'], average=None)
        elif metric_name == 'Recall':
            metric_dict[key] = recall_score(df['true'], df['pred'], average=None)
    df = pd.DataFrame(metric_dict)
    df = df*100
    df = df.applymap(lambda x: round(x, 2))
    mean = df.apply('mean', axis=1).round(2) 
    std  = df.apply('std', axis=1).round(2) 
    df['mean'] = mean
    df['std'] = std
    df['metrics'] = metric_name
    
    return df



def numberClassChannel(database_type):
    if database_type=='A':
        number_class = 4
        number_channel = 22
    elif database_type=='B':
        number_class = 2
        number_channel = 3
    elif database_type=='C':
        number_class = 4
        number_channel = 128
    return number_class, number_channel




    



class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)

        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer

        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0  # works like mute the min-max scale in the function of scale_cam_image
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def __call__(self, input_tensor, target_category=None):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        # 正向传播得到网络输出logits(未经过softmax)
        output = self.activations_and_grads(input_tensor)
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            # ! 这里进行了修改 output[0]是输入的尺寸，output[1]是输出的尺寸?
            output=output[1]
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        print('the loss is', loss)
        loss.backward(retain_graph=True)
        # loss.backward(torch.ones_like(output), retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap  # + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def center_crop_img(img: np.ndarray, size: int):
    h, w, c = img.shape

    if w == h == size:
        return img

    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    img = cv2.resize(img, dsize=(new_w, new_h))

    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h+size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w+size]

    return img