# TCANet for motor imagery EEG classification
## TCANet: A Temporal Convolutional Attention Network for Motor Imagery EEG Decoding [[Paper](https://link.springer.com/article/10.1007/s11571-025-10275-5)] [PDF (https://rdcu.be/eq1n8)]

core idea: Multi-scale CNN + TCN + multi-head self-atttention

Our research builds upon and improves the [CTNet](https://github.com/snailpt/CTNet) and [MSCFormer](https://github.com/snailpt/MSCFormer).

### Abstract:
Decoding motor imagery electroencephalogram (MI-EEG) signals is fundamental to the development of brain–computer interface (BCI) systems. However, robust decoding remains a challenge due to the inherent complexity and variability of MI-EEG signals. This study proposes the Temporal Convolutional Attention Network (TCANet), a novel end-to-end model that hierarchically captures spatiotemporal dependencies by progressively integrating local, fused, and global features. Specifically, TCANet employs a multi-scale convolutional module to extract local spatiotemporal representations across multiple temporal resolutions. A temporal convolutional module then fuses and compresses these multi-scale features while modeling both short- and long-term dependencies. Subsequently, a stacked multi-head self-attention mechanism refines the global representations, followed by a fully connected layer that performs MI-EEG classification. The proposed model was systematically evaluated on the BCI IV-2a and IV-2b datasets under both subject-dependent and subject-independent settings. In subject-dependent classification, TCANet achieved accuracies of 83.06% and 88.52% on BCI IV-2a and IV-2b respectively, with corresponding Kappa values of 0.7742 and 0.7703, outperforming multiple representative baselines. In the more challenging subject-independent setting, TCANet achieved competitive performance on IV-2a and demonstrated potential for improvement on IV-2b. 

### Overall Framework:
![architecture of TCANet](https://raw.githubusercontent.com/snailpt/TCANet/refs/heads/main/TCANet_architecture.png)


### Dataset & prepare processing
the same as [CTNet](https://github.com/snailpt/CTNet)



#### Comparison of Subject-specific classification accuracy (in %) and kappa on the BCI IV-2a dataset.
| Method \ Subject      | A01   | A02   | A03   | A04   | A05   | A06   | A07   | A08   | A09   | Average |
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| ShallowConvNet   | 82.64 | 56.94 | 90.97 | 68.06 | 71.18 | 57.64 | 75.69 | 82.29 | 75.69 | 73.46   |
| DeepConvNet      | 79.17 | 51.74 | 87.85 | 75.69 | 76.39 | 60.07 | 93.06 | 79.51 | 84.38 | 76.43   |
| EEGNet          | 85.76 | 65.28 | 88.89 | 69.79 | 71.18 | 57.29 | 74.65 | 80.90 | 84.72 | 75.38   |
| EEGInception     | 70.14 | 53.82 | 70.49 | 68.40 | 73.26 | 53.82 | 68.75 | 72.22 | 68.75 | 66.63   |
| TSception       | 62.15 | 39.58 | 73.26 | 54.86 | 64.93 | 47.22 | 59.72 | 63.19 | 63.54 | 58.72   |
| EEGTCNet       | 79.51 | 65.97 | 92.36 | 69.44 | 73.96 | 60.42 | 85.07 | 81.94 | 76.04 | 76.08   |
| ADFCNN          | 88.19 | 60.07 | 92.01 | 78.82 | 70.49 | 65.97 | 83.68 | 84.03 | 81.60 | 78.32   | 
| MSCFormer       | 86.46 | 61.46 | 93.75 | 80.90 | 77.78 | 69.44 | 91.32 | 83.68 | 78.47 | 80.36   |
| **TCANet (proposed)** | 88.89 | 70.14 | 92.71 | 79.86 | 77.78 | 74.31 | 92.71 | 85.76 | 85.42 | **83.06** | 

Note: Comparison of experimental results when data augmentation only generates 1 times the original training sample size



#### Comparison of Subject-specific classification accuracy (in %) and kappa on the BCI IV-2b dataset.
| Method \ Subject      | A01   | A02   | A03   | A04   | A05   | A06   | A07   | A08   | A09   | Average | 
|-----------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|---------|
| ShallowConvNet        | 71.88 | 64.29 | 83.75 | 96.88 | 92.81 | 84.69 | 90.94 | 90.94 | 85.62 | 84.64   |
| DeepConvNet       | 79.06 | 65.71 | 82.19 | 97.50 | 95.31 | 80.62 | 91.25 | 92.19 | 89.06 | 85.88   |
| EEGNet            | 75.94 | 66.07 | 85.31 | 98.44 | 94.38 | 84.38 | 91.25 | 94.69 | 87.50 | 86.44   |
| EEGInception      | 77.81 | 66.07 | 85.62 | 98.12 | 98.12 | 87.81 | 90.31 | 95.31 | 90.00 | 87.69   |
| TSception        | 75.31 | 63.57 | 75.31 | 95.00 | 90.31 | 74.38 | 84.38 | 90.00 | 80.62 | 80.99   |
| EEGTCNet         | 76.88 | 65.00 | 84.69 | 96.88 | 89.69 | 86.56 | 91.88 | 94.69 | 85.94 | 85.80   |
| ADFCNN          | 79.38 | 62.86 | 82.50 | 97.19 | 95.31 | 84.38 | 91.25 | 92.50 | 87.50 | 85.87   |
| MSCFormer        | 75.00 | 68.57 | 80.00 | 98.44 | 95.94 | 85.00 | 93.75 | 94.38 | 88.44 | 86.61   |
| **TCANet (proposed)** | 82.50 | 70.71 | 86.88 | 97.81 | 94.69 | 87.19 | 92.50 | 95.94 | 88.44 | **88.52** | 


#### Comparison of cross-subject classification accuracy (in %) and kappa on the BCI IV-2a & IV-2b datasets.
| Method \ Subject       | BCI IV-2a | BCI IV-2b |
|------------------------|-----------|-----------|
| ShallowConvNet     | 58.64     | 74.92     |
| DeepConvNet        | 61.86     | 76.10     |
| EEGNet           | **62.44** | 75.92     |
| EEGInception        | 58.66     | 74.64     |
| TSception       | 50.70     | 71.67     |
| EEGTCNet           | 57.91     | 75.80     |
| ADFCNN             | 60.40     | **76.24** |
| MSCFormer        | 59.39     | 74.92     |
| **TCANet (proposed)**  | 60.98     | 74.61     |



### Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

Zhao, W., Lu, H., Zhang, B. et al. TCANet: a temporal convolutional attention network for motor imagery EEG decoding. Cogn Neurodyn 19, 91 (2025). https://doi.org/10.1007/s11571-025-10275-5

