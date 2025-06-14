# TCANet for motor imagery EEG classification
## TCANet: A Temporal Convolutional Attention Network for Motor Imagery EEG Decoding [[Paper](https://link.springer.com/article/10.1007/s11571-025-10275-5)]

core idea: Multi-scale CNN + TCN + multi-head self-atttention

Our research builds upon and improves the [CTNet](https://github.com/snailpt/CTNet) and [MSCFormer](https://github.com/snailpt/MSCFormer).

### Abstract:
Decoding motor imagery electroencephalogram (MI-EEG) signals is fundamental to the development of brain–computer interface (BCI) systems. However, robust decoding remains a challenge due to the inherent complexity and variability of MI-EEG signals. This study proposes the Temporal Convolutional Attention Network (TCANet), a novel end-to-end model that hierarchically captures spatiotemporal dependencies by progressively integrating local, fused, and global features. Specifically, TCANet employs a multi-scale convolutional module to extract local spatiotemporal representations across multiple temporal resolutions. A temporal convolutional module then fuses and compresses these multi-scale features while modeling both short- and long-term dependencies. Subsequently, a stacked multi-head self-attention mechanism refines the global representations, followed by a fully connected layer that performs MI-EEG classification. The proposed model was systematically evaluated on the BCI IV-2a and IV-2b datasets under both subject-dependent and subject-independent settings. In subject-dependent classification, TCANet achieved accuracies of 83.06% and 88.52% on BCI IV-2a and IV-2b respectively, with corresponding Kappa values of 0.7742 and 0.7703, outperforming multiple representative baselines. In the more challenging subject-independent setting, TCANet achieved competitive performance on IV-2a and demonstrated potential for improvement on IV-2b. 

### Overall Framework:
![architecture of TCANet](https://raw.githubusercontent.com/snailpt/TCANet/refs/heads/main/TCANet_architecture.png)


### Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊
Zhao, W., Lu, H., Zhang, B. et al. TCANet: a temporal convolutional attention network for motor imagery EEG decoding. Cogn Neurodyn 19, 91 (2025). https://doi.org/10.1007/s11571-025-10275-5

