'''
TCANet: A Temporal Convolutional Attention Network for Motor Imagery EEG Decoding

The paper is currently undergoing proofreading. 
Once it is officially published online, 
the complete source code will be promptly shared.

Author: zhaowei701@163.com

'''
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
        filter_1: int = 16,
        kern_length: int = 32,
        drop_prob: float = 0.5,
        depth: int = 5,
        kernel_size: int = 8,
        filters: int = 48,
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
        # CNN block
        self.mseegnet = MSCNet(
            number_channel=self.n_chans,
            dropout_rate=self.drop_prob,
            pooling_size=POOLING_SIZE
        )

        # TCN Block
        self.tcn_block = TCNBlock(
            input_dimension=96,
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
        Forward pass of the TCANet model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_chans, n_times).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs).
        """
        x = self.mseegnet(x)
        x = self.tcn_block(x)
        sa = self.sa(x)
        x = self.drop(sa + x)

        features = self.flatten(x)
        x = self.classifier(features)

        return x, features   
