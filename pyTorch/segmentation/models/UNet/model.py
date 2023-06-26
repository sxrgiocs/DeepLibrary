import torch.nn as nn
from modules import DoubleConv, Down, Up, Out


class UNet(nn.Module):
    """
    3D U-Net model for semantic segmentation. Based on the original paper by
    Ronneberger et al. https://arxiv.org/abs/1505.04597. Works for both 2D and
    3D images. The network is composed by 4 encoder filters which encode the
    input image, one bottleneck layer to reduce dimensionality and 4 decoder
    layers to produce the segmentation. The architecture is designed such as
    the number of filters is automaticall scalable by a factor chosen by the
    user when initializing the class.

    Parameters
    ----------
    in_channels : int
        Number of input channels. Works for both RGB and grayscale images.
    n_classes : int
        Number of output classes. Refers number of labels in the segmentation
        plus the background (eg. 4 labels + background -> nclasses = 5).
    filter_factor : float
        Factor to adjust the number of filters in each layer. Defaults to 1,
        keeping the same number of filters as in the paper by Ronneberger.

    Attributes
    ----------
    encoder : nn.Sequential
        Encoder module responsible for downsampling the input image.
    bottleneck : DoubleConv
        Bottleneck module representing the central part of the U-Net.
    decoder : nn.Sequential
        Decoder module responsible for upsampling and recovering the spatial
        resolution.
    out : Out
        Output module that produces the final segmentation prediction.

    Methods
    -------
    forward(x)
        Forward pass of the U-Net model.
    _encoder(in_channels, filters)
        Function used to create the encoder path of the U-Net
    _decoder(filters)
        Function used to create the decoder path of the U-Net
    """

    def __init__(self, in_channels, n_classes, filter_factor=1):
        super(UNet, self).__init__()

        filters = [64, 128, 256, 512]
        filters = [int(f // filter_factor) for f in filters]

        self.encoder = self._encoder(in_channels, filters)
        self.bottleneck = DoubleConv(filters[-1], filters[-1] * 2)
        self.decoder = self._decoder(filters)
        self.out = Out(filters[0], n_classes)

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        out : torch.Tensor
            Output tensor representing the segmentation prediction. It is
            a probabilistic segmentation since either a Sigmoid or Softmax
            function is applied.
        """

        # Create an empty list to store the encoders for the skip-connections
        encoders = []
        for encoder in self.encoder:
            x = encoder(x)
            encoders.append(x)

        x = self.bottleneck(x)

        for i, decoder in enumerate(self.decoder):
            x = decoder(x, encoders[-(i + 1)])

        out = self.out(x)

        return out

    def _encoder(self, in_channels, filters):
        """
        Function used to create the encoder path of the U-Net. It takes a list
        containing the number of features as an input argument, iterates
        through it and appends each encoder layer to an empty list. Then, the
        list is used to create a torch.nn.Sequential object that can be used in
        the forward pass.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        filters : list[int]
            List of number of filters for each encoder layer.

        Returns
        -------
        encoder : nn.Sequential
            Encoder path of the U-Net.
        """

        # Create empty list to store each encoder layer
        encoder_layers = []
        # First decoder layer takes the input channels, so create it separately
        encoder_layers.append(DoubleConv(in_channels, filters[0]))
        # Create the rest of the layers using a for loop
        for i in range(len(filters) - 1):
            encoder_layers.append(Down(filters[i], filters[i + 1]))

        encoder = nn.Sequential(*encoder_layers)

        return encoder

    def _decoder(self, filters):
        """
        Function uses to create the decoder path of the U-Net. It takes a list
        containing the number of features as an input argument, iterates
        through it and appends each decoder layer to an empty list. Then, the
        list is used to create a torch.nn.Sequential object that can be used in
        the forward pass.

        Parameters
        ----------
        filters : list[int]
            List of number of filters for each encoder layer.

        Returns
        -------
        decoder : nn.Sequential
            Decoder path of the U-Net.
        """

        # Create empty list to store each decoder layer
        decoder_layers = []
        # Create the layers using a for loop
        for i in range(len(filters) - 1, -1, -1):
            decoder_layers.append(Up(filters[i] * 2, filters[i]))

        decoder = nn.Sequential(*decoder_layers)

        return decoder
