import numpy as np
import yaml
from typing import List, Tuple, Dict, Union
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn as nn

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    pos_enc = pos * angle_rates
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    return tf.convert_to_tensor(pos_enc, dtype=tf.float32)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_enc = positional_encoding(seq_len, d_model)

    def call(self, inputs):
        return inputs + self.pos_enc

    def get_config(self):
        return {"seq_len": self.seq_len, "d_model": self.d_model}

class SqueezeAndExcitation(nn.Module):
    """Squeeze and excitation module"""

    def __init__(self, channels: int, ratio: int):
        super(SqueezeAndExcitation, self).__init__()
        self._gap = torch.nn.AdaptiveAvgPool3d(1)
        self._fc1 = torch.nn.Linear(channels, channels // ratio, bias=False)
        self._fc2 = torch.nn.Linear(channels // ratio, channels, bias=False)
        self._relu = nn.ReLU(inplace=True)
        self._sigmoid = nn.Sigmoid()

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        b, c, _, _, _ = x_in.shape
        x = self._gap(x_in)
        x = x.view(b, c)
        x = self._relu(self._fc1(x))
        x = self._sigmoid(self._fc2(x))
        x = x.view(b, c, 1, 1, 1)
        x = x_in * x
        return x


class Encoder(nn.Module):
    """Encoder section of the COAE"""

    def __init__(
        self,
        bottleneck_size: int,
        num_encoder_downsamples: int,
        num_layers_E_0: int,
        num_filters_E_0: int,
        activation_E_0: str,
        num_layers_E_1: int,
        num_filters_E_1: int,
        activation_E_1: str,
        num_layers_E_2: int,
        num_filters_E_2: int,
        activation_E_2: str,
        num_layers_E_3: int,
        num_filters_E_3: int,
        activation_E_3: str,
    ):
        super(Encoder, self).__init__()
        self._bottleneck_size = bottleneck_size
        # cap filter counts to prevent overflow
        num_filters_E_0 = min(num_filters_E_0, 80)
        num_filters_E_1 = min(num_filters_E_0, 128)
        num_layers = [num_layers_E_0, num_layers_E_1, num_layers_E_2, num_layers_E_3]
        num_filters = [1, num_filters_E_0, num_filters_E_1, num_filters_E_2, num_filters_E_3][
            : num_encoder_downsamples + 1
        ]
        activations = [activation_E_0, activation_E_1, activation_E_2, activation_E_3]
        self._setup_activation_dict()
        self._build_layers(num_encoder_downsamples, num_layers, num_filters, activations)

    def _setup_activation_dict(self) -> None:
        self._activation_dict = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }

    def _setup_pooling(self, num_encoder_downsamples: int) -> List[Tuple[int]]:
        """MaxPool3d kernel sizes per downsample stage."""
        pools = []
        if num_encoder_downsamples == 1:
            pools.append((36, 36, 45))
        elif num_encoder_downsamples == 2:
            pools.append((9, 6, 9))
            pools.append((4, 6, 5))
        elif num_encoder_downsamples == 3:
            pools.append((4, 6, 5))
            pools.append((3, 3, 3))
            pools.append((3, 2, 3))
        elif num_encoder_downsamples == 4:
            pools.append((3, 3, 5))
            pools.append((3, 3, 3))
            pools.append((2, 2, 3))
            pools.append((2, 2, 1))
        else:
            raise ValueError(
                f"Only valid choices for `num_encoder_downsamples` are 1, 2, 3, or 4]. You input: {num_encoder_downsamples}"
            )
        return pools

    def _build_layers(
        self,
        num_encoder_downsamples: int,
        num_layers: List[int],
        num_filters: List[int],
        activations: List[str],
    ) -> None:
        self._layers = nn.ModuleList()
        pools = self._setup_pooling(num_encoder_downsamples)
        for i in range(num_encoder_downsamples):
            for j in range(num_layers[i]):
                self._layers.append(
                    nn.Conv3d(
                        in_channels=num_filters[i] if j == 0 else num_filters[i + 1],
                        out_channels=num_filters[i + 1],
                        kernel_size=(3, 3, 3),
                        stride=(1, 1, 1),
                        padding="same",
                    )
                )
                self._layers.append(self._activation_dict[activations[i].lower()])
                self._layers.append(SqueezeAndExcitation(num_filters[i + 1], 8))
            self._layers.append(nn.BatchNorm3d(num_filters[i + 1]))
            self._layers.append(nn.MaxPool3d(kernel_size=pools[i]))
        self._output_filters = num_filters[i + 1]
        self._layers.append(nn.Flatten())
        self._layers.append(
            nn.Linear(in_features=2 * num_filters[i + 1], out_features=self._bottleneck_size)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for layer in self._layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """Decoder section of the COAE"""

    def __init__(
        self,
        bottleneck_size: int,
        num_decoder_upsamples: int,
        num_layers_D_0: int,
        num_filters_D_0: int,
        activation_D_0: str,
        num_layers_D_1: int,
        num_filters_D_1: int,
        activation_D_1: str,
        num_layers_D_2: int,
        num_filters_D_2: int,
        activation_D_2: str,
        num_layers_D_3: int,
        num_filters_D_3: int,
        activation_D_3: str,
    ):
        super(Decoder, self).__init__()
        self._bottleneck_size = bottleneck_size
        num_layers = [num_layers_D_0, num_layers_D_1, num_layers_D_2, num_layers_D_3]
        num_filters = self._process_filters(
            num_decoder_upsamples,
            num_filters_D_0,
            num_filters_D_1,
            num_filters_D_2,
            num_filters_D_3,
        )
        activations = [activation_D_0, activation_D_1, activation_D_2, activation_D_3]
        self._setup_activation_dict()
        self._build_layers(num_decoder_upsamples, num_layers, num_filters, activations)

    def _process_filters(
        self,
        num_decoder_upsamples: int,
        num_filters_D_0: int,
        num_filters_D_1: int,
        num_filters_D_2: int,
        num_filters_D_3: int,
    ) -> List[int]:
        """Filter counts per upsample stage, ending in 1 output channel."""
        filters_dict = {
            1: [num_filters_D_0, 1],
            2: [num_filters_D_0, min(num_filters_D_1, 80), 1],
            3: [num_filters_D_0, min(num_filters_D_1, 128), min(num_filters_D_2, 80), 1],
            4: [
                num_filters_D_0,
                num_filters_D_1,
                min(num_filters_D_2, 128),
                min(num_filters_D_3, 80),
                1,
            ],
        }
        num_filters = filters_dict[num_decoder_upsamples]
        return num_filters

    def _setup_activation_dict(self) -> None:
        self._activation_dict = {
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "relu": nn.ReLU(),
            "leakyrelu": nn.LeakyReLU(0.1),
            "softplus": nn.Softplus(),
        }

    def _setup_upsampling(self, num_decoder_upsamples: int) -> List[Tuple[int]]:
        """Upsample scale factors per stage."""
        unpools = []
        if num_decoder_upsamples == 1:
            unpools.append((36, 36, 45))
        elif num_decoder_upsamples == 2:
            unpools.append((4, 6, 5))
            unpools.append((9, 6, 9))
        elif num_decoder_upsamples == 3:
            unpools.append((3, 2, 3))
            unpools.append((3, 3, 3))
            unpools.append((4, 6, 5))
        elif num_decoder_upsamples == 4:
            unpools.append((2, 2, 1))
            unpools.append((2, 2, 3))
            unpools.append((3, 3, 3))
            unpools.append((3, 3, 5))
        else:
            raise ValueError(
                f"Only valid choices for `num_decoder_upsamples` are 1, 2, 3, or 4]. You input: {num_decoder_upsamples}"
            )
        return unpools

    def _build_layers(
        self,
        num_decoder_upsamples: int,
        num_layers: List[int],
        num_filters: List[int],
        activations: List[str],
    ) -> None:
        self._initial_layers = nn.ModuleList(
            [
                nn.Linear(in_features=self._bottleneck_size, out_features=2 * num_filters[0]),
                self._activation_dict["leakyrelu"],
            ]
        )
        self._first_filter = num_filters[0]
        self._layers = nn.ModuleList()
        unpools = self._setup_upsampling(num_decoder_upsamples)
        for i in range(num_decoder_upsamples):
            self._layers.append(nn.Upsample(scale_factor=unpools[i]))
            if (i + 1) != num_decoder_upsamples:
                for j in range(num_layers[i]):
                    self._layers.append(
                        nn.Conv3d(
                            in_channels=num_filters[i] if j == 0 else num_filters[i + 1],
                            out_channels=num_filters[i + 1],
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding="same",
                        )
                    )
                    self._layers.append(self._activation_dict[activations[i].lower()])
                    self._layers.append(SqueezeAndExcitation(num_filters[i + 1], 4))
                self._layers.append(nn.BatchNorm3d(num_filters[i + 1]))
        self._layers.append(
            nn.Conv3d(
                in_channels=num_filters[-2],
                out_channels=num_filters[-1],
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding="same",
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._initial_layers:
            x = layer(x)
        # x = x.view(x.size(0), self._first_filter, 2, 1, 1)
        x = x.view(-1, self._first_filter, 2, 1, 1)
        for layer in self._layers:
            x = layer(x)
        return x


class COAE(nn.Module):
    """Convolutional Orthogonal Autoencoder, built from a hyperparameter config."""

    def __init__(self, config: dict):
        super(COAE, self).__init__()
        self._bottleneck_size = config.get("bottleneck_size", 10)
        self._alpha = config.get("alpha", 1.0)
        self.encoder = Encoder(
            bottleneck_size=self._bottleneck_size,
            num_encoder_downsamples=config.get("num_encoder_downsamples", 4),
            num_layers_E_0=config.get("num_layers_E_0", 1),
            num_layers_E_1=config.get("num_layers_E_1", 1),
            num_layers_E_2=config.get("num_layers_E_2", 1),
            num_layers_E_3=config.get("num_layers_E_3", 1),
            num_filters_E_0=config.get("num_filters_E_0", 16),
            num_filters_E_1=config.get("num_filters_E_1", 32),
            num_filters_E_2=config.get("num_filters_E_2", 64),
            num_filters_E_3=config.get("num_filters_E_3", 128),
            activation_E_0=config.get("activation_E_0", "relu"),
            activation_E_1=config.get("activation_E_1", "relu"),
            activation_E_2=config.get("activation_E_2", "relu"),
            activation_E_3=config.get("activation_E_3", "relu"),
        )
        self.decoder = Decoder(
            bottleneck_size=self._bottleneck_size,
            num_decoder_upsamples=config.get("num_decoder_upsamples", 4),
            num_layers_D_0=config.get("num_layers_D_0", 1),
            num_layers_D_1=config.get("num_layers_D_1", 1),
            num_layers_D_2=config.get("num_layers_D_2", 1),
            num_layers_D_3=config.get("num_layers_D_3", 1),
            num_filters_D_0=config.get("num_filters_D_0", 128),
            num_filters_D_1=config.get("num_filters_D_1", 64),
            num_filters_D_2=config.get("num_filters_D_2", 32),
            num_filters_D_3=config.get("num_filters_D_3", 16),
            activation_D_0=config.get("activation_D_0", "relu"),
            activation_D_1=config.get("activation_D_1", "relu"),
            activation_D_2=config.get("activation_D_2", "relu"),
            activation_D_3=config.get("activation_D_3", "relu"),
        )
        # self.encoder = torch.nn.parallel.DataParallel(self.encoder)
        # self.decoder = torch.nn.parallel.DataParallel(self.decoder)

    def _load_config(self, filename: str) -> Dict[str, Union[str, int, float]]:
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        return config.get("model")

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.encoder(x)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._encode(x)
        xhat = self._decode(z)
        return xhat, z
