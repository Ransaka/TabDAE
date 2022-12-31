import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, encoding_dim, dropout_rate=0.2,activation=nn.ReLU,output_activation=nn.Sigmoid):
        super(DenoisingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        # Build the encoder network
        encoder_layers = []
        in_dim = input_dim
        for i, out_dim in enumerate(hidden_dims):
            out_dim = int(in_dim / 2) if i > 0 else out_dim
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.BatchNorm1d(out_dim))
            encoder_layers.append(activation())
            encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, encoding_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build the decoder network
        decoder_layers = []
        in_dim = encoding_dim
        for i, out_dim in enumerate(reversed(hidden_dims)):
            out_dim = int(in_dim / 2) if i > 0 else out_dim
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.BatchNorm1d(out_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        decoder_layers.append(output_activation())
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x, add_noise=True):
        if add_noise:
            x = x + torch.randn_like(x) * 0.1
        encoding = self.encoder(x)
        reconstructed = self.decoder(encoding)
        return reconstructed