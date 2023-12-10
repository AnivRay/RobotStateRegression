import numpy as np
import torch
import torch.nn as nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        # self.past_actions_count = past_actions_count
        # self.state_vector_len = state_vector_len
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)

class MLP_Fourier(nn.Module):
    def __init__(self, input_size, output_size, past_actions_count, numFreqs=16):
        super(MLP_Fourier, self).__init__()
        self.past_actions_count = past_actions_count
        self.state_vector_len = input_size // past_actions_count
        self.fourier_features = FourierFeatures(num_freq=numFreqs, inputSize=self.state_vector_len)
        self.layers = nn.Sequential(
            nn.Linear(self.state_vector_len * numFreqs * 2 * self.past_actions_count, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        x = x.reshape(-1, self.past_actions_count, self.state_vector_len)
        # print("Before fourier ", x.size())
        x_fourier = self.fourier_features(x)
        # print("After fourier ", x_fourier.size())
        x = x_fourier.reshape(-1, self.past_actions_count * x_fourier.size(-1))
        return self.layers(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Create constant 'pe' matrix with values dependant on position and i
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float))
        x = x + self.pe[:x.size(0), :]
        return x

class FourierFeatures(nn.Module):
    def __init__(self, num_freq=128, inputSize=35): # , sigma=1):
        super(FourierFeatures, self).__init__()

        self.num_freq = num_freq
        self.inputSize = inputSize
        
        # Create a tensor for multiplication factors [1, 2, 3]
        multiplication_factors = torch.rand(self.num_freq).to(device) # torch.arange(self.num_freq).to(device) / self.num_freq
        # Repeat the multiplication factors tensor to match the length of the repeated elements tensor
        self.repeated_factors = multiplication_factors.repeat(inputSize)
        # self.sigma = 1 # sigma

        # self.freq = nn.Linear(in_features=self.inputSize, out_features=self.num_freq)
        # print('non-learnable frequencies: {}'.format(self.sigma))
        # with torch.no_grad(): # fix these weights
        #     self.freq.weight = nn.Parameter(torch.normal(mean=0, std=self.sigma, size=(self.num_freq, self.inputSize)), requires_grad=False)
        #     self.freq.bias = nn.Parameter(torch.zeros(self.num_freq), requires_grad=False)

    # Expects input x ~ (batch_size, sequence_len, state_vector_len)
    def forward(self, x):
        # print(x.size(), self.freq.weight.size())
        # x = self.freq(x)

        x = torch.repeat_interleave(x, self.num_freq, dim=-1)
        # Multiply element-wise
        x = x * self.repeated_factors

        x = torch.cat([torch.sin(2 * np.pi * x), torch.cos(2 * np.pi * x)], 
                    dim=-1)
        
        # print("x size after cat ", x.size())
        
        assert x.shape[-1] == 2 * self.num_freq * self.inputSize

        return x

class ActionTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, nhead, nlayer, action_len):
        super(ActionTransformer, self).__init__()
        self.past_actions_count = action_len
        self.embedding = nn.Linear(in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayer)

        self.decoder = nn.Linear(d_model * action_len, out_dim)

    def forward(self, x):
        x = x.reshape(-1, self.past_actions_count, self.embedding.in_features)
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.size(0), -1)
        x = self.decoder(x)

        return x

class ActionTransformerFourier(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, nhead, nlayer, action_len, numFreqs):
        super(ActionTransformerFourier, self).__init__()
        self.in_dim = in_dim
        self.past_actions_count = action_len
        self.fourier_features = FourierFeatures(num_freq=numFreqs, inputSize=in_dim)
        self.embedding = nn.Linear(2 * numFreqs * in_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayer)

        self.decoder = nn.Linear(d_model * action_len, out_dim)

    def forward(self, x):
        x = x.reshape(-1, self.past_actions_count, self.in_dim)
        x = self.fourier_features(x)
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        x = x.reshape(x.size(0), -1)
        x = self.decoder(x)

        return x