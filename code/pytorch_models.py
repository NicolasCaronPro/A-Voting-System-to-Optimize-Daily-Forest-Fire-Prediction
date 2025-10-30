from dgl.nn.pytorch.conv import GraphConv, GATConv
import torch
import torch.nn as nn
import dgl.function as fn
    
class MLPLayer(torch.nn.Module):
    def __init__(self, in_feats, hidden_dim, device):
        super(MLPLayer, self).__init__()
        self.mlp = nn.Linear(in_feats, hidden_dim, weight_initializer='glorot', bias=True, bias_initializer='zeros').to(device)
        #self.mlp = torch.nn.Linear(in_feats, hidden_dim).to(device)
    def forward(self, x):
        return self.mlp(x)
    
class NetMLP(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, end_channels, output_channels, n_sequences, device, task_type, return_hidden=False, horizon=0):
        super(NetMLP, self).__init__()
        self.layer1 = MLPLayer(in_dim * n_sequences + end_channels, hidden_dim[0], device) if horizon > 0 else MLPLayer(in_dim * n_sequences, hidden_dim[0], device)
        self.layer3 = MLPLayer(hidden_dim[0], hidden_dim[1], device)
        self.layer4 = MLPLayer(hidden_dim[1], end_channels, device)
        self.layer2 = MLPLayer(end_channels, output_channels, device)
        self.task_type = task_type
        self.n_sequences = n_sequences
        self.soft = torch.nn.Softmax(dim=1)
        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.decoder = None
        self._decoder_input = None
        self.output_channels = output_channels
        self.in_dim = in_dim
        self.horizon = horizon
        self.n_sequences = self.n_sequences

    def forward(self, features, z_prev=None, edges=None):
        if self.horizon > 0:
            if z_prev is None:
                z_prev = torch.zeros((features.shape[0], self.end_channels * self.n_sequences))

        features = features.view(features.shape[0], features.shape[1] * self.n_sequences)
        
        if self.horizon > 0:
            features = torch.cat((features, z_prev), dim=1)

        x = F.relu(self.layer1(features))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        hidden = x
        logits = self.layer2(x)
        if self.task_type == 'classification':
            output = self.soft(logits)
        else:
            output = logits

        self._decoder_input = hidden

        return output, logits, hidden
    
    def define_horizon_decodeur(self):
        self.decode1 = torch.nn.Linear(self.end_channels + (self.in_dim * self.n_sequences) + self.output_channels, self.end_channels)
        self.decode2 = torch.nn.Linear(self.end_channels, self.output_channels)

    def forward_horizon(self, z, y_prev=None, X_futur=None):
        B = z.shape[0]
        if y_prev is None:
            y_prev = torch.zeros((B, getattr(self, "_decoder_output_dim", self.output_channels)), device=z.device)
        if X_futur is None:
            X_futur = torch.zeros((B, (self.in_dim * self.n_sequences)), device=z.device)
        else:
            X_futur = X_futur.view(X_futur.shape[0], X_futur.shape[1] * self.n_sequences)
        
        x = torch.cat((z, y_prev, X_futur), dim=1)

        x = self.decode1(x)
        hidden = F.relu(x)
        logits = self.decode2(hidden)

        if self.task_type == 'classification':
            output = self.soft(logits)
        else:
            output = logits

        return output, logits, hidden
    
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False, return_hidden=False, horizon=0):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            if i == 0:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i] + end_channels if horizon > 0 else channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            else:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output layer
        self.linear1 = torch.nn.Linear(channels[-1], lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, x, edges=None, z_prev=None):
        # Couche d'entrée

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)
        else:
            z_prev = z_prev.view(x.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

class GRU(torch.nn.Module):
    def __init__(self, in_channels, gru_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.0, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False, horizon=0):
        super(GRU, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.gru_size = gru_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels

        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=in_channels + self.end_channels if horizon > 0 else in_channels,
            hidden_size=gru_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(gru_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(gru_size).to(device)

        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output linear layer
        self.linear1 = torch.nn.Linear(gru_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)

        # Reshape to (batch, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.gru_size).to(self.device)

        # GRU forward
        x, _ = self.gru(x, h0)

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        x = self.act_func(self.linear1(x))
        hidden = self.act_func(self.linear2(x))
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden

class LSTM(torch.nn.Module):
    def __init__(self, in_channels, lstm_size, hidden_channels, end_channels, n_sequences, device,
                 act_func='ReLU', task_type='regression', dropout=0.03, num_layers=1,
                 return_hidden=False, out_channels=None, use_layernorm=False, horizon=0):
        super(LSTM, self).__init__()

        self.device = device
        self.return_hidden = return_hidden
        self.num_layers = num_layers
        self.hidden_size = hidden_channels
        self.task_type = task_type
        self.is_graph_or_node = False
        self.lstm_size = lstm_size
        self.end_channels = end_channels
        self.n_sequences = n_sequences
        self.decoder = None
        self._decoder_input = None
        self.horizon = horizon
        self.out_channels = out_channels

        # LSTM block
        self.lstm = torch.nn.LSTM(
            input_size=in_channels + end_channels if horizon > 0 else in_channels,
            hidden_size=self.lstm_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        ).to(device)

        # Optional normalization layer
        if use_layernorm:
            self.norm = torch.nn.LayerNorm(self.lstm_size).to(device)
        else:
            self.norm = torch.nn.BatchNorm1d(self.lstm_size).to(device)

        # Dropout after LSTM
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        # Output layer
        self.linear1 = torch.nn.Linear(self.lstm_size, hidden_channels).to(device)
        self.linear2 = torch.nn.Linear(hidden_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Task-dependent activation
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)

    def forward(self, X, edge_index=None, graphs=None, z_prev=None):
        """
        Parameters:
            X: Tensor of shape (batch_size, features, sequence_length)

        Returns:
            output: Final prediction tensor
            (optionally) hidden_repr: The hidden state before final layer
        """
        batch_size = X.size(0)

        if z_prev is None:
            z_prev = torch.zeros((X.shape[0], self.end_channels, self.n_sequences), device=X.device, dtype=X.dtype)
        else:
            z_prev = z_prev.view(X.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            X = torch.cat((X, z_prev), dim=1)

        # (batch_size, seq_len, features)
        x = X.permute(0, 2, 1)

        # Initial hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.lstm_size).to(self.device)

        # LSTM forward
        x, _ = self.lstm(x, (h0, c0))

        # Last time step output
        x = x[:, -1, :]  # shape: (batch_size, hidden_size)

        # Normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
        
class DilatedCNN(torch.nn.Module):
    def __init__(self, channels, dilations, lin_channels, end_channels, n_sequences, device, act_func, dropout, out_channels, task_type, use_layernorm=False, return_hidden=False, horizon=0):
        super(DilatedCNN, self).__init__()

        # Initialisation des listes pour les convolutions et les BatchNorm
        self.cnn_layer_list = []
        self.batch_norm_list = []
        self.num_layer = len(channels) - 1
        
        # Initialisation des couches convolutives et BatchNorm
        for i in range(self.num_layer):
            if i == 0:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i] + end_channels if horizon > 0 else channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            else:
                self.cnn_layer_list.append(torch.nn.Conv1d(channels[i], channels[i + 1], kernel_size=3, padding='same', dilation=dilations[i], padding_mode='replicate').to(device))
            if use_layernorm:
                self.batch_norm_list.append(torch.nn.LayerNorm(channels[i + 1]).to(device))
            else:
                self.batch_norm_list.append(torch.nn.BatchNorm1d(channels[i + 1]).to(device))

        self.dropout = torch.nn.Dropout(dropout)
        
        # Convertir les listes en ModuleList pour être compatible avec PyTorch
        self.cnn_layer_list = torch.nn.ModuleList(self.cnn_layer_list)
        self.batch_norm_list = torch.nn.ModuleList(self.batch_norm_list)
        
        # Dropout after GRU
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        # Output layer
        self.linear1 = torch.nn.Linear(channels[-1], lin_channels).to(device)
        self.linear2 = torch.nn.Linear(lin_channels, end_channels).to(device)
        self.output_layer = torch.nn.Linear(end_channels, out_channels).to(device)

        # Activation function
        self.act_func = getattr(torch.nn, act_func)()

        self.return_hidden = return_hidden
        self.device = device
        self.end_channels = end_channels
        self.horizon = horizon
        self.out_channels = out_channels
        self.n_sequences = n_sequences

        # Output activation depending on task
        if task_type == 'classification':
            self.output_activation = torch.nn.Softmax(dim=-1).to(device)
        elif task_type == 'binary':
            self.output_activation = torch.nn.Sigmoid().to(device)
        else:
            self.output_activation = torch.nn.Identity().to(device)  # For regression or custom handling

    def forward(self, x, edges=None, z_prev=None):
        # Couche d'entrée

        if z_prev is None:
            z_prev = torch.zeros((x.shape[0], self.end_channels, self.n_sequences), device=x.device, dtype=x.dtype)
        else:
            z_prev = z_prev.view(x.shape[0], self.end_channels, self.n_sequences)
        
        if self.horizon > 0:
            x = torch.cat((x, z_prev), dim=1)

        # Couches convolutives dilatées avec BatchNorm, activation et dropout
        for cnn_layer, batch_norm in zip(self.cnn_layer_list, self.batch_norm_list):
            x = cnn_layer(x)
            x = batch_norm(x)  # Batch Normalization
            x = self.act_func(x)
            x = self.dropout(x)
        
        # Garder uniquement le dernier élément des séquences
        x = x[:, :, -1]

        # Activation and output
        #x = self.act_func(x)
        x = self.act_func(self.linear1(x))
        #x = self.dropout(x)
        hidden = self.act_func(self.linear2(x))
        #x = self.dropout(x)
        logits = self.output_layer(hidden)
        output = self.output_activation(logits)
        return output, logits, hidden
