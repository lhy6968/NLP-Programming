import torch
import torch.nn as nn
import torch.optim as optim


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.embedding_layer = nn.Embedding(input_size, hidden_size)
        self.rnn_layer = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inputs):
        embedded_result = self.embedding_layer(inputs)
        output_1, output_2 = self.rnn(embedded_result)
        y = self.fc(output_1[:, -1, :])
        return y