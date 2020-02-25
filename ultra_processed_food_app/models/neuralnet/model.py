import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):

    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        self.loss_function = nn.CrossEntropyLoss()

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for l in layers:
            all_layers.append(nn.Linear(input_size, l))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(l))
            all_layers.append(nn.Dropout(p))
            input_size = l

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def train_model(self, categorical_train_data, numerical_train_data, train_outputs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        epochs = 300
        aggregated_losses = []

        for i in range(epochs):
            i += 1
            y_pred = self.forward(categorical_train_data, numerical_train_data)
            # print(y_pred)
            # print(y_pred.shape)
            # print(train_outputs)
            # print(train_outputs.shape)
            single_loss = self.loss_function(y_pred, train_outputs)
            aggregated_losses.append(single_loss)

            if i % 25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for d, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, d]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        # print('before: ')
        # print(x_numerical)
        x_numerical = self.batch_norm_num(x_numerical)
        # print('after: ')
        # print(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

    def get_predictions(self, categorical_test_data, numerical_test_data, test_outputs=None):
        if test_outputs is None:
            test_outputs = []
        with torch.no_grad():
            y_val = self.forward(categorical_test_data, numerical_test_data)
            if len(test_outputs) == 0:
                loss = self.loss_function(y_val, test_outputs)
                print(f'Loss: {loss:.8f}')

        y_val = np.argmax(y_val, axis=1)
        return y_val

