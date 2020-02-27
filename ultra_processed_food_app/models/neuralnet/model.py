import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys

sys.path.insert(1, '../../')
import get_usda_ingredients

class Model(nn.Module):

    def __init__(self, column_names, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)
        self.loss_function = nn.CrossEntropyLoss()
        self.column_names = column_names

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

        x_numerical = self.batch_norm_num(x_numerical)

        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x

    def get_predictions(self, categorical_test_data, numerical_test_data, test_outputs=None):
        if test_outputs is None:
            test_outputs = []
        with torch.no_grad():
            y_val = self.forward(categorical_test_data, numerical_test_data)
            if len(test_outputs) != 0:
                loss = self.loss_function(y_val, test_outputs)
                print(f'Loss: {loss:.8f}')

        y_val = np.argmax(y_val, axis=1)
        return y_val

    def get_score(self, ingredients_str):
        ingredient_list = get_usda_ingredients.ingredients_to_list(ingredients_str)
        new_data_frame = pd.DataFrame(np.zeros((1, len(self.column_names))))
        new_data_frame.columns = self.column_names
        for col in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
            if col in ingredient_list:
                new_data_frame[col] = 1
            else:
                new_data_frame[col] = 0
        for category in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
            new_data_frame[category] = new_data_frame[category].astype('category')
        new_data_frame['num_ingredients'] = len(ingredient_list)
        new_data_frame = new_data_frame.append(new_data_frame, ignore_index=True)
        cat_arr = []
        for col in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
            cat_arr.append(new_data_frame[col].cat.codes.values)
        num_arr = []
        for col in new_data_frame.columns[len(new_data_frame.columns) - 5: len(new_data_frame.columns) - 4]:
            num_arr.append(new_data_frame[col].values)
        cat_test_data = np.stack(cat_arr, 1)
        cat_test_data = torch.tensor(cat_test_data, dtype=torch.int64)
        num_test_data = np.stack(num_arr, 1)
        num_test_data = torch.tensor(num_test_data, dtype=torch.float)
        return self.get_predictions(cat_test_data, num_test_data)[0] + 1
