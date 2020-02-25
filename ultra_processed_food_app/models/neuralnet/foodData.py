import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '../../')
import get_usda_ingredients
class FoodDataset:
    """Food dataset."""
    def __init__(self, csv_file, test_frac):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file, sep='\t')
        self.data_frame = self.data_frame.drop(0, axis=0).reset_index()
        self.setup_data_frame()
        self.categorical_columns = self.data_frame.columns[0:len(self.data_frame.columns)-5]
        self.numerical_columns = ['num_ingredients']
        self.outputs = ['hu_nova_score']
        for category in self.categorical_columns:
            self.data_frame[category] = self.data_frame[category].astype('category')
        cat_arr = []
        for col in self.categorical_columns:
            cat_arr.append(self.data_frame[col].cat.codes.values)
        self.categorical_data = np.stack(cat_arr, 1)
        self.categorical_data = torch.tensor(self.categorical_data, dtype=torch.int64)
        num_arr = []
        for col in self.numerical_columns:
            num_arr.append(self.data_frame[col].values)
        self.numerical_data = np.stack(num_arr, 1)
        self.numerical_data = torch.tensor(self.numerical_data, dtype=torch.float)
        self.data_frame = self.data_frame.astype({self.outputs[0]: 'int64'})
        i=0
        for row in self.data_frame[self.outputs[0]]:
            self.data_frame[self.outputs[0]].iloc[i] = row - 1
            i+=1
        self.output_data = torch.tensor(self.data_frame[self.outputs].values).flatten()
        categorical_column_sizes = [(c, len(self.data_frame[c].cat.categories) + 1) for c in self.categorical_columns]
        self.categorical_embedding_sizes = [(c, min(50, (c + 1) // 2)) for _, c in categorical_column_sizes]
        sample_size = round(len(self.output_data) * (1 - test_frac))
        self.categorical_train_data = self.categorical_data[:sample_size]
        self.categorical_test_data = self.categorical_data[sample_size:]
        self.numerical_train_data = self.numerical_data[:sample_size]
        self.numerical_test_data = self.numerical_data[sample_size:]
        self.train_outputs = self.output_data[:sample_size]
        self.test_outputs = self.output_data[sample_size:]
    def __len__(self):
        return len(self.data_frame)
    def setup_data_frame(self):
        all_food_ingredients = []
        for row in self.data_frame['ingredients']:
            ingredients = get_usda_ingredients.ingredients_to_list(row)
            all_food_ingredients = np.concatenate([all_food_ingredients, ingredients])
        new_data_frame = pd.DataFrame(np.zeros((self.data_frame.shape[0], 1)))
        for food_ingredients in all_food_ingredients:
            new_data_frame[food_ingredients] = 0
        new_data_frame['num_ingredients'] = 0
        new_data_frame = new_data_frame.drop(0, axis=1)
        new_data_frame = new_data_frame.loc[:, ~new_data_frame.columns.duplicated()]
        for col in self.data_frame.drop('ingredients', axis=1).columns:
            new_data_frame[col] = self.data_frame[col]
        for index, row in self.data_frame.iterrows():
            ingredients = get_usda_ingredients.ingredients_to_list(self.data_frame['ingredients'].iloc[index])
            new_data_frame['num_ingredients'].iloc[index] = len(ingredients)
            for ingredient in ingredients:
                new_data_frame[ingredient].iloc[index] = 1
        new_data_frame = new_data_frame.drop('index', axis=1)
        self.data_frame = new_data_frame