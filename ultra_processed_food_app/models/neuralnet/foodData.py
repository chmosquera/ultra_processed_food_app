import imports


class FoodDataset:
    """Food dataset."""

    def __init__(self, csv_file, cat_vars, num_vars, output_vars, test_frac):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = imports.pd.read_csv(csv_file)
        self.categorical_columns = cat_vars
        self.numerical_columns = num_vars
        self.outputs = output_vars

        for category in self.categorical_columns:
            self.data_frame[category] = self.data_frame[category].astype('category')

        cat_arr = []
        for col in self.categorical_columns:
            cat_arr.append(self.data_frame[col].cat.codes.values)
        self.categorical_data = imports.np.stack(cat_arr, 1)
        self.categorical_data = imports.torch.tensor(self.categorical_data, dtype=imports.torch.int64)

        num_arr = []
        for col in self.categorical_columns:
            num_arr.append(self.data_frame[col].cat.codes.values)
        self.numerical_data = imports.np.stack(num_arr, 1)
        self.numerical_data = imports.torch.tensor(self.numerical_data, dtype=imports.torch.float)

        self.output_data = imports.torch.tensor(self.data_frame[self.outputs].values).flatten()

        for v in self.categorical_columns: self.data_frame[v] = self.data_frame[v].astype('category').cat.as_ordered()
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
        features = ['pclass', 'survived', 'sex', 'age']
        self.data_frame = self.data_frame.loc[:, features]
        self.data_frame.loc[:, 'pclass'] = self.data_frame['pclass'].fillna(self.data_frame['pclass'].mode()).astype(int)
        self.data_frame.loc[:, 'age'] = self.data_frame['age'].fillna(self.data_frame['age'].median())
        self.data_frame.loc[:, 'age'] = (self.data_frame['age'] / 10).astype(str).str[0].astype(int) * 10
        print(self.data_frame)
