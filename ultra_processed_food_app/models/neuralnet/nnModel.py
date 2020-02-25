from sklearn.externals import joblib


class NNModel:

    def __init__(self, file_name):
        self.model = joblib.load(file_name)

    def get_score(self, ingredients_str):
        return self.model.get_score(ingredients_str)
