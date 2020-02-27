import joblib
import torch.nn as nn

import ultra_processed_food_app.models.neuralnet.model


class NNModel:

    def __init__(self, file_name):
        self.hh = joblib.load(file_name)

    def get_score(self, ingredients_str):
        return self.hh.get_score(ingredients_str)
