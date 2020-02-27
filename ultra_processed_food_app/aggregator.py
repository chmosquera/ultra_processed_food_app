import random
import operator
import sys

sys.path.insert(1, 'models/neuralnet/')
import nnModel

# a placeholder model class to illustrate how to use the Aggregator class below
class DummyModel:
    def __init__(self, fav_score, fav_weight):
        self.fav_score = fav_score
        self.fav_weight = fav_weight if fav_weight < 1 else 0.75

    def get_score(self, ingredients):
        do_return_favorite = random.random() < self.fav_weight

        score = self.fav_score

        if(not do_return_favorite):
            while(score == self.fav_score):
                score = random.randint(1, 4)

        return score


# An aggregator class that initializes AI models and then polls their responses to an ingredient list
# in order to determine the concensus NOVA score.
# IMPORTANT: Each model passed to this class MUST HAVE a get_score(ingredients) method
class Aggregator:  
    
    # Inputs:
    #   - model_classes: a list of references to AI model python classes.
    #   - model_file_saves: an array the same length as model_classes, where each element is a list of filenames where
    #                       saved, trained models are stored.
    # Example initialization: 'Aggregator([DummyModel, DummyModel], [['model_save1'],['model_save2', 'model_save3']])'

    def __init__(self, model_classes, model_file_saves):
        self.models = []

        for i in range(len(model_classes)):
            for model_file in model_file_saves[i]:
                print(model_file)
                model_class = model_classes[i](model_file)
                self.models.append(model_class)


    # Input: ingredients - a string version of a list of ingredients.
    # Output: an integer score 1-4
    def get_score(self, ingredients):

        votes = []

        for model in self.models:
            votes.append(model.get_score(ingredients))

        print("================\n" + ' '.join([str(elem) for elem in votes]) + "\n================")

        tallies = [0,0,0,0]

        for vote in votes:
            tallies[vote - 1] += 1

        index, value = max(enumerate(tallies), key=operator.itemgetter(1))

        return index + 1

        

#aggy = Aggregator([DummyModel, DummyModel, DummyModel], 30)
#print(aggy.get_score('poop, farts'))