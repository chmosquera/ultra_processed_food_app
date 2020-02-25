import random
import operator

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
    #   - num_models: the total desired number of models. Each model in model_classes will will an equal 
    #                 number of models generated, adding up to num_models.
    # Example initialization: 'Aggregator([DummyModel, DummyModel, DummyModel], 30)'

    def __init__(self, model_classes, num_models):  
        self.models = []  

        #split models initialization for cases where num_models / len(model_classes) has some remainder

        # builds the first (model_classes - 1) * num_models models
        for model_index in range(len(model_classes) - 1):
            for i in range(num_models // len(model_classes)):
                self.models.append(model_classes[model_index](random.randint(1,4), random.random()))

        # builds the last ((model_classes - 1) / models_classes) * num_models models
        for i in range(num_models - ((num_models // len(model_classes)) * (num_models - 1))):
            self.models.append(model_classes[model_index](random.randint(1,4), random.random()))


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