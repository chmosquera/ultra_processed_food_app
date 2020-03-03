from joblib import load


class FoodRandomForest:
    # Files: list of strings representing the filenames
    # The first file is the vectorizer file and the second file is the model
    def __init__(self, file_name):
        self.tfidf = load('rf-models/tfidfvectorizer.joblib')
        self.classifier = load(file_name)

    # Takes in a string containing the ingredients e.g. "tomato, salt, pepper"
    # Returns an integer of [1,4]
    def get_score(self, ingredient):
        vector = self.tfidf.transform([ingredient])  # Turn list of ingredients into vector
        return self.classifier.predict(vector)[0]
