from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import foodData
# import model
from sklearn.externals import joblib
import nnModel
import sys
def main():
    load_model = 'False'

    food_data = foodData.FoodDataset(
        "usda_products_1_human.csv", 0.2)
    # if load_model == 'False':
    #     nn_model = model.Model(food_data.data_frame.columns, food_data.categorical_embedding_sizes, food_data.numerical_data.shape[1], 4, [500, 250, 125],
    #                            p=0.5)
    #     nn_model.train_model(food_data.categorical_train_data, food_data.numerical_train_data, food_data.train_outputs)
    #     # Save the model as a pickle in a file
    #     joblib.dump(nn_model, 'nnmodel3.pkl')
    # else:
    #     # Load the model from the file
    #     nn_model = joblib.load('filename.pkl')
    nn_model = nnModel.NNModel('nnmodel3.pkl')

    # Use the loaded model to make predictions
    # y_val = nn_model.get_predictions(food_data.categorical_test_data, food_data.numerical_test_data,
    #                                  test_outputs=food_data.test_outputs)
    y_val = nn_model.get_score('ALMONDMILK (FILTERED WATER, ALMONDS), CANE SUGAR, CONTAINS 2% OR LESS OF: VITAMIN AND MINERAL BLEND (CALCIUM CARBONATE, VITAMIN E ACETATE, ZINC GLUCONATE, VITAMIN A PALMITATE, RIBOFLAVIN [B2], VITAMIN B12, VITAMIN D2), SEA SALT, NATURAL FLAVOR, SUNFLOWER LECITHIN, LOCUST BEAN GUM, GELLAN GUM.')
    print(y_val)
    # if len(food_data.test_outputs) != 0:
    #     print(confusion_matrix(food_data.test_outputs, y_val))
    #     print(classification_report(food_data.test_outputs, y_val))
    #     print(accuracy_score(food_data.test_outputs, y_val))

if __name__ == "__main__":
    main()
