import imports
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import foodData
import model


def main():
    food_data = foodData.FoodDataset(
        "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv",
        ['pclass', 'sex'],
        ['age'],
        ['survived'],
        0.2)
    food_data.setup_data_frame()
    nn_model = model.Model(food_data.categorical_embedding_sizes, food_data.numerical_data.shape[1], 2, [500, 250, 125], p=0.5)

    # loss_function = imports.nn.CrossEntropyLoss()
    # optimizer = imports.torch.optim.Adam(nn_model.parameters(), lr=0.001)
    #
    # epochs = 300
    # aggregated_losses = []
    #
    # for i in range(epochs):
    #     i += 1
    #     y_pred = nn_model(food_data.categorical_train_data, food_data.numerical_train_data)
    #     single_loss = loss_function(y_pred, food_data.train_outputs)
    #     aggregated_losses.append(single_loss)
    #
    #     if i % 25 == 1:
    #         print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
    #
    #     optimizer.zero_grad()
    #     single_loss.backward()
    #     optimizer.step()
    #
    # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

    # with imports.torch.no_grad():
    #     y_val = nn_model(food_data.categorical_test_data, food_data.numerical_test_data)
    #     loss = loss_function(y_val, food_data.test_outputs)
    # print(f'Loss: {loss:.8f}')
    #
    # y_val = imports.np.argmax(y_val, axis=1)

    loss_function = nn_model.train_model(food_data.categorical_train_data, food_data.numerical_train_data, food_data.train_outputs)
    y_val = nn_model.get_predictions(food_data.categorical_test_data, food_data.numerical_test_data, loss_function)
    print(confusion_matrix(food_data.test_outputs, y_val))
    print(classification_report(food_data.test_outputs, y_val))
    print(accuracy_score(food_data.test_outputs, y_val))


if __name__ == "__main__":
    main()