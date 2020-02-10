from usda import UsdaClient

client = UsdaClient('AemedCUPSQHBrbfoJdkfrdFSbtS9ogDP7YpCWDTN')

foods_list = client.list_foods(5)
for _ in  range(5):
        food_item = next(foods_list)
        print(food_item.name)