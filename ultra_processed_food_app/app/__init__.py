import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from get_usda_ingredients import get_usda_ingredients
from sklearn.externals import joblib
import numpy as np
import torch

import fdc
import openfoodfacts
import sys

sys.path.insert(1, 'models/neuralnet/')
from foodData import FoodDataset
import model

USDA_API_KEY = 'AemedCUPSQHBrbfoJdkfrdFSbtS9ogDP7YpCWDTN'

REPORT_TEMPLATE = '''\
[b]fdcid:[/b] {reportData.fdcid}
[b]Brand:[/b] {reportData.brand}

{reportData.description}

[b]Ingredients:[/b] {reportData.ingredients}

[size=24][b]NOVA Score:[/b] {reportData.nova} [/size]
'''

class MainScreen(Screen):
    def on_pre_enter(self, *args):
        if('scanner_input' in self.ids):
            self.ids.scanner_input.refocus()
            self.ids.scanner_input.text = ""
            App.get_running_app().reportData = None
        return super().on_pre_enter(*args)

class NovaScoreScreen(Screen):
    def on_pre_enter(self, *args):
        self.ids.report_label.update()
        return super().on_pre_enter(*args)

class ScannerInput(TextInput):

    report_target = ObjectProperty(None)


    def __init__(self, **kwargs):
        super(ScannerInput, self).__init__(**kwargs)
        self.fdcClient = fdc.FdcClient(USDA_API_KEY)
        self.food_data = FoodDataset("usda_products_1_human.csv", 0.2)

        self.nn_model = joblib.load('filename.pkl')

    def on_parent(self, widget, parent):
        self.refocus()

    def refocus(self):
        self.focus = True

    def on_focus(self, instance, value):
        if(not value and self.focus == False):
            Clock.schedule_once(lambda _: self.refocus())

    def on_text_validate(self):
        searchResult = self.fdcClient.search(self.text, productLimit=1)
        productResult = openfoodfacts.get_product(self.text)
        app = App.get_running_app()
        try:
            res1 = next(searchResult)
            fdcId = res1['fdcId']
            description = res1['description']
            app.reportData = ReportData(fdcId, description)


            app.reportData.brand = res1['brandOwner'] if 'brandOwner' in res1 else "<Unavailable>"

            app.reportData.ingredients = res1['ingredients'] if 'ingredients' in res1 else "<Unavailable>"
            app.reportData.score = res1['score'] if 'score' in res1 else "<Unavailable>"

            novaAvailable = productResult['status'] == 1 and 'nova_group' in productResult['product']
            ingredient_list = get_usda_ingredients(app.reportData.ingredients)
            new_data_frame = self.food_data.data_frame.iloc[0]
            for col in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
                if col in ingredient_list:
                    new_data_frame[col] = 1
                else:
                    new_data_frame[col] = 0
            for category in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
                new_data_frame[category] = new_data_frame[category].astype('category')

            cat_arr = []
            for col in new_data_frame.columns[0:len(new_data_frame.columns) - 5]:
                cat_arr.append(new_data_frame[col].cat.codes.values)
            cat_test_data = np.stack(cat_arr, 1)
            cat_test_data = torch.tensor(cat_test_data, dtype=torch.int64)
            app.reportData.nova = self.nn_model.get_predictions(cat_test_data, [len(ingredient_list)]) + 1
            # app.reportData.nova = productResult['product']['nova_group'] if novaAvailable else "<Unavailable>"


        except StopIteration:
            app.reportData = None
        except KeyError:
            app.reportData = None

        app.root.current = "NovaScoreScreen"
        app.root.transition.direction = "left"

class ReportLabel(Label):

    def update(self):
        app = App.get_running_app()
        if(app.reportData != None):
            self.text = REPORT_TEMPLATE.format(reportData = app.reportData)
            self.halign = 'left'
        else:
            self.text = "No matching item found!"
            self.halign = 'center'

class ReportData:
    def __init__(self, fdcid, description):
        self.fdcid = fdcid
        self.description = description
        self.brand = None
        self.ingredients = None
        self.score = None
        self.nova = None

kv_file = Builder.load_file("app/interface.kv")

class UltraProcessedFoodApp(App):

    def __init__(self, **kwargs):
        self.reportData = None
        super(UltraProcessedFoodApp, self).__init__(**kwargs)

    def build(self):
        return kv_file

    def run(self, **kwargs):
        super(UltraProcessedFoodApp, self).run(**kwargs)



if __name__ == "__main__":
    UltraProcessedFoodApp().run()
