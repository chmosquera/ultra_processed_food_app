import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
import get_usda_ingredients
from sklearn.externals import joblib
import numpy as np
import pandas as pd
import torch

import fdc
import openfoodfacts
from aggregator import Aggregator
from aggregator import DummyModel

import sys

sys.path.insert(1, 'models/neuralnet/')
from models.neuralnet import nnModel
sys.path.insert(1, 'models/randomforest/')
from models.randomforest import foodRandomForest

USDA_API_KEY = 'AemedCUPSQHBrbfoJdkfrdFSbtS9ogDP7YpCWDTN'

NOVA_BLURBS = [
    # Nova 1
    "[i]Minimally Processed[/i]. Foods in this group are the least processed of all. "
    "Typically they are the direct product of a single plant or animal. Common examples include raw nuts, fruit, "
    "leaf vegetables, uncured meat, eggs, or milk. While most Nova 1 foods are of some raw form, a very small amount "
    "of processing is permitted for preservation.",
    # Nova 2
    "[i]Processed Ingredients[/i]. Products in this group are best characterized relative to their raw counterparts in NOVA group 1."
    "Group 2 products are typically a combination of raw natural ingredients that have been lightely refined or processed. Some examples of "
    "acceptable processing for this group include pressing, grinding, milling, and spray drying. Often products in this group are meant for use as "
    "ingredients. Thusly, cited examples include vegetable oils, maple syrup, honey, corn starch, or salted butter.",
    # Nova 3
    "[i]Processed Foods[/i]. NOVA 3 products are mixtures of ingredients from the prior two groups. Foods that combine naturally based ingredients and "
    "a small number of simple preservative methods belong in this group. Some examples include fruit in syrup, canned meat, smoked meat, cheese, and some fresh breads.",
    # Nova 4
    "[i]ULTRA-Processed Foods[/i]. NOVA 4 products are \"industrial formulations\" consisting of many ingredients both natural and artificial. "
    "Foods may fall into this category due to a large number of naturally processed ingredients, but often the include more significantly processed ingredients or artificial ingredients. "
    "Examples of such ingredients are \"casein, lactose, whey, gluten [...], hydrolysed porteins, high fructose corn syrup, emulsifiers, and non-sugar sweeteners.\""
]

REPORT_TEMPLATE = '''\
[b]fdcid:[/b] {reportData.fdcid}
[b]Brand:[/b] {reportData.brand}

{reportData.description}

[b]Ingredients:[/b] {reportData.ingredients}

[size=24][b]NOVA Score:[/b] {reportData.nova} [/size]
{blurb}
[size=12]Monteiro, Carlos A., et al. "NOVA. The star shines bright." [i]World Nutrition[/i] 7.1-3 (2016): 28-38.[/size]
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

            # num_arr = [new_data_frame['num_ingredients'].values, new_data_frame['num_ingredients'].values]


            # app.reportData.nova = productResult['product']['nova_group'] if novaAvailable else "<Unavailable>"
            nova_score = app.get_running_app().models_aggregator.get_score(app.reportData.ingredients)

            novaAvailable = productResult['status'] == 1 and 'nova_group' in productResult['product']
            #app.reportData.nova = productResult['product']['nova_group'] if novaAvailable else "<Unavailable>"
            app.reportData.nova = nova_score

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
            self.text = REPORT_TEMPLATE.format(reportData = app.reportData, blurb = NOVA_BLURBS[int(app.reportData.nova)-1])
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
        #insert your model class names here. Replace 'DummyModel' with your ai model class name
        self.models_aggregator = Aggregator([nnModel.NNModel, foodRandomForest.FoodRandomForest],
                                            [['nnmodels/nnmodel0.pkl',
                                              'nnmodels/nnmodel1.pkl',
                                              'nnmodels/nnmodel2.pkl',
                                              'nnmodels/nnmodel3.pkl',
                                              'nnmodels/nnmodel4.pkl',
                                              'nnmodels/nnmodel5.pkl',
                                              'nnmodels/nnmodel6.pkl',
                                              'nnmodels/nnmodel7.pkl',
                                              'nnmodels/nnmodel8.pkl',
                                              'nnmodels/nnmodel9.pkl',
                                              'nnmodels/nnmodel10.pkl',
                                              'nnmodels/nnmodel11.pkl',
                                              'nnmodels/nnmodel12.pkl',
                                              'nnmodels/nnmodel13.pkl',
                                              'nnmodels/nnmodel14.pkl',
                                              'nnmodels/nnmodel15.pkl',
                                              'nnmodels/nnmodel16.pkl',
                                              'nnmodels/nnmodel17.pkl',
                                              'nnmodels/nnmodel18.pkl',
                                              'nnmodels/nnmodel19.pkl',
                                              'nnmodels/nnmodel20.pkl',
                                              'nnmodels/nnmodel21.pkl',
                                              'nnmodels/nnmodel22.pkl',
                                              'nnmodels/nnmodel23.pkl',
                                              'nnmodels/nnmodel24.pkl',
                                              'nnmodels/nnmodel25.pkl',
                                              'nnmodels/nnmodel26.pkl',
                                              'nnmodels/nnmodel27.pkl',
                                              'nnmodels/nnmodel28.pkl',
                                              'nnmodels/nnmodel29.pkl',
                                              'nnmodels/nnmodel30.pkl'
                                              ],
                                             ['rf-models/randomforest0.joblib',
                                              'rf-models/randomforest1.joblib',
                                              'rf-models/randomforest2.joblib',
                                              'rf-models/randomforest3.joblib',
                                              'rf-models/randomforest4.joblib',
                                              'rf-models/randomforest5.joblib',
                                              'rf-models/randomforest6.joblib',
                                              'rf-models/randomforest7.joblib',
                                              'rf-models/randomforest8.joblib',
                                              'rf-models/randomforest9.joblib',
                                              'rf-models/randomforest10.joblib',
                                              'rf-models/randomforest11.joblib',
                                              'rf-models/randomforest12.joblib',
                                              'rf-models/randomforest13.joblib',
                                              'rf-models/randomforest14.joblib',
                                              'rf-models/randomforest15.joblib',
                                              'rf-models/randomforest16.joblib',
                                              'rf-models/randomforest17.joblib',
                                              'rf-models/randomforest18.joblib',
                                              'rf-models/randomforest19.joblib',
                                              'rf-models/randomforest20.joblib',
                                              'rf-models/randomforest21.joblib',
                                              'rf-models/randomforest22.joblib',
                                              'rf-models/randomforest23.joblib',
                                              'rf-models/randomforest24.joblib',
                                              'rf-models/randomforest25.joblib',
                                              'rf-models/randomforest26.joblib',
                                              'rf-models/randomforest27.joblib',
                                              'rf-models/randomforest28.joblib',
                                              'rf-models/randomforest29.joblib'
                                              ]])
        super(UltraProcessedFoodApp, self).__init__(**kwargs)

    def build(self):
        return kv_file

    def run(self, **kwargs):
        super(UltraProcessedFoodApp, self).run(**kwargs)



if __name__ == "__main__":
    UltraProcessedFoodApp().run()
