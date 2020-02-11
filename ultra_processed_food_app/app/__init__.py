import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

class MainWindow(Screen):
    pass

class NovaScoreWindow(Screen):
    pass

class WindowManager(ScreenManager):
    pass


kv_file = Builder.load_file("interface.kv")

class UltraProcessedFoodAppApp(App):
    def build(self):
        return kv_file



if __name__ == "__main__":
    UltraProcessedFoodAppApp().run()
