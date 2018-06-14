from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine

from QTGui.controller.main_controller import MainController

if __name__ == "__main__":
    import sys

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    mainController = MainController()
    engine.rootContext().setContextProperty("main_controller", mainController)
    engine.load("gui/main.qml")

    engine.quit.connect(app.quit)
    sys.exit(app.exec_())
