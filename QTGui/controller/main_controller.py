from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


class MainController(QObject):
    clickResult = pyqtSignal(str, arguments=['text_result'])

    def __init__(self):
        QObject.__init__(self)

    @pyqtSlot()
    def on_click_button(self):
        self.clickResult.emit("ciao")
