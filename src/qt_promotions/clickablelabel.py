from PyQt5 import QtCore, QtGui, QtWidgets

class ClickableLabel(QtWidgets.QLabel):
    on_click = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.is_selected = False

    def mousePressEvent(self, event):
        if not self.is_selected:
            self.setStyleSheet("border: 3px solid rgb(87, 227, 137); background-color: rgb(222, 221, 218);")
            self.is_selected = True
        else:
            self.setStyleSheet("background-color: rgb(222, 221, 218);")
            self.is_selected = False

        self.on_click.emit(self.objectName())