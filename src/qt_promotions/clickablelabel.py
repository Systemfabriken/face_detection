from PyQt5 import QtCore, QtGui, QtWidgets

BORDER_STYLE_SELECTED = "border: 5px solid rgb(87, 227, 137);" # Green
BORDER_STYLE_UNSELECTED = "" # Transparent
BORDER_STYLE_INVALID = "border: 5px solid rgb(255, 0, 0);" # Red

class ClickableLabel(QtWidgets.QLabel):
    on_click = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super(ClickableLabel, self).__init__(parent)
        self.is_selected = False
        self.border_style = BORDER_STYLE_UNSELECTED
        self.is_invalid = False

    def mousePressEvent(self, event):
        if not self.is_selected:
            print("Selected " + self.objectName())
            self.border_style = BORDER_STYLE_SELECTED
            self.setStyleSheet((BORDER_STYLE_SELECTED if not self.is_invalid else BORDER_STYLE_INVALID) + " background-color: rgb(222, 221, 218);")
            self.is_selected = True
        else:
            print("Unselected " + self.objectName())
            self.border_style = BORDER_STYLE_UNSELECTED
            self.setStyleSheet((BORDER_STYLE_UNSELECTED if not self.is_invalid else BORDER_STYLE_INVALID) + " background-color: rgb(222, 221, 218);")
            self.is_selected = False

        self.on_click.emit(self.objectName())

    def set_invalid(self, is_invalid):
        print("is invalid: " + str(is_invalid))
        self.is_invalid = is_invalid
        self.setStyleSheet(BORDER_STYLE_INVALID if self.is_invalid else self.border_style + " background-color: rgb(222, 221, 218);")

    