import PyQt5.QtWidgets as QtWidgets


from ui_generated.pyqt5.modify_person_dialog import Ui_Dialog
from models.person import Person

class ModifyPersonDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        self.person_instance = None

    def get_name(self):
        return self.ui.nameLineEdit.text()

    def set_name(self, name):
        self.ui.nameLineEdit.setText(name)

    def get_status(self) -> Person.Status:
        if self.ui.allowed_radio_button.isChecked():
            return Person.Status.ALLOWED
        elif self.ui.denied_radio_button.isChecked():
            return Person.Status.DENIED
        else:
            return Person.Status.PENDING

    def set_status(self, status: Person.Status):
        if status == Person.Status.ALLOWED:
            self.ui.allowed_radio_button.setChecked(True)
        elif status == Person.Status.DENIED:
            self.ui.denied_radio_button.setChecked(True)
        elif status == Person.Status.PENDING:
            self.ui.pending_radio_button.setChecked(True)
        else:
            raise ValueError("Invalid status")

    def get_person(self) -> Person:
        return Person(self.get_name(), self.get_status())

    def set_person(self, person: Person):
        self.person_instance = person

    def get_person_from_dialog(self):
        return self.get_name(), self.get_status()

    def set_person_to_dialog(self, person):
        self.set_name(person.name)
        self.set_status(person.status)