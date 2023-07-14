from enum import Enum
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import cv2
import numpy as np
import dlib
import face_recognition
import pickle

# Local imports
from ui_generated.pyqt5.classification_main_window import Ui_MainWindow
from ui_generated.pyqt5.modify_person_dialog import Ui_Dialog as ModifyPersonDialog
from face_identification.classifier import extract_single_face_multi_images, extract_single_face_single_image, extract_features, extract_feature, train_classifier, predict
from modify_person_dialog import ModifyPersonDialog
from models.person import Person

def get_available_cameras(max_cameras=10) -> list[int]:
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap is None or not cap.isOpened():
            break
        print(f"Found camera {i}")
        print(f"  width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  fps: {cap.get(cv2.CAP_PROP_FPS)}")
        print(f"  brightness: {cap.get(cv2.CAP_PROP_HW_DEVICE)}")
        cameras.append(i)
        cap.release()
    return cameras

def convert_to_qt_format(cv_img):
    rgbImage = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgbImage.shape
    bytesPerLine = ch * w
    return QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)

class CameraThread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, camera_port):
        super().__init__()
        self.camera_port = camera_port
        self.cap: cv2.VideoCapture = cv2.VideoCapture(self.camera_port)
        self.curr_img_320_240 = None
        self.raw_frame = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                self.raw_frame = frame
                convertToQtFormat = convert_to_qt_format(frame)
                p = convertToQtFormat.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                self.curr_img_320_240 = convertToQtFormat.scaled(320, 240, QtCore.Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

    def stop(self):
        print("Stopping camera thread...")
        self.cap.release()

class ImageDisplayMode(Enum):
    IMAGES = 0
    FACES = 1

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setWindowFlags(QtCore.Qt.Window)
        self.setupUi(self)

        self.select_camera_box.clear()
        self.select_camera_box.addItem("No Camera Selected")
        self.select_camera_box.addItems(["Camera " + str(i) for i in get_available_cameras()])
        self.select_camera_box.currentIndexChanged.connect(self.select_camera)
        self.camera_thread = None

        self.image_display_mode: ImageDisplayMode = ImageDisplayMode.IMAGES
        self.image_display_mode_selector.currentIndexChanged.connect(self.select_image_display_mode)

        self.right_above_label.on_click.connect(self.on_image_clicked)
        self.right_center_label.on_click.connect(self.on_image_clicked)
        self.right_below_label.on_click.connect(self.on_image_clicked)
        self.front_above_label.on_click.connect(self.on_image_clicked)
        self.front_center_label.on_click.connect(self.on_image_clicked)
        self.front_below_label.on_click.connect(self.on_image_clicked)
        self.left_above_label.on_click.connect(self.on_image_clicked)
        self.left_center_label.on_click.connect(self.on_image_clicked)
        self.left_below_label.on_click.connect(self.on_image_clicked)
        self.currently_selected_label = None

        self.model_path = ""
        self.actionSave.triggered.connect(self.save_model)
        self.actionSave_As.triggered.connect(self.save_model_as)
        self.actionOpen.triggered.connect(self.open_model)
        self.actionExport_Classifier.triggered.connect(self.export_classifier)

        self.capture_image_button.setEnabled(False)
        self.capture_image_button.clicked.connect(self.capture_image)
        self.model: dict = dict()
        self.model["images"] = dict()
        self.model["currently_selected_label"] = None
        self.model["features"] = None
        self.model["persons"] = dict()
        self.model["classifier"] = None
        self.selected_person: Person = None

        self.database_table.setRowCount(0)
        self.database_table.selectionModel().selectionChanged.connect(self.row_selected)

        self.add_person_button.clicked.connect(self.add_person)
        self.remove_person_button.clicked.connect(self.remove_person)
        self.edit_person_button.clicked.connect(self.edit_person)
        self.open_image_button.clicked.connect(self.open_image)
        self.test_predict_button.clicked.connect(self.test_predict)

        self.init_from_model()

        self.show()

    def add_person(self):
        dialog = ModifyPersonDialog()
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            person = dialog.get_person()
            if person.name in self.model["persons"]:
                self.statusbar.showMessage("Person already exists")
                return
            self.add_person_to_table(person)
            print(f"Adding person {person.name} with status {person.status.value}")
        else:
            print("Adding person cancelled")

    def edit_person(self):
        if self.selected_person is None:
            self.statusbar.showMessage("No person selected")
            return
        dialog = ModifyPersonDialog()
        dialog.set_name(self.selected_person.name)
        dialog.set_status(self.selected_person.status)
        dialog.setWindowTitle("Edit Person")
        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            old_person = self.selected_person
            person = dialog.get_person()
            # Copy all data from the old person to the new person.
            person.face = old_person.face
            # Remove the old person from the dictionary.
            self.remove_person()
            # Add the new person to the dictionary.
            self.add_person_to_table(person)
            print(f"Edited person {person.name} with status {person.status.value}")

    def remove_person(self):
        if self.selected_person is None:
            self.statusbar.showMessage("No person selected")
            return
        self.database_table.removeRow(self.database_table.currentRow())
        del self.model["persons"][self.selected_person.name]
        self.statusbar.showMessage(self.selected_person.name + " removed")
        self.selected_person = None

    def open_image(self):
        print("Opening image...")
        try:
            if self.selected_person is None or self.currently_selected_label is None:
                self.statusbar.showMessage("No person selected or no image selected")
                return
            file_name = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
            if not file_name[0]:
                return
            image = cv2.imread(file_name[0])
            perspective = self.get_perspective(self.currently_selected_label)
            self.selected_person.face.add_image(perspective, image)
            # label = getattr(self, self.currently_selected_label)
            self.show_person(self.selected_person, self.image_display_mode)
            # face_image = self.selected_person.face.get_face_image(perspective)
            # qimage = convert_to_qt_format(face_image.face() if self.select_image_display_mode is ImageDisplayMode.FACES else face_image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            # label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        except Exception as e:
            print(e)
            self.statusbar.showMessage("Error opening image")
        finally:
            self.statusbar.showMessage("Image opened")

    def add_person_to_table(self, person):
        row_position = self.database_table.rowCount()
        self.database_table.insertRow(row_position)
        self.database_table.setItem(row_position, 0, QtWidgets.QTableWidgetItem(person.name))
        self.database_table.setItem(row_position, 1, QtWidgets.QTableWidgetItem(person.status.value))
        
        # Add the Person object to the dictionary.
        self.model["persons"][person.name] = person
        self.statusbar.showMessage(person.name + " added")

    def row_selected(self, selected):
        selected_rows = selected.indexes()
        if selected_rows:
            row = selected_rows[0].row()
            # Retrieve the name of the person from the table.
            selected_person_name = self.database_table.item(row, 0).text()
            # Retrieve the corresponding Person object from the dictionary.
            self.selected_person = self.model["persons"][selected_person_name]
            self.show_person(self.selected_person, self.image_display_mode)
            print(f"Selected person: {self.selected_person.name} with status {self.selected_person.status.value}")

    def save_model(self):
        print("Saving model...")
        if self.model_path == "":
            self.save_model_as()
            return
        self.model["currently_selected_label"] = self.currently_selected_label
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(e)
        finally:
            print("Model saved")   

    def save_model_as(self):
        print("Saving model as...")
        file_name = QFileDialog.getSaveFileName(self, "Save Database", self.model_path, "Model (*.secsystdb)")
        filename: str = file_name[0]

        # Check and append extension if not present
        if not filename.endswith('.secsystdb'):
            filename += '.secsystdb'

        if filename:
            self.model_path = filename
            self.save_model()
        print("Model saved as: " + self.model_path)

    def load_model(self):
        if self.model_path == "":
            self.open_model()
            return
        print("Loading model...")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded")
        self.init_from_model()

    def open_model(self):
        print("Opening model...")
        file_name = QFileDialog.getOpenFileName(self, "Open Database", self.model_path, "Model (*.secsystdb)")
        if file_name[0]:
            self.model_path = file_name[0]
            self.load_model()
        print("Model opened")

    def init_from_model(self):
        self.currently_selected_label = self.model["currently_selected_label"]
        if self.currently_selected_label is not None:
            self.on_image_clicked(self.currently_selected_label)
        self.show_model()

    def export_classifier(self):
        if self.model_path == "":
            self.statusbar.showMessage("No model selected")
            return

        print("Exporting classifier...")
        file_name = QFileDialog.getSaveFileName(self, "Export Classifier", self.model_path, "Classifier (*.secsystsvm)")
        if not file_name[0]:
            return    
        clf_path = file_name[0]

        # Check and append extension if not present
        if not clf_path.endswith('.secsystsvm'):
            clf_path += '.secsystsvm'

        # Extract features from all faces in the database.
        features = []
        labels = []
        if self.model["persons"] is None:
            self.statusbar.showMessage("No persons in database")
            return
        
        for person_name in self.model["persons"]:
            person: Person = self.model["persons"][person_name]
            person_features = person.face.get_features()
            features.extend(person_features)
            labels.extend([person.name] * len(person_features))
            print(f"Extracted {len(person_features)} features from {person.name}")

        # Train the classifier.
        clf = train_classifier(features, labels)
        self.model["classifier"] = clf

        # Save the classifier.
        with open(clf_path, "wb") as f:
            pickle.dump(clf, f)

        print("Classifier exported")

    @QtCore.pyqtSlot(QtGui.QImage)
    def setCameraImage(self, image):
        self.camera_image_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def select_camera(self, index):
        if index == 0:
            self.statusbar.showMessage("No Camera Selected")
            self.capture_image_button.setEnabled(False)
        else:
            self.statusbar.showMessage("Camera " + str(index) + " selected")
            self.capture_image_button.setEnabled(True)

        if self.camera_thread is not None:
            self.camera_thread.terminate()
            self.camera_thread.wait()

        if index != 0:
            self.camera_thread = CameraThread(index - 1)
            self.camera_thread.changePixmap.connect(self.setCameraImage)
            self.camera_thread.start()

    def select_image_display_mode(self, index):
        if index == 0:
            self.statusbar.showMessage("Displaying Images Prior To Processing")
            self.image_display_mode = ImageDisplayMode.IMAGES
        elif index == 1:
            self.statusbar.showMessage("Displaying Images After Processing")
            self.image_display_mode = ImageDisplayMode.FACES
        
        if self.selected_person is not None:
            self.show_person(self.selected_person, self.image_display_mode)

    @QtCore.pyqtSlot(str)
    def on_image_clicked(self, name):
        if self.currently_selected_label is None:
            self.currently_selected_label = name
        else:   
            if self.currently_selected_label != name:
                label = getattr(self, self.currently_selected_label)
                label.mousePressEvent(None)
                self.currently_selected_label = name
            else:
                self.currently_selected_label = None
        
        if self.currently_selected_label is not None:
            self.statusbar.showMessage("Currently selected: " + name)
        else:
            self.statusbar.showMessage("No image selected")

    def get_perspective(self, name) -> str:
        if name == "left_above_label":
            return "left_above"
        elif name == "left_center_label":
            return "left_center"
        elif name == "left_below_label":
            return "left_below"
        elif name == "right_above_label":
            return "right_above"
        elif name == "right_center_label":
            return "right_center"
        elif name == "right_below_label":
            return "right_below"
        elif name == "front_above_label":
            return "center_above"
        elif name == "front_center_label":
            return "center_center"
        elif name == "front_below_label":
            return "center_below"
        else:
            raise Exception("Invalid perspective")

    def capture_image(self):
        print("Capturing image...")
        if self.camera_thread is not None and self.currently_selected_label is not None and self.selected_person is not None:
            raw_frame = self.camera_thread.raw_frame
            if raw_frame is None:
                self.statusbar.showMessage("No frame captured")
                return
            perspective = self.get_perspective(self.currently_selected_label)
            self.selected_person.face.add_image(perspective, raw_frame)
            label = getattr(self, self.currently_selected_label)
            face_image = self.selected_person.face.get_face_image(perspective)
            qimage = convert_to_qt_format(face_image.face() if self.select_image_display_mode is ImageDisplayMode.FACES else face_image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.statusbar.showMessage("No camera selected or no image selected or no person selected")

    def test_predict(self):
        print("Testing prediction...")
        if self.model["classifier"] is None:
            self.statusbar.showMessage("No classifier in database")
            return
        
        clf = self.model["classifier"]

        if self.camera_thread is not None:
            raw_frame = self.camera_thread.raw_frame

        if raw_frame is None:
            self.statusbar.showMessage("No frame captured")
            return
        
        # Extract faces from the frame.
        face_box = extract_single_face_single_image(raw_frame)
        if face_box[0] == 0 and face_box[1] == 0 and face_box[2] == 0 and face_box[3] == 0:
            self.statusbar.showMessage("No face detected")
            return
        
        face = raw_frame[face_box[1]:face_box[3], face_box[0]:face_box[2]]
        features = extract_feature(face)


        # Predict the person.
        person_name = predict(clf, features)
        self.statusbar.showMessage("Predicted: " + person_name)

    def show_model(self):      
        if self.selected_person is not None:
            self.show_person(self.selected_person, self.image_display_mode)

        person_names = list(self.model["persons"].keys())
        self.database_table.setRowCount(len(person_names))
        for idx, person_name in enumerate(person_names):
            person = self.model["persons"][person_name]
            self.database_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(person.name))
            self.database_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(person.status.value))

    def show_person(self, person: Person, mode: ImageDisplayMode):
        image = person.face.left_above
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.left_above_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.left_above_label.clear()
            self.left_above_label.setText("No Image")

        image = person.face.left_center
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.left_center_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.left_center_label.clear()
            self.left_center_label.setText("No Image")

        image = person.face.left_below
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.left_below_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.left_below_label.clear()
            self.left_below_label.setText("No Image")

        image = person.face.right_above
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.right_above_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.right_above_label.clear()
            self.right_above_label.setText("No Image")

        image = person.face.right_center
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.right_center_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.right_center_label.clear()
            self.right_center_label.setText("No Image")

        image = person.face.right_below
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.right_below_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.right_below_label.clear()
            self.right_below_label.setText("No Image")

        image = person.face.center_above
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            self.front_above_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.front_above_label.clear()
            self.front_above_label.setText("No Image")

        image = person.face.center_center
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)            
            self.front_center_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.front_center_label.clear()
            self.front_center_label.setText("No Image")

        image = person.face.center_below
        if image is not None:
            qimage = convert_to_qt_format(image.face() if mode is ImageDisplayMode.FACES else image.image).scaled(320, 240, QtCore.Qt.KeepAspectRatio)            
            self.front_below_label.setPixmap(QtGui.QPixmap.fromImage(qimage))
        else:
            self.front_below_label.clear()
            self.front_below_label.setText("No Image")

if __name__ == '__main__':
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')
    window = MainWindow()
    sys.exit(app.exec_())
    