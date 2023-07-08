from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np
import dlib
import face_recognition
import pickle

# Local imports
from ui_generated.pyqt5.classification_main_window import Ui_MainWindow
from face_identification.classifier import extract_faces, extract_features

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

        self.actionSave_Current.triggered.connect(self.save_model)
        self.actionLoad.triggered.connect(self.load_model)

        self.capture_image_button.setEnabled(False)
        self.capture_image_button.clicked.connect(self.capture_image)
        self.model: dict = dict()
        self.model["images"] = dict()
        self.model["currently_selected_label"] = None
        self.model["features"] = None

        self.init_from_model()

        self.process_images_button.setEnabled(False)
        self.process_images_button.clicked.connect(self.process_images)

        self.show()

    def save_model(self):
        print("Saving model...")
        self.model["currently_selected_label"] = self.currently_selected_label
        with open("model.pickle", "wb") as f:
            pickle.dump(self.model, f)
        print("Model saved")

    def load_model(self):
        print("Loading model...")
        with open("model.pickle", "rb") as f:
            self.model = pickle.load(f)
        print("Model loaded")
        self.init_from_model()

    def init_from_model(self):
        self.currently_selected_label = self.model["currently_selected_label"]
        if self.currently_selected_label is not None:
            self.on_image_clicked(self.currently_selected_label)
        self.show_model()
        if len(self.model["images"]) > 0:
            self.process_images_button.setEnabled(True)
        else:
            self.process_images_button.setEnabled(False)

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

    def capture_image(self):
        if self.camera_thread is not None and self.currently_selected_label is not None:
            label = getattr(self, self.currently_selected_label)
            label.setPixmap(QtGui.QPixmap.fromImage(self.camera_thread.curr_img_320_240))
            self.model["images"][self.currently_selected_label] = (self.camera_thread.raw_frame, None)
            if len(self.model["images"]) > 0:
                self.process_images_button.setEnabled(True)

    def process_images(self):
        print("Processing images...")
        self.statusbar.showMessage("Processing images...")

        # Extracting faces
        frame_names = list(self.model["images"].keys())
        faces = extract_faces([self.model["images"][frame_name][0] for frame_name in frame_names])
        print("Found " + str(len(faces)) + " faces")
        for idx, face in faces:
            self.model["images"][frame_names[idx]] = (self.model["images"][frame_names[idx]][0], face)
        self.display_faces()

        # Extracting features
        faces_list = [face for idx, face in faces if face is not None]
        self.model["features"] = extract_features(faces_list)

        self.statusbar.showMessage("Done processing images")

    def display_faces(self):
        print("Displaying faces...")
        frame_names = list(self.model["images"].keys())
        for frame_name in frame_names:
            face = self.model["images"][frame_name][1]
            label = getattr(self, frame_name)
            if face is None:
                label.set_invalid(True)
                continue
            else:
                label.set_invalid(False)
            # Display face
            converted_face = convert_to_qt_format(face).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
            label.setPixmap(QtGui.QPixmap.fromImage(converted_face))

    def show_model(self):
        frame_names = list(self.model["images"].keys())
        for frame_name in frame_names:
            frame = self.model["images"][frame_name][0]
            face = self.model["images"][frame_name][1]
            label = getattr(self, frame_name)
            if face is None:
                qimage = convert_to_qt_format(frame).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
                label.setPixmap(QtGui.QPixmap.fromImage(qimage))
                continue
            else:
                qimage = convert_to_qt_format(face).scaled(320, 240, QtCore.Qt.KeepAspectRatio)
                label.setPixmap(QtGui.QPixmap.fromImage(qimage))

if __name__ == '__main__':
    import sys
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)  # Enable high DPI scaling
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Breeze')
    window = MainWindow()
    sys.exit(app.exec_())
    