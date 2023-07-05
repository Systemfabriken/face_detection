from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
import numpy as np

from ui_generated.pyqt5.classification_main_window import Ui_MainWindow

def get_available_cameras(max_cameras=10):
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

# class Ui_MainWindow(object):
#     def setupUi(self, MainWindow):
#         MainWindow.setObjectName("MainWindow")
#         MainWindow.resize(800, 600)
#         self.centralwidget = QtWidgets.QWidget(MainWindow)
#         self.centralwidget.setObjectName("centralwidget")

#         # Create a 3x3 grid of labels to display the images
#         self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
#         self.labels = [[QtWidgets.QLabel(self.centralwidget) for _ in range(3)] for _ in range(3)]
#         for i in range(3):
#             for j in range(3):
#                 self.gridLayout.addWidget(self.labels[i][j], i, j)

#         # Create buttons to start the capture process, capture an image, and confirm the image
#         self.startButton = QtWidgets.QPushButton("Start", self.centralwidget)
#         self.captureButton = QtWidgets.QPushButton("Capture", self.centralwidget)
#         self.confirmButton = QtWidgets.QPushButton("Confirm", self.centralwidget)
#         self.gridLayout.addWidget(self.startButton, 3, 0)
#         self.gridLayout.addWidget(self.captureButton, 3, 1)
#         self.gridLayout.addWidget(self.confirmButton, 3, 2)

#         MainWindow.setCentralWidget(self.centralwidget)

#         # Connect the buttons to their respective methods
#         self.startButton.clicked.connect(self.start_capture)
#         self.captureButton.clicked.connect(self.capture_image)
#         self.confirmButton.clicked.connect(self.confirm_image)

#         self.cap = None
#         self.current_image = None
#         self.current_position = (0, 0)

#     def start_capture(self):
#         # Open the webcam
#         self.cap = cv2.VideoCapture(0)

#     def capture_image(self):
#         # Capture an image from the webcam
#         ret, frame = self.cap.read()
#         if ret:
#             # Convert the image to QPixmap and display it on the current label
#             height, width, channel = frame.shape
#             bytesPerLine = 3 * width
#             qImg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888).rgbSwapped()
#             self.labels[self.current_position[0]][self.current_position[1]].setPixmap(QtGui.QPixmap.fromImage(qImg))

#             # Store the current image for confirmation
#             self.current_image = frame

#     def confirm_image(self):
#         # Save the current image to a file
#         cv2.imwrite(f"image_{self.current_position[0]}_{self.current_position[1]}.jpg", self.current_image)

#         # Move to the next position
#         self.current_position = ((self.current_position[0] + (self.current_position[1] + 1) // 3) % 3, (self.current_position[1] + 1) % 3)

class CameraThread(QtCore.QThread):
    changePixmap = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, camera_port):
        super().__init__()
        self.camera_port = camera_port
        self.cap = cv2.VideoCapture(self.camera_port)
        self.curr_img_320_240 = None
        self.raw_frame = None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.raw_frame = frame
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
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

        self.set_images_enabled(False)
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

        self.capture_image_button.setEnabled(False)
        self.capture_image_button.clicked.connect(self.capture_image)
        self.raw_frames = []

        self.preprocess_images_button.setEnabled(False)
        self.preprocess_images_button.clicked.connect(self.preprocess_images)
        self.net = cv2.dnn.readNetFromCaffe("./src/face-detection/deploy.prototxt.txt", "./src/face-detection/res10_300x300_ssd_iter_140000.caffemodel")

        self.show()

    @QtCore.pyqtSlot(QtGui.QImage)
    def setCameraImage(self, image):
        self.camera_image_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def select_camera(self, index):
        if index == 0:
            self.statusbar.showMessage("No Camera Selected")
            self.set_images_enabled(False)
        else:
            self.statusbar.showMessage("Camera " + str(index) + " selected")
            self.set_images_enabled(True)

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
            self.capture_image_button.setEnabled(True)
        else:
            self.statusbar.showMessage("No image selected")
            self.capture_image_button.setEnabled(False)

    def set_images_enabled(self, enabled):
        self.right_above_label.setEnabled(enabled)
        self.right_center_label.setEnabled(enabled)
        self.right_below_label.setEnabled(enabled)
        self.front_above_label.setEnabled(enabled)
        self.front_center_label.setEnabled(enabled)
        self.front_below_label.setEnabled(enabled)
        self.left_above_label.setEnabled(enabled)
        self.left_center_label.setEnabled(enabled)
        self.left_below_label.setEnabled(enabled)

    def capture_image(self):
        if self.camera_thread is not None:
            label = getattr(self, self.currently_selected_label)
            label.setPixmap(QtGui.QPixmap.fromImage(self.camera_thread.curr_img_320_240))
            self.raw_frames.append(self.camera_thread.raw_frame)
            if len(self.raw_frames) == 9:
                self.preprocess_images_button.setEnabled(True)

    def preprocess_images(self):
        print("Preprocessing images...")
        for frame in self.raw_frames:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())