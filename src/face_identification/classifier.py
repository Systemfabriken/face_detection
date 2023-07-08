
import cv2
import numpy as np
import dlib

# Initialize the face encoder
# Note: This assumes you have dlib's face recognition model in your path
# You can download it from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
face_encoder = dlib.face_recognition_model_v1("./src/dlib_face_recognition_resnet_model_v1.dat")
net = cv2.dnn.readNetFromCaffe("./src/face-detection/deploy.prototxt.txt", "./src/face-detection/res10_300x300_ssd_iter_140000.caffemodel")

def extract_face(image: cv2.Mat) -> tuple[int, int, int, int]:
    """
    Extracts a face from an image using OpenCV's DNN face detector.
    :param image: image
    :return: face bounding box (x, y, w, h)
    """
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Find the detection with the highest confidence
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    face = (0, 0, 0, 0)
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        face = box.astype("int")

    return face

def extract_faces(images: list[cv2.Mat]) -> list[tuple[int, cv2.Mat]]:
    """
    Extracts faces from images using OpenCV's DNN face detector. Returns a list of faces and their corresponding image index.
    :param images: list of images
    :return: list of tuples (index of origin image, face image)
    """
    faces = []
    for idx in range(len(images)):
        frame = images[idx]
        (startX, startY, endX, endY) = extract_face(frame)
        if (startX, startY, endX, endY) == (0, 0, 0, 0):
            faces.append((idx, None))
            continue
        face = frame[startY:endY, startX:endX]
        faces.append((idx, face))
    return faces

def extract_feature(face: cv2.Mat) -> np.ndarray:
    """
    Extracts features from a face using dlib's face recognition model.
    :param face: face
    :return: features
    """
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
    rgb = cv2.resize(rgb, (150, 150))
    encoding = face_encoder.compute_face_descriptor(rgb)
    return np.array(encoding)

def extract_features(faces: list[cv2.Mat]) -> list:
    """
    Extracts features from faces using dlib's face recognition model.
    :param faces: list of faces
    :return: list of features
    """
    return [extract_feature(face) for face in faces]

def train_classifier():
    # Use scikit-learn to train a classifier on the features
    pass

def evaluate_classifier():
    # Use scikit-learn to evaluate the classifier using cross-validation
    pass

def predict():
    # Use scikit-learn to predict the class of a new face
    pass
