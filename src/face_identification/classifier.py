
import cv2
import numpy as np
import dlib

def extract_faces(images: list[cv2.Mat]) -> list[tuple[int, cv2.Mat]]:
    """
    Extracts faces from images using OpenCV's DNN face detector. Returns a list of faces and their corresponding image index.
    :param images: list of images
    :return: list of tuples (index of origin image, face image)
    """

    net = cv2.dnn.readNetFromCaffe("./src/face-detection/deploy.prototxt.txt", "./src/face-detection/res10_300x300_ssd_iter_140000.caffemodel")
    faces = []

    for idx in range(len(images)):
        frame = images[idx]
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Find the detection with the highest confidence
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        face = None
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

        faces.append((idx, face))

    return faces

def extract_features(faces: list[cv2.Mat]) -> list:
    """
    Extracts features from faces using dlib's face recognition model.
    :param faces: list of faces
    :return: list of features
    """

    # Initialize the face encoder
    # Note: This assumes you have dlib's face recognition model in your path
    # You can download it from http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
    face_encoder = dlib.face_recognition_model_v1("./src/dlib_face_recognition_resnet_model_v1.dat")

    features = []

    for face in faces:
        # rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        rgb = cv2.resize(rgb, (150, 150))
        # Compute the facial embedding
        encoding = face_encoder.compute_face_descriptor(rgb)
        features.append(np.array(encoding))

    return features

def train_classifier():
    # Use scikit-learn to train a classifier on the features
    pass

def evaluate_classifier():
    # Use scikit-learn to evaluate the classifier using cross-validation
    pass

def predict():
    # Use scikit-learn to predict the class of a new face
    pass
