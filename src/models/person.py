
import numpy as np
import cv2
from face_identification.classifier import extract_face, extract_feature, extract_features

class FaceImage:
    def __init__(self, image: cv2.Mat, face: tuple[int, int, int, int]):
        self.image: cv2.Mat = image
        self.face_box = extract_face(image)
        # Throw exception if face is not found
        if not self.is_face():
            raise Exception("Face not found")
        self.feature: np.ndarray = extract_feature(self.face())

    def is_face(self) -> bool:
        return self.face != (0, 0, 0, 0)
    
    def face(self) -> cv2.Mat:
        return self.image[self.face_box[0]:self.face_box[2], self.face_box[1]:self.face_box[3]]

class Face:
    def __init__(self):
        self.images: dict[str, FaceImage] = {}

        self.attributes = [
            "left_above",
            "left_center",
            "left_below",
            "right_above",
            "right_center",
            "right_below",
            "center_above",
            "center_center",
            "center_below"
        ]

        # Different perspectives of the face used for training the classifier
        self.left_above: FaceImage = None
        self.left_center: FaceImage = None
        self.left_below: FaceImage = None
        self.right_above: FaceImage = None
        self.right_center: FaceImage = None
        self.right_below: FaceImage = None
        self.center_above: FaceImage = None
        self.center_center: FaceImage = None
        self.center_below: FaceImage = None

    def get_features(self) -> list:
        """     
        Extracts features from faces using dlib's face recognition model.
        :param faces: list of faces
        :return: list of features
        """
        faces = [getattr(self, attr_name).face() for attr_name in self.attributes if getattr(self, attr_name) is not None]
        return extract_features(faces)

    def add_image(self, perspective: str, image: cv2.Mat):
        if perspective not in self.attributes:
            raise Exception("Invalid angle")
        self.setattr(perspective, FaceImage(image, extract_face(image)))

class Person():
    def __init__(self, name: str):
        self.name: str = name
        self.face: Face = None


    
