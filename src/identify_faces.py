import cv2
import pickle
import argparse
import os

# Move the current working directory one level up
os.chdir('..')

# Load your classifier
with open('Sodra_Vagen_4.secsystsvm', 'rb') as f:
    clf = pickle.load(f)

# Import functions from your module
from face_identification.classifier import (
    extract_multi_faces_single_image, 
    get_face_from_image, 
    extract_feature, 
    predict
)

# Function to draw boxes and labels
def draw_boxes_and_labels(image: cv2.Mat, faces: list[tuple[int, int, int, int]], clf) -> None:
    """
    Draw bounding boxes and labels on image
    :param image: image
    :param faces: list of faces
    :param clf: trained classifier
    """
    # Define font and color for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color

    # Draw box and label for each face
    for (idx, (startX, startY, endX, endY)) in faces:
        if (startX, startY, endX, endY) == (0, 0, 0, 0):
            continue

        # Draw bounding box
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Extract face using bounding box
        face = get_face_from_image(image, (startX, startY, endX, endY))
        
        # Extract feature from face
        feature = extract_feature(face)
        
        # Predict label
        label = predict(clf, feature)
        
        # Draw label
        cv2.putText(image, label, (startX, startY-10), font, font_scale, font_color, 2)

     # Save the image
    cv2.imwrite("output.jpg", image)

# Set up command line arguments
parser = argparse.ArgumentParser(description='Perform face detection on an image.')
parser.add_argument('image_path', type=str, help='The path to the image file.')

# Parse the command line arguments
args = parser.parse_args()

# Load image using the command line argument
image = cv2.imread(args.image_path)

# Extract faces from image
faces = extract_multi_faces_single_image(image)

# Draw bounding boxes and labels
draw_boxes_and_labels(image, faces, clf)
