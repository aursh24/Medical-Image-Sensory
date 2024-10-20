import cv2
import numpy as np
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    enhanced_image = cv2.equalizeHist(gray_image)

    return enhanced_image

def segment_image(enhanced_image):
    _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def extract_features(contours):
    coordinates = []
    values = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            coordinates.append((cX, cY))
            values.append(np.random.rand() * 100)  

    return coordinates, values

def perform_spatial_analysis(coordinates, values):
    w = weights.KNN.from_array(np.array(coordinates), k=4)

    moran = esda.Moran(values, w)

    return moran.I, moran.p_sim

def visualize_results(coordinates, values):
    plt.scatter(*zip(*coordinates), c=values, cmap='hot', alpha=0.5)
    plt.colorbar(label='Values')
    plt.title("Spatial Distribution of Features")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def main(image_path):
    enhanced_image = preprocess_image(image_path)

    contours = segment_image(enhanced_image)

    coordinates, values = extract_features(contours)

    moran_i, p_value = perform_spatial_analysis(coordinates, values)

    visualize_results(coordinates, values)

    print(f"Moran's I: {moran_i}, p-value: {p_value}")

if __name__ == "__main__":
    image_path = '/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/1_colon_10x.tif'  
    main(image_path)
