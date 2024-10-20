import cv2
import numpy as np
import pandas as pd
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    """Read an image and convert it to grayscale."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced_image = cv2.equalizeHist(gray_image)
    return enhanced_image

def segment_image(enhanced_image):
    """Segment the image to find contours."""
    _, binary_image = cv2.threshold(enhanced_image, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_features(contours):
    """Extract coordinates and area of contours."""
    coordinates = []
    areas = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contour) 
            coordinates.append((cX, cY))
            areas.append(area)  

    return coordinates, areas

def perform_spatial_analysis(coordinates, values):
    """Perform spatial analysis using Moran's I."""
    w = weights.KNN.from_array(np.array(coordinates), k=4)
    moran = esda.Moran(values, w)
    return moran.I, moran.p_sim

def visualize_results(coordinates, values):
    """Visualize the results with a colorful scatter plot."""
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(*zip(*coordinates), c=values, cmap='viridis', alpha=0.6, edgecolor='w')
    plt.colorbar(scatter, label='Area Values')  
    plt.title("Spatial Distribution of Contour Areas")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()

def save_to_csv(coordinates, areas, output_path):
    """Save the dataset to a CSV file."""
    df = pd.DataFrame(coordinates, columns=['X', 'Y'])
    df['Area'] = areas
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

def main(image_path, output_path):
    """Main function to process the image and generate the dataset."""
    enhanced_image = preprocess_image(image_path)
    contours = segment_image(enhanced_image)
    coordinates, areas = extract_features(contours)

    moran_i, p_value = perform_spatial_analysis(coordinates, areas)
    visualize_results(coordinates, areas)

    print(f"Moran's I: {moran_i}, p-value: {p_value}")

    save_to_csv(coordinates, areas, output_path)

if __name__ == "__main__":
    image_path = '/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/1_colon_10x.tif'  
    output_path = '/home/sristy/Desktop/Medical-Image-Sensory/spatial correlation coefficient/contour_data.csv'
    main(image_path, output_path)
