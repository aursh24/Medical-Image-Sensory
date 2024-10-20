import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same dimensions")
    
    return image1, image2

def downsample_image(image, factor):
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_AREA)

def morans_i(image1, image2, x, y, width, height):
    flat_image1 = image1.flatten()
    flat_image2 = image2.flatten()

    mean1 = np.mean(flat_image1)
    mean2 = np.mean(flat_image2)

    numerator = 0
    denominator = (flat_image1[y * width + x] - mean1) ** 2

    neighbors = [
        (x - 1, y), (x + 1, y),  
        (x, y - 1), (x, y + 1)   
    ]

    for nx, ny in neighbors:
        if 0 <= nx < width and 0 <= ny < height:
            neighbor_idx = ny * width + nx
            numerator += (flat_image1[y * width + x] - mean1) * (flat_image2[neighbor_idx] - mean2)

    I = (numerator / denominator) if denominator != 0 else 0
    return I

def main(image_path1, image_path2, output_csv, num_points=100, downsample_factor=1):
    image1, image2 = load_and_prepare_images(image_path1, image_path2)

    if downsample_factor > 1:
        image1 = downsample_image(image1, downsample_factor)
        image2 = downsample_image(image2, downsample_factor)

    height, width = image1.shape

    selected_indices = np.random.choice(height * width, num_points, replace=False)
    results = []

    for idx in selected_indices:
        x = idx % width
        y = idx // width
        morans_value = morans_i(image1, image2, x, y, width, height)
        results.append((x, y, morans_value))

    df = pd.DataFrame(results, columns=['X', 'Y', "Moran's I"])
    df.to_csv(output_csv, index=False)

    plt.figure(figsize=(6, 6))
    plt.imshow(image1, cmap='gray')
    plt.title('Selected Points on Image 1')
    plt.axis('off')

    for (x, y, _) in results:
        plt.scatter(x, y, color='red', s=10) 

    plt.show()

if __name__ == "__main__":
    image_path1 = "/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/1_colon_10x.tif"
    image_path2 = "/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/2_colon_10x.tif"
    output_csv = "/home/sristy/Desktop/Medical-Image-Sensory/spatial correlation coefficient/moran_results.csv"
    downsample_factor = 2 
    main(image_path1, image_path2, output_csv, num_points=100, downsample_factor=downsample_factor)
