import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_and_prepare_images(image_path1, image_path2):
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
    
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same dimensions")
    
    return image1, image2

def downsample_image(image, factor):
    return cv2.resize(image, (image.shape[1] // factor, image.shape[0] // factor), interpolation=cv2.INTER_AREA)

def generate_systematic_points(image, num_points, offset=5):
    height, width = image.shape
    y_coords = np.linspace(0, height - 1, int(np.sqrt(num_points)))
    x_coords = np.linspace(0, width - 1, int(np.sqrt(num_points)))
    points = [(int(x), int(y)) for y in y_coords for x in x_coords]
    
    noisy_points = [(x + np.random.randint(-offset, offset + 1), 
                     y + np.random.randint(-offset, offset + 1)) for x, y in points]
    
    noisy_points = [(x, y) for x, y in noisy_points if 0 <= x < width and 0 <= y < height]
    
    return list(zip(*noisy_points[:num_points]))

def calculate_good_matches(points1, points2, good_match_threshold=30):
    good_matches = 0
    results = []

    x_coords1, y_coords1 = points1
    x_coords2, y_coords2 = points2

    for x1, y1, x2, y2 in zip(x_coords1, y_coords1, x_coords2, y_coords2):
        distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        results.append((x1, y1, x2, y2, distance))
        
        if distance < good_match_threshold:
            good_matches += 1

    return good_matches, results

def main(image_path1, image_path2, num_points=100, downsample_factor=1, good_match_threshold=30):
    image1, image2 = load_and_prepare_images(image_path1, image_path2)

    if downsample_factor > 1:
        image1 = downsample_image(image1, downsample_factor)
        image2 = downsample_image(image2, downsample_factor)

    points1 = generate_systematic_points(image1, num_points)
    points2 = generate_systematic_points(image2, num_points)

    good_matches_count, results = calculate_good_matches(points1, points2, good_match_threshold)
    good_match_percentage = (good_matches_count / num_points) * 100 if num_points > 0 else 0

    print(f"Good Matches Percentage: {good_match_percentage:.2f}%")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image1, cmap='gray')
    plt.title('Image 1 with Points')
    plt.axis('off')
    for (x1, y1) in zip(points1[0], points1[1]):
        plt.scatter(x1, y1, color='red', s=10)

    plt.subplot(1, 2, 2)
    plt.imshow(image2, cmap='gray')
    plt.title('Image 2 with Points')
    plt.axis('off')
    for (x2, y2) in zip(points2[0], points2[1]):
        plt.scatter(x2, y2, color='blue', s=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path1 = "/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/6_colon_10x.tif"
    image_path2 = "/home/sristy/Desktop/Medical-Image-Sensory/Colon/10x/10_colon_10x.tif"
    downsample_factor = 2 
    main(image_path1, image_path2, num_points=100, downsample_factor=downsample_factor)
