import cv2
import numpy as np

# Function to calculate PSNR
def calculate_psnr(original, distorted):
    mse = np.mean((original - distorted) ** 2)
    if mse == 0:
        return float('inf')  # Perfect match, no difference
    max_pixel_value = 255.0

    #if mse != 0 then PSNR is calculated
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

# Function to detect and match features using SIFT
def sift_feature_matching(img1, img2):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Use FLANN-based matcher
    index_params = dict(algorithm=1, trees=5)  # FLANN index for SIFT
    search_params = dict(checks=50)  # Search checks
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform knn matching with k=2
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return len(kp1), len(kp2), len(good_matches)

# Load original and distorted images
original_img = cv2.imread('D:/ORB Aursh Personal/Colon/10x/1_colon_10x.tif', cv2.IMREAD_GRAYSCALE)    #Referrence Image
distorted_img = cv2.imread('D:/ORB Aursh Personal/Colon/10x/2_colon_10x.tif', cv2.IMREAD_GRAYSCALE)   #Image to be Registered

# Calculate PSNR
psnr_value = calculate_psnr(original_img, distorted_img)
print(f"PSNR between original and distorted image: {psnr_value} dB")

# Perform SIFT feature matching
sift_kp1, sift_kp2, sift_matches = sift_feature_matching(original_img, distorted_img)
print(f"SIFT - \nKeypoints in Original: {sift_kp1}\nKeypoints in Image to Reg(Distorted): {sift_kp2}\nGood Matches: {sift_matches}\n")
