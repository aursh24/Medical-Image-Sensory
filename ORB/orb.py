''' 
Steps for Matching the Image:  
Import 2 Images
Convert 2 Gray scale
Initialise ORB Detector
Find Key Points and describe them
Match Keypoints - Brute Force Matcher
RANSAC (It Rejects bad Keypoints)
Register two images (use Homology)
'''


import cv2
import numpy as np

img1 = cv2.imread("D:/ORB Aursh Personal/Best Images/after.jpg") #Image to be registered
img2 = cv2.imread("D:/ORB Aursh Personal/Input/before.jpg")  #Reference Image

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)


#initialising orb
orb = cv2.ORB_create(500)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

#Match the descriptors

matches = matcher.match(des1, des2, None)

matches = sorted(matches, key = lambda x:x.distance)


#creating an empty list with zeros

points1 = np.zeros((len(matches), 2), dtype= np.float32)
points2 = np.zeros((len(matches), 2), dtype= np.float32)

# filling the empty list and unwrapping the keypoints 
# to mark the difference b/w 2 distinct cordinates of the 2 images
# basically calculating the homography

for i, match in enumerate(matches):
    points1[i, :] = kp1[match.queryIdx].pt   #it gives the list of the descriptors 
    points2[i, :] = kp2[match.trainIdx].pt    #it gives the list of training descriptors

h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)  #RANSAC keeps the inlier points(good points) and excludes outliers(bad points)

#use homography

height, width, channels = img2.shape

img1Reg = cv2.warpPerspective(img1, h, (width, height))

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None)
cv2.imshow("Keypoint Image", img3)
cv2.imshow("Registered Image", img1Reg)
cv2.waitKey(0)