import cv2 
import matplotlib.pyplot as plt
import numpy as np
import os
 
#read images
directoryName =  os.path.dirname(__file__)

img1 = cv2.imread(directoryName + "/examples/1.jpg")  
img2 = cv2.imread(directoryName + "/examples/2.jpg")

#SIFT
sift = cv2.SIFT_create()

keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

#find matches using knn algorithm
match = cv2.BFMatcher()
matches = match.knnMatch(descriptors_1, descriptors_2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
        
#parameters for the drawMatches() function    
draw_params = dict(matchColor = (255, 0, 0), 
                       singlePointColor = None,
                       flags = 4)

img3 = cv2.drawMatches(img1, keypoints_1, img2,keypoints_2, good, None, **draw_params)
plt.imshow(img3)
plt.title("Matches after the implementation of SIFT detector")
plt.savefig(directoryName + '/SiftDetectorImplementation.png')
plt.show()
plt.close()


#HOMOGRAPHY - RANSAC

#m is a DMatch object. From this object we want only the correcponding points between
#the two images, in order to calculate homography.
src_pts = np.float32([ keypoints_1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ keypoints_2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

#calculate the Homography using Ransac algorithm, in order to get the model with the most inliers.
HomographyMatrix, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

draw_params4 = dict(matchColor = (255, 0, 0), 
                                singlePointColor = None,
                                matchesMask = matchesMask, # draw only good inliers points
                                flags = 2)

img4  = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good, None, **draw_params4)

plt.imshow(img4)
plt.title("Matches after Ransac's algorithm implementation")
plt.savefig(directoryName + '/AfterRansacImplementation.png')
plt.show()
plt.close()


#Modifications to the angle of the second image, in order to get stitched to the first one.

#z contains some details about the colour of the image
h, w, z = img1.shape
InitialMatrix = np.array([[0, w -1, w -1,0],[0,0, h -1, h-1],[1,1,1,1]])

# Finding the final coordinates (xi, yi) of the corners of the
# image after transformation.

#Final matrix is the inner product of the HomographyMatrix and the InitialMatrix
FinalMatrix = np.dot(HomographyMatrix, InitialMatrix)
[x, y, c] = FinalMatrix
x = np.divide(x, c)
y = np.divide(y, c)

min_x, max_x =int(round(min(x))),int(round(max(x)))
min_y, max_y =int(round(min(y))),int(round(max(y)))

NewWidth = max_x
NewHeight = max_y
Correction = [0,0]

if min_x <0:
    NewWidth -= min_x
    Correction[0] =abs(min_x)
if min_y <0: 
    NewHeight -= min_y
    Correction[1] =abs(min_y)
    
    
if NewWidth < (img1.shape[1] + Correction[0]):
    NewWidth = img1.shape[1] + Correction[0]
if NewHeight < (img1.shape[0] + Correction[1]):
    NewHeight = img1.shape[0] + Correction[1]
    
    
x = np.add(x, Correction[0])
y = np.add(y, Correction[1])

OldPoints = np.float32([[0,0],[w -1,0],[w -1, h-1],[0, h -1]])
NewPoints = np.float32(np.array([x, y]).transpose())

#this function takes an input of 4 pairs of corresponding points and outputs the tranfsormation matrix
TransformationMatrix = cv2.getPerspectiveTransform(OldPoints,NewPoints)

#here the perspective transformation of img2 is taking place
StitchedImage = cv2.warpPerspective(img2, TransformationMatrix,(NewHeight, NewWidth))

StitchedImage[Correction[1]:Correction[1]+img1.shape[0],Correction[0]:Correction[0]+img1.shape[1]]= img1

plt.imshow(StitchedImage)
plt.title("Stitched Image")
plt.savefig(directoryName + '/StitchedImage.png')
plt.show()
plt.close()