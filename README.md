In this project Image Stitching is being reviewd with the use of 2, 3 and 4 images. 

* The common attributes between the images are detected with the use of SIFT algorith.
* Afterwards, RANSAC algorithm is implemented to find the model with the largest number of inliers with the use of the homography matrix.
* Lastly, the image is warped and blended and the final stitched image is being saved.

The ImageStitching2Images.py script stitches images 1.jpg, 2.jpg that are inside examples folder. During the running of this code the images SiftDetectorImplementation, AfterRansacImplementation and StitchedImage are created.

Inside ImageStitching3Images there is a python script that stithes the 3 images that are included in this file. In order for this code to run correctly, the StitchedImage must by deleted.

Inside ImageStitching3Images there is a python script that stithes the 4 images included in this file. The results and the images created from this script can be shown inside the .ipynb file. 
