## Lane Detection and Turn Prediction

Implementation of Lane Detection and Turn Prediction with Histogram Equalization methods

## How to run the code

-- Recommended Python : 3

-- Recommended IDE : SPYDER

-- Problem_1_Hist_Equalization.py : Code for Histogram Equalization

-- Problem_2_DataSet_1.py : Code for Lane Detection of Data Set 1

-- Problem_2_DataSet_2.py : Code for Lane Detection of Data Set 1

-- Video_Stitch.py : Code to stitch together images of Data Set 1


	> For the Problem 1, if you need to see the outputs of Gray Scale Histogram Equalization, HSV Histogram Equalization or 
	the RGB Histogram Equalization, just uncomment the respective lines in the main while loop.

	> Please make sure to have the 'Night Drive - 2689.mp4' video in the same folder as the code.

	> The images provided for the data set 1 are stitched to make a video file called 'Data_Set1_vid.mp4'.

	> Please make sure to have the 'challenge_video.mp4' video as well as the 'Data_Set1_vid.mp4' in the same folder as the codes.

	> The data folder zip file will need to be downloaded and unzipped and the Video_stitch.py code to be inside the unzipped folder
	to properly get a stitched video of the Data Set 1.

	> If you wish to see a slightly transparent version of the filled lane as output, uncomment line 179 in Problem_2_DataSet_1.py

	> If you wish to see a slightly transparent version of the filled lane as output, uncomment line 194 in Problem_2_DataSet_2.py

	> The video writing commands out.write(line 207) in Problem_2_DataSet_2.py and (line 192) in Problem_2_DataSet_1.py are commented so as not to lose the original video.
	  Please make sure to unhash it if you wish to write a new video.

> If you wish to see a particular window of the pipeline such as warped image, mask, cropped image etc, then uncomment the respective line 
in the code.
