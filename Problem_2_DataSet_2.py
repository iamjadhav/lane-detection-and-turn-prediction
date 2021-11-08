import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("challenge_video.mp4")
#image shape --> (1240,720)

out = cv2.VideoWriter('challenge_output_Final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1200,617))

#undistorting the frame using the calibration and distance matrix 
def undistort(frame_dist):
    
    K = np.array([[1.15422732e+03,0.00000000e+00,6.71627794e+02],[0.00000000e+00,1.14818221e+03,3.86046312e+02],[0.00000000e+00,0.00000000e+00,1.00000000e+00]])
    D = np.array([[ -2.42565104e-01,-4.77893070e-02,-1.31388084e-03,-8.79107779e-05,2.20573263e-02]])
    
    height, width = frame_dist.shape[:2]
    
    K_new, roi = cv2.getOptimalNewCameraMatrix(K , D , (width , height) , 1 , (width , height) )
                                                      
    correct = cv2.undistort(frame_dist, K, D, None, K_new)

    l, m, n, o = roi
    
    correct = correct[m:m + o, l:l + n]

    return correct

#calculating the radius of curvature
def roc(coef_left,coef_right,points_left,points_right):
    
    radius = ((1+((2*coef_right[0]*points_right[50][1])+coef_right[1])**2)**(3/2))/abs(2*coef_right[0])
    return radius
        
#determining the homography and creating the warped image
def homography(warped):
    
    source = np.array([[560, 50], [145, 220],[1180, 220], [725, 50]])#main
    
    dest = np.array([[0, 0], [0, 500], [200, 500],[200, 0]])
    
    H , status = cv2.findHomography(source, dest)

    warped = cv2.warpPerspective(warped, H, (200,500))
    
    return warped

#determining the inverse homography and creating the unwarped image
def unwarp(unwarped):
    
    source = np.array([[560, 50], [145, 220],[1180, 220], [725, 50]])#main
     
    dest = np.array([[0, 0], [0, 500], [200, 500],[200, 0]])
    
    H , status = cv2.findHomography(source, dest)

    unwarped = cv2.warpPerspective(unwarped, np.linalg.inv(H), (crop.shape[1],crop.shape[0]))
    
    return unwarped
    
#calculating the histogram to determine two intensity peaks corresponding to the lanes
def histogram(image):
    
    whites = list()
    index = list()
    
    for i in range(image.shape[1]):
        z = np.where(image[:,i] > 0)
        whites.append(len(z[0]))
        index.append(i)
        
    one = whites[:120]
    two = whites[120:]
    max1 = max(one)
    max2 = max(two)
    col1 = whites.index(max1)
    col2 = whites.index(max2)
    
    #bins = 200
    #plt.plot(index,whites, bins)
    #plt.show()
    return col1,col2

#determining the direction of the road using the image and the lane centers
def direction(img_center,left_max,right_max,res_img,radius_dir):
    
    l_center = left_max + (right_max - left_max)/2
    dev = l_center - img_center
    
    if ( -14 < dev < -9.5):
        res_img = cv2.putText(res_img,'Straight',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("Straight")
    elif ( -9.5 < dev < -3):
        res_img = cv2.putText(res_img,'Right',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("Right")
    else:#( 6 < dev )
        res_img = cv2.putText(res_img,'Left',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("Left")
     
    
while True:
    
    ret, frame = cap.read()
    if ret == True:
    
        frame1 = frame.copy()
        
        undistorted = undistort(frame1)
        
        #blur = cv2.GaussianBlur(undistorted,(3,3),0)
        med_blur = cv2.medianBlur(undistorted,ksize=3)
        
        #cropping the denoised frame to get only the road
        crop = med_blur[360:, :]
        crop_1 = med_blur[:360, :]
        
        #crop_full = np.concatenate((crop_1,crop), axis = 0)
        
        #cv2.imshow('Crop Full',crop_full)
        
        warped = homography(crop)
        
        hsl = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)
        
        lower_yellow = np.array([15, 100, 20])
        higher_yellow = np.array([60, 200, 250])
        
        mask_yellow = cv2.inRange(hsl, lower_yellow, higher_yellow)
        
        yellow_hsl = cv2.bitwise_and(hsl, hsl, mask = mask_yellow).astype(np.uint8)
        
        lower_white = np.array([0, 180, 0])
        higher_white = np.array([255, 255, 255])
        
        mask_white = cv2.inRange(hsl, lower_white, higher_white)
        
        white_hsl = cv2.bitwise_and(hsl, hsl, mask=mask_white).astype(np.uint8)
        
        #combining the white and yellow hsl masks
        combined_hsl = cv2.bitwise_or(white_hsl,yellow_hsl)
        
        #getting the maximum intensity column indices on the left and right side
        col_l,col_r = histogram(combined_hsl)
        
        try:
            #finding the white pixels in the max column as well as its neighbouring columns
            leftpts = np.where(combined_hsl[:,col_l-10:col_l+11] > 0)
            rightpts = np.where(combined_hsl[:,col_r-10:col_r+11] > 0)
            #calculating the coefficients of the polynomials of left and right lanes
            coeff_left = np.polyfit(leftpts[0], leftpts[1]+(col_l-10), 2)
            coeff_right = np.polyfit(rightpts[0], rightpts[1]+(col_r-10), 2)
        except:
            continue   
        
        x = np.arange(500)
        x_sqr = np.square(x)
        ones = np.ones(500)
        
        x_pts = np.stack((x_sqr,x,ones))
        
        #calculating y = dot product of the coefficients and x
        points_left = np.dot(coeff_left,x_pts).astype(np.uint8)
        points_right = np.dot(coeff_right,x_pts).astype(np.uint8)
        
        pnts_left = np.vstack((points_left,x)).T
        pnts_right = np.vstack((points_right,x)).T
        
        #reversing the right lane y points in order to plot using polylines
        pnts_right[:,0] = pnts_right[::-1,0]
        pnts_right[:,1] = pnts_right[::-1,1]
        
        ptss = np.concatenate((pnts_left,pnts_right), axis = 0)
        
        #drwaing the lanes and filling the polygon using fillpoly
        #warped = cv2.polylines(warped, [ptss], False, (255, 0, 0),2)
        
        cv2.fillPoly(warped, [ptss], (50, 205, 50))
        
        newwarp = unwarp(warped)
        
        #thresholding the unwarped grayscale image
        newwarp_gray = cv2.cvtColor(newwarp, cv2.COLOR_BGR2GRAY)
        
        _, un_thresh = cv2.threshold(newwarp_gray, 0, 250, cv2.THRESH_BINARY_INV)
        
        #extracting only the road from the unwarped image to use as a mask 
        mask_inv = cv2.bitwise_and(crop, crop, mask = un_thresh).astype(np.uint8)
        
        #adding the resultant mask with the unwarped image
        result_half = cv2.add(mask_inv, newwarp)
        
        #result_half = cv2.addWeighted(crop, 1, newwarp, 0.3, 0)                     #for slightly ransparent lanes
        
        #stitching the upper half of the image
        result = np.concatenate((crop_1,result_half), axis = 0)
        
        radius_curve = roc(coeff_left,coeff_right,pnts_left,pnts_right)
        
        image_center = int(warped.shape[1]/2)        
        
        direction(image_center,col_l,col_r,result,radius_curve)
        
        cv2.imshow('Result',result)
        
        out.write(result)
          
        #cv2.imshow('Frame', frame1)
        #cv2.imshow('Undistorted', undistorted)
        #cv2.imshow('GBlur', blur)
        #cv2.imshow('Cropped', crop)
        #cv2.imshow('HSV White', white_hsl)
        #cv2.imshow('HSV Yellow', yellow_hsl)
        #cv2.imshow('HSV Combined', combined_hsl)
        #cv2.imshow('Warped', warped)
        #cv2.imshow('Un-Warped', un_warped)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):             
            break
    else:
        break 
    
#cv2.waitKey(0)
out.release()
cv2.destroyAllWindows() 