import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("Data_Set1_vid.avi")
#image shape --> (1392,512)

out = cv2.VideoWriter('Data_Set1_Output_Final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1282,372))

#undistorting the frame using the calibration and distance matrix 
def undistort(frame_dist):
    
    K = np.array([[9.037596e+02,0.000000e+00,6.957519e+02],[0.000000e+00,9.019653e+02,2.242509e+02],[0.000000e+00,0.000000e+00,1.000000e+00]])
    
    D = np.array([-3.639558e-01,1.788651e-01,6.029694e-04,-3.922424e-04,-5.382460e-02])
    
    height, width = frame_dist.shape[:2]
    
    K_new, roi = cv2.getOptimalNewCameraMatrix(K , D , (width , height) , 1 , (width , height) )
                                                      
    correct = cv2.undistort(frame_dist, K, D, None, K_new)

    l, m, n, o = roi
    
    correct = correct[m:m + o, l:l + n]

    return correct

#determining the homography and creating the warped image    
def homography(warped):

    source = np.array([[515, 50], [260, 180],[850, 180], [685, 50]])#main
    
    dest = np.array([[0, 0], [0, 500], [200, 500],[200, 0]])
    
    H , status = cv2.findHomography(source, dest)

    warped = cv2.warpPerspective(warped, H, (200,500))
    
    return warped

#determining the inverse homography and creating the unwarped image
def unwarp(unwarped):
    
    source = np.array([[515, 50], [260, 180],[850, 180], [685, 50]])#main
    
    dest = np.array([[0, 0], [0, 500], [200, 500],[200, 0]])
    
    H , status = cv2.findHomography(source, dest)

    unwarped = cv2.warpPerspective(unwarped, np.linalg.inv(H), (crop.shape[1],crop.shape[0]))
    
    return unwarped

#calculating the radius of curvature
def roc(coef_left,coef_right,points_left,points_right):
    
    radius = ((1+((2*coef_right[0]*points_right[50][1])+coef_right[1])**2)**(3/2))/abs(2*coef_right[0])
    return radius    

#calculating the histogram to determine two intensity peaks corresponding to the lanes
def histogram(image):
    
    whites = list()
    index = list()
    for i in range(image.shape[1]):
        z = np.where(image[:,i] > 0)
        whites.append(len(z[0]))
        index.append(i)
    one = whites[:100]
    two = whites[100:]
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
    
    if ( dev < -5):
        res_img = cv2.putText(res_img,'Left',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("left")
    elif ( dev < 12):
        res_img = cv2.putText(res_img,'Straight',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("Straight")
    else:
        res_img = cv2.putText(res_img,'Right',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        res_img = cv2.putText(res_img,"ROC {}".format(radius_dir),(50,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)
        #print("Right")
    
    
while True:
    
    ret, frame = cap.read()
    if ret == True:
    
        frame1 = frame.copy()
        
        undistorted = undistort(frame1)
        
        #blur = cv2.GaussianBlur(undistorted,(7,7),0)
        med_blur = cv2.medianBlur(undistorted,ksize=3)
        
        #cropping the denoised frame to get only the road
        crop = med_blur[160:, :]
        crop_1 = med_blur[:160, :]
        
        #crop_full = np.concatenate((crop_1,crop), axis = 0)
        
        #cv2.imshow('Crop Full',crop_full)
        
        warped = homography(crop)
        
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        
        lower_white = np.array([0, 0, 200])
        higher_white = np.array([255, 255, 255])
        
        mask = cv2.inRange(hsv, lower_white, higher_white)
        
        #getting the maximum intensity column indices on the left and right side
        col_l,col_r = histogram(mask)
        
        #finding the white pixels in the max column as well as its neighbouring columns
        leftpts = np.where(mask[:,col_l-7:col_l+8] > 0)
        rightpts = np.where(mask[:,col_r-7:col_r+8] > 0)     
        
        #calculating the coefficients of the polynomials of left and right lanes
        coeff_left = np.polyfit(leftpts[0], leftpts[1]+(col_l-7), 2)
        coeff_right = np.polyfit(rightpts[0], rightpts[1]+(col_r-7), 2)
        
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
        warped = cv2.polylines(warped, [ptss], False, (255, 0, 0),2)
        
        cv2.fillPoly(warped, [ptss], (50, 205, 50))
        
        newwarp = unwarp(warped)
        
        #thresholding the unwarped grayscale image
        newwarp_gray = cv2.cvtColor(newwarp, cv2.COLOR_BGR2GRAY)
        
        _, un_thresh = cv2.threshold(newwarp_gray, 0, 250, cv2.THRESH_BINARY_INV)
        
        #extracting only the road from the unwarped image to use as a mask 
        mask_inv = cv2.bitwise_and(crop, crop, mask = un_thresh).astype(np.uint8)
        
        #adding the resultant mask with the unwarped image
        result_half = cv2.add(mask_inv, newwarp)
        
        #result_half = cv2.addWeighted(crop, 1, newwarp, 0.3, 0)                       #for slightly ransparent lanes
        
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
        #cv2.imshow('HSV', mask)
        #cv2.imshow('Warped', warped)
        #cv2.imshow('Un-Warped', un_warped)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):             
            break
    else:
        break 
    
    
#cv2.waitKey(0)
out.release()
cv2.destroyAllWindows() 
