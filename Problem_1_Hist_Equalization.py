import numpy as np
import cv2

cap = cv2.VideoCapture("Night Drive - 2689.mp4")

out = cv2.VideoWriter('Problem_1_Final.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))

#to calculate the histogram of the image passed
def histogram(g):
    lis = list()
    
    for i in range(256):
        c = np.where(g == i)
        lis.append( [ i , len(c[0]) ] )
        
    return lis

#calculating the cumulative distributive function
def cumulative(lis,height,width):
    cdf = list()
    z = 0
    for i in range(len(lis)):
        z = z + (lis[i][1]/(height * width))
        cdf.append(round(z*255))
     
    return cdf

#equalization of a grayscale image
def GrayEqualization(gray):

    gray_copy = gray.copy()
    
    height,width = gray.shape
        
    lis = histogram(gray)
        
    hnew = cumulative(lis,height,width)
    
    hnew = np.asarray(hnew)
    
    gray_copy[:,:] = hnew[gray[:,:]]  
    
    gray_equalized = gray_copy
    
    cv2.imshow('Gray Equalized',gray_equalized)
    
    return lis,gray_equalized

#equalzation of an image in the HSV color space
def HSVEqualization(b11):
    
    b11_copy = b11.copy()
    
    hsv = cv2.cvtColor(b11, cv2.COLOR_BGR2HSV)
    
    #taking the V channel of the hsv image
    vlist = hsv[:,:,2]
    
    height,width = vlist.shape
        
    lis = histogram(vlist)
        
    hnew = cumulative(lis,height,width)
    
    hnew = np.asarray(hnew)
    
    #putting the new v channel intensities 
    hsv[:,:,2] = hnew[hsv[:,:,2]] 
    
    b11_copy = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('HSV Equalized',b11_copy)
    
    return lis,b11_copy

#equalzation of an image in the RGB color space
def RGBEqualization(b22):

    b22_copy = b22.copy()
    
    rlist = b22[:,:,0] 
    glist = b22[:,:,1]
    blist = b22[:,:,2]
    
    height,width = rlist.shape
        
    lis1 = histogram(rlist)
    lis2 = histogram(glist)
    lis3 = histogram(blist)
        
    hnew = cumulative(lis1,height,width)
    hnew1 = cumulative(lis2,height,width)
    hnew2 = cumulative(lis3,height,width)
    
    hnew = np.asarray(hnew)
    hnew1 = np.asarray(hnew1)
    hnew2 = np.asarray(hnew2)
            
    b22_copy[:,:,0] = hnew[b22[:,:,0]] 
    b22_copy[:,:,1] = hnew1[b22[:,:,1]]
    b22_copy[:,:,2] = hnew2[b22[:,:,2]]
           
    cv2.imshow('RGB Equalized',b22_copy)
    
    return b22_copy

while True:
    
    ret, frame = cap.read()
    if ret == True:
    
        #frame1 = frame.copy()
        
        b = cv2.resize(frame,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA)
        b1 = b.copy()
        b2 = b.copy()
        
        #lis,gray_equal = GrayEqualization(gray)
        #out.write(gray_equal)
        
        lis1,hsv_equal = HSVEqualization(b1)
        out.write(hsv_equal)
        
        #rgb_equal = RGBEqualization(b2)
        #out.write(rgb_equal)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):             
            break
    else:
        break

#cv2.waitKey(0)
out.release()
cv2.destroyAllWindows() 