import cv2
import numpy as np
import os

out = cv2.VideoWriter('Data_Set1_vid.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1392,512))

for i in range(303):
    
    if i < 10:
        i = '0' + '0' + '{}'.format(i)
    elif 10 <= i < 100:
        i = '0' + '{}'.format(i)
        
    img = cv2.imread('0000000{}.png'.format(i))
    cv2.imshow('Image',img)
    out.write(img)
    cv2.waitKey(1)
    
out.release()
cv2.destroyAllWindows()
