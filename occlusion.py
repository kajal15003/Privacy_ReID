import numpy as np
import glob
import os
import random
import cv2
cameras = ['query_without_blackout']
regions = ['full_body']
start= 0
end = 153
for cam in cameras:
    path = cam#'/data/home/kajal2/triplet-reid/image_root/query1'
    image_list = sorted(glob.glob(path+'/*.jpg'))
    for i in range(len(image_list)):

        #person_number = '/person_{:04d}'.format(i)
        person_name = image_list[i]
        #print person_name
        y= np.ones((128,256))

               
        im = cv2.resize(cv2.imread(person_name),(128,256))
        #print im.size
        #cv2.imwrite('test.jpg',im)

        numx= random.choice(range(0, 70, 1))
        numy= random.choice(range(0, 198, 1))
        #print numx, numy
        y[numx:numx+58 , numy:numy + 58]= 0
        y_3d= np.array([y, y, y])
        im= np.multiply(y_3d.transpose(2,1,0), im)

        #os.remove(person_name)
        final_dir = os.path.join('query')
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
        cv2.imwrite(os.path.join(final_dir,os.path.basename(person_name)),im)


        #print "Done",i
        #print images
