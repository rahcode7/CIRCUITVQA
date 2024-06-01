"""
Count based

How many resisters are on the left of the ammeter ?
How many gates are connected to the and gates directly ?

"""

import cv2 
import pandas as pd 
import os 
import ast 
import cv2
from utils.preprocessing import xml_processor,randomize_qs
import utils.templates as templates
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys 
import math
from shapely.geometry import Polygon, LineString


np.set_printoptions(sys.maxsize)



#IMAGE_PATH = "datasets/train/images/d1_254_png.rf.0b2e9377b97914d428d85e23cbf099b1.jpg"
#XML_PATH = "datasets/train/xml/d1_254_png.rf.0b2e9377b97914d428d85e23cbf099b1.xml"

IMAGE_PATH = "datasets/train/images/d1_autockt_-66_png.rf.a30811b86284a145cd9ed227d824e5c1.jpg"
XML_PATH = "datasets/train/xml/d1_autockt_-66_png.rf.a30811b86284a145cd9ed227d824e5c1.xml"

#IMAGE_PATH = "datasets/train/images/d5_IMG_20220810_193854_jpg.rf.a2d16599be5d3788ddf63b7c9ceabb57.jpg"
#XML_PATH = "datasets/train/xml/d5_IMG_20220810_193854_jpg.rf.a2d16599be5d3788ddf63b7c9ceabb57.xml"


DATA_PATH = "datasets/"
OUTPUT_PATH = "datasets/questions/position"


if __name__ == "__main__":

    # Add labels to gates 
    image = cv2.imread(IMAGE_PATH)
    symbol_df = xml_processor(XML_PATH)
    print(symbol_df)

    # Line detection 
    image = mpimg.imread(IMAGE_PATH)

    # Convert the image to gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(gray, threshold1 =100, threshold2=200)#,apertureSize=3)

     ###Read gray image
    result = image.copy()
    thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    

    # Detect dots
    # resultd = image.copy()
    # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # min_area =  10000
    # black_dots = []
    # for c in cnts:
    #     area = cv2.contourArea(c)
    #     if area > min_area:
    #         cv2.drawContours(resultd, [c], -1, (36, 255, 12), 2)
    #         black_dots.append(c)
    # cv2.imwrite('image1_jnct.jpg', resultd)


    # create the detector with default parameters.
    # im = gray
    # detector = cv2.SimpleBlobDetector_create()
 
    # # Detect dots.
    # keypoints = detector.detect(im)

    # print("Black Dots Count is:",len(keypoints))

    # # Draw detected blobs as red circles.
    # im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,250), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite('image1_jnct.jpg', im_with_keypoints)

            # Hz
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
    # detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    # cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print(cnts)
    # for c in cnts:
    #     cv2.drawContours(result, [c], -1, (36,255,12), 2)   
    # cv2.imwrite('image1_cnt.jpg', result)

    # Detect vertical lines
    # result2 = image.copy()
    # vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,10))
    # detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    # cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # print("Initial contours",cnts)
    # for c in cnts:
    #     cv2.drawContours(result2, [c], -1, (36,255,12), 2)
    # cv2.imwrite('image1_cnt_vrt.jpg', result2)




#     lines = cv2.HoughLinesP(edges,
#     rho=6,
#     #theta=np.pi / 60,
#     theta = math.pi/2, # horiz
# #    theta= 90,
#     threshold=100,
#     lines=np.array([]),
#     minLineLength=100,
#     maxLineGap=25)   
    
#     #print(np.array(lines))
#     for line in lines:
#         print(line[0])    
#         x1, y1, x2, y2 = line[0]
#         cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
#     # Show result
    
#     #cv2.imshow("Result Image", image)
#     cv2.imwrite("image_1.jpg", image)
    # THRESHOLD_GRP = 100
    # def find_if_close(cnt1,cnt2):
    #     row1,row2 = cnt1.shape[0],cnt2.shape[0]
    #     for i in range(row1):
    #         for j in range(row2):
    #             dist = np.linalg.norm(cnt1[i]-cnt2[j])
    #             if abs(dist) < THRESHOLD_GRP:
    #                 return True 
    #             elif i==row1-1 and j==row2-1:
    #                 return False
    # contours = cnts
    # LENGTH = len(contours)
    # status = np.zeros((LENGTH,1))

    # for i,cnt1 in enumerate(contours):
    #     x = i    
    #     if i != LENGTH-1:
    #         for j,cnt2 in enumerate(contours[i+1:]):
    #             x = x+1
    #             dist = find_if_close(cnt1,cnt2)
    #             if dist == True:
    #                 val = min(status[i],status[x])
    #                 status[x] = status[i] = val
    #             else:
    #                 if status[x]==status[i]:
    #                     status[x] = i+1

    # unified = []
    # maximum = int(status.max())+1
    # for i in range(maximum):
    #     pos = np.where(status==i)[0]
    #     if pos.size != 0:
    #         cont = np.vstack([contours[i] for i in pos])
    #         hull = cv2.convexHull(cont)
    #         unified.append(hull)
    # print("Grpd cntrs",unified)

    # cv2.drawContours(image,unified,-1,(0,255,0),2)
    # cv2.drawContours(thresh,unified,-1,255,-1)
    # cv2.imwrite('image1_vrt_grp.jpg', result2)

    # Line intersections
    

    # poly = Polygon([(5,5), (10,10), (10,0)])
    # a = LineString([(35,67), (173, 242)])
    # print(a.intersects(poly))