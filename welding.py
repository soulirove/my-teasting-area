# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:47:48 2021

@author: Souli
""" 
###Importing the Libreries
import os
import cv2
import glob
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.linear_model import LinearRegression
###########################################

"""
THE AREA BELOW WILL CONTAIN THE FUNCTIONS THAT ARE SHARED FOR BOTH STATES OF THE MAIN SUBJECT.


"""
###Scalling Function
def scale(img):
  img = img[img.shape[0]-150:,img.shape[1]-300:]
  img_not = cv2.bitwise_not(img)
  gray = cv2.cvtColor(img_not,cv2.COLOR_BGR2GRAY)
  ret,th1 = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
  histogram = np.sum(th1, axis=0)
  nullx=np.where(histogram!=0)
  scale = 1/len(nullx[0])
  return scale
###############################################################

###Plotting function
def plotting(img,order,title=''):
  #fig = plt.figure(figsize=(26,12))
  plt.subplot(order)# 1 row, 2 cols, and this is the first figure
  plt.imshow(img,'gray')
  plt.title(str(order)+title,fontsize=10)
###################################################

###rescaling function
def rescaling(img):
    img[(img!=0)]=1
    return img
######################################

###K-Means Clustering function 
def K_Means_Clustering(img,klusters):
    #img = cv2.imread('home.jpg')
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = klusters
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    gray = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
    gray_blur= cv2.medianBlur(gray, 5)
    myList = sorted(set(gray[300,:400]))
    # print(myList)
    # fig = plt.figure(figsize=(26,12))
    # for i in range(len(myList)):
    #     new_gray = gray_blur.copy()
    #     new_gray[(new_gray!=myList[i])]=0
    #     order = '1' + str(len(myList)) + str(i+1)
    #     plotting(new_gray,order)
#[129, 141, 158, 188] optimal for the work    
    return gray_blur, myList
##########################################

###Image Segmentation with Watershed Algorithm function 
def Watershed(img,mask):
    #setting up the kernal
    kernal =np.ones((9,9), np.uint8)
    #making a copy of the image 
    img_copy = img.copy()
    sure_bg = cv2.dilate(mask,kernal,iterations=10)
    dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
    #print(dist_transform[:,300])
    ret2, sure_fg = cv2.threshold(dist_transform,0.6*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    markers[unknown==1] = 0
    markers = cv2.watershed(img,markers)
    threshhold= np.max(markers)
    #img[markers == threshhold] = [0,0,255] 
    img_copy[markers == threshhold] = [0,0,255] 
    #print(unknown[:,300]) 
    #img2 = color.label2rgb(markers, bg_label=0)
    #regions = measure.regionprops(markers, intensity_image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    #print(img2[100,500])
    # #ploting 
    # fig = plt.figure(figsize=(20,40))
    # #plotting(img,111,' Watersheded')
    # plotting(img_copy,111,' Watersheded')
##seperating the mask
    mask = np.zeros_like(img)
    mask[markers == threshhold] = [0,0,255] 
    mask_gray = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
    mask_gray_binary = np.uint8(mask_gray/np.max(mask_gray))
    # #print(mask_gray_binary[300,:])
    # #ploting 
    # fig = plt.figure(figsize=(10,20))
    # plotting(mask_gray_binary,111,' Mask')

    return img_copy,mask_gray_binary,mask
##################################################################################################

###histogram
def X_axis(img):
  histogram = np.sum(img, axis=0)
  #plt.plot(histogram)
  #midpoint = np.int(histogram.shape[0]/2)
  #plt.plot(midpoint,histogram[midpoint],'go--')
  nullx=np.where(histogram!=0)
  h_array=np.asarray(nullx)
  leftx_base=h_array[0,0]
  rightx_base=h_array[0,-1]
  width = np.abs(leftx_base-rightx_base)
  # print('Left most point on the X-axis = '+str(leftx_base))
  # print('Right most point on the X-axis = '+str(rightx_base))
  # print('Width = '+ str(np.abs(leftx_base-rightx_base)))
  # plt.plot(leftx_base,histogram[leftx_base],'r+')
  # plt.plot(rightx_base,histogram[rightx_base],'r+')
  return leftx_base,rightx_base,width
##########################################

###histogram 
def Y_axis(img):
  histogram = np.sum(img, axis=1)
  #plt.plot(histogram)
  #midpoint = np.int(histogram.shape[0]/2)
  #plt.plot(midpoint,histogram[midpoint],'go--')
  nully=np.where(histogram!=0)
  v_array=np.asarray(nully)
  topy_base=v_array[0,0]
  buttomy_base=v_array[0,-1]
  height = np.abs(topy_base-buttomy_base)
  # plt.plot(topy_base,histogram[topy_base],'r+')
  # plt.plot(buttomy_base,histogram[buttomy_base],'r+')
  # print('Top most point on the Y-axis = '+str(topy_base))
  # print('Buttom most point on the Y-axis = '+str(buttomy_base))
  # print('Height= '+ str(np.abs(topy_base-buttomy_base)))
  return topy_base,buttomy_base,height
############################################################

###Finding centroid
def centroid_Finder(img):
  # convert image to grayscale image
  gray_mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  # convert the grayscale image to binary image
  ret,thresh = cv2.threshold(gray_mask,75,255,0)

  # calculate moments of binary image
  M = cv2.moments(thresh)

  # calculate x,y coordinate of center
  cX = int(M["m10"] / M["m00"])

  cY = int(M["m01"] / M["m00"])

  # put text and highlight the center
  cv2.circle(mask, (cX, cY), 5, (255, 255, 255), -1)

  cv2.putText(mask, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

  # #ploting 
  # fig = plt.figure(figsize=(10,20))
  # plotting(mask,111,' Mask')
  # print(cX,cY)
  return cX,cY
########################################################################################


"""
THE AREA BELOW CONTAINS THE FUNCTIONS FOR THE FIRST CASE STUDY 
WITH THE SUBJECT BEING LIGHTER THAN THE SURROUNDING ENVIRONMENT:
    
"""

###Morphological Transformations function 
def Morphological_Transformations(img,List):
    #setting up the kernal
    kernal =np.ones((9,9), np.uint8)
    #setting up a dictionary of new copies 
    d = {}
    for x in range(0,len(List)):
        d["img{0}".format(x)] = img.copy()
        #highlighting the main area
        d[f'img{x}'][(d[f'img{x}']!=List[x])]=0
        
    # closing small holes inside the foreground objects
    closing = cv2.morphologyEx(d[f'img{str(len(List)-2)}'], cv2.MORPH_CLOSE, kernal)
    #rescale to 0 and 1 
    rescaling(closing)
    
    #joining broken parts of the object
    dilation = cv2.dilate(d[f'img{str(len(List)-3)}'], kernal, iterations=7)  
    #rescale to 0 and 1 
    rescaling(dilation)
    
    #joining broken parts of the object
    close = cv2.erode(d[f'img{str(len(List)-1)}'], kernal, iterations=2) 
    #rescale to 0 and 1 
    rescaling(close)  
    
    #a copy of the first threshold
    minimal = d[f'img{0}']
    #rescale to 0 and 1
    rescaling(minimal)
    
##Extracting secondairy important object

    #setting up an array of zeros
    binary_formation =  np.zeros_like(minimal)
    
    #subtracting the closing from the dilation
    binary_formation[(closing == 1) & (dilation == 0)] = 1
    
    #joining broken parts of the object
    close_formation= cv2.erode(binary_formation, kernal, iterations=1) 
    #rescale to 0 and 1 
    rescaling(close_formation)
    
    #setting up an array of zeros
    binary_output = np.zeros_like(minimal)
    
    #adding  the close to  the close_formation
    binary_output[(close_formation == 1) | (close == 1)] = 1
    
##Extracting the main object
    #noise reduction
    median_binary_output = cv2.medianBlur(binary_output,51)
    # closing small holes inside the foreground objects
    closing_median_binary_output = cv2.morphologyEx(median_binary_output, cv2.MORPH_CLOSE, kernal, iterations=6)
    
    
    #closing small holes inside the foreground objects
    #closing_median_binary_output = cv2.morphologyEx(binary_output, cv2.MORPH_CLOSE, kernal, iterations=4)
    #noise reduction
    #median_binary_output = cv2.medianBlur(closing_median_binary_output,75)
    
    # #plotting the resutls 
    # fig = plt.figure(figsize=(26,12))
    # plotting(median_binary_output,121, ' median_binary_output')
    # plotting(closing_median_binary_output,122, ' closing_median_binary_output')
    
    return closing_median_binary_output, minimal
#################################################################################################

###horizontal line Y-axis finder
def horizontal_line_finder(img,extra,centroid_Y):
  #midpoint_horizontal= median_horizontal.shape[1]//2
  centroid_horizontal_right = extra[centroid_Y-40:centroid_Y+40,extra.shape[1]-100:extra.shape[1]]
  centroid_horizontal_left = extra[centroid_Y-40:centroid_Y+40,0:100]

  #setting up the kernal
  kernel =np.ones((1,500), np.uint8)

  #joining broken parts on the horizontal line
  dilation_centroid_horizontal_right= cv2.dilate(centroid_horizontal_right, kernel, iterations=2)
  dilation_centroid_horizontal_left= cv2.dilate(centroid_horizontal_left, kernel, iterations=2)

  #ploting 
  #fig = plt.figure(figsize=(20,40))
  #plotting(centroid_horizontal_left,141,' centroid_horizontal_left')
  #plotting(dilation_centroid_horizontal_left,142,' dilation_centroid_horizontal_left')
  #plotting(centroid_horizontal_right,143,' centroid_horizontal_right')
  #plotting(dilation_centroid_horizontal_right,144,' dilation_centroid_horizontal_right')

  histogram_left = np.sum(dilation_centroid_horizontal_left, axis=1)
  histogram_right = np.sum(dilation_centroid_horizontal_right, axis=1)

  #plt.plot(histogram_left)
  #plt.plot(histogram_right)

  #finding the Horizontal line coordinates  and plotting them on the histogram
  left_not_void = np.where(histogram_left!=0)
  right_not_void = np.where(histogram_right!=0)

  left_array = np.asarray(left_not_void)
  right_array = np.asarray(right_not_void)

  left_section_horizontal_line = left_array[0,0]
  right_section_horizontal_line = right_array[0,0]

  centroid_horizontal_differance_left = left_section_horizontal_line - 40
  centroid_horizontal_differance_right = right_section_horizontal_line - 40

  #plt.plot(left_section_horizontal_line,histogram[left_section_horizontal_line],'r+')
  #plt.plot(right_section_horizontal_line,histogram[right_section_horizontal_line],'r+')

  #print('Horizontal line on the left side is '+str(centroid_horizontal_differance_left)+' pixels from the centroid')
  #print('Horizontal line on the right side is '+str(centroid_horizontal_differance_right)+' pixels from the centroid')

  Horizontal_line_coordinates_right_side = centroid_Y + centroid_horizontal_differance_right
  Horizontal_line_coordinates_left_side = centroid_Y + centroid_horizontal_differance_left
  #print('Horizontal line coordinates on the left side = '+str(Horizontal_line_coordinates_left_side)+', Horizontal line coordinates on the right side = '+str(Horizontal_line_coordinates_right_side))

  #making a copy of the image 
  img_copy_for_horizontal_lines = img.copy()

  #making the B points line
  B_left = [0,Horizontal_line_coordinates_left_side]
  B_right = [img_copy_for_horizontal_lines.shape[1],Horizontal_line_coordinates_right_side]
  cv2.line(img_copy_for_horizontal_lines,tuple(B_left),tuple(B_right),(1,1,0),1)

  #drawing the 2 other strictly horizontal
  cv2.line(img_copy_for_horizontal_lines,tuple(B_left),(img_copy_for_horizontal_lines.shape[1],Horizontal_line_coordinates_left_side),(1,1,0),1)
  cv2.line(img_copy_for_horizontal_lines,tuple(B_right),(0,Horizontal_line_coordinates_right_side),(1,1,0),1)


  # #plotting the figure
  # fig = plt.figure(figsize=(26,12))
  # plotting(img_copy_for_horizontal_lines,111,' lines horizontal')

  return Horizontal_line_coordinates_left_side,Horizontal_line_coordinates_right_side, B_left, B_right
#######################################################################

###Angles Extraction
###Angles Extraction
####################

###finding gradient function
def gradient(pt1,pt2):
  return ((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))
#########################################################################

###angle calculator function
def getangle(base_point,vertical,third):
  m1= gradient(base_point,vertical)
  m2= gradient(base_point,third)
  angR = math.atan((m2-m1)/(1+(m2*m1)))
  angD = round(math.degrees(angR))
  angD_positive=np.absolute(angD)
  return angD_positive
  #print('the right angle= '+ str(angD_positive))
######################################################################

###Rotating the main Image
#Function to rotate the image and give back the horizontal line 2 extreme points
def rotate(img,left_point,right_point,angle):
  # if left_point[1] >= right_point[1]:
  #   # grab the dimensions of the image
  #   (h, w) = img.shape[:2]
  #   # rotate our image by image angle degrees around one of the horizontal line extreme point 
  #   M = cv2.getRotationMatrix2D(tuple(left_point), -angle, 1.0)
  #   rotated = cv2.warpAffine(img, M, (w, h))
  #   #give back the horizontal line 2 extreme points
  #   left,right = left_point, [w,Horizontal_line_coordinates_left_side]
  #   #drawing the strictly horizontal
  #   cv2.line(rotated,tuple(left_point),(w,Horizontal_line_coordinates_left_side),(255,255,0),1)
  #   #plotting the figure
  #   #fig = plt.figure(figsize=(26,12))
  #   #plotting(rotated,111," Rotated by 1 Degrees")
  # else:    
  #   # grab the dimensions of the image
  #   (h, w) = img.shape[:2]
  #   # rotate our image by image angle degrees around one of the horizontal line extreme point 
  #   M = cv2.getRotationMatrix2D(tuple(right_point), angle, 1.0)
  #   rotated = cv2.warpAffine(img, M, (w, h))
  #   #give back the horizontal line 2 extreme points
  #   left,right = [0,Horizontal_line_coordinates_right_side],right_point
  #   #drawing the strictly horizontal
  #   cv2.line(rotated,tuple(right_point),(0,Horizontal_line_coordinates_right_side),(255,255,0),1)
  #   #plotting the figure
  #   #fig = plt.figure(figsize=(26,12))
  #   #plotting(rotated,111," Rotated by 1 Degrees")
  # return rotated,left,right

  if left_point[1] > right_point[1]:
    # grab the dimensions of the image
    (h, w) = img.shape[:2]
    # rotate our image by image angle degrees around one of the horizontal line extreme point 
    M = cv2.getRotationMatrix2D(tuple(left_point), -angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #give back the horizontal line 2 extreme points
    left,right = left_point, [w,Horizontal_line_coordinates_left_side]
    #drawing the strictly horizontal
    cv2.line(rotated,tuple(left_point),(w,Horizontal_line_coordinates_left_side),(255,255,0),1)
    #plotting the figure
    #fig = plt.figure(figsize=(26,12))
    #plotting(rotated,111," Rotated by 1 Degrees")
  elif left_point[1] < right_point[1]:    
    # grab the dimensions of the image
    (h, w) = img.shape[:2]
    # rotate our image by image angle degrees around one of the horizontal line extreme point 
    M = cv2.getRotationMatrix2D(tuple(right_point), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    #give back the horizontal line 2 extreme points
    left,right = [0,Horizontal_line_coordinates_right_side],right_point
    #drawing the strictly horizontal
    cv2.line(rotated,tuple(right_point),(0,Horizontal_line_coordinates_right_side),(255,255,0),1)
    #plotting the figure
    #fig = plt.figure(figsize=(26,12))
    #plotting(rotated,111," Rotated by 1 Degrees")
  else:
      print('No rotation needed')
      rotated = img
      left,right = left_point,right_point
      
  return rotated,left,right
#######################################################################

###Function that gives back the top part edges
def canny_crop(rotated,left,right):
  #cropping the image 
  cropped = rotated[:left[1],left[0]:right[0]]
  blur = cv2.bilateralFilter(cropped,47,75,75)
  gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray,100,200)
  # #plotting the figure
  # fig = plt.figure(figsize=(26,12))
  # plotting(edges,111," edges ")
  return edges
###########################################################

###Find X point function
def X_point(img,Y):
  img =  img.copy()
  img = img[Y,:]
  index_occurrences = np.where(img == 255)[0]
  if index_occurrences.shape[0] > 1:
    leftx_base = index_occurrences[0]
    rightx_base = index_occurrences[-1]
    #print('Left most point on the X-axis with the following ' +str(Y)+' Y-axis coordinates = '+str(leftx_base))
    #print('Right most point on the X-axis with the following ' +str(Y)+' Y-axis coordinates = '+str(rightx_base))
    #print('Width = '+ str(np.abs(leftx_base-rightx_base)))
  elif index_occurrences.shape[0] == 1:
    leftx_base =  rightx_base = index_occurrences[0]
    #print('There is only one point X with the following ' +str(Y)+' Y-axis coordinates = ' +str(leftx_base))
  else:
    #print('There is no point X ')
    leftx_base = rightx_base = 0
  return leftx_base, rightx_base
###############################################################

###Find Y point function
def Y_point(img,X):
  img =  img.copy()
  img = img[:,X]
  index_occurrences = np.where(img == 255)[0]
  if index_occurrences.shape[0] > 1:
    Y = index_occurrences[-1]
    #print('The point with the following X-axis coordinates ' +str(X)+' correspond to the following coordinates on the Y-axis = '+str(Y))
  elif index_occurrences.shape[0] == 1:
    Y = index_occurrences[0]
    #print('The point with the following X-axis coordinates ' +str(X)+' correspond to the following coordinates on the Y-axis = '+str(Y))
  else:
    #print('There is no coordinates on the Y-axis for The point with the following X-axis coordinates ' +str(X))
    Y = 0
  return Y
#######################################################################

###readjusting the image to get rid of all noise
def readjusting(img,precision):
  img = img.copy()
  final = []
  for i in range(precision):
    i_multiplier = i*10
    begging, end = img.shape[0]-1-i_multiplier,img.shape[0]-12-i_multiplier
    for i_left in range (begging,end,-1):
      starting_points = X_point(img,i_left)
      starting_points_ahead = X_point(img,i_left-1)
      if starting_points[0]-starting_points_ahead[0]>0:
        break
    if (begging - i_left) < 10:
      final.append(i_left-1)

    for i_right in range (begging,end,-1):
      starting_points = X_point(img,i_right)
      starting_points_ahead = X_point(img,i_right-1)
      if starting_points[-1]-starting_points_ahead[-1]<0:
        break
    if (begging - i_right) < 10:
      final.append(i_right-1)
  img = img[:min(final),:]
  return img
####################################################################################

###extreme points coordinates
"""
The 3 angles in this case are: A,B,C.
A: being the intersection point of the horizantal line and the object.

B: is a point on the horizontal line prefebly y=0 or y=shape[1].

C: is the leftest or rightest point on the object in case that wasn't A already (first case senario)..... otherwise it is a point on the objects that is far form A with 10 pixcels (second case senario).
"""
###Finding point A
def A_Locator(img):
  leftx_A,rightx_A = X_point(img,img.shape[0]-1)
  A_point_left =[leftx_A,img.shape[0]]
  A_point_right =[rightx_A,img.shape[0]]
  #print('Point A left = '+str(A_point_left))
  #print('Point A right = '+str(A_point_right))
  return A_point_left, A_point_right
############################################################

###Finding point C with Linear Regression
def C_Locator(img,plt,precision=10):
  PLT = X_point(img,plt)
  #print(PLT[0],PLT[1])
  X_L = []
  X_R = []
  Y_A = []
  for i in range(precision):
    y = img.shape[0]-1-i 
    Y = X_point(img,y)
    Y_A.append(y)
    X_L.append(Y[0])
    X_R.append(Y[1])
  #print(' Y_A = ',Y_A,'\n','X_L = ', X_L ,'\n','X_R = ', X_R)
  X_L = pd.DataFrame(data= X_L)
  X_R = pd.DataFrame(data= X_R)

  model_X_L = LinearRegression()
  model_X_L.fit(X_L,Y_A)

  model_X_R = LinearRegression()
  model_X_R.fit(X_R,Y_A)

  C_point_left_Y = model_X_L.predict([[PLT[0]]])
  C_point_right_Y = model_X_R.predict([[PLT[1]]])
  C_point_left_Y = C_point_left_Y.tolist()
  C_point_right_Y = C_point_right_Y.tolist()
  C_point_left_Y = round(C_point_left_Y[0])
  C_point_right_Y = round(C_point_right_Y[0])
  C_point_left, C_point_right = [PLT[0],C_point_left_Y] , [PLT[1],C_point_right_Y]
  return C_point_left, C_point_right
#####################################################################################

###The angles of the shape
def angles(A_point_left, A_point_right, C_point_left, C_point_right):
  #the three main points 
  B_point_left, B_point_right = [0,final.shape[0]],[final.shape[1],final.shape[0]]
  left_side_angle = getangle(A_point_left,B_point_left,C_point_left)
  right_side_angle = getangle(A_point_right,B_point_right,C_point_right)
  #print(left_side_angle,right_side_angle)
  #print('The left side angle of the shape and the horizontal line is equal to <'+str(left_side_angle)+'°> and, the right side angle of the shape and the horizontal line is equal to <'+str(right_side_angle)+'°>')
  return left_side_angle, right_side_angle, B_point_left, B_point_right
####################################################################################

###drawing the angles on the image
def drawing(rotated,A_point_left, A_point_right, C_point_left, C_point_right, B_point_left, B_point_right,left_side_angle, right_side_angle):
  #making a copy of the image 
  img_copy_for_angle = rotated.copy()
  
  cv2.putText(img_copy_for_angle,str(left_side_angle)+'°',(A_point_left[0]-50,A_point_left[1]-25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
  cv2.line(img_copy_for_angle,tuple(A_point_left),tuple(C_point_left),(255,0,255),2)

  cv2.putText(img_copy_for_angle,str(right_side_angle)+'°',(A_point_right[0]+25,A_point_right[1]-25),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
  cv2.line(img_copy_for_angle,tuple(A_point_right),tuple(C_point_right),(255,0,255),2)

  cv2.line(img_copy_for_angle,tuple(B_point_left),tuple(B_point_right),(255,0,255),3)
  return img_copy_for_angle
######################################################################################

"""
THE AREA BELOW CONTAINS THE FUNCTIONS FOR THE SECOND CASE STUDY 
WITH THE SUBJECT BEING DARKER THAN THE SURROUNDING ENVIRONMENT:
    
"""



"""
THE AREA BELOW IS THE AREA OF THE MAIN CODE:


"""
# path = "F:\inegi\SecondTake"
# #path = "F:\inegi\SecondTake"
# images = []
# Names = []
# if (path.find("\\") != 0):
#     newPath = path.replace(os.sep, '/')
#     folders = os.listdir(newPath)
#     for folder in folders:
#         derectory = newPath + '/' + folder + '/*.tif'
#         test_images_path = glob.glob(derectory)
#         for test_image_path in test_images_path:
#             image = cv2.imread(test_image_path)
#             img,ThresholdList = K_Means_Clustering(image,5) 
#             print('Loading new image from this depository: ' + test_image_path)
#             print('Clusters list: ')
#             print(ThresholdList)
#             Transformations,extra = Morphological_Transformations(img,ThresholdList)
#             Watersheded,mask = Watershed(image,Transformations)
#             fig = plt.figure(figsize=(20,40))
#             plotting(Watersheded,121,'Watersheded '+ test_image_path)
#             plotting(mask,122,'Mask '+ test_image_path )
#     else:
#         newPath = "F:/inegi/" + path
#         image = cv2.imread(newPath)
#         img,ThresholdList = K_Means_Clustering(image,5) 
#         print(ThresholdList)
#         Transformations,extra = Morphological_Transformations(img,ThresholdList)
#         Watersheded,mask = Watershed(image,Transformations)
#         fig = plt.figure(figsize=(20,40))
#         plotting(Watersheded,121,'Watersheded')
#         plotting(mask,122,'Mask')
                  




# newPath = path.replace(os.sep, '/')
# print(newPath)
# print(os.listdir(newPath))
# derectory = os.listdir(newPath)
# derectory = newPath+'/'+derectory[0]+'/*.tif'
# test_images_path = glob.glob(derectory)
# for test_image_path in test_images_path:
#     images = []
#     #print('this is the link',test_image_path)
#     #fig = plt.figure(figsize=(20,40))
#     img = cv2.imread(test_image_path)
#     images.append(img)
#     # tuple(images)
# print(type(images))



specific_image = "4_obj1x_23_bri=47_gain=1.4_sat=84_gamm=1.4_hue=312_sat=44 in_focus_4.tif"
# specific_image = "4_obj1x_23_bri=47_gain=1.4_sat=84_gamm=1.4_hue=312_sat=44 in_focus_1.tif"
new_path = 'F:/inegi/'+specific_image
image = cv2.imread(new_path)
dark = False
area= False
process = False
z=0
d=0
print('Process Starting')

"""
THIS IS FIRST PART OF THE WHILE WILL BE FOR LIGHT METALS IN CASE THEY ARE OF 
DARK NATURE THE AREA WILL ALWAYS BE SMALL TO BE ACCEPTED OR TO BIG TO BE ACCEPTED. 
SO WE MOVE FOR THE SECOND WHILE LOOP FOR DARK METALS:
"""
while (dark == False) and (process == False):
    print('Clustering Please Wait')
    while (area == False) and (z<10):
        condition = False
    #print('Clustering Please Wait')
        while (condition == False) and (z<10):
            print('Please Wait')
            z+=1
            img,ThresholdList = K_Means_Clustering(image,6)
            print(ThresholdList)
            if len(ThresholdList) >= 4:
                condition = True
        
        Transformations,extra = Morphological_Transformations(img,ThresholdList)
        Watersheded,mask_gray_binary,mask = Watershed(image,Transformations)
        counter = 0
        for i in range(mask.shape[0]):
          for j in range(mask.shape[1]):
            if mask_gray_binary[i,j]==1:
              counter +=1
        print(counter)
        if (counter > 70000) and (counter < 170000):
            area = True
            print('Requirements Satisfied')
        print('this is the number of iteration z = ',z)
    #print('this is the area status',area)
    if (area == False):
        print('_________________________________________________________________')
        print("The main subject isn't lighter than the surroundings environment.\nThe algorithm will change to detect the main object.\nThis time will try to see if the object is darker than the surrounding environment")
        dark = True
        condition = False
    else:
        leftx_base,rightx_base,width = X_axis(mask)
        topy_base,buttomy_base,height = Y_axis(mask)
        centroid_X, centroid_Y = centroid_Finder(mask)
        centroid = [centroid_X, centroid_Y]
        Horizontal_line_coordinates_left_side,Horizontal_line_coordinates_right_side, B_left, B_right = horizontal_line_finder(image,extra,centroid_Y)
        # Getting the image angle in accordance to the horizontal line
        a,b,c = B_right, [0,Horizontal_line_coordinates_right_side], [0,Horizontal_line_coordinates_left_side]#the three main points 
        image_angle = getangle(a,b,c)
        rotated,left,right = rotate(image,B_left,B_right,image_angle)
        edges = canny_crop(rotated,left,right)
        # X_point(edges,227)
        # Y_point(edges,260)
        final = readjusting(edges,6)
        A_point_left, A_point_right = A_Locator(final)
        C_point_left, C_point_right = C_Locator(final,final.shape[0]-75,60)
        left_side_angle, right_side_angle, B_point_left, B_point_right = angles(A_point_left, A_point_right, C_point_left, C_point_right)
        draw = drawing(rotated,A_point_left, A_point_right, C_point_left, C_point_right, B_point_left, B_point_right,left_side_angle, right_side_angle)
        Reinforcement = edges.shape[0] - topy_base
        Penetration = height - Reinforcement
        Dilution = (Penetration/height)*100
        scale = scale(image)
        #plotting the figure
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_draw = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        gray_Watersheded = cv2.cvtColor(Watersheded, cv2.COLOR_BGR2GRAY)
        
        fig = plt.figure(figsize=(20,40))
        plotting(gray_image,121,'Original Image')
        plotting(mask_gray_binary,122,'Mask')
        
        fig = plt.figure(figsize=(20,40))
        plotting(gray_Watersheded,121,'Watersheded')
        plotting(gray_draw,122,'Angles drawn')
        
        ##the needed data printing 
        print('Extracting needed Data')
        print('--->')
        print('--->')
        print('--->')
        print('--->')
        print('All the data below is in calculated in pixel.\n Some metrics will be shown in metric system at the bottom')
        print('The image has the following dimensions ['+str(image.shape[0])+','+str(image.shape[1])+'].')
        print('The area of interest has the following dimensions:')
        print('The Area = '+str(counter)+'.')
        print('The Width = '+str(width)+'.')
        print('The Height= '+str(height)+'.')
        print('The Reinforcement = '+str(Reinforcement)+'.')
        print('The Penetration = '+str(Penetration)+'.')
        print('The percentage of Dilution = '+str(Dilution)+'.')
        print('The center of the welding have the following coordinates: '+str(centroid)+'.')
        print('')
        print('The image have a small tilt therefore the horizontal line have different height on the extreme left and the extrem right.')
        print('The horizontal line Y-axis coordinates on the left side = '+str(Horizontal_line_coordinates_left_side)+', Horizontal line Y-axis coordinates on the right side = '+str(Horizontal_line_coordinates_right_side)+'.')
        print('The image is tilted by '+str(image_angle)+'° from the real horizontal.')
        print('The image to be cropped above the horizontal line to give a better reading of the angles.')
        print('The cropped image have the following dimensions ['+str(edges.shape[0])+','+str(edges.shape[1])+'].')
        print('')
        print('The angles 3 points in this case will be represented by A,B,C.')
        print('A: being the intersection point of the horizantal line and the object.')
        print('B: is a point on the horizontal line prefebly y=0 or y=shape[1].')
        print("C: is a point on the object that will be most of the time 10 pixels above point A but it can change if there is a need for it.")
        print('')
        print('Point A on the left side has the following coordinates '+str(A_point_left)+', Point A on the right side has the following coordinates '+str(A_point_right)+'.')
        print('Point B on the left side has the following coordinates '+str(B_point_left)+', Point B on the right side has the following coordinates '+str(B_point_right)+'.')
        print('Point C on the left side has the following coordinates '+str(C_point_left)+', Point C on the right side has the following coordinates '+str(C_point_right)+'.')
        print('The left side angle of the shape and the horizontal line is equal to '+str(left_side_angle)+'° and, the right side angle of the shape and the horizontal line is equal to '+str(right_side_angle)+'°.')
        print('')
        print('The Area = '+str(round(counter*scale,3))+'mm^2.')
        print('The Width = '+str(round(width*scale,3))+'mm.')
        print('The Height= '+str(round(height*scale,3))+'mm.')
        print('The Reinforcement = '+str(round(Reinforcement*scale,3))+'mm.')
        print('The Penetration = '+str(round(Penetration*scale,3))+'mm.')
        print('The percentage of Dilution = '+str(round(Dilution,3))+'%.')
        print('The Left angle = '+str(left_side_angle)+'°.')
        print('The Right angle = '+str(right_side_angle)+'°.')
        process = True
        
        
        
# print('this is dark ',dark)
# print('this is area ',area)
# print('this is condition ',condition)
# print('this is status ',process)
        
"""
HERE WILL BE THE SECOND WHILE LOOP FOR DARK METALS. 
THIS MEANS THE FIRST PART WAS NOT ABLE TO GET THE SHAPE. 
THE AREA WAS TOO BIG OR TOO SMALL. SAME RULES AND BUT REVERSED FUNCTION WILL BE USED HERE. 
THE PRICIPLE IS THE SAME. 
SOME FUNCTION WILL BE UNIQUE ONLY TO THIS PART SAME AS THE FIRST PART HAD SOME UNIQUE FUNCTIONS TO IT ALONE.
"""
        
# while (dark == True) and (process == False):
#     d+=1
#     print('it s dark time now!!!')
#     dark = False
#     print('Reclustering Please Wait')
#     while (area == False) and (z<10):
#         condition = False
#     #print('Clustering Please Wait')
#         while (condition == False) and (z<10):
#             print('Please Wait')
#             z+=1
#             img,ThresholdList = K_Means_Clustering(image,6)
#             print(ThresholdList)
#             if len(ThresholdList) >= 4:
#                 condition = True