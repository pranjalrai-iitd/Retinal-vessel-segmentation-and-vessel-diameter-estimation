"""
===========================
@Author  : Pranjal Rai
@Version: 1.0    12/07/2020
Selection of vessel segment
===========================
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import segment


# Some global variables
ims=0
para=True
para1=True
contour=0
ps=0
imggg=0
C=0
parts = []
index = -1
ims1 = 0
ps1=0


# Find the closest contour to the selected point
def cont(ps, j, point, C):

    nonzero = cv2.findNonZero(ps)
    distances = np.sqrt((nonzero[:,:,0] - point[1]) ** 2 + (nonzero[:,:,1] - point[0]) ** 2)
    nearest_index = np.argmin(distances)
    x, y = nonzero[nearest_index][0]
    
    cont,_ = cv2.findContours(ps, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in cont:
        if (cv2.pointPolygonTest(c,(x, y),True)>=0):
            cv2.drawContours(j, c, -1, (255, 255, 255), 4)
            cv2.drawContours(C, c, -1, (255, 255, 255), 1)
            return c

# Draws and returns the selected contour
def draw_c(event,x,y,flags,param):
    global ims, para, contour, ps, imggg, C
    if para==True:
        ims = imggg.copy()
    if event == cv2.EVENT_MOUSEMOVE and para == True:
        cv2.rectangle(ims,(x-8,y-8),(x+8, y+8),(255,255,255),2)
        cv2.line(ims,(0, y), (x,y),(255,255,255), 2)
        cv2.line(ims,(x,y),(imggg.shape[1]-1, y),(255,255,255),2)
        cv2.line(ims,(x, 0), (x,y),(255,255,255),2)
        cv2.line(ims,(x,y),(x, imggg.shape[0]-1),(255,255,255),2)
        
    if event == cv2.EVENT_LBUTTONDOWN and para == True:
        para = False
        contour = cont(ps, ims, (y, x), C)



# Selects the minimum distance between a part of contour and the selected point
def distance_min(ar, point):
    distance = 100000000
    for p in ar:
        d = math.sqrt((p[0][0]-point[0])**2+(p[0][1]-point[1])**2)
        if (d<distance):
            distance = d
    return distance

# selects the contour part closest to the selected point
def select_part (parts, point):
    idx = -1
    distance = 100000000
    for i,part in enumerate(parts):
        d = distance_min(part, point)
        if(d<distance):
            distance = d
            idx = i
    return idx

# draws and returns the selected contour part
def draw_part(event,x,y,flags,param):
    global ps1, para1, parts, ims1, index
    if para1==True:
        ims1 = ps1.copy()

    if event == cv2.EVENT_MOUSEMOVE and para1==True:
        cv2.rectangle(ims1,(x-8,y-8),(x+8, y+8),(255,255,255),2)
        cv2.line(ims1,(0, y), (x,y),(255,255,255), 2)
        cv2.line(ims1,(x,y),(ps1.shape[1]-1, y),(255,255,255),2)
        cv2.line(ims1,(x, 0), (x,y),(255,255,255),2)
        cv2.line(ims1,(x,y),(x, ps1.shape[0]-1),(255,255,255),2)
        
    if event == cv2.EVENT_LBUTTONDOWN and para1==True:
        para1 = False
        point = (x, y)
        index = select_part(parts, point)
        for i, part in enumerate(parts):
            if(i!=index):
                 cv2.drawContours(ims1, part, -1, (0, 0, 0), 5)
        cv2.drawContours(ims1, parts[index], -1, (255, 255, 255), 5)




def select(img):
    global ims, para, contour, ps, imggg, C, parts, index, para1, ims1, ps1


    # Segmentation
    im, cl, d = segment.seg(img)

    # Setting values of the global variables
    ps = cl.copy()
    imggg = im
    j = imggg.copy()
    ims = imggg.copy()
    C = np.zeros(ims.shape, np.uint8)
    contour = None
    para = True
        
    # Selction of contour
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', (int(im.shape[1]/2), int(im.shape[0]/2)))
    cv2.moveWindow('image', 40,0) 
    cv2.setMouseCallback('image',draw_c)

    while(1):
        cv2.imshow('image',ims)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()


    # Finding all the points on the contour 
    Cp = cv2.cvtColor(C, cv2.COLOR_BGR2GRAY)
    point = cv2.findNonZero(Cp)

    # Taking as input the number of parts to which the contour should be divided
    num_parts = input("In how many parts you want to divide the selected vessel (Please enter an integer <=5):  ")
    #print(point.shape)
    print("Select the required part and press Q.")

    parts = np.array_split(point, int(num_parts), axis=0)
    colbgr = [(193, 182, 255), (255, 0, 102), (255, 128, 0), (0, 255, 255), (10, 200, 10)]
    Cparts = np.zeros(C.shape)

    for i, part in enumerate(parts):
        if (i>=5):
            cv2.drawContours(Cparts, part, -1, (255,255,255), 5)
        else:
            cv2.drawContours(Cparts, part, -1, colbgr[i], 5)

    # Global variables
    para1 = True
    ps1 = Cparts
    ims1 = ps1.copy()
    parts = np.array_split(point, int(num_parts), axis=0)


    # At max supports 10 parts
    if (int(num_parts)<=1 or int(num_parts)>10):
        num_parts = 1
        C_parts_selected = parts[0]

        return (C_parts_selected, d)


    # Selecting the part if number of parts <=10
    cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', (int(im.shape[1]/2), int(im.shape[0]/2)))
    cv2.moveWindow('image1', 40,0) 
    cv2.setMouseCallback('image1', draw_part)

    while(1):
        cv2.imshow('image1',ims1)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cv2.destroyAllWindows()



    C_parts_selected = parts[index]


    return C_parts_selected, d


