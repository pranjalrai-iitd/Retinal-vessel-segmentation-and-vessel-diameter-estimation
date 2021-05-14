"""
===========================
@Author  : Pranjal Rai
@Version: 1.0    12/07/2020
Vessel diameter estimation
===========================
"""



import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import vessel_select


# Find perpendicular direction
def perpendicluar(x, y, vx, vy, d):
    mag = math.sqrt(vx * vx + vy * vy)
    if (vy != 0):
        vx = vx / mag
        vy = vy / mag
        temp = vx
        vx = -1 * vy
        vy = temp
        Cx = (x + vx * d)
        Cy = (y + vy * d)
        Dx = (x - vx * d)
        Dy = (y - vy * d)
    else:
        Cx = (x)
        Cy = (y + d)
        Dx = (x)
        Dy = (y - d)

    return (Cx,Cy,Dx,Dy)


# Bilinear interpolation to find values at non-integral positions
def interpolate(img, x, y):
    y1 = int(y)
    y2 = int(y)+1
    x1 = int(x)
    x2 = int(x)+1
    if(x==x1 and y==y1):
        return np.array([img[x1, y1]])[0]

    if(x==x1 and y!=y1):
        val = ((y2-y)/(y2-y1))*(img[x1, y1])+((y-y1)/(y2-y1))*(img[x1, y2])
        return np.array([val])[0]

    if(x!=x1 and y==y1):
        val = ((x2-x)/(x2-x1))*(img[x1, y1])+((x-x1)/(x2-x1))*(img[x2, y1])
        return np.array([val])[0]

    if(x!=x1 and y!=y1):
        val = (np.matmul(np.matmul(np.array([x2-x, x-x1]),np.array([[img[x1,y1], img[x1, y2]],[img[x2,y1], img[x2,y2]]])), np.array([[y2-y],[y-y1]])))/((x2-x1)*(y2-y1))
        return val[0]

# K-Means clustering of the points along a normal
def cluster(Z):
    z = Z.reshape((-1,1))
    z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((Z.shape))
    l = np.array(label.flatten()).reshape(Z.shape)
    return res2, l

# Selecting the lowest intensity cluster
def dia_color_min(l):
    l = l.reshape(-1, 1)
    #temp = np.ones(l.shape)*255
    temp = np.zeros(l.shape)
    val = np.min(l)
    temp_r = l.copy()
    for i in range(l.shape[0]):
        if(temp_r[i, 0] == val):
            temp_r[i, 0] = 255
    _, temp = cv2.threshold(l, val, 255, cv2.THRESH_BINARY_INV) 
    return temp


def diameter(img):

    g = img[:,:,1]
    # Selected centerline
    point, dl = vessel_select.select(img)



    # Finding normal points for each centreline point and stroing their coordinates and values
    tempo = g.copy()
    nrp = []
    nrv = []
    for j in range(3, point.shape[0] - 3, 1):
        vx = point[j+ 3][0][0] - point[j- 3][0][0]
        vy = point[j+ 3][0][1] - point[j- 3][0][1]
        (Cx, Cy, Dx, Dy) = perpendicluar(point[j][0][0], point[j][0][1], vx,
                                         vy, int(dl)) #disctance to be 40
        cv2.line(tempo, (int(Cx), int(Cy)), (int(Dx), int(Dy)), 255, thickness=1)
        D = int(2*dl) #dyatance is 40
        normal_points = []
        values = []
        normal_points.append((Dy, Dx))
        values.append(interpolate(g, Dy, Dx))
        for i in range(1, D): #from 1 to D-1 for D-1 points at distance of 1 pixe
                part = ((((D-i)*Dy+i*Cy))/D, (((D-i)*Dx+i*Cx))/D)
                normal_points.append(part)
                values.append(interpolate(g, (((D-i)*Dy+i*Cy))/D, (((D-i)*Dx+i*Cx))/D ))
        normal_points.append((Cy, Cx))
        values.append(interpolate(g, Cy, Cx))
        nrp.append(np.array(normal_points))
        nrv.append(np.array(values))
    nrp = np.array(nrp)
    nrv = np.array(nrv)

    # Interpolated N0rmals
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', (int(tempo.shape[1]/2), int(tempo.shape[0]/2)))
    #cv2.moveWindow('image', 40,30)  # Move it to (40,30)
    #cv2.imshow('image', tempo)  

    #cv2.waitKey()
    #cv2.destroyAllWindows()
     


     
    X = np.arange(0, nrv.shape[1]).reshape(1, -1)
    Y = np.arange(0, nrv.shape[0]).reshape(-1, 1)
    Z = nrv

    # 3D feature map
    
    '''
    fig = plt.figure(figsize=(12,7))
    ax1 = fig.add_subplot(111, projection='3d')


    mycmap = plt.get_cmap('gist_earth')
    ax1.set_title('Vessel histogram map')
    ax1.set_xlabel('Points normal to the vessel profile')
    ax1.set_ylabel('Points along the vessel profile')
    ax1.set_zlabel('Interpolated intensity')
    surf1 = ax1.plot_surface(X, Y, Z, cmap=mycmap)
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    plt.show()
    '''

    # Clustering points along each normal
    L = np.zeros(Z.shape)
    Zc = np.zeros(Z.shape)
    for i in range(Z.shape[0]):
        Zc[i, :], L[i,:] = cluster(Z[i,:])

    im_dia = np.ones(Z.shape)

    # CLR correction i.e, joining broken lines along every normal
    for i in range(L.shape[0]):
        pred = dia_color_min(Zc[i])  
        kernel = np.ones((5, 1), np.uint8)
        d_dia = cv2.dilate(pred, kernel)
        e_dia = cv2.erode(d_dia, kernel) 
        im_dia[i] = e_dia.reshape(-1)

    #plt.figure(figsize=(10, 10))

    #plt.subplot(3, 1, 1)
    #plt.imshow(im_dia.T, cmap='gray')

    # joining broken lines along the centreline
    kernel = np.ones((5, 1), np.uint8)
    im_d_dia = cv2.dilate(im_dia, kernel)
    im_e_dia = cv2.erode(im_d_dia, kernel) 

    #plt.subplot(3, 1, 2)
    #plt.imshow(im_e_dia.T, cmap='gray')

    # using symmetry to remove unwanted false positives
    im_e_dia_rev = im_e_dia.copy()
    for i in range(im_e_dia.T.shape[0]):
        im_e_dia_rev.T[i] = im_e_dia.T[im_dia.T.shape[0]-1-i]
        
    pred_dia_cl = cv2.bitwise_and(im_e_dia, im_e_dia_rev, mask=None)

    # Finally smoothing the profile edges by joining pixels which are horizontally at a distance less than 20px
    kernel = np.ones((20, 1), np.uint8)
    pred_d_dia = cv2.dilate(pred_dia_cl, kernel)
    pred_e_dia = cv2.erode(pred_d_dia, kernel) 

    #plt.subplot(3, 1, 3)
    #plt.imshow(pred_e_dia.T, cmap='gray')


    #Storing the diameters
    diameters = []

    for i in range(im_e_dia.shape[0]):
        dia = np.sum(1*(pred_e_dia[i]!=0))
        diameters.append(dia)
        


    diameters = np.array(diameters).reshape(len(diameters), 1)
    print('Average diameter length:', np.mean(diameters))
    print('Median diameter length:', np.median(diameters))
    print('Standard deviation:', np.std(diameters)/len(diameters))



    # Drawing rough diameters on the image 
    final_ann = img.copy()
    
    for j in range(3, point.shape[0] - 3, 1):
        vx = point[j+ 3][0][0] - point[j- 3][0][0]
        vy = point[j+ 3][0][1] - point[j- 3][0][1]
        (Cx, Cy, Dx, Dy) = perpendicluar(point[j][0][0], point[j][0][1], vx,
                                         vy, diameters[i]//2) #disctance to be 40
        cv2.line(final_ann, (int(Cx), int(Cy)), (int(Dx), int(Dy)), (0, 0, 0), thickness=1)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', (int(final_ann.shape[1]/2), int(final_ann.shape[0]/2)))
    cv2.moveWindow('image', 40,30)  # Move it to (40,30)
    cv2.imshow('image', final_ann)
    cv2.waitKey()
    cv2.destroyAllWindows()

