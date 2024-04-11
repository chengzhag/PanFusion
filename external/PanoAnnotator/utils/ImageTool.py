import numpy as np
# import matplotlib.pyplot as plt
import os

from .. import utils
from ..configs import Params as pm

# from PyQt5.QtGui import QImage, QPixmap
from skimage import morphology, filters, draw, transform
from PIL import Image

def imageROI(data, lt, rb):

    regionDate = data[lt[0]:rb[0], lt[1]:rb[1]]
    return regionDate

def imageRegionMean(data, center, steps):

    lt, rb = imageRegionBox(center, steps, data.shape)
    roi = imageROI(data, lt, rb)
    mean = np.nanmean(roi)
    return mean

def imageRegionBox(center, steps, size):

    lt = (center[0] - steps[0], center[1] - steps[1])
    rb = (center[0] + steps[0], center[1] + steps[1])

    lt = checkImageBoundary(lt, size)
    rb = checkImageBoundary(rb, size)
    return lt, rb

def imagePointsBox(posList):

    X = [pos[0] for pos in posList]
    Y = [pos[1] for pos in posList]

    lt = (min(X), min(Y))
    rb = (max(X), max(Y))
    return lt, rb

def checkImageBoundary(pos, size):
        
    x = sorted([0, pos[0], size[0]])[1]
    y = sorted([0, pos[1], size[1]])[1]
    return (x, y)

def data2Pixmap(data):

    imgData = data * 255
    imgData = imgData.astype(dtype=np.uint8)
    image = QImage(imgData, data.shape[1], data.shape[0], 
                    QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(image)
    return pixmap

def imageResize(data, size):

    dataR = transform.resize(data, size, mode='constant')
    return dataR

def imageDilation(data, rad):

    ans = np.zeros(data.shape, dtype=float)
    for i in range(data.shape[2]):
        channel = data[:,:,i]
        ans[:,:,i] = morphology.dilation(channel, 
                            morphology.diamond(rad))
    return ans

def imageGaussianBlur(data, sigma):

    ans = np.zeros(data.shape, dtype=float)
    for i in range(data.shape[2]):
        channel = data[:,:,i]
        ans[:,:,i] = filters.gaussian(channel, sigma)
    return ans

def imagesMSE(data1, data2):

    if not data1.shape == data2.shape:
        print('size error')
    #data1r = transform.resize(data1, size, mode='constant')
    #data2r = transform.resize(data2, size, mode='constant')

    #data1r[data1r==0] = np.nan
    #data2r[data2r==0] = np.nan
    #mse = np.nanmean((data1r - data2r)**2)
    mse = np.mean((data1 - data2)**2)

    return mse
    
def imageDrawLine(data, p1, p2, color):

    rr, cc = draw.line(p1[1],p1[0],p2[1],p2[0])
    draw.set_color(data, [rr,cc], list(color))

def imageDrawPolygon(data, points, color):

    X = np.array([p[0] for p in points])
    Y = np.array([p[1] for p in points])
    rr, cc = draw.polygon(Y,X)
    draw.set_color(data, [rr,cc], list(color))

def imageDrawWallDepth(data, polygon, wall, plane_map=None, wall_idx=None):

    size = (data.shape[1], data.shape[0])
    polyx = np.array([p[0] for p in polygon])
    polyy = np.array([p[1] for p in polygon])

    posy, posx = draw.polygon(polyy, polyx)

    for i in range(len(posy)):
        coords = utils.pos2coords((posx[i],posy[i]), size)
        vec =  utils.coords2xyz(coords, 1)

        point = utils.vectorPlaneHit(vec, wall.planeEquation)
        depth = 0 if point is None else utils.pointsDistance((0,0,0), point)
        # color = (depth, depth, depth)
        if posy[i] >= 0 and posy[i] < data.shape[0] and posx[i] >= 0 and posx[i] < data.shape[1]:
            curr = data[posy[i],posx[i]]
            if (curr == 0) or (curr > depth):
                draw.set_color(data, [posy[i],posx[i]], depth)
                if plane_map is not None:
                    plane_map[posy[i],posx[i]] = wall_idx

def showImage(image):

    plt.figure()
    plt.imshow(image)
    plt.show()

def saveImage(image, path):

    im = Image.fromarray(np.uint8(image*255))
    im.save(path)

def saveDepth(depth, path):

    data = np.uint16(depth*4000)

    array_buffer = data.tobytes()
    img = Image.new("I", data.T.shape)
    img.frombytes(array_buffer, 'raw', "I;16")
    img.save(path)

def saveMask(mask, path):

    mask = mask[:,:,0]
    im = Image.fromarray(np.uint8(mask*255))
    im.save(path)