import numpy as np
import os
import math

from .. import data
from .. import utils
from ..configs import Params as pm
# import estimator

class Annotation(object):

    def __init__(self, scene):
        
        self.__scene = scene
        #self.pushPred = estimator.PushPredLite(scene)
        # self.pushPred = estimator.PushPred(scene)

        self.__layoutPoints = []
        self.__layoutWalls = []
        self.__layoutObjects2d = []

        self.__layoutFloor = None
        self.__layoutCeiling = None

        self.__cameraHeight = pm.defaultCameraHeight
        self.__layoutHeight = pm.defaultLayoutHeight

    #####
    #Layout generation
    #####
    def genLayoutWallsByPoints(self, points):

        self.clearLayoutWalls()
        self.__scene.selectObjs = []

        pnum = len(points)
        for i in range(0, pnum):
            plane = data.WallPlane(self.__scene, 
                             [points[i], points[(i+1)%pnum]])
            self.__layoutWalls.append(plane)
       
        self.__layoutFloor = data.FloorPlane(self.__scene, False)
        self.__layoutCeiling = data.FloorPlane(self.__scene, True)

    def calcManhLayoutPoints(self, points):

        self.__layoutPoints = []

        manhxyz = utils.alignManhattan(points)
        for xyz in manhxyz:
            gp = data.GeoPoint(self.__scene, None, xyz)
            self.__layoutPoints.append(gp)

    def genManhLayoutWalls(self):
        
        self.clearLayoutWalls()

        self.calcManhLayoutPoints(self.__layoutPoints)
        self.genLayoutWallsByPoints(self.__layoutPoints)
    
    def genSplitPoints(self,  wall, point):

        p1 = data.GeoPoint(self.__scene, None, point)
        p2 = data.GeoPoint(self.__scene, None, point)

        wP1idx = self.__layoutPoints.index(wall.gPoints[0])
        wP2idx = self.__layoutPoints.index(wall.gPoints[1])

        if abs(wP1idx - wP2idx) == len(self.__layoutPoints) - 1: 
            self.__layoutPoints.extend((p1, p2))
        else:
            idx = max([wP1idx,wP2idx])
            self.__layoutPoints[idx:idx] = [p1, p2]
            
        self.genLayoutWallsByPoints(self.__layoutPoints)

    def genObject2d(self, points, wall):

        p1 = data.GeoPoint(self.__scene, None, points[0])
        p2 = data.GeoPoint(self.__scene, None, points[1])

        obj = data.Object2D(self.__scene, [p1, p2], wall)
        self.__layoutObjects2d.append(obj)        

    #####
    #Layout operation
    #####
    def delLayoutWalls(self, walls):

        for wall in walls:
            for point in wall.gPoints:
                if point in self.__layoutPoints:
                    self.__layoutPoints.remove(point)

        self.genManhLayoutWalls()

    def mergeLayoutWalls(self, walls):

        tmp = []
        for wall in walls:
            for point in wall.gPoints:
                if point not in tmp:
                    tmp.append(point)
                elif point in self.__layoutPoints:
                    self.__layoutPoints.remove(point)

        self.genManhLayoutWalls()
    
    def delLayoutObject2ds(self, obj2ds):

        for obj in obj2ds:
            if obj in self.__scene.selectObjs:
                self.__scene.selectObjs.remove(obj)
            if obj in obj.attach.attached: 
                obj.attach.attached.remove(obj)
            self.__layoutObjects2d.remove(obj)

    def mergeTrivialWalls(self, val):
        
        restart = True
        while restart:
            for wall in self.__layoutWalls:
                if wall.width < val:
                    restart = True
                    self.delLayoutWalls([wall])
                    break
                restart = False
    
    def clearLayoutWalls(self):

        #self.delLayoutObject2ds(self.__layoutObjects2d)
        self.__layoutObjects2d = []
        self.__layoutWalls = []

    def cleanLayout(self):

        self.__layoutPoints = []
        self.__layoutWalls = []
        self.__layoutObjects2d = []
        
        self.__layoutFloor = None
        self.__layoutCeiling = None
        self.__cameraHeight = pm.defaultCameraHeight
        self.__layoutHeight = pm.defaultLayoutHeight
    
    #####
    # Objects manual operation
    #####
    def moveWallByNormal(self, wall, val):

        wall.moveByNormal(val)
        self.updateLayoutGeometry()
    
    def moveFloor(self, val):
        self.__cameraHeight += val
        self.__layoutHeight += val
        self.updateLayoutGeometry()
    
    def moveCeiling(self, val):
        self.__layoutHeight += val
        self.updateLayoutGeometry()

    #####
    # Objects automatic operation
    #####
    def moveWallByPred(self, wall, val):
        #self.pushPred.optimizeWallBF(wall, val)
        self.pushPred.optimizeWallGS(wall, val)

    #####
    # Auto-Calulation part
    #####
    def calcInitLayout(self):
        
        self.cleanLayout()

        samplePoints = []
        for x in np.arange(0.0, 1.0, 0.01):
            coords = (x, 0.5)
            geoPoint = data.GeoPoint(self.__scene, coords)
            samplePoints.append(geoPoint)
        
        self.calcManhLayoutPoints(samplePoints)
        self.genLayoutWallsByPoints(self.__layoutPoints)

        #self.mergeTrivialWalls(0.5)
        self.mergeTrivialWalls(1.0)
        #self.pushPred.optimizeLayoutBF()
        self.pushPred.optimizeLayoutGS()
        #self.mergeTrivialWalls(0.5)

    def updateLayoutGeometry(self):

        for wall in self.__layoutWalls:
            wall.updateGeometry()
        self.__layoutFloor.updateGeometry()
        self.__layoutCeiling.updateGeometry()

    #####
    #Getter & Setter
    #####
    def setCameraHeight(self, cameraH):
        self.__cameraHeight = cameraH

    def setLayoutHeight(self, layoutH):
        self.__layoutHeight = layoutH

    def getLayoutPoints(self):
        return self.__layoutPoints
    def setLayoutPoints(self, points):
        self.__layoutPoints = points
        self.genLayoutWallsByPoints(self.__layoutPoints)
        self.updateLayoutGeometry()

    def getLayoutWalls(self):
        return self.__layoutWalls
    
    def getLayoutFloor(self):
        return self.__layoutFloor

    def getLayoutCeiling(self):
        return self.__layoutCeiling

    def getLayoutObject2d(self):
        return self.__layoutObjects2d
    def setLayoutObject2d(self, obj2ds):
        self.__layoutObjects2d = obj2ds
        self.updateLayoutGeometry()

    def getCameraHeight(self):
        return self.__cameraHeight

    def getCam2CeilHeight(self):
        return self.__layoutHeight - self.__cameraHeight

    def getLayoutHeight(self):
        return self.__layoutHeight