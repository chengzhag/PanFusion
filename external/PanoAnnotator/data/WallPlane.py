import random

from .. import data
from .. import utils 

wpInstanceCount = 0

class WallPlane(object):

    def __init__(self, scene, gPoints):

        self.__scene = scene

        if(len((gPoints))<2):
            print("Two point at least")
            
        self.gPoints = gPoints
        self.attached = []
        self.color = (random.random(), random.random(), 
                      random.random())

        self.normal = (0, 0, 0)
        self.planeEquation = (0, 0, 0, 0)
        self.width = 0 

        self.corners = []
        self.edges = []
        self.bbox2d = ((0,0),(1,1))

        self.id = 0

        self.init()

        global wpInstanceCount
        wpInstanceCount += 1
        self.id = wpInstanceCount
    
    def init(self):

        self.updateGeometry()

    def moveByNormal(self, val):

        vec = utils.vectorMultiplyC(self.normal, val)
        for gp in self.gPoints:
            gp.moveByVector(vec)
    
        for obj2d in self.attached:
            obj2d.moveByNormal(val)

        self.updateGeometry()

    def updateGeometry(self):

        self.updateCorners()
        self.updateEdges()
        self.updateBbox2d()

        self.normal = utils.pointsNormal(self.corners[0].xyz,self.corners[1].xyz,
                                        self.corners[3].xyz)
        self.color = utils.normal2color(self.normal)
        self.planeEquation = utils.planeEquation(self.normal, self.corners[0].xyz)
        self.width =  utils.pointsDistance(self.corners[0].xyz, self.corners[1].xyz)

        for obj2d in self.attached:
            obj2d.updateGeometry()

    def updateCorners(self):

        gps = self.gPoints
        scene = self.__scene
        cameraH = scene.label.getCameraHeight()
        cam2ceilH = scene.label.getCam2CeilHeight()

        self.corners = [data.GeoPoint(scene, None, 
                        (gps[0].xyz[0], cam2ceilH, gps[0].xyz[2])),
                        data.GeoPoint(scene, None, 
                        (gps[1].xyz[0], cam2ceilH, gps[1].xyz[2])),
                        data.GeoPoint(scene, None, 
                        (gps[1].xyz[0], -cameraH, gps[1].xyz[2])),
                        data.GeoPoint(scene, None, 
                        (gps[0].xyz[0], -cameraH, gps[0].xyz[2]))]
    
    def updateEdges(self):

        scene = self.__scene
        self.edges = [data.GeoEdge(scene, (self.corners[0], self.corners[1])),
                    data.GeoEdge(scene, (self.corners[1], self.corners[2])),
                    data.GeoEdge(scene, (self.corners[2], self.corners[3])),
                    data.GeoEdge(scene, (self.corners[3], self.corners[0]))]

    def updateBbox2d(self):

        coords = []
        for c in [e.coords for e in self.edges]:
            coords += c 
        self.bbox2d = utils.imagePointsBox(coords)

    #manh only
    def checkRayHit(self, vec, orig=(0,0,0)):

        point = utils.vectorPlaneHit(vec, self.planeEquation)
        if point is None:
            return False, None
        
        cs = self.corners
        if cs[2].xyz[1] <= point[1] <= cs[0].xyz[1]:

            p1 = (point[0], cs[0].xyz[1], point[2])
            dis1 = utils.pointsDistance(p1, cs[0].xyz)
            dis2 = utils.pointsDistance(p1, cs[1].xyz)
            dis3 = utils.pointsDistance(cs[0].xyz, cs[1].xyz)

            if dis1 + dis2 <= dis3 * 1.0005:
                return True, point

        return False, None
