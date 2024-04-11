from .. import data
from .. import utils 

fpInstanceCount = 0

class FloorPlane(object):

    def __init__(self, scene, isCeiling=False):

        self.__scene = scene

        self.__isCeiling = isCeiling

        self.gPoints = scene.label.getLayoutPoints()
        self.walls = scene.label.getLayoutWalls()
        self.color = (0,0,0)
        
        self.normal = (0, -1, 0) if isCeiling else (0, 1, 0)
        self.height = 0
        self.planeEquation = (0, 0, 0, 0)

        self.corners = []
        self.edges = []
        self.bbox2d = ((0,0),(1,1))

        self.id = 0

        self.init()

        global fpInstanceCount
        fpInstanceCount += 1
        self.id = fpInstanceCount

    def init(self):
    
        self.updateGeometry()
    
    def updateGeometry(self):

        cameraH = self.__scene.label.getCameraHeight()
        cam2ceilH =  self.__scene.label.getCam2CeilHeight()
        self.height = cam2ceilH if self.__isCeiling else cameraH 
        self.planeEquation = self.normal + (self.height,)
        self.color = utils.normal2color(self.normal)

        self.updateCorners()
        self.updateEdges()
        self.updateBbox2d()
        
    def updateCorners(self):

        self.corners = []
        for gp in self.gPoints:
            if self.__isCeiling:
                xyz = (gp.xyz[0], self.height, gp.xyz[2])
            else:
                xyz = (gp.xyz[0], -self.height, gp.xyz[2])
            corner = data.GeoPoint(self.__scene, None, xyz)
            self.corners.append(corner)
    
    def updateEdges(self):
        
        self.edges = []
        cnum = len(self.corners)
        for i in range(cnum):
            edge = data.GeoEdge(self.__scene, 
                                (self.corners[i], self.corners[(i+1)%cnum]))
            self.edges.append(edge)
    
    def updateBbox2d(self):

        coords = []
        for c in [e.coords for e in self.edges]:
            coords += c 
        self.bbox2d = utils.imagePointsBox(coords)

    def isCeiling(self):
        return self.__isCeiling

