from .. import utils 

gpInstanceCount = 0

class GeoPoint(object):

    def __init__(self, scene, coords=None, xyz=None):

        self.__scene = scene

        self.coords = coords
        self.color = (0, 0, 0)
        self.depth = 0
        self.xyz = xyz

        self.type = 0 # [convex, concave, occul]
        self.id = 0

        self.initByScene()

        global gpInstanceCount
        gpInstanceCount += 1
        self.id = gpInstanceCount

    def initByScene(self):

        if self.coords == None:
            self.coords = utils.xyz2coords(self.xyz)

        coordsT = (self.coords[1], self.coords[0])

        colorData = self.__scene.getPanoColorData()

        colorPos = utils.coords2pos(coordsT, colorData.shape)
        rgb = colorData[colorPos[0]][colorPos[1]]
        self.color = (rgb[0], rgb[1], rgb[2])

        depthData = self.__scene.getPanoDepthData()
        
        depthPos = utils.coords2pos(coordsT, depthData.shape)
        depthMean = utils.imageRegionMean(depthData, depthPos, (5, 5))
        self.depth = depthMean
        #self.depth = depthData[depthPos[0]][depthPos[1]]

        if self.xyz == None:
            self.xyz = utils.coords2xyz(self.coords, self.depth)
        
        #self.calcGeometryType()

    def moveByVector(self, vec):

        self.xyz = utils.vectorAdd(self.xyz, vec)
        self.coords = utils.xyz2coords(self.xyz)

    '''
    def calcGeometryType(self):

        coordsT = (self.coords[1], self.coords[0])
        depthData = self.__scene.getPanoDepthData()

        depthPos = utils.coords2pos(coordsT, depthData.shape)
        depth = depthData[depthPos[0]][depthPos[1]]
        if depth <= 0:
            return

        #print(depthPos)
        lt, rb = utils.calcCenterRegionPos(depthPos, 
                                            ( int(50/depth), int(50/depth))
                                            ,depthData.shape)
        #print("{0} {1}".format(lt, rb))
        
        cb = (rb[0], depthPos[1])
        #print("cb {0}".format(cb))
        regionL = utils.getRegionData(depthData, lt, cb)
        #print(regionL.shape)
        ct = (lt[0], depthPos[1])
        #print("ct {0}".format(ct))
        regionR = utils.getRegionData(depthData, ct, rb)
        #print(regionR.shape)

        avgL = np.nanmean(regionL)
        avgR = np.nanmean(regionR)

        #print("L : {0}   R : {1}".format(avgL, avgR))
        if abs(avgL - avgR) > 0.75:
            self.type = 2
    '''

    