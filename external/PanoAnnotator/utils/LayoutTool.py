import numpy as np

from ..configs import Params as pm
from .. import utils


def alignManhattan(gps):

    class Edge:
        def __init__(self, axis, p1):
            self.axis = axis
            self.points = [p1]
            self.center = (0, 0, 0)

    n = len(gps)
    if n < 2:
        print('cant align manh world')
        return
    
    #create edges, calculate axis type and contain points
    edges = []
    for i in range(n):

        dist = utils.pointsDirectionPow(gps[i].xyz, gps[(i+1)%n].xyz, 2)
        axis = 0 if dist[0] >= dist[2] else 1

        if len(edges) == 0:
            edges.append(Edge(axis, gps[i]))
        elif not edges[-1].axis == axis:
            edges[-1].points.append(gps[i])
            edges.append(Edge(axis, gps[i]))
        elif edges[-1].axis == axis:
            edges[-1].points.append(gps[i])

    #merge last edge to first if they have same axis
    if edges[0].axis == edges[-1].axis:
        edges[0].points += edges[-1].points
        edges.pop()

    #calculate each edge's center position
    for edge in edges:
        pList = [p.xyz for p in edge.points]
        edge.center = utils.pointsMean(pList)

    #calculate manhattan corner points
    manhPoints = []
    for i in range(len(edges)):
        if edges[i].axis == 0:
            manhPoints.append((edges[i-1].center[0], 0, edges[i].center[2]))
        elif edges[i].axis == 1:
            manhPoints.append((edges[i].center[0], 0, edges[i-1].center[2]))

    return manhPoints


def genWallPolygon2d(size, wall):

    size = (size[1], size[0])
    
    isCrossUp, ul, ur = wall.edges[0].checkCross()
    isCrossDown, dl, dr = wall.edges[2].checkCross()

    polygon = []; vertex = []
    for edge in wall.edges:
        vertex.extend([s for s in edge.sample])
        polygon.extend([utils.coords2pos(c,size) for c in edge.coords])

    if not (isCrossUp or isCrossDown):
        return False, polygon
    else:
        iur = vertex.index(ur); iul = iur + 1
        idr = vertex.index(dr); idl = idr - 1
        
        uh = int((polygon[iur][1] + polygon[iul][1])/2)
        dh = int((polygon[idr][1] + polygon[idl][1])/2)        
        polygon1 = polygon[:iur] + [(size[0],uh), (size[0],dh)] + polygon[idr:]
        polygon2 = [(0,uh)] + polygon[iul:idl] + [(0,dh)]
        return True, (polygon1,polygon2)
    
def genLayoutNormalMap(scene, size, plane_map):
    
    normalMap = np.zeros(size)
    normalMap[:int(size[0]/2),:] = scene.label.getLayoutCeiling().color
    normalMap[int(size[0]/2)+1:,:] = scene.label.getLayoutFloor().color
    
    for wall_idx, wall in enumerate(scene.label.getLayoutWalls()):
        mask = plane_map == wall_idx
        if wall.planeEquation[3] > 0:
            continue
        normalMap[mask] = wall.color
    
    return normalMap


def genLayoutOMap(scene, size, plane_map):

    oMap = np.zeros(size)
    oMap[:,:,0] = 1
        
    for wall_idx, wall in enumerate(scene.label.getLayoutWalls()):
        mask = plane_map == wall_idx
        if wall.planeEquation[3] > 0:
            continue
        
        color = utils.normal2ManhColor(wall.normal)
        oMap[mask] = color
    
    return oMap

def genLayoutDepthMap(scene, size, get_plane_map=False):

    depthMap = np.zeros(size)
    if get_plane_map:
        planeMap = np.zeros(size, dtype=np.uint8)

    for y in range(0, size[0]):
        for x in range(0, size[1]):
            coords = utils.pos2coords((y,x), size)
            coordsT = utils.posTranspose(coords)
            vec =  utils.coords2xyz(coordsT, 1)
            if y <= int(size[0]/2):
                plane = scene.label.getLayoutCeiling().planeEquation
                if get_plane_map:
                    planeMap[y,x] = len(scene.label.getLayoutWalls())
            else:
                if get_plane_map:
                    planeMap[y,x] = len(scene.label.getLayoutWalls()) + 1
                plane = scene.label.getLayoutFloor().planeEquation
            point = utils.vectorPlaneHit(vec, plane)
            depth = 0 if point is None else utils.pointsDistance((0,0,0), point)
            depthMap[y,x] = depth

    for wall_idx, wall in enumerate(scene.label.getLayoutWalls()):
        # if wall.planeEquation[3] > 0:
        #     continue
        isCross, polygon = genWallPolygon2d(size, wall)
        if not isCross:
            utils.imageDrawWallDepth(depthMap, polygon, wall, plane_map=planeMap, wall_idx=wall_idx)
        else:
            utils.imageDrawWallDepth(depthMap, polygon[0], wall, plane_map=planeMap, wall_idx=wall_idx)
            utils.imageDrawWallDepth(depthMap, polygon[1], wall, plane_map=planeMap, wall_idx=wall_idx)

    if get_plane_map:
        return depthMap, planeMap
    return depthMap

def genLayoutEdgeMap(scene, size):

    edgeMap = np.zeros(size)

    sizeT = (size[1],size[0])
    for wall in scene.label.getLayoutWalls():
        # if wall.planeEquation[3] > 0:
        #     continue
        for edge in wall.edges:
            color = utils.normal2ManhColor(edge.vector)
            #color = (1, 1, 1)
            for i in range(len(edge.coords)-1):
                isCross, l, r = utils.pointsCrossPano(edge.sample[i],
                                                    edge.sample[i+1])
                if not isCross:
                    pos1 = utils.coords2pos(edge.coords[i], sizeT)
                    pos2 = utils.coords2pos(edge.coords[i+1], sizeT)
                    utils.imageDrawLine(edgeMap, pos1, pos2, color)
                else:
                    lpos = utils.coords2pos(utils.xyz2coords(l), sizeT)
                    rpos = utils.coords2pos(utils.xyz2coords(r), sizeT)
                    ch = int((lpos[1] + rpos[1])/2)
                    utils.imageDrawLine(edgeMap, lpos, (0,ch), color)
                    utils.imageDrawLine(edgeMap, rpos, (sizeT[0],ch), color)
        
    edgeMap = utils.imageDilation(edgeMap, 1)
    edgeMap = utils.imageGaussianBlur(edgeMap, 2)
    return edgeMap


def genLayoutObj2dMap(scene, size):

    obj2dMap = np.zeros(size)

    for obj2d in scene.label.getLayoutObject2d():
        isCross, polygon = genWallPolygon2d(size, obj2d)
        if not isCross:
            utils.imageDrawPolygon(obj2dMap, polygon, obj2d.color)
        else:
            utils.imageDrawPolygon(obj2dMap, polygon[0], obj2d.color)
            utils.imageDrawPolygon(obj2dMap, polygon[1], obj2d.color)

    return obj2dMap

def normal2ManhColor(normal):
    vec = [abs(e) for e in list(normal)]
    axis = vec.index(max(vec))

    if axis == 0:
        color = (0,0,1)
    elif axis == 1:
        color = (1,0,0)
    elif axis == 2:
        color = (0,1,0)
    
    return color