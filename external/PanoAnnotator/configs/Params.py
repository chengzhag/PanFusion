
class Params(object):

    fileDefaultOpenPath = "D:"
    labelFileDefaultName = "label.json"

    colorFileDefaultName = "color.png"
    depthFileDefaultName = "depth.png"
    linesFileDefaultName = "None" #"lines.png"
    omapFileDefaultName = "None" #"omap.png"

    #GPU
    isDepthPred = True
    isGUI = True

    #Annotation
    layoutMapSize = [512, 1024, 3]

    defaultCameraHeight = 1.8
    defaultLayoutHeight = 3.2


    #Input
    keyDict = {'none':0, 'ctrl':1, 'shift':2, 'alt':3, 'object':4}

    layoutHeightSampleRange = 0.3
    layoutHeightSampleStep = 0.01 

    #MonoView
    monoViewFov = (-1, 90)

    #PanoTool
    pcSampleStride = 30
    meshProjSampleStep = 30