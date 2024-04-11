
progMax = 0
progCount = 0

def resetProgress(scene, maxVal=1):

    global progMax
    progMax = maxVal
    global progCount
    progCount = 0
    setProgressVal(scene)

def updateProgress(scene):

    global progMax
    global progCount
    progCount += 1
    if progCount >= progMax:
        resetProgress(scene)
    else:
        setProgressVal(scene)

def setProgressVal(scene):

    global progMax
    global progCount
    
    mainWindows = scene.getMainWindows()
    val = float(progCount)/progMax * 100
    mainWindows.updataProgressView(val)