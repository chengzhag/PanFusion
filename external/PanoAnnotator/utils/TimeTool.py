import time

timeStartRun = 0
timeStartFPS = 0

def getFPS():

    global timeStartFPS
    durning = time.clock() - timeStartFPS
    if not durning == 0:
        fps = 1.0 / (time.clock() - timeStartFPS)
    else:
        fps = 0.0
    timeStartFPS = time.clock()
    return fps

def resetTimer():
    global timeStartRun
    timeStartRun = time.clock()

def getRunTime():
    global timeStartRun
    print(time.clock() - timeStartRun)
    timeStartRun = time.clock()

