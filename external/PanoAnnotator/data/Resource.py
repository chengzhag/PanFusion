import os
import numpy as np

from ..configs import Params as pm

from PIL import Image as Image
# from PyQt5.QtGui import QPixmap

class Resource(object):

    def __init__(self, name):

        self.name = name

        self.path = ''
        self.image = None #(w,h)
        self.data = None #(h,w)
        self.pixmap = None

    def initByImageFile(self, filePath):
        
        if os.path.exists(filePath):
            self.path = filePath
            self.image = Image.open(filePath).convert('RGB')
            self.data = np.asarray(self.image).astype(float)    
            if pm.isGUI:       
                self.pixmap = QPixmap(filePath)
            return True
        else :
            print("No default {0} image found".format(self.name))
            return False

    def initByImageFileDepth(self, filePath):
        
        if os.path.exists(filePath):
            self.path = filePath
            self.image = Image.open(filePath)
            self.data = np.asarray(self.image).astype(float)           
            return True
        else :
            print("No default {0} image found".format(self.name))
            return False