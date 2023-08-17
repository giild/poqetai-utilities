import numpy
import LayerActivation

from marshmallow import Schema, fields

class Activationdetails:

    def __init__(self):
        self.modelName = ''
        self.modeFile = ''
        self.recordName = ''
        self.recordLabel = ''
        self.epoch = ''
        self.documentType = "ProfileInference"
        self.modelType = "keras"
        self.activationData = []
        return
    
    def addLayerActivation(self, data):
        self.activationData.append(data)