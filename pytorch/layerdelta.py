import floatdelta
from marshmallow import Schema, fields

# The initial attempt at serializing the data to JSON used Marshmallow, but 
# that ended up with limitations. Basically, it didn't work for generic lists
# of layer diff. Eventually marshamallow will be removed, since it doesn't
# work the way we want.

# Diff object encapsulates the difference between a single layer for two checkpoints
# it has array for the weight and bias dif and some basic stats
class Conv2dLayerDelta:

    def __init__(self, layerindex, layername, kheight, kwidth, channel, filter):
        self.index = layerindex
        self.layername = layername
        self.height = kheight
        self.width = kwidth
        self.channels = channel
        self.filters = filter
        self.type = 'tf.keras.layers.Conv2D'
        self.layerindex = 0
        self.deltaarray = []
        self.diffcount = 0
        self.paramcount = 0
        self.deltasum = 0.0
        self.deltamax = 0.0
        self.biasarray = []
        self.biasparamcount = 0
        self.biasdiffcount = 0
        self.biasdeltasum = 0.0
        self.biasdeltamax = 0.0

    def AddArray(self, data):
        self.deltaarray.append(data)
        #print('delta array len: ', len(self.deltaarray))

    def incrementDeltaCount(self):
        self.diffcount +=1
    
    def incrementParamCount(self):
        self.paramcount +=1

    def AddDelta(self, dval):
        self.deltasum += dval
        if dval > self.deltamax:
            self.deltamax = dval
    
    def incrementBiasDeltaCount(self):
        self.biasdiffcount +=1
    
    def AddBiasDelta(self, dval):
        self.biasdeltasum += dval
        if dval > self.biasdeltamax:
            self.biasdeltamax = dval

    def incrementBiasParamCount(self):
        self.biasparamcount +=1

    @property
    def name(self):
        return self.layername

class DenseLayerDelta:
    def __init__(self, layerindex, name) -> None:
        self.layername = name
        self.index = layerindex
        self.type = 'tf.keras.layers.Dense'
        self.width = 0
        self.height = 0
        self.deltaarray = []
        self.diffcount = 0
        self.paramcount = 0
        self.deltasum = 0.0
        self.deltamax = 0.0
        self.biasarray = []
        self.biasdiffcount = 0
        self.biasdeltasum = 0.0
        self.biasparamcount = 0
        self.biasdeltamax = 0.0

    def AddArray(self, data):
        self.deltaarray.append(data)
        #print('delta array len: ', len(self.deltaarray)) v

    def incrementParamCount(self):
        self.paramcount +=1
    
    def incrementDeltaCount(self):
        self.diffcount +=1

    def AddDelta(self, dval):
        self.deltasum += dval
        if dval > self.deltamax:
            self.deltamax = dval

    def AddBiasDelta(self, dval):
        self.biasdeltasum += dval
        if dval > self.biasdeltamax:
            self.biasdeltamax = dval

    def incrementBiasDeltaCount(self):
        self.biasdiffcount +=1

    def incrementBiasParamCount(self):
        self.biasparamcount +=1

