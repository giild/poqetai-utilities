import layerdelta

# modelDelta encapsulates the difference between two checkpoint files for a training run
# the delta currently handles Conv2D and Dense layers.
class ModelDelta:

    def __init__(self, name, filename1, filename2):
        self.modelname = name
        self.modelfile1 = filename1
        self.modelfile2 = filename2
        self.epoch1 = ''
        self.epoch2 = ''
        self.layerdeltas = []

    @property
    def name(self):
        return self.modelname
    
    def addLayerDelta(self, delta):
        self.layerdeltas.append(delta)
