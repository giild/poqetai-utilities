import tensorflow as tf

# class encapsulates the prediction results and groups them into positive, negative, falsepositive and falsenegative
class ValidationResult:
    def __init__(self) -> None:
        self.checkpointfile = ''
        self.epoch = '1'
        self.starttime = 0
        self.endtime = 0
        self.positiveCount = 0
        self.falsePositiveCount = 0
        self.results = {'FalsePositive': None, 'Positive': None}
        # populate the result with the default list

    # false positives are organized into dictionary of the classid
    # and a nested dictionary of the wrong predictionid
    def addFalsePositive(self, filename, classid, predictid):
        # add false positive
        fp = self.results['FalsePositive']
        if (fp == None):
            # false positive dictionay is null, so we just create dictionary of dictionary
            fp = {classid: {predictid:[filename]}}
        else:
            if (classid in fp):
                cfp = fp[classid]
                # check if the predictionid entry exists
                if (predictid in cfp):
                    cfp[predictid].append(filename)
                else:
                    cfp[predictid] = [filename]
            else:
                fp[classid] = {predictid:[filename]}
        self.results['FalsePositive'] = fp
        self.falsePositiveCount = self.falsePositiveCount + 1

    def addPositive(self, filename, classid):
        # add positive
        p = self.results['Positive']
        if (p == None):
            p = {classid: [filename]}
        else:
            if (classid in p):
                p[classid].append(filename)
            else:
                p[classid] = [filename]
        self.results['Positive'] = p
        self.positiveCount = self.positiveCount + 1
    
    def getFalsePositives(self):
        # get false positive
        fp = self.results['FalsePositive']
        return fp
    
    def getFalseNegatives(self):
        # get false negative
        fn = self.results['FalseNegative']
        return fn
    
    def getPositives(self):
        # get positive
        p = self.results['Positive']
        return p
    
    def getNegatives(self):
        # get negative
        n = self.results['Negative']
        return n

    def getSummary(self):
        # get summary
        summary = {}
        summary['starttime'] = self.starttime
        summary['endtime'] = self.endtime
        summary['FalsePositive'] = self.getFalsePositives()
        summary['Positive'] = self.getPositives()
        return summary