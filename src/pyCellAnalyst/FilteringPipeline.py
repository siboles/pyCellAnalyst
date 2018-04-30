import SimpleITK as sitk
from .Filters import Filter

class FilteringPipeline(object):
    def __init__(self, inputImage=None):
        self.inputImage = inputImage
        self.pipeline = []
        self.outputImages = []

    def addFilter(self, f):
        self.pipeline.append(f)

    def generateVisualSummary(self, indices=None):
        if indices is None:
            indices = range(len(self.outputImages))
        for i, ind in enumerate(indices):
            self.outputImages[ind].writeAsVTK(name='img{:02d}'.format(i))

    def execute(self):
        self.pipeline[0].inputImage = self.inputImage
        self.pipeline[0].execute()
        self.outputImages.append(self.pipeline[0].outputImage)
        for i, f in enumerate(self.pipeline[1:]):
            f.inputImage = self.outputImages[i]
            f.execute()
            self.outputImages.append(f.outputImage)

