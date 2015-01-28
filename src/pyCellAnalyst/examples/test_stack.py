from pyCellAnalyst import Volume

v = Volume('test_stack',pixel_dim=[0.41,0.41,0.3],regions=[[14,25,9,32,34,26],[16,6,11,36,28,25],[54,36,8,36,27,32],[60,19,6,33,37,34],[14,91,10,26,31,29],[6,76,11,29,30,26],[54,89,7,38,30,34],[60,71,9,36,28,32]],segmentation='Entropy',handle_overlap=True)

v.writeLabels()
v.writeSurfaces()
v.getDimensions()
