from pyCellAnalyst import Volume

vol = Volume("dat/real_cells",tratio=0.4,pixel_dim=[0.411,0.411,0.698726714681/2],stain='cell',display=False,left=[216,407,9],right=[335,305,28],counter=0)


for i in xrange(len(vol.volumes)):
    print 'Cell %d' %i
    print 'Volume: %3.4f um^3' % vol.volumes[i]
    print 'Ellipsoid axis closest to x-direction: %2.4f um' % vol.dimensions[i][0]
    print 'Ellipsoid axis closest to y-direction: %2.4f um' % vol.dimensions[i][1]
    print 'Ellipsoid axis closest to z-direction: %2.4f um' % vol.dimensions[i][2]

