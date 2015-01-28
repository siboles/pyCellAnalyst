from pyCellAnalyst import Volume
import xlrd

#wb = xlrd.open_workbook('right_lat_Cond_All_regions.xlsx')
wb = xlrd.open_workbook('right_lat_Cond_All_regions_trunc.xlsx')

data_d = ["dat/ACL2_D/right_lat_cond/unloaded",
"dat/ACL2_D/right_lat_cond/10",
"dat/ACL2_D/right_lat_cond/15",
"dat/ACL2_D/right_lat_cond/20",
"dat/ACL2_D/right_lat_cond/30",
"dat/ACL2_D/right_lat_cond/40",
"dat/ACL2_D/right_lat_cond/60"]

names = wb.sheet_names()
for sn in xrange(len(names)):
    s = wb.sheet_by_name(names[sn])
    n = s.nrows
    bottom_left = []
    top_right = []
    for r in xrange(n-1): #will skip the first header row
        v = s.row_values(r+1,start_colx=1)
        if r%2 == 0:
            bottom_left.append([v[0],v[1]+v[3],v[4]])
        else:
            top_right.append([v[0]+v[2],v[1],v[4]])
    counter = 0
    for i in xrange(len(bottom_left)):
        vol = Volume(data_d[sn],tratio=0.1,pixel_dim=[0.411,0.411,0.698726714681/2],stain='cell',display=True,left=bottom_left[i],right=top_right[i],counter=counter)
        '''
        for j in xrange(len(vol.volumes)):
            print 'Cell %d' % (counter+j)
            print 'Volume: %3.4f um^3' % vol.volumes[j]
            print 'Ellipsoid axis closest to x-direction: %2.4f um' % vol.dimensions[j][0]
            print 'Ellipsoid axis closest to y-direction: %2.4f um' % vol.dimensions[j][1]
            print 'Ellipsoid axis closest to z-direction: %2.4f um' % vol.dimensions[j][2]
        counter += len(vol.volumes)
        '''
