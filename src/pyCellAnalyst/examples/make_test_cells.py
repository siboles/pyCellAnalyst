import gts,subprocess,random
import numpy as np
ini_pos = [[-10,0,0],
[0,-10,0],
[10,0,0],
[0,10,0],
[0,0,10],
[0,0,-10]]

fin_pos = [[-15,0,0],
[0,-15,0],
[15,0,0],
[0,15,0],
[0,0,4.4444],
[0,0,-4.4444]]
datf = open("TestStrains.dat",'w')
for i in xrange(6):
    F = np.zeros((3,3),float)
    for j in xrange(3):
        for k in xrange(3):
            if j == k:
                F[j,k] = random.uniform(.5,1.5)
            else:
                F[j,k] = 0.0
                #F[j,k] = random.uniform(0,.2)
    #F[1,0] = 0; F[2,0] = 0; F[2,1] = 0
    cell = gts.sphere(4)
    scalex = random.uniform(3,5)
    scaley = random.uniform(3,5)
    scalez = random.uniform(.8,1.5)
    cell.scale(dx=scalex,dy=scaley,dz=scalez)
    cell.translate(dx=ini_pos[i][0],dy=ini_pos[i][1],dz=ini_pos[i][2])
    fname = "MechVerification/Material/cell%02d" % i
    fid = open(fname+'.gts', 'w')
    cell.write(fid)
    fid.close()
    subprocess.call("gts2stl < "+fname+".gts > "+fname+".stl",shell=True)

    cell = gts.sphere(4)
    cell.scale(dx=scalex,dy=scaley,dz=scalez)
    for v in cell.vertices():
        new_coords = np.dot(F,np.array([[v.x],[v.y],[v.z]],float))
        v.set(new_coords[0],new_coords[1],new_coords[2])
    cell.translate(dx=fin_pos[i][0],dy=fin_pos[i][1],dz=fin_pos[i][2])
    fname = "MechVerification/Spatial/cell%02d" % i
    fid = open(fname+'.gts', 'w')
    cell.write(fid)
    fid.close()
    subprocess.call("gts2stl < "+fname+".gts > "+fname+".stl",shell=True)

    E = 0.5*(np.dot(F.T,F)-np.eye(3))

    datf.write("The exact strain tensor for cell %d:\n" % i)
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[0,0], E[0,1], E[0,2]))
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[1,0], E[1,1], E[1,2]))
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[2,0], E[2,1], E[2,2]))

subprocess.call("rm MechVerification/Material/*.gts",shell=True)
subprocess.call("rm MechVerification/Spatial/*.gts",shell=True)





