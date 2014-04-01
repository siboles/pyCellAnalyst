import gts,subprocess,random
import numpy as np
ini_pos = [[-10,0,0],
[0,-10,0],
[10,0,0],
[0,10,0],
[0,0,10],
[0,0,-10]]

while True:
    F = np.zeros((3,3),float)
    for j in xrange(3):
        for k in xrange(3):
            if j == k:
                F[j,k] = random.uniform(.5,1.5)
            else:
                F[j,k] = random.uniform(-0.2,0.2)
    if np.linalg.det(F) > 0:
        break
fin_pos = []
for i in xrange(6):
    fin_pos.append(list(np.dot(F,np.array(ini_pos[i],float))))
E = 0.5*(np.dot(F.T,F)-np.eye(3))
datf = open("TestStrains.dat",'w')
datf.write("The exact strain tensor for the ECM:\n")
datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[0,0], E[0,1], E[0,2]))
datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[1,0], E[1,1], E[1,2]))
datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[2,0], E[2,1], E[2,2]))
for i in xrange(6):
    a = np.trace(F)/3.
    while True:
        ldil = random.uniform(0.2,1.5)
        ldev = random.uniform(0.01,1.5)
        nF = ldil*a*np.eye(3)+ldev*(F-a*np.eye(3))
        if np.linalg.det(nF) > 0.1:
            break
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
        new_coords = np.dot(nF,np.array([[v.x],[v.y],[v.z]],float))
        v.set(new_coords[0],new_coords[1],new_coords[2])
    cell.translate(dx=fin_pos[i][0],dy=fin_pos[i][1],dz=fin_pos[i][2])
    fname = "MechVerification/Spatial/cell%02d" % i
    fid = open(fname+'.gts', 'w')
    cell.write(fid)
    fid.close()
    subprocess.call("gts2stl < "+fname+".gts > "+fname+".stl",shell=True)

    E = 0.5*(np.dot(nF.T,nF)-np.eye(3))

    datf.write("The exact strain tensor for cell %d:\n" % i)
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[0,0], E[0,1], E[0,2]))
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[1,0], E[1,1], E[1,2]))
    datf.write("%1.4f\t%1.4f\t%1.4f\n" % (E[2,0], E[2,1], E[2,2]))
    datf.write("The volume ratio for cell %d: %1.4f\n" % (i, np.linalg.det(nF)))


subprocess.call("rm MechVerification/Material/*.gts",shell=True)
subprocess.call("rm MechVerification/Spatial/*.gts",shell=True)
