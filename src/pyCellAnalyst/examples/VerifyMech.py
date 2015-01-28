from pyCellAnalyst import CellMech
import numpy as np

ecm_truth = np.array([[-0.1003,-0.0353,0.0288],[-0.0353,-0.2974,0.0308],[0.0288,0.0308,-0.0301]])

cell_truth = [ np.array([[0.1050, -0.0393,0.0321],[-0.0393,-0.1203,0.0370],[0.0321, 0.0370, 0.1784]]),
np.array([[0.0861, -0.0152,0.0124],[-0.0152,-0.0045,0.0159],[0.0124, 0.0159, 0.1118]]),
np.array([[0.3005, -0.0505,0.0412],[-0.0505,0.0106, 0.0479],[0.0412, 0.0479, 0.3944]]),
np.array([[-0.2439,-0.0246,0.0200],[-0.0246,-0.3797,0.0208],[0.0200, 0.0208, -0.1940]]),
np.array([[0.2347, -0.0480,0.0392],[-0.0480,-0.0404,0.0452],[0.0392, 0.0452, 0.3246]]),
np.array([[-0.2067,-0.0419,0.0340],[-0.0419,-0.4226,0.0284],[0.0340, 0.0284, -0.1098]])]
proc = CellMech("MechVerification/Material","MechVerification/Spatial")
print "The ECM strain tensor:"
print proc.ecm_strain
true_principal = np.sort(np.linalg.eig(ecm_truth)[0])
calc_principal = np.sort(np.linalg.eig(proc.ecm_strain)[0])
ecm_error = abs(calc_principal-true_principal)
cell_error = []
for i,s in enumerate(proc.cell_strains):
    print "The strain tensor calculated for cell %d:" % i
    print proc.cell_strains[i]
    true_principal = np.sort(np.linalg.eig(cell_truth[i])[0])
    calc_principal = np.sort(np.linalg.eig(s)[0])
    cell_error.append(abs(calc_principal-true_principal))
print "Mean error in Cell Principal Strains"
print np.mean(np.array(cell_error),axis=0)
print np.std(np.array(cell_error),axis=0)
print "Max error in ECM Principal Strains"
print ecm_error
    


