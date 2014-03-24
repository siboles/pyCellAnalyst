from pyCellAnalyst import CellMech

proc = CellMech("MechVerification/Material","MechVerification/Spatial")
print "The ECM strain tensor:"
print proc.ecm_strain
for i in xrange(len(proc.cell_strains)):
    print "The strain tensor calculated for cell %d:" % i
    print proc.cell_strains[i]
