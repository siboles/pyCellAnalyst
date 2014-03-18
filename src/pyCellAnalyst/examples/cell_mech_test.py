from pyCellAnalyst import CellMech
import numpy as np

proc = CellMech("dat/processed/0_strain","dat/processed/10_strain")

print proc.ecm_strain
print proc.cell_strains
    

