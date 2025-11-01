from acatal.tool import concentration2formula
import numpy as np
cons = np.loadtxt('generated_cons.txt', dtype=float)

formulas = concentration2formula(cons)
np.savetxt('formulas.txt', formulas, fmt='%s')
