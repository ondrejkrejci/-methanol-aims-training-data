import os
from ase import units
from ase.io import *
from ase.optimize import BFGS
from mace.calculators.mace import MACECalculator

### INPUTS HERE ###

infile = 'pre_opt.xyz'
fmax = 0.03 # 0.05 - loose; 0.01 - precise - but for DFT
steps = 100 # maximal number of optimization steps

device = 'cpu' # 'cuda'
cwd = os.getcwd()
mace_fname = os.path.join(cwd, "model_1/model_1.model")
default_dtype="float32"

#######################
atoms = read(infile)

#const = atoms.constraints[0].index
#print("constraints",const) # just checking!

mace_calc = MACECalculator(
        model_paths=mace_fname,
        device=device,
        default_dtype=default_dtype,
    )

atoms.set_calculator( mace_calc )

opt = BFGS(atoms, trajectory='opt_MACE.traj')
opt.run(fmax=fmax,steps=steps)
energy = atoms.get_potential_energy()
print("opt_energy:",energy)