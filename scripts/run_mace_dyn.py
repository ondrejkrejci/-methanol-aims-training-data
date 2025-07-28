import os
from ase import units
from ase.io import *
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from mace.calculators.mace import MACECalculator



### INPUTS HERE ###

infile = 'pre_opt.xyz'
temperature = 700 # K
timestep = 0.5 # fs
friction = 0.01 #
nsave = 50 #
NSTEPS = 3000 # * timestep = 1.5 ps
path_save_config = "MD_out.traj"

device = 'cpu' # 'cuda'
cwd = os.getcwd()
mace_fname = os.path.join(cwd, "model_1/model_1.model")
default_dtype="float32"

#######################
atoms = read(infile)

#const = atoms.constraints[0].index
#print("constraints",const) # just checking!

MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

mace_calc = MACECalculator(
        model_paths=mace_fname,
        device=device,
        default_dtype=default_dtype,
    )

atoms.set_calculator( mace_calc )

dyn = Langevin(
        atoms=atoms,
        timestep=timestep * units.fs,
        temperature_K=temperature,
        friction=friction,
        logfile='md.log'
    )

def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


dyn.attach(printenergy, interval=nsave)
save_config = Trajectory(path_save_config, 'w', atoms)
dyn.attach(save_config, interval=nsave) #, dyn=dyn, fname=path_save_config)
printenergy()
dyn.run(NSTEPS)
