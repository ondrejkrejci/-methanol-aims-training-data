import os
import numpy as np
from ase.io import read,write

cwd = os.getcwd()

# set following #

ratio=0.9 # 0.9-0.8 makes sense 
original_data = os.path.join(cwd, "methanol_700K.extxyz")
training_data_path = os.path.join(cwd, "new_training_data.xyz")
test_file_path = os.path.join(cwd, "new_test_data.xyz")
atomic_energies = None 
# set to None if you do not want to calculate Atomization Energies
# example for otherwise:
#atomic_energies ={'H':-13.663181292231226, 'C':-1029.2809654211628, 'N':-1485.3076, 'O':-2043.5670}}

# Do not change afterwards, unless you know what you are doing #

total_test=[]
total_train=[]

def split_dataset(array, ratio):
    # Shuffle the array
    np.random.shuffle(array)
    
    # Calculate the split index
    split_index = int(len(array) * ratio)
    
    # Split the array
    train_set = array[:split_index]
    test_set = array[split_index:]
    
    return train_set, test_set

print(ratio,":",1-ratio)

for i in range(1):
    print("i:",i)
    configs = read(original_data,index=":")
    l=len(configs)
    
    print("number of geometries:",l)
    if atomic_energies is not None:
        for atoms in configs:
            total_energy = atoms.get_potential_energy()

            # Calculate the sum of atomic energies based on the atomic species
            atomic_energy_sum = sum(atomic_energies[symbol] for symbol in atoms.get_chemical_symbols())

            # Calculate the atomization energy
            ato_energy = total_energy - atomic_energy_sum

            # Store the Atomization energy in the ASE info dictionary
            atoms.info["atomization_energy"] = ato_energy
    print("geom[0]",configs[0])
    
    # Example usage
    #data = np.arange(l)
    train, test = split_dataset(configs[1:], ratio)
    
    #print("Train set:", train)
    #print("Test set:", test)
    
    #write("training_"+str(i)+".extxyz",train)
    #write("test_"+str(i)+".extxyz",test)
    for it in train:
        total_train.append(it)
    for it in test:
        total_test.append(it)

total_train.insert(0,configs[0]) # 0 is the fully-optized.geometry


print("total lengths train",len(total_train),"test",len(total_test))

write(training_data_path , total_train)
write(test_file_path , total_test)
