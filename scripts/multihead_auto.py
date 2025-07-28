## training_auto.py

import os
import subprocess
from ase.io import read, write
import numpy as np
import time

##########################################################################
# parts to adjust in the begginig, then later in training part as well : #
# ***********************************************************************#

number_of_models=1
seeds_of_models=[42]

# paths for the training, (subset of) original training data, and test file; has to end with xyz, but extxyz format
cwd = os.getcwd()
training_data_path = os.path.join(cwd, "dft_results.xyz")
test_file_path = os.path.join(cwd, "new_test_data.xyz")

pt_data_path = os.path.join(cwd, "new_training_data.xyz")
foundation_model_path = os.path.join(cwd, "model_1/model_1.model")


# for data splitting if needed:
split = False
split_seed = 42
all_data_file = 'dft_results.extxyz'
#training starts from here

# here, what most of actual changes are needed
def train_models(num_models, seeds, training_data_path, test_file_path, pt_data_path, foundation_model_path):
    for i in range(num_models):
        seed = seeds[i]

        # Define the command for training
        command = f"""
        python3 "/projappl/project_2006995/MACE_gpu/scripts/run_train.py" \
            --name="model_multihead_{i + 1}" \
            --train_file="{training_data_path}" \
            --valid_fraction=0.05 \
            --test_file="{test_file_path}" \
            --E0s="{{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1485.3076, 8:-2043.5670}}" \
            --energy_key="energy" \
            --forces_key="forces" \
            --model="MACE" \
            --num_interactions=2 \
            --num_channels=128 \
            --max_L=2 \
            --num_cutoff_basis=5 \
            --correlation=2 \
            --r_max=5.0 \
            --forces_weight=1000 \
            --energy_weight=100 \
            --batch_size=1 \
            --valid_batch_size=1 \
            --eval_interval=1 \
            --max_num_epochs=200 \
            --error_table="PerAtomRMSE" \
            --ema \
            --ema_decay=0.99 \
            --amsgrad \
            --default_dtype="float32" \
            --device="cuda" \
            --restart_latest \
            --distributed \
            --save_cpu \
            --seed={seed} \
            --multiheads_finetuning=True \
            --foundation_model={foundation_model_path} \
            --pt_train_file={pt_data_path}
        """
        # For more info have a look at: 
        # https://colab.research.google.com/github/imagdau/Tutorials/blob/main/T01-MACE-Practice-I.ipynb
        # have a look at a second chapter there
        # Some other quick info:
        # energies should be normally defined like this:
        #    --E0s="{{1:-13.663181292231226, 6:-1029.2809654211628, 7:-1485.3076, 8:-2043.5670}}" \
        # But average could be used as well, but it is ideall for use of atomization energies, when the 1 atom energies are subtracted from the total (potential) energy, example can be seen in splitting_data.py script
        #    --EOs="average" \ 
        # If your energies are high, then you need to use "float64"
        # If your forces errors are low, but energies are not, try to use SWA, for some parts
        #    --swa \
        #    --start_swa=450 \
        #    --swa_forces_weight=100 \
        # Important is to look to loss function and number of epochs, piece of knowledge from the documentation:
        # An heuristic for initial settings, is to consider the number of gradient update constant to 200 000, which can be computed as $text{max-num-epochs}*frac{text{num-configs-training}}{text{batch-size}}$.
        # Here we have just a try to look, how it works 
        
        # ********************************************************************** #
        # parts to adjust ending here                                            #
        ##########################################################################
        
        # Create a folder for each model based on the seed
        model_folder = f"model_{i + 1}"
        if not os.path.exists(model_folder):
            os.makedirs(model_folder, exist_ok=True)

        # Change directory to the model folder
        os.chdir(model_folder)
        print(os.getcwd())

        # Run the training command
        subprocess.run(command, shell=True)

        # Move back to the original directory
        os.chdir("../")
        print(os.getcwd())

    print("Training models completed.")

if split:
    # Read the .extxyz file
    structures = read(all_data_file, format='extxyz', index=':')

    # Get the number of structures
    num_structures = len(structures)
    print(f"No. of structures: {num_structures}", flush=True)

    # Set a seed for reproducibility
    np.random.seed(split_seed)

    # Generate random indices
    random_indices = np.random.permutation(num_structures)

    # Specify the ratio for training and testing sets
    train_ratio = 0.8  # 80% for training
    test_ratio = 0.2   # 20% for testing

    # Calculate the number of structures for training and testing
    num_train = int(train_ratio * num_structures)
    num_test = int(test_ratio * num_structures)

    # Split the random indices into training and testing sets
    train_indices = random_indices[:num_train]
    test_indices = random_indices[num_train:num_train + num_test]

    # Create training and testing sets
    train_set = [structures[i] for i in train_indices]
    test_set = [structures[i] for i in test_indices]

    # Write the training set to a new .extxyz file
    write(training_data_path, train_set, format='extxyz')

    # Write the testing set to a new .extxyz file
    write(test_file_path, test_set, format='extxyz')

start = time.time()

# for training 1 model 
train_models(number_of_models, seeds_of_models, training_data_path, test_file_path, pt_data_path, foundation_model_path)

end = time.time()
print("Script completed in " + str(end - start) + " seconds")

