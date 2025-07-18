from  prediction_lib import PowerSpectrum
# Example usage:
data_file_ase = "./aims.xyz" # can be any file,from an MD (AIMD or MLIP MD) run with the whole trajectory that ase can read - traj,xyz,extxyz, json ....  

# IR Spectrum
power_spectrum = PowerSpectrum(data_file_ase, resolution=1500)
power_spectrum.compute_spectrum(molecule_idx=0)
power_spectrum_data = power_spectrum.get_spectrum()
print("Power spectrum:", power_spectrum_data)

# Normalize the IR spectrum intensities between 0 and 1
power_frequencies, power_intensities = power_spectrum_data
# Filter frequencies greater than 100
filtered_indices_ = power_frequencies > 100
ref_frequencies_ = power_frequencies[filtered_indices_]
ref_filtered_intensities_ = power_intensities[filtered_indices_]

# Normalize the filtered intensities between 0 and 1
ref_normalized_ir_intensities_ = (ref_filtered_intensities_ - np.min(ref_filtered_intensities_)) / (np.max(ref_filtered_intensities_) - np.min(ref_filtered_intensities_))

# Plot the filtered and normalized IR spectrum
plt.figure(figsize=(9,5))
plt.plot(ref_frequencies_, ref_normalized_ir_intensities_, c='green', label='Power spectrum', linewidth=2)
plt.xlabel('Wave number (cm$^{-1}$)', fontsize=26)
plt.ylabel('Absorbance (a.u)', fontsize=26)
plt.legend(fontsize=14, loc='upper left')
plt.xticks(range(0, 4100, 500))
plt.xlim(0, 4000)
plt.tick_params(axis='y', which='both', length=0, labelsize=0)
plt.tick_params(axis='x', which='both', direction='inout', length=10, width=2, labelsize=22, bottom=True, top=True)
plt.tick_params(axis='x', which='minor', direction='inout', length=5, width=1)
plt.tight_layout()

plt.show()
