import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase import Atoms
from ase.units import _c
from ase import units as ase_units
#import pandas as pd

# auto-correlation code

__all__ = ["VibrationalSpectrum", "PowerSpectrum", "IRSpectrum"]


def cosine_sq_window(n_points: int):
    points = np.arange(n_points)
    window = np.cos(np.pi * points / (n_points - 1) / 2) ** 2
    return window


def fft_autocorrelation(data: np.array, n_lags: int):
    print("data,mean,std",data,np.mean(data),np.std(data));exit()
    data = (data - np.mean(data)) / np.std(data)
    n_points = data.shape[0]
    fft_forward = np.fft.fft(data, n=2 * n_points)
    fft_autocorr = fft_forward * np.conjugate(fft_forward)
    fft_backward = np.fft.ifft(fft_autocorr)[:n_points] / n_points
    autocorrelation = np.real(fft_backward[: n_lags + 1])
    return autocorrelation

class VibrationalSpectrum:
    def __init__(self, data_file: str, resolution: int = 4096, window: callable = cosine_sq_window):
        self.data_file = data_file
        self.resolution = resolution
        self.window = window
        
        self.timestep = 0.5
        spectral_range = 0.5 / self.timestep / (_c / 1e13)
        spectral_resolution = spectral_range / (4 * resolution)
        logging.info(
            "Spectral resolutions: {:12.3f} [cm^-1]".format(spectral_resolution)
        )
        logging.info("Spectral range:       {:12.3f} [cm^-1]".format(spectral_range))

        self.res = spectral_resolution
        self.frequencies = []
        self.intensities = []

        
    def compute_spectrum(self, molecule_idx: int = 0):
        relevant_data = self._get_data(molecule_idx)
        print("relevant_data",relevant_data)
        autocorrelation = self._compute_autocorrelations(relevant_data)
        autocorrelation = self._process_autocorrelation(autocorrelation)

        self.frequencies = []
        self.intensities = []

        for autocorr in autocorrelation:
            frequencies, intensities = self._compute_spectrum(autocorr)
            self.frequencies.append(frequencies)
            self.intensities.append(intensities)

        self._process_spectrum()

    
        
    def _compute_spectrum(self, autocorrelation: np.array):
        """
        Compute the spectrum from the autocorrelation function.

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.

        Returns:
            (numpy.array,numpy.array):
                frequencies:
                    Vibrational frequencies in inverse centimeters.
                intensities:
                    Intensities of the vibrational bands.
        """
        print("self.resolution",self.resolution)
        data = autocorrelation[: self.resolution]

        # Various tricks for nicer spectra
        # 1) Apply window function
        n_unpadded = data.shape[0]
        if self.window is not None:
            data *= self.window(n_unpadded)
        # 2) Zero padding
        data_padded = np.zeros(4 * n_unpadded)
        data_padded[:n_unpadded] = data
        # 3) Mirror around origin
        data_mirrored = np.hstack((np.flipud(data_padded), data_padded))
        # Compute the spectrum
        n_fourier = 8 * n_unpadded
        intensities = np.abs(
            self.timestep * np.fft.fft(data_mirrored, n=n_fourier)[: n_fourier // 2]
        )
        frequencies = np.arange(n_fourier // 2) / (n_fourier * self.timestep)
        # Conversion to inverse cm
        frequencies /= ase_units._c / 1e13
        return frequencies, intensities

    @staticmethod
    def _compute_autocorrelations(data: np.array):
        """
        Compute the autocorrelation function of the data. A separate autocorrelation is computred
        for every array dimension except the first axis.

        Args:
            data (numpy.array): Function array.

        Returns:
            numpy.array: Autocorrelation of the inputs.
        """
        n_data = data.shape[0]
        data_dim = data.shape[1:]
        n_lags = n_data - 2

        # Flatten data for easier iteration
        reshaped_data = data.reshape((n_data, -1))
        n_fields = reshaped_data.shape[1]

        # Compute all individual autocorrelations
        autocorrelations = np.zeros((n_fields, n_lags + 1))
        for field in range(n_fields):
            autocorrelations[field, ...] = fft_autocorrelation(
                reshaped_data[..., field], n_lags
            )

        # Reconstruct shape of original property
        autocorrelations = autocorrelations.reshape((*data_dim, -1))
        return autocorrelations
    
    def _get_data(self, molecule_idx: int):
        """
        Placeholder for extracting teh required data from the HDF5 dataset.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
                                Uses the same conventions as schnetpack.md.System.
        """
        raise NotImplementedError

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Placeholder for postprocessing the autocorrelation functions (e.g. weighting).

        Args:
            autocorrelation (numpy.array): Autorcorrelation function.
        """
        raise NotImplementedError

    def _process_spectrum(self):
        """
        Placeholder function if postprocessing should be applied to the spectrum (e.g. quantum coorections).
        """
        pass

    def get_spectrum(self):
        """
        Returns all computed spectra in the form of a list of tuples of frequencies and intensities.

        Returns:
            list: List of tuples of frequencies and intensities of all computed spectra.
        """
        spectrum = list(zip(self.frequencies, self.intensities))
        if len(spectrum) == 1:
            return spectrum[0]
        else:
            return spectrum

class PowerSpectrum(VibrationalSpectrum):
    """
    Compute power spectra from a molecular dynamics ASE extxyz file.

    Args:
        data_file (str): Path to the ASE extxyz file containing velocity data.
        resolution (int, optional): Resolution used when computing the spectrum.
    """

    def __init__(self, data_file: str, resolution: int = 4096, velocity_handle: str = "velocities"):
        super(PowerSpectrum, self).__init__(data_file, resolution=resolution)
        self.velocity_handle = velocity_handle

    def _get_data(self, molecule_idx: int = None):
        """
        Extract molecular velocities from the ASE dataset.

        Args:
            molecule_idx (int): Index of the molecule for which the spectrum should be computed.
        
        Returns:
            numpy.array: Array holding molecular velocities.
        """
        file_extension = os.path.splitext(self.data_file)[-1]
        
        if file_extension in [".xyz", ".traj", ".pdb", ".in", ".out", ".extxyz"]:  # Add other ASE-supported extensions if needed
            # Read from ASE file
            atoms = read(self.data_file, index=":")
            
            # Initialize an array to store velocities for all molecules
            velocities_array = []
        
            # Iterate through each molecule in the 'atoms' list
            for molecule in atoms:
                # Obtain the velocities for the current molecule
                velocities = molecule.get_velocities()
            
                # Append the velocities to the array
                velocities_array.append(velocities)
        
            # Convert the list of velocities to a NumPy array
            velocities_array = np.array(velocities_array)
        
        elif file_extension == ".txt":
            # Read from .txt file if velocities are stored in text format
            velocities_array = np.array(np.loadtxt(self.data_file))
        
        else:
            raise ValueError("Unsupported file type: {}".format(file_extension))

        # If needed, apply preprocessing to velocities (e.g., central differences or other)
        relevant_data = velocities_array 
        return relevant_data

    def _process_autocorrelation(self, autocorrelation: np.array):
        """
        Sum over number of atoms and the three Cartesian components.

        Args:
            autocorrelation (numpy.array): Autocorrelation function.
A        
        Returns:
            numpy.array: Updated autocorrelation.
        """
        vdos_autocorrelation = np.sum(autocorrelation, axis=1)
        vdos_autocorrelation = np.mean(vdos_autocorrelation, axis=0)
        return [vdos_autocorrelation]
    
    
class IRSpectrum(VibrationalSpectrum):
    def __init__(self, data_file: str, resolution: int = 4096, dipole_moment_handle: str = "dipole"):
        super(IRSpectrum, self).__init__(data_file, resolution=resolution)
        self.dipole_moment_handle = dipole_moment_handle

    def _get_data(self, molecule_idx: int = None):
        file_extension = os.path.splitext(self.data_file)[-1]
        
        if file_extension in [".xyz", ".traj", ".pdb", ".in", ".out", ".extxyz"]:  # Add other ASE-supported extensions if needed
            # Read from ASE file
            atoms = read(self.data_file, index=":")
            
            # Initialize an array to store dipole moments for all molecules
            dipole_moments_array = []
        
            # Iterate through each molecule in the 'atoms' list
            for molecule in atoms:
                # Obtain the dipole moment for the current molecule
                dipole_moment = molecule.get_dipole_moment()
            
                # Append the dipole moment to the array
                dipole_moments_array.append(dipole_moment)
        
            # Convert the list of dipole moments to a NumPy array
            dipole_moments_array = np.array(dipole_moments_array)
        
        elif file_extension == ".txt":
            # Read from .txt file
            dipole_moments_array = np.array(np.loadtxt(self.data_file))
        
        else:
            raise ValueError("Unsupported file type: {}".format(file_extension))

        # Compute the time derivative using central differences if needed
        relevant_data = (dipole_moments_array[2:, ...] - dipole_moments_array[:-2, ...]) / (2 * self.timestep)
        
        return relevant_data

    def _process_autocorrelation(self, autocorrelation: np.array):
        dipole_autocorrelation = np.sum(autocorrelation, axis=0)
        return [dipole_autocorrelation]
