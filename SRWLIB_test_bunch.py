# -*- coding: utf-8 -*-
#############################################################################
# SRWLIB test code: Simulating synchrotron radiation for the full CLEAR beamline-
# consisting of 11 quadrupole magnets, 4 corrector magnets, and 2 dipole magnets
# v 0.01
#############################################################################

from __future__ import print_function
import os
from copy import deepcopy
from srwlib import *
from uti_plot import *
import time
import numpy as np  # Correctly import numpy
import csv
import matplotlib.pyplot as plt
import struct


print('SRWLIB Python Example 13 for Bunch instead of a single-particle beam:')
print('Simulating emission and propagation of Bending Magnet Synchrotron Radiaiton wavefront through a simple beamline')

# Data Folder and File Names
strExDataFolderName = 'data_example13_bunch'
strSpecOutFileName0 = 'bunch_res_spec.dat'
strIntOutFileName0 = 'bunch_res_int_se.dat'
strIntOutFileName1 = 'bunch_res_int_prop_se.dat'
strIntOutFileName2 = 'bunch_res_int_prop_me.dat'

# Bending Magnet
B = 0.4  # Dipole magnetic field [T]
LeffBM = 4.  # Magnet length [m] (exaggerated)
BM = SRWLMagFldM(B, 1, 'n', LeffBM)
magFldCnt = SRWLMagFldC([BM], [0], [0], [0])  # Container of magnetic field elements and their positions in 3D

# Electron Bunch Parameters
n = 1000  # Number of electrons in the bunch
eBeam = SRWLPartBeam()
eBeam.Iavg = 0.5 * n  # Average current [A]
#eBeam.partStatMom1 = SRWLPartBeamMom1()

eBeam.partStatMom1.x = 0.  # Initial horizontal position [m]
eBeam.partStatMom1.y = 0.  # Initial vertical position [m]
eBeam.partStatMom1.z = 0.  # Initial longitudinal position [m]
eBeam.partStatMom1.xp = 0.  # Initial horizontal angle [rad]
eBeam.partStatMom1.yp = 0.  # Initial vertical angle [rad]


#beam parameters are initialized with randomization, each parameter is assigned an array of values corresponding 
#to the spread of particles within the bunch
'''eBeam.partStatMom1.x = np.random.normal(loc=0., scale=0.1, size=n)
eBeam.partStatMom1.y = np.random.normal(loc=0., scale=0.1, size=n)
eBeam.partStatMom1.z = np.random.normal(loc=0., scale=0.1, size=n)
eBeam.partStatMom1.xp = np.random.normal(loc=0., scale=0.1, size=n)
eBeam.partStatMom1.yp = np.random.normal(loc=0., scale=0.1, size=n)'''

eBeam.partStatMom1.gamma = 3. / 0.51099890221e-03  # Relative energy


# Introduce energy spread
energy_spread = 0.01  # Relative energy spread
for i in range(n):
    eBeam.partStatMom1.gamma += energy_spread * (2 * np.random.random() - 1)  # Random spread around central energy


# Randomly distribute energies around the central energy for each particle in the bunch
#gamma_values = eBeam.partStatMom1.gamma + energy_spread * (2 * np.random.rand(np) - 1)
#gamma_values = eBeam.partStatMom1.gamma + energy_spread * (2 * np.random.rand(n) - 1)

# Set bunch statistical moments
#eBeam.partStatMom1.gamma = np.mean(gamma_values)  # Set the mean energy
#eBeam.arStatMom2[10] = np.var(gamma_values) / eBeam.partStatMom1.gamma**2  # Set energy spread

# New grid size and resolution
grid_size_x = 75e-3  # 75 mm in meters
grid_size_y = 75e-3  # Assuming square grid for simplicity

# New resolution (adjust as needed, e.g., 1024, 2048, etc.)
nx_new = 100 #2048
ny_new = 100 #2048

# Introduce transverse emittance
emittance_x = 10e-6  # Horizontal emittance [m*rad]
emittance_y = 10e-6  # Vertical emittance [m*rad]
# Wavefront for Spectral Intensity
wfrSp = SRWLWfr()
wfrSp.allocate(500, 1, 1)
wfrSp.mesh.zStart = 5.  # Longitudinal position for initial wavefront [m]
wfrSp.mesh.eStart = 0.1  # Initial photon energy [eV]
wfrSp.mesh.eFin = 10000.  # Final photon energy [eV]
wfrSp.partBeam = eBeam

# Wavefront for Initial Wavefront
wfr = SRWLWfr()
wfr.allocate(1, 10, 10)
distSrcLens = 5.  # Distance from geometrical source point to lens [m]
wfr.mesh.zStart = distSrcLens  # Longitudinal position for initial wavefront [m]
wfr.mesh.eStart = 0.123984  # Initial photon energy [eV]
wfr.mesh.eFin = wfr.mesh.eStart  # Final photon energy [eV]
horAng = 0.03  # Horizontal angle [rad]
#wfr.mesh.xStart = -0.5 * horAng * distSrcLens  # Initial horizontal position [m]
#wfr.mesh.xFin = 0.5 * horAng * distSrcLens  # Final horizontal position [m]
verAng = 0.02  # Vertical angle [rad]
#wfr.mesh.yStart = -0.5 * verAng * distSrcLens  # Initial vertical position [m]
#wfr.mesh.yFin = 0.5 * verAng * distSrcLens  # Final vertical position [m]
wfr.partBeam = eBeam

# Define the mesh for the wavefront
wfr.mesh.xStart = -grid_size_x / 2  # Start position in x
wfr.mesh.xFin = grid_size_x / 2     # End position in x
wfr.mesh.yStart = -grid_size_y / 2  # Start position in y
wfr.mesh.yFin = grid_size_y / 2     # End position in y
wfr.mesh.nx = nx_new                # Number of points in x
wfr.mesh.ny = ny_new                # Number of points in y

print("initial nx: ", wfr.mesh.nx)
#print("ny: ", wfr.mesh.ny)
#print("xStart: ", wfr.mesh.xStart)
#print("xFin: ", wfr.mesh.xFin)
#print("yStart: ", wfr.mesh.yStart)
#print("yFin: ", wfr.mesh.yFin)

# Update step sizes
dx = grid_size_x / (nx_new - 1)
dy = grid_size_y / (ny_new - 1)

# Save initial mesh
initial_mesh = deepcopy(wfr.mesh)

# Optical Elements and Propagation Parameters
distLensImg = distSrcLens
focLen = wfr.mesh.zStart * distLensImg / (distSrcLens + distLensImg)
optLens = SRWLOptL(_Fx=focLen, _Fy=focLen)
optDrift = SRWLOptD(distLensImg)
propagParLens = [1, 1, 1., 0, 0, 1., 2., 1., 2., 0, 0, 0]
propagParDrift = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

optBL = SRWLOptC([optLens, optDrift], [propagParLens, propagParDrift])

# Bunch trajectory
trj = SRWLPrtTrj()
trj.allocate(n)
trj.arX = [eBeam.partStatMom1.x] * n
trj.arY = [eBeam.partStatMom1.y] * n
trj.arZ = [eBeam.partStatMom1.z] * n
trj.arXp = [eBeam.partStatMom1.xp] * n
trj.arYp = [eBeam.partStatMom1.yp] * n
trj.arE = [eBeam.partStatMom1.gamma] * n

# Perform SR Calculation for Spectral Intensity
print('Performing initial SR spectrum calculation...')
srwl.CalcElecFieldSR(wfrSp, 0, magFldCnt, [2, 0.01, 0, 0, 20000, 1, -1])
print('   Extracting intensity and saving it to a file ...')
meshSp = deepcopy(wfrSp.mesh)
arSp = array('f', [0] * meshSp.ne)
srwl.CalcIntFromElecField(arSp, wfrSp, 6, 0, 0, meshSp.eStart, 0, 0)
srwl_uti_save_intens_ascii(arSp, meshSp, os.path.join(os.getcwd(), strExDataFolderName, strSpecOutFileName0))

# Restore initial mesh after SRW calculations
wfr.mesh = deepcopy(initial_mesh)
print("new nx value: ", wfr.mesh.nx)

# Perform SR Calculation for Initial Wavefront
print('Performing initial electric field wavefront calculation ...')
srwl.CalcElecFieldSR(wfr, 0, magFldCnt, [2, 0.01, 0, 0, 20000, 1, 0.8])
print('   Extracting intensity and saving it to a file ...')
mesh0 = deepcopy(wfr.mesh)
arI0 = array('f', [0] * mesh0.nx * mesh0.ny)
srwl.CalcIntFromElecField(arI0, wfr, 6, 0, 3, mesh0.eStart, 0, 0)
srwl_uti_save_intens_ascii(arI0, mesh0, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName0))

# Restore initial mesh after SRW calculations
wfr.mesh = deepcopy(initial_mesh)
print("1st post-calculation nx value: ", wfr.mesh.nx)

# Propagate Wavefront
print('Simulating electric field wavefront propagation ...')
srwl.PropagElecField(wfr, optBL)
print('   Extracting intensity and saving it to a file ...')

# Restore initial mesh after SRW calculations
#wfr.mesh = deepcopy(initial_mesh)
print("2nd post-calculation nx value: ", wfr.mesh.nx)

mesh1 = deepcopy(wfr.mesh)
arI1s = array('f', [0] * mesh1.nx * mesh1.ny)
srwl.CalcIntFromElecField(arI1s, wfr, 6, 0, 3, mesh1.eStart, 0, 0)
srwl_uti_save_intens_ascii(arI1s, mesh1, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName1))

# Plotting
uti_plot1d(arSp, [meshSp.eStart, meshSp.eFin, meshSp.ne], labels=('Photon Energy', 'Spectral Intensity', 'On-Axis SR Intensity Spectrum'), units=['eV', 'ph/s/0.1%bw/mm^2'])
unitsIntPlot = ['m', 'm', 'ph/s/.1%bw/mm^2']
uti_plot2d1d(arI0, [mesh0.xStart, mesh0.xFin, mesh0.nx], [mesh0.yStart, mesh0.yFin, mesh0.ny], labels=('Horizontal position', 'Vertical position', 'Intensity Before Lens'), units=unitsIntPlot)
uti_plot2d1d(arI1s, [mesh1.xStart, mesh1.xFin, mesh1.nx], [mesh1.yStart, mesh1.yFin, mesh1.ny], labels=('Horizontal position', 'Vertical position', 'Single-E Intensity in Image Plane'), units=unitsIntPlot)
uti_plot_show()

print("final post-calculation nx value: ", wfr.mesh.nx)


#******** EXTRA part Starts here on#********

#*********Saving the electric field information (real and imaginary)

# Define a function to save the electric field data to CSV files
def save_electric_field_to_csv(wfr, filename_real, filename_imag):
    nx = wfr.mesh.nx
    ny = wfr.mesh.ny
    ne = wfr.mesh.ne
    
    
    # Calculate the photon energy values
    photon_energies = np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, ne)

    # Calculate position grids
    x_grid = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin, nx)
    y_grid = np.linspace(wfr.mesh.yStart, wfr.mesh.yFin, ny)
    
    #print("x_grid = ", x_grid)
    #print("y_grid = ", y_grid)


    # Extract the real and imaginary parts of the electric field and save to CSV files
    with open(filename_real + '_Ex.csv', mode='w', newline='') as file_real_ex, \
         open(filename_real + '_Ey.csv', mode='w', newline='') as file_real_ey, \
         open(filename_imag + '_Ex.csv', mode='w', newline='') as file_imag_ex, \
         open(filename_imag + '_Ey.csv', mode='w', newline='') as file_imag_ey:

        writer_real_ex = csv.writer(file_real_ex)
        writer_real_ey = csv.writer(file_real_ey)
        writer_imag_ex = csv.writer(file_imag_ex)
        writer_imag_ey = csv.writer(file_imag_ey)

        # Write headers
        writer_real_ex.writerow(['Photon Energy', 'Vertical Position', 'Horizontal Position', 'Real Ex'])
        writer_real_ey.writerow(['Photon Energy', 'Vertical Position', 'Horizontal Position', 'Real Ey'])
        writer_imag_ex.writerow(['Photon Energy', 'Vertical Position', 'Horizontal Position', 'Imaginary Ex'])
        writer_imag_ey.writerow(['Photon Energy', 'Vertical Position', 'Horizontal Position', 'Imaginary Ey'])

        for ie in range(ne):
            for iy in range(ny):
                for ix in range(nx):
                    idx = 2 * (ix + iy * nx + ie * nx * ny)
                    x_pos = x_grid[ix]
                    y_pos = y_grid[iy]

                    real_ex = wfr.arEx[idx]     # Real part
                    imag_ex = wfr.arEx[idx + 1] # Imaginary part
                    real_ey = wfr.arEy[idx]     # Real part
                    imag_ey = wfr.arEy[idx + 1] # Imaginary part

                    writer_real_ex.writerow([photon_energies[ie], y_pos, x_pos, real_ex])
                    writer_real_ey.writerow([photon_energies[ie], y_pos, x_pos, real_ey])
                    writer_imag_ex.writerow([photon_energies[ie], y_pos, x_pos, imag_ex])
                    writer_imag_ey.writerow([photon_energies[ie], y_pos, x_pos, imag_ey])

# Example usage
save_electric_field_to_csv(wfr, 'data_example13_bunch/electric_field_real', 'data_example13_bunch/electric_field_imag')

# Plot the electric field data
def plot_electric_field(wfr, component='Ex', part='real'):
    nx = wfr.mesh.nx
    ny = wfr.mesh.ny
    # Use the values directly
    #nx = nx_new
    #ny = ny_new
    #print("nx value check: ", wfr.mesh.nx)

    ne = wfr.mesh.ne
    
    # Initialize arrays to store the electric field component
    arE = np.zeros((ny, nx), dtype=np.float64)

    for iy in range(ny):
        for ix in range(nx):
            idx = 2 * (ix + iy * nx)
            if component == 'Ex':
                if part == 'real':
                    arE[iy, ix] = wfr.arEx[idx]
                elif part == 'imag':
                    arE[iy, ix] = wfr.arEx[idx + 1]
            elif component == 'Ey':
                if part == 'real':
                    arE[iy, ix] = wfr.arEy[idx]
                elif part == 'imag':
                    arE[iy, ix] = wfr.arEy[idx + 1]

    plt.figure(figsize=(10, 8))
    plt.imshow(arE, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin], aspect='auto')
    plt.colorbar(label=f'{component} ({part})')
    plt.title(f'{component} ({part}) Electric Field')
    plt.xlabel('Horizontal Position [m]')
    plt.ylabel('Vertical Position [m]')
    plt.show()

# Example plotting
plot_electric_field(wfr, component='Ex', part='real')
plot_electric_field(wfr, component='Ex', part='imag')
plot_electric_field(wfr, component='Ey', part='real')
plot_electric_field(wfr, component='Ey', part='imag')

print(f"nx: {wfr.mesh.nx}, ny: {wfr.mesh.ny}, ne: {wfr.mesh.ne}")

#srwl.CalcElecFieldSR(wfrSp, 0, magFldCnt, arPrecSR)
print('Electric field array (Ex):', wfr.arEx[:10])  # Print first 10 elements
print('Electric field array (Ey):', wfr.arEy[:10])  # Print first 10 elements

#***************************************************************************************************

# Attempt of writing the extracted field info into a ZBF file
# Function to write ZBF file
def write_ZBF(nx, ny, dx, dy, isPol, lam, ExR, ExI, EyR, EyI, gamma, filename):
    is_polarized = isPol  # use 1 for a polarized beam
    unittype = 0  # 3 for m, 0 for mm
    wavelength = lam  # in meters
    
    rayleigh = (gamma * gamma * lam) / (2 * np.pi)
    ficl = 0.0  # unused for this ZBF
    position = 0.0
    waist = (gamma * lam) / (2 * np.pi)
    if unittype == 0:
        wavelength = wavelength * 1000
        rayleigh = rayleigh * 1000
        waist = waist * 1000
        dx = dx * 1000
        dy = dy * 1000

    print("Creating ZBF file...")

    print("Allocating memory for beam...")
    numbytes = np.dtype(np.float64).itemsize * (2 * (nx * ny) + 1)
    print(numbytes)
    cax = np.zeros(numbytes // np.dtype(np.float64).itemsize, dtype=np.float64)

    cay = None
    if is_polarized == 1:
        cay = np.zeros(numbytes // np.dtype(np.float64).itemsize, dtype=np.float64)

    print("Putting SR data in beam...")
    xc = 1 + nx // 2
    yc = 1 + ny // 2
    k = 0
    for j in range(ny):
        for i in range(nx):
            cax[k] = ExR[j, i]  # real part
            cax[k + 1] = ExI[j, i]  # imaginary part
            if is_polarized:
                cay[k] = EyR[j, i]  # real part
                cay[k + 1] = EyI[j, i]  # imaginary part
            k += 2

    print(f"Opening beam file \"{filename}\" for writing...")
    open(filename, 'w').close()
    with open(filename, "ab") as out:
        # Write the format version number
        i = 0
        out.write(i.to_bytes(4, 'little'))

        # Write the beam nx size
        out.write(nx.to_bytes(4, 'little'))

        # Write the beam ny size
        out.write(ny.to_bytes(4, 'little'))

        # Write the is_polarized flag
        out.write(is_polarized.to_bytes(4, 'little'))

        # Write out the current units
        out.write(unittype.to_bytes(4, 'little'))

        # 4 unused integers
        i = 0
        out.write(i.to_bytes(4, 'little'))
        out.write(i.to_bytes(4, 'little'))
        out.write(i.to_bytes(4, 'little'))
        out.write(i.to_bytes(4, 'little'))

        # dx
        out.write(struct.pack('d', dx))

        # dy
        out.write(struct.pack('d', dy))

        # Beam parameters
        out.write(struct.pack('d', position))
        out.write(struct.pack('d', rayleigh))
        out.write(struct.pack('d', wavelength))
        out.write(struct.pack('d', waist))

        # Fiber coupling
        out.write(struct.pack('d', ficl))

        # 4 unused doubles
        x = 0.0
        out.write(struct.pack('d', x))
        out.write(struct.pack('d', x))
        out.write(struct.pack('d', x))
        out.write(struct.pack('d', x))

        # Now the beam itself 
        cax.astype(np.float64).tofile(out)
        if is_polarized == 1:
            cay.astype(np.float64).tofile(out)
    print(f'nx {nx}, ny {ny}, shape {(2 * nx * ny)}')
    print(" !\n\n\n")

# Extracting wavefront data from SRW (example)
# Assuming wfr is your SRW wavefront object
nx = wfr.mesh.nx
ny = wfr.mesh.ny 
# Use the values directly
#nx = nx_new
#ny = ny_new

print(f"nx_new: {nx_new}, ny_new: {ny_new}")

dx = (wfr.mesh.xFin - wfr.mesh.xStart) / (nx - 1)
dy = (wfr.mesh.yFin - wfr.mesh.yStart) / (ny - 1)

freq = 0.0605 #3e15 # in PetaHz
#sourceE = srwl_uti_ph_en_conv(freq, 'Hz', 'eV')
sourceE = 2.5 # in eV
print('energy = ', sourceE) # photon energy in eV
#dis = 1.678 #Distance from geometrical source point to lens [m]
#lam = (1.242E-6)/sourceE      #λ(m)= hc/E(eV)
lam = 4.95936e-7 #in meters
print('wavelength = ', lam) # wavelength in meters
beamE = 200 
mass = 0.511
#pol = 0
gamma = beamE/mass

#lam = 1.24e-10  # Wavelength in meters (example value, adjust as needed)
#gamma = 1.0  # Example gamma value (adjust as needed)

# Extracting real and imaginary parts of Ex and Ey
ExR = np.array(wfr.arEx[::2]).reshape((ny, nx))
ExI = np.array(wfr.arEx[1::2]).reshape((ny, nx))
EyR = np.array(wfr.arEy[::2]).reshape((ny, nx))
EyI = np.array(wfr.arEy[1::2]).reshape((ny, nx))

# Determine if the beam is polarized
isPol = 1 if np.any(EyR) or np.any(EyI) else 0

# Call the function to write the ZBF file
filename = "data_example13_bunch/output_wavefront.zbf"
write_ZBF(nx, ny, dx, dy, isPol, lam, ExR, ExI, EyR, EyI, gamma, filename)


print("final nx value: ", wfr.mesh.nx)
print("final ny value: ", wfr.mesh.ny)

