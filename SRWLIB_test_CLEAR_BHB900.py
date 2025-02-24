# -*- coding: utf-8 -*-
#############################################################################
# SRWLIB test code: Simulating synchrotron radiation for a partial CLEAR beamline-
# consisting of three quadrupole magnets, one corrector magnet, and a dipole magnet
# v 0.01
#############################################################################

from __future__ import print_function  # Python 2.7 compatibility
from srwlib import *
from uti_plot import *  # required for plotting
import time
import numpy as np
import csv
import matplotlib.pyplot as plt
import struct
import math
from scipy.ndimage import gaussian_filter1d # for smoothening
from scipy.integrate import simps # computes the approximation of a definite integral by Simpson's rule
import srwlpy as srwl


print('SRWLIB test code: Simulating Synchrotron Radiation for CLEAR Beamline- @BHB400 dipole')
#print('Consisting of three quadrupole magnets, one corrector magnet, and a dipole magnet')
print('status1')
# ***********Data Folder and File Names
strExDataFolderName = 'data_test_beamline_CLEAR' # test data sub-folder name
strSpecOutFileName = 'ex_beamline_res_spec.dat'  # file name for output SR spectrum vs photon energy data
strIntOutFileName0 = 'ex_beamline_res_int.dat'  # file name for output intensity vs horizontal position data
strIntOutFileName1 = 'ex13_res_int_prop_se.dat' #file name for output propagated single-electron SR intensity vs X and Y data
strIntOutFileName2 = 'ex13_res_int_prop_me.dat'  #file name for output propagated multi-electron SR intensity vs X and Y data


# ***********Quadrupole Magnets
B_quad = 0.3  # Quadrupole magnetic field [T]
L_quad = 0.29 # changed on 04.10.24 from previous value of 0.4  # Quadrupole magnet length [m]

quad1 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad2 = SRWLMagFldM(-B_quad, 1, 'n', L_quad)
quad3 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
print('status2')

# ***********Corrector Magnet
B_corr = 0.017 # changed on 04.10.24 from previous value of 0.2  # Corrector magnet magnetic field [T]
L_corr = 0.11  # Corrector magnet length in [m]

corr = SRWLMagFldM(B_corr, 1, 'n', L_corr)

# ***********Dipole Magnet
B_dipole = 0.4  # Dipole magnetic field [T]
L_dipole = 0.5  # Dipole magnet length [m]

dipole = SRWLMagFldM(B_dipole, 1, 'n', L_dipole)
magFldCnt = SRWLMagFldC([dipole], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D
##magFldCnt = SRWLMagFldC([quad1, quad2, quad3, corr, dipole], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0])
print('status3')


#BM = SRWLMagFldM(B, 1, 'n', LeffBM)
#magFldCnt = SRWLMagFldC([BM], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D

# ***********Electron Beam 
eBeam = SRWLPartBeam()
#eBeam.Iavg = 0.00000005 #changed to of 50 nanoAmps at 02/10/24; previously was too high at 0.5  # Average current [in A]
eBeam.Iavg = 0.0000000005 # 500 picoAmps
#eBeam.Iavg = 0.5 
# 1st order statistical moments:    
eBeam.partStatMom1.x = 0.  # Initial horizontal position of central trajectory [m]
eBeam.partStatMom1.y = 0.  # Initial vertical position of central trajectory [m]
eBeam.partStatMom1.z = 0.  # Initial longitudinal position of central trajectory [m]
eBeam.partStatMom1.xp = 0.  # Initial horizontal angle of central trajectory [rad]
eBeam.partStatMom1.yp = 0.  # Initial vertical angle of central trajectory [rad]
#eBeam.partStatMom1.gamma = 3. / 0.51099890221e-03  # Relative energy; correspond to a beam energy of about 3 GeV
eBeam.partStatMom1.gamma = 220. / 0.51099890221  # Relative energy for 220 MeV beam

# 2nd order statistical moments:
eBeam.arStatMom2[0] = (2.0e-03)**2 # changed 12.02.25 onward, previously was (127.346e-06)**2  # <(x-x0)^2> [m^2] : spots size sigma_x
eBeam.arStatMom2[1] = -10.85e-09  # <(x-x0)*(x'-x'0)> [m] : Coupling/correlation terms (between positions_x and angles_x)
eBeam.arStatMom2[2] = (2.0e-02)**2 # changed 12.02.25 onward, previously was (92.3093e-06)**2  # <(x'-x'0)^2> : Divergence(angular spread) sigma_x^'
eBeam.arStatMom2[3] = (2.0e-03)**2 # changed 12.02.25 onward, previously was (13.4164e-06)**2  # <(y-y0)^2> : spots size sigma_y
eBeam.arStatMom2[4] = 0.0072e-09  # <(y-y0)*(y'-y'0)> [m] : Coupling/correlation terms (between positions_y and angles_y)
eBeam.arStatMom2[5] = (2.0e-02)**2 # changed 12.02.25 onward, previously was (0.8022e-06)**2  # <(y'-y'0)^2> : Divergence(angular spread) sigma_y^'
eBeam.arStatMom2[10] = (0.89e-02)**2  # <(E-E0)^2>/E0^2 i.e. energy spread was 0.89%
print('status4')

# ***********Radiation Sampling for the On-Axis SR Spectrum
wfrSp = SRWLWfr()  # Wavefront structure (placeholder for data to be calculated)
wfrSp.allocate(500, 1, 1)  # Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)
#wfrSp.allocate(1, 1, 1)  # the ne value was changed for calculation (decreased on 24.10.24)
wfrSp.mesh.zStart = 1.  # Longitudinal position for initial wavefront [m]

wfrSp.mesh.eStart = 0.1 #2.175 # changed on 19/09/2024 to correspond 530 nm; previously was: 0.1  # Initial photon energy [eV]
wfrSp.mesh.eFin = 100 #2.339 # changed on 19/09/2024 to correspond 570 nm; previously was: 100 #10000.  # Final photon energy [eV]

wfrSp.mesh.xStart = 0.  # Initial horizontal position [m]
wfrSp.mesh.xFin = wfrSp.mesh.xStart  # Final horizontal position [m]
wfrSp.mesh.yStart = 0.  # Initial vertical position [m]
wfrSp.mesh.yFin = 0.  # Final vertical position [m]

wfrSp.partBeam = eBeam  # e-beam data is contained inside the wavefront struct

# ***********Radiation Sampling for the Initial Wavefront (before first optical element)
wfr = SRWLWfr()  # Wavefront structure (placeholder for data to be calculated)
#wfr.allocate(1, 50, 50)  # Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)
wfr.allocate(1, 450, 450)  # the nx and ny values were increased for the sake of calculation


distSrcLens = 1 #changed on 18/09/2024, old default value was: 5.  # Distance from geometrical source point to Lens/observing plane [in m]
wfr.mesh.zStart = distSrcLens  # Longitudinal position for initial wavefront [m]

#wfr.mesh.eStart = 2.254 # changed on 19/09/2024 to correspond 550 nm; previously was: 0.123984  # Initial photon energy [eV]
#wfr.mesh.eFin = wfr.mesh.eStart  # Final photon energy [eV]
wfr.mesh.eStart = 2.175
wfr.mesh.eFin = 2.339

charAng = 1 / eBeam.partStatMom1.gamma # characteristic angular spread of SR is on the order of 1/γ
horAng = 3 * charAng  # new value 5 times the characteristic angle
verAng = 3 * charAng  # new value as well

#horAng = 0.03  # Horizontal angle [rad]
wfr.mesh.xStart = -0.5 * horAng * distSrcLens  # Initial horizontal position [m]
wfr.mesh.xFin = 0.5 * horAng * distSrcLens  # Final horizontal position [m]
#verAng = 0.02  # Vertical angle [rad]
wfr.mesh.yStart = -0.5 * verAng * distSrcLens  # Initial ver tical position [m]
wfr.mesh.yFin = 0.5 * verAng * distSrcLens  # Final vertical position [m]

wfr.partBeam = eBeam  # e-beam data is contained inside the wavefront struct

print("horAng & verAng = ", horAng)
print("charAng= ", charAng)

print("wfr.mesh.xStart= ", wfr.mesh.xStart)
print("wfr.mesh.xFin= ", wfr.mesh.xFin)
print("wfr.mesh.yStart= ", wfr.mesh.yStart)
print("wfr.mesh.yFin= ", wfr.mesh.yFin)

# ***********Optical Elements and their Corresponding Propagation Parameters
distLensImg = distSrcLens  # Distance from lens to image plane
focLen = wfr.mesh.zStart * distLensImg / (distSrcLens + distLensImg)
print('distLensImg or distSrcLens value = ', distLensImg)
print('focal length value = ', focLen)
optLens = SRWLOptL(_Fx=focLen, _Fy=focLen)  # Thin lens with 100 mm focal length
optDrift = SRWLOptD(distLensImg) #Drift space from lens to image plane


# ********** Part for adding the extra element for zemax compatibality (i.e. thin paraxial lens)# **********
# Define the lens and drift space after the lens
focLenthin = 0.1  # 100 mm = 0.1 meters
optLensthin = SRWLOptL(_Fx=focLenthin, _Fy=focLenthin)  # Thin lens with 100 mm focal length
optDriftB = SRWLOptD(0.2) # i.e. 200 mm. distance               #SRWLOptD(distLensImg) #Drift space from lens to image plane
optDriftA = SRWLOptD(0.2) # 'B' represents before and 'A' represents after as Drift space from lens to source also need to be added
#Propagation paramaters (SRW specific)
#                [0][1][2] [3][4] [5] [6] [7] [8]
propagParLens =  [0, 0, 1., 0, 0, 1., 2., 1., 2., 0, 0, 0]
propagParDrift = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
optBL = SRWLOptC([optLens, optDrift], [propagParLens, propagParDrift]) # this has been commented into the code on 14/11/2024!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Propagate the wavefront through the lens and drift space
#srwlib.srwl.PropagElecField(wfr, optical_elements)
print('status5')

optQuad1 = SRWLOptL(L_quad)
optQuad2 = SRWLOptL(L_quad)
optQuad3 = SRWLOptL(L_quad)
optCorr = SRWLOptL(L_corr)
optDipole = SRWLOptL(L_dipole)
optDrift1 = SRWLOptD(0.5) #SRWLOptD(distLensImg)
optDrift2 = SRWLOptD(0.5) #SRWLOptD(distLensImg)

#propagParQuad1 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0] #this was the original or old one
propagParQuad1 =  [0, 0, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad2 =  [0, 0, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad3 =  [0, 0, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]

#propagParCorr =  [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0] #this was the original or old one
propagParCorr =  [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

#propagParDipole = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0] #this was the original or old one
propagParDipole = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

#propagParDrift1 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0] #this was the original or old one
propagParDrift1 = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift2 = [0, 0, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

beamline_elements = [optQuad1, optQuad2, optQuad3, optDrift1, optCorr, optDrift2, optDipole]
#now instead of that, modified version after adding a thin lens
#beamline_elements = [optQuad1, optQuad2, optQuad3, optDrift1, optCorr, optDrift2, optDipole, optDriftB, optLens, optDriftA]

propagation_parameters = [propagParQuad1, propagParQuad2, propagParQuad3, propagParDrift1, propagParCorr, propagParDrift2, propagParDipole]
#now instead of that, modified version after adding a thin lens
#propagation_parameters = [propagParQuad1, propagParQuad2, propagParQuad3, propagParDrift1, propagParCorr, propagParDrift2, propagParDipole, propagParDrift, propagParLens, propagParDrift]

print('status6')

#beamline_elements = [optQuad1, optDrift1, optDipole1, optDrift2, optCorr1]
#propagation_parameters = [propagParQuad1, propagParDrift1, propagParDipole1, propagParDrift2, pro  pagParCorr1]
#optBL = SRWLOptC(beamline_elements, propagation_parameters) # this has been commented out on 14/11/2024!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#optBL = SRWLOptC([optLens, optDrift], [propagParLens, propagParDrift]) 

print('status7')

# beamline - Container of optical elements (together with their corresponding wavefront propagation parameters / instructions)
#optBL = SRWLOptC([optQuad1, optQuad2, optQuad3, optCorr, optDipole])

print(f"n_x0: {wfr.mesh.nx}, n_y0: {wfr.mesh.ny}")

# Define the Stokes structure for polarization calculation
#stokes = srwl.SRWLStokes()
#stokes.allocate(101, 101, 1)  # Set grid for angles

# ***********BM SR Calculation
# Precision parameters
meth = 2  # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
relPrec = 0.005  # Relative precision
zStartInteg = 0  # Longitudinal position to start integration (effective if < zEndInteg)
zEndInteg = 0  # Longitudinal position to finish integration (effective if > zStartInteg)
npTraj = 20000  # Number of points for trajectory calculation
useTermin = 1  # Use "terminating terms" (i.e. asymptotic expansions at zStartInteg and zEndInteg) or not (1 or 0 respectively)

print('   Performing initial SR spectrum calculation ... ', end='')
t0 = time.time()
sampFactNxNyForProp = -1  # Sampling factor for adjusting nx, ny (effective if > 0)
arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp]
srwl.CalcElecFieldSR(wfrSp, 0, magFldCnt, arPrecSR)  # Calculating electric field
print('done in', round(time.time() - t0), 's')

print('   Extracting intensity and saving it to a file ... ', end='')
t0 = time.time()
meshSp = deepcopy(wfrSp.mesh)
arSp = array('f', [0] * meshSp.ne)  # "Flat" array to take 1D intensity data (vs E)
srwl.CalcIntFromElecField(arSp, wfrSp, 6, 0, 0, meshSp.eStart, 0, 0)  # Extracting intensity vs photon energy
srwl_uti_save_intens_ascii(arSp, meshSp, os.path.join(os.getcwd(), strExDataFolderName, strSpecOutFileName))
print('done in', round(time.time() - t0), 's')

print('   Performing initial electric field wavefront calculation ... ', end='')
t0 = time.time()
sampFactNxNyForProp = - 1 #0.8  # Sampling factor for adjusting nx, ny (effective if > 0)
arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp]
srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecSR)  # Calculating electric field
print('done in', round(time.time() - t0), 's')

print('status8')

print('   Extracting intensity and saving it to a file ... ', end='')
t0 = time.time()
mesh0 = deepcopy(wfr.mesh)
arI0 = array('f', [0] * mesh0.nx * mesh0.ny)  # "Flat" array to take 2D intensity data (vs X & Y)
srwl.CalcIntFromElecField(arI0, wfr, 6, 0, 3, mesh0.eStart, 0, 0)  # Extracting intensity vs horizontal and vertical positions
srwl_uti_save_intens_ascii(arI0, mesh0, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName0))
print('done in', round(time.time() - t0), 's')

print('status9')

# ***********Wavefront Propagation
print('   Simulating single-electron electric field wavefront propagation ... ', end='')
t0 = time.time()
srwl.PropagElecField(wfr, optBL)
print('done in', round(time.time() - t0), 's')

# ***********Extracting Intensity from Calculated Electric Field and Saving it to File
print('   Extracting intensity from calculated electric field and saving it to file ... ', end='')
t0 = time.time()
mesh1 = deepcopy(wfr.mesh)
arI1 = array('f', [0] * mesh1.nx * mesh1.ny)  # "Flat" array to take 2D single-electron intensity data (vs X & Y)
#mesh1.xStart = -0.001
#mesh1.yStart = -0.001
srwl.CalcIntFromElecField(arI1, wfr, 1, 0, 3, mesh1.eStart, 0, 0)  # Extracting single-electron intensity vs X & Y
srwl_uti_save_intens_ascii(arI1, mesh1, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName1))

arI1m = deepcopy(arI1) #"Flat" array to take 2D multi-electron intensity data (vs X & Y)
srwl.CalcIntFromElecField(arI1m, wfr, 1, 1, 3, mesh1.eStart, 0, 0) #Calculating multi-electron intensity vs X & Y using convolution method (assuming it to be valid!)
srwl_uti_save_intens_ascii(arI1m, mesh1, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName2))
print('done in', round (time.time() - t0), 's')

#arI1m_2d = arI1m.reshape(mesh1.ny, mesh1.nx)


# ***********Plotting the Calculation Results
uti_plot1d(arSp, [meshSp.eStart, meshSp.eFin, meshSp.ne], labels=('Photon Energy', 'Spectral Intensity', 'On-Axis SR Intensity Spectrum'), units=['eV', 'ph/s/0.1%bw/mm^2'])
#plt.xticks(plt.xticks()[0] * 1000)  # Convert keV back to eV for tick marks

unitsIntPlot = ['m', 'm', 'ph/s/.1%bw/mm^2']
uti_plot2d1d(arI0, [mesh0.xStart, mesh0.xFin, mesh0.nx], [mesh0.yStart, mesh0.yFin, mesh0.ny], labels=('Horizontal position', 'Vertical position', 'Intensity Before Lens'), units=unitsIntPlot)
uti_plot2d1d(arI1, [mesh1.xStart, mesh1.xFin, mesh1.nx], [mesh1.yStart, mesh1.yFin, mesh1.ny], labels=('Horizontal position', 'Vertical position', 'Single-E Intensity in Image Plane'), units=unitsIntPlot)
uti_plot2d1d(arI1m, [mesh1.xStart, mesh1.xFin, mesh1.nx], [mesh1.yStart, mesh1.yFin, mesh1.ny], labels=('Horizontal position', 'Vertical position', 'Multi-E Intensity in Image Plane'), units=unitsIntPlot)
uti_plot_show()  # show all graphs (blocks script execution; close all graph windows to proceed)


'''
file = "OSR_multiintensity.png"
# Full path for the file
full_path = os.path.join(strExDataFolderName, file)
# Save the figure
plt.savefig(full_path, dpi=300, bbox_inches='tight')
'''

# Calculate horizontal and vertical beam sizes (σ_x and σ_y)
sigma_x = math.sqrt(eBeam.arStatMom2[0])  # √<x²>
sigma_y = math.sqrt(eBeam.arStatMom2[3])  # √<y²>

print("Horizontal beam size (σ_x):", sigma_x, "m")
print("Vertical beam size (σ_y):", sigma_y, "m")
#freq = 5 # THz

#sourceE = srwl_uti_ph_en_conv(freq, 'THz', 'eV')
#dis = 1.678 #Distance from geometrical source point to lens [m]

#lam = (1.242E-6)/sourceE
#beamE = 3000
#mass = 0.511
#pol = 0
#gamma = beamE/mass

print(f"n_x1: {wfr.mesh.nx}, n_y1: {wfr.mesh.ny}, n_e1: {wfrSp.mesh.ne}")


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
save_electric_field_to_csv(wfr, 'data_test_beamline_CLEAR/electric_field_real', 'data_test_beamline_CLEAR/electric_field_imag')

print(f"n_x2: {wfr.mesh.nx}, n_y2: {wfr.mesh.ny}, n_e2: {wfr.mesh.ne}")
print('status test0')
# Plot the electric field data
def plot_electric_field(wfr, component='Ex', part='real'):
    nx = wfr.mesh.nx
    ny = wfr.mesh.ny
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

print('status test00')

# Example plotting
plot_electric_field(wfr, component='Ex', part='real')
plot_electric_field(wfr, component='Ex', part='imag')
plot_electric_field(wfr, component='Ey', part='real')
print('status test000')
plot_electric_field(wfr, component='Ey', part='imag')
print('status test0000')

print(f"nx: {wfr.mesh.nx}, ny: {wfr.mesh.ny}, ne: {wfr.mesh.ne}")

#srwl.CalcElecFieldSR(wfrSp, 0, magFldCnt, arPrecSR)
print('Electric field array (Ex):', wfr.arEx[:10])  # Print first 10 elements
print('Electric field array (Ey):', wfr.arEy[:10])  # Print first 10 elements

print(f"n_x3: {wfr.mesh.nx}, n_y3: {wfr.mesh.ny}, n_e3: {wfr.mesh.ne}")
# Get the dimensions of the wavefront
nx = wfr.mesh.nx
ny = wfr.mesh.ny
# need to make the 1D E-field arrays into a 2D ones
# Reshape the real and imaginary parts into 2D arrays (already present in your code)
Ex_real = np.array(wfr.arEx[::2]).reshape(ny, nx)  # Real part of Ex
Ex_imag = np.array(wfr.arEx[1::2]).reshape(ny, nx)  # Imaginary part of Ex
Ey_real = np.array(wfr.arEy[::2]).reshape(ny, nx)  # Real part of Ey
Ey_imag = np.array(wfr.arEy[1::2]).reshape(ny, nx)  # Imaginary part of Ey

# Calculate the intensity (spatial profile)
#intensity = np.abs(Ex)**2 + np.abs(Ey)**2
#intensity = (Ex_real**2 + Ex_imag**2) + (Ey_real**2 + Ey_imag**2)
intensity = (np.abs(Ex_real)**2 + np.abs(Ex_imag)**2) + (np.abs(Ey_real)**2 + np.abs(Ey_imag)**2)


# Plot the intensity as a 2D heatmap
plt.figure(figsize=(10, 8))
plt.imshow(intensity, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin], cmap='hot', origin='lower')
plt.colorbar(label='Intensity')
plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
#plt.title('Optical Synchrotron Radiation Intensity Profile')
plt.title('OSR Intensity Profile')
# Specify the directory where you want to save the plot
#save_directory = "/path/to/your/directory/"
# Specify the filename
filename = "OSR_intensity_profile.png"
# Full path for the file
full_path = os.path.join(strExDataFolderName, filename)
# Save the figure
plt.savefig(full_path, dpi=300, bbox_inches='tight')
plt.show()


# Generate multi-electron intensity data
print("Calculating multi-electron electric fields...")

'''
# Deep copy the original wavefront to avoid modifying single-electron fields
wfr_multi = deepcopy(wfr)

# Reinitialize arEx and arEy with properly allocated memory
#wfr_multi.arEx = array('f', [0] * 2 * nx * ny)  # 2 for real and imaginary parts
#wfr_multi.arEy = array('f', [0] * 2 * nx * ny)  # 2 for real and imaginary parts (as polarization is used)
wfr_multi.arEx = array('f', [0] * 2 * wfr_multi.mesh.nx * wfr_multi.mesh.ny)
wfr_multi.arEy = array('f', [0] * 2 * wfr_multi.mesh.nx * wfr_multi.mesh.ny)


# Perform multi-electron field calculations
srwl.CalcIntFromElecField(
    wfr_multi.arEx,  # Output Electric Field (X-component)
    wfr_multi,       # Wavefront object
    6,               # Extracting intensity
    1,               # Multi-electron mode (convolution)
    3,               # Polarization (total)
    wfr_multi.mesh.eStart,  # Photon energy
    0,               # Horizontal position
    0                # Vertical position
)

# Extract electric field arrays for multi-electron case
#Ex_real_multi = np.array(wfr_multi.arEx[::2]).reshape(wfr_multi.mesh.ny, wfr_multi.mesh.nx)
#Ex_imag_multi = np.array(wfr_multi.arEx[1::2]).reshape(wfr_multi.mesh.ny, wfr_multi.mesh.nx)
#Ey_real_multi = np.array(wfr_multi.arEy[::2]).reshape(wfr_multi.mesh.ny, wfr_multi.mesh.nx)
#Ey_imag_multi = np.array(wfr_multi.arEy[1::2]).reshape(wfr_multi.mesh.ny, wfr_multi.mesh.nx)

# Convert `array.array` to 2D arrays for Ex and Ey (Real and Imaginary parts)
Ex_real_multi = np.array(wfr_multi.arEx[::2]).reshape((ny, nx))  # Real part of Ex
Ex_imag_multi = np.array(wfr_multi.arEx[1::2]).reshape((ny, nx))  # Imaginary part of Ex
Ey_real_multi = np.array(wfr_multi.arEy[::2]).reshape((ny, nx))  # Real part of Ey
Ey_imag_multi = np.array(wfr_multi.arEy[1::2]).reshape((ny, nx))  # Imaginary part of Ey

# Calculate intensity from electric fields
intensity_multi = (Ex_real_multi**2 + Ex_imag_multi**2) + (Ey_real_multi**2 + Ey_imag_multi**2)


# Convert `array.array` to 2D arrays for Ex and Ey (Real and Imaginary parts)
Ex_real_multi = [[wfr_multi.arEx[2 * (i * nx + j)] for j in range(nx)] for i in range(ny)]
Ex_imag_multi = [[wfr_multi.arEx[2 * (i * nx + j) + 1] for j in range(nx)] for i in range(ny)]
Ey_real_multi = [[wfr_multi.arEy[2 * (i * nx + j)] for j in range(nx)] for i in range(ny)]
Ey_imag_multi = [[wfr_multi.arEy[2 * (i * nx + j) + 1] for j in range(nx)] for i in range(ny)]

# Calculate the multi-electron intensity
intensity_multi = [[(Ex_real_multi[i][j] ** 2 + Ex_imag_multi[i][j] ** 2) +
                    (Ey_real_multi[i][j] ** 2 + Ey_imag_multi[i][j] ** 2)
                    for j in range(nx)] for i in range(ny)]

# Calculate the multi-electron intensity
#intensity_multi = (np.abs(Ex_real_multi) ** 2 + np.abs(Ex_imag_multi) ** 2) + \
 #                 (np.abs(Ey_real_multi) ** 2  + np.abs(Ey_imag_multi) ** 2)
'''
print('multi-E intensity continues')
      
      
# Pre-allocate array for multi-electron intensity
#arI1m = np.zeros(wfr.mesh.nx * wfr.mesh.ny, dtype=np.float64)
arI1m = array('f', [0] * mesh1.nx * mesh1.ny)
# Calculate intensity directly
srwl.CalcIntFromElecField(
    arI1m,          # Output array for intensity
    wfr,            # Wavefront object
    6,              # Intensity calculation option (6 = multi-electron intensity)
    1,              # Multi-electron mode (convolution)
    3,              # Polarization (3 = total intensity)
    mesh1.eStart,  # Photon energy
    0,              # Horizontal position
    0               # Vertical position
)

# Reshape and plot
#intensity_multi = arI1m.reshape(wfr.mesh.ny, wfr.mesh.nx)
# Instead, Manually reshaping of array.array into a 2D list
intensity_multi = [arI1m[i * mesh1.nx:(i + 1) * mesh1.nx] for i in range(mesh1.ny)]
intensity_multi = np.array([arI1m[i * mesh1.nx:(i + 1) * mesh1.nx] for i in range(mesh1.ny)]) # Converting intensity_multi to a NumPy array

# backtracking electric field components from the multi-intensity
Ex_real_multi = np.sqrt(intensity_multi / 2)  # Approximate Ex from intensity (half assumed for Ex and Ey)
Ex_imag_multi = np.zeros_like(Ex_real)       # Imaginary part assumed 0
Ey_real_multi = np.sqrt(intensity_multi / 2)  # Same for Ey
Ey_imag_multi = np.zeros_like(Ey_real)

# Calculate/Reconstruct intensity from electric fields
intensity_multi_recon = (Ex_real_multi**2 + Ex_imag_multi**2) + (Ey_real_multi**2 + Ey_imag_multi**2)



# Plot the multi-electron intensity as a 2D heatmap
plt.figure(figsize=(10, 8))
plt.imshow(intensity_multi_recon, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin],
           cmap='hot', origin='lower')
plt.colorbar(label='Intensity')
plt.xlabel('Horizontal Position (m)')
plt.ylabel('Vertical Position (m)')
plt.title('Multi-Electron OSR Intensity Profile')

# Save the multi-electron intensity plot
filename_multi = "OSR_multi_electron_intensity_profile.png"
full_path_multi = os.path.join(strExDataFolderName, filename_multi)
plt.savefig(full_path_multi, dpi=300, bbox_inches='tight')

plt.show()



# beam profile (showing beam size, position, and any asymmetry)
# Extracting the intensity along the center of the horizontal axis (y = 0)
horizontal_profile = intensity[int(ny/2), :]  # ny/2 corresponds to y = 0

# Create the x_values array based on the mesh data
x_values = np.linspace(wfr.mesh.xStart, wfr.mesh.xFin,  wfr.mesh.nx)
'''
plt.plot(x_values, horizontal_profile)
#plt.plot(np.linspace(-0.6, 0.6, nx), horizontal_profile)
plt.xlabel('Horizontal Position (mm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Horizontal Beam Profile from OSR')
plt.grid(True)
plt.show()

# For smoothening the horizontal slice data from the OSR map
smoothed_intensity = gaussian_filter1d(intensity, sigma=2)  # You can adjust sigma

plt.plot(x_values, smoothed_intensity)
#plt.plot(np.linspace(-0.6, 0.6, nx), smoothed_intensity)
plt.title("Smoothed Horizontal Beam Profile from OSR")
plt.xlabel("Horizontal Position (mm)")
plt.ylabel("Intensity (a.u.)")
plt.show()
'''

#phase info

# to calculate Phase for Each Component (Ex and Ey)
# Calculate the phase for Ex and Ey separately
phase_Ex = np.arctan2(Ex_imag, Ex_real)  # Phase for Ex component
phase_Ey = np.arctan2(Ey_imag, Ey_real)  # Phase for Ey component

'''
# Plot the phase for Ex
plt.figure(figsize=(10, 8))
plt.imshow(phase_Ex, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin], aspect='auto')
plt.colorbar(label='Phase (radians)')
plt.title('Phase of Ex Component')
plt.xlabel('Horizontal Position [m]')
plt.ylabel('Vertical Position [m]')
plt.show()

# Plot the phase for Ey
plt.figure(figsize=(10, 8))
plt.imshow(phase_Ey, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin], aspect='auto')
plt.colorbar(label='Phase (radians)')
plt.title('Phase of Ey Component')
plt.xlabel('Horizontal Position [m]')
plt.ylabel('Vertical Position [m]')
plt.show()


# Now to calculate the Overall Phase of the Combined Field

# Combine real and imaginary parts for total field
E_real = Ex_real + Ey_real  # Total real part
E_imag = Ex_imag + Ey_imag  # Total imaginary part

# Calculate the overall phase
phase_total = np.arctan2(E_imag, E_real)

# Plot the overall phase
plt.figure(figsize=(10, 8))
plt.imshow(phase_total, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin], aspect='auto')
plt.colorbar(label='Phase (radians)')
plt.title('Overall Phase of Electric Field')
plt.xlabel('Horizontal Position [m]')
plt.ylabel('Vertical Position [m]')
plt.show()
'''
# Define the photon energy array
photon_energy_Sp = np.linspace(wfrSp.mesh.eStart, wfrSp.mesh.eFin,  wfrSp.mesh.ne)
photon_energy = np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne)


# Extract the photon flux for each energy point
#spectral_flux_Sp = np.array([wfrSp.arS[i] for i in range( wfrSp.mesh.ne)])
#spectral_flux = np.array([wfr.arS[i] for i in range( wfr.mesh.ne)])

spectral_flux_Sp = np.array(arSp)
spectral_flux = np.array(arI1)

print('status10')

# Plot the wfr_Sp spectral intensity (photon flux) as a function of photon energy
plt.plot(photon_energy_Sp, spectral_flux_Sp)
plt.title("Spectral Intensity (Photon Flux) vs Photon Energy for wfrSp")
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Photon Flux (photons/eV)")
plt.grid(True)
print('status10a')
#plt.show()
print('status10b')

print('status test0')

'''# Plot the wfr spectral intensity (photon flux) as a function of photon energy
plt.plot(photon_energy, spectral_flux)
plt.title("Spectral Intensity (Photon Flux) vs Photon Energy for wfr")
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Photon Flux (photons/eV)")
plt.grid(True)
plt.show()
'''

# Total power radiated
print('status test1')
# Set up the desired OSR bandwidth (e.g., between 10 eV and 500 eV)
osr_energy_min = 2.175 #1  # Minimum energy for OSR in eV
osr_energy_max = 2.339 #100  # Maximum energy for OSR in eV
print('status test2')

# Create an array of energies
#energies = np.linspace(meshSp.eStart, meshSp.eFin, meshSp.ne)
#photon_energy_Sp = np.linspace(wfrSp.mesh.eStart, wfrSp.mesh.eFin, wfrSp.mesh.ne)


# Extract the photon flux across the OSR bandwidth
osr_photon_flux_Sp = np.array([arSp[i] for i in range(wfr.mesh.ne) if osr_energy_min <= wfr.mesh.eStart + i * (wfr.mesh.eFin - wfr.mesh.eStart) / wfr.mesh.ne <= osr_energy_max])
osr_photon_flux = np.array([arI1[i] for i in range(wfr.mesh.ne) if osr_energy_min <= wfr.mesh.eStart + i * (wfr.mesh.eFin - wfr.mesh.eStart) / wfr.mesh.ne <= osr_energy_max])

osr_energies = np.array([e for e in photon_energy if osr_energy_min <= e <= osr_energy_max])

# Integrate the photon flux over the OSR bandwidth to get total radiated power
#total_power_osr_Sp = simps(osr_photon_flux_Sp, np.linspace(osr_energy_min, osr_energy_max, len(osr_photon_flux_Sp)))
total_power_osr_Sp = simps(osr_photon_flux_Sp, osr_energies)
total_power_osr = simps(osr_photon_flux, np.linspace(osr_energy_min, osr_energy_max, len(osr_photon_flux)))


print('status test3')

total_power_osr_Sp = simps(arSp, photon_energy_Sp)
#total_power_osr = simps(arI1, photon_energy)

# Print total power in OSR bandwidth
print(f"Total Power Radiated in OSR Bandwidth for wfrSp ({osr_energy_min}-{osr_energy_max} eV): {total_power_osr_Sp} W")
#print(f"Total Power Radiated in OSR Bandwidth for wfr ({osr_energy_min}-{osr_energy_max} eV): {total_power_osr} W")

print('status10A')

# s- and p- polarization calculation
'''
# Function to calculate intensity for a specific polarization
def calculate_polarized_intensity(wfr, mesh, pol):
    arI = array('f', [0] * mesh.nx * mesh.ny)
    srwl.CalcIntFromElecField(arI, wfr, pol, 0, 3, mesh.eStart, 0, 0)
    return np.array(arI).reshape(mesh.ny, mesh.nx)

# Calculate s-polarization (horizontal)
intensity_s = calculate_polarized_intensity(wfr, wfr.mesh, 1)

# Calculate p-polarization (vertical)
intensity_p = calculate_polarized_intensity(wfr, wfr.mesh, 2)

# Total intensity (if needed)
intensity_total = intensity_s + intensity_p

# Now you can plot or analyze these separately
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(intensity_s, cmap='viridis')
plt.title('S-polarization')
plt.colorbar()

plt.subplot(132)
plt.imshow(intensity_p, cmap='viridis')
plt.title('P-polarization')
plt.colorbar()

plt.subplot(133)
plt.imshow(intensity_total, cmap='viridis')
plt.title('Total Intensity')
plt.colorbar()


#plt.tight_layout()
plt.show()


# Define the photon energy array
photon_energy_Sp = np.linspace(wfrSp.mesh.eStart, wfrSp.mesh.eFin,  wfrSp.mesh.ne)
photon_energy = np.linspace(wfr.mesh.eStart, wfr.mesh.eFin, wfr.mesh.ne)
spectral_flux_Sp = np.array(arSp)
spectral_flux = np.array(arI1)
# Plot the wfr_Sp spectral intensity (photon flux) as a function of photon energy
plt.plot(photon_energy_Sp, spectral_flux_Sp)
plt.title("Spectral Intensity (Photon Flux) vs Photon Energy for wfrSp")
plt.xlabel("Photon Energy (eV)")
plt.ylabel("Photon Flux (photons/eV)")
plt.grid(True)
plt.show()


# Now Plot their angular distributions

def calculate_angular_distribution(wfrSp, meshSp, pol, num_points=1000):
    arSp = np.zeros(num_points)
    gamma = wfrSp.partBeam.partStatMom1.gamma
    theta_max = 5 / gamma  # Extend to 5 times the characteristic angle
    thetas = np.linspace(-theta_max, theta_max, num_points)
    print ('theta max value: ', theta_max)
    print ('gamma value: ', gamma)

    
    for i, theta in enumerate(thetas):
        meshSp.xStart = meshSp.xFin = theta * meshSp.zStart
        #srwl.CalcIntFromElecField(arSp[i:i+1], wfrSp, pol, 0, 0, meshSp.eStart, 0, 0)
        #srwl.CalcIntFromElecField(arSp[i:i+1], wfrSp, 6, pol, 0, meshSp.eStart, 0, 0)
        srwl.CalcIntFromElecField(arSp, wfrSp, 6, 0, 0, meshSp.eStart, 0, 0)  # Extracting intensity vs photon energy

    
    return thetas, arSp

# Calculate angular distributions
thetas, intensity_s = calculate_angular_distribution(wfrSp, wfrSp.mesh, 1)  # s-polarization
thetas, intensity_p = calculate_angular_distribution(wfrSp, wfrSp.mesh, 2)  # p-polarization
intensity_total = intensity_s + intensity_p
'''
print('status11A')

def calculate_angular_distribution(wfr, pol, num_points=1000): 
    #arSp = np.zeros(num_points)
    gamma = wfr.partBeam.partStatMom1.gamma
    theta_max = 5 / gamma  # Extend to 5 times the characteristic angle
    #thetas = np.linspace(-theta_max, theta_max, num_points)
    print ('theta max value: ', theta_max)
    print ('gamma value: ', gamma)
    
    # Create a 2D mesh for observation
    mesh1 = deepcopy(wfr.mesh)
    mesh1.xStart = -theta_max * mesh1.zStart  # Start angle (horizontal)
    mesh1.xFin = theta_max * mesh1.zStart     # End angle (horizontal)
    mesh1.nx = num_points                      # Number of horizontal observation points
    # Vertical direction (assuming no variation)
    mesh1.yStart = -theta_max * mesh1.zStart
    mesh1.yFin = theta_max * mesh1.zStart
    mesh1.ny = num_points
    #meshSp.yStart = meshSp.yFin = 0             # Single point in the vertical direction
    #meshSp.ny = 1                               # No variation in vertical direction
    
    # Now calculate the intensity
    ar1 = array('f', [0] * num_points * num_points)
    #ar1 = array('f', [0] * mesh1.nx * mesh1.ny) #"Flat" array to take 2D single-electron intensity data (vs X & Y)
    srwl.CalcIntFromElecField(ar1, wfr, 6, pol, 3, mesh1.eStart, 0, 0)  # 6: intensity, pol: polarization, 3: vs horizontal angle
    #srwl.CalcIntFromElecField(arSp, wfrSp, 6, 0, 0, meshSp.eStart, 0, 0) #Extracting intensity vs photon energy
    
    
    # Convert to numpy array and reshape
    intensity_2d = np.array(ar1).reshape(num_points, num_points)
    
    # Extract horizontal profile (middle row of the 2D distribution)
    intensity_profile = intensity_2d[num_points // 2, :]
    
    # Calculate corresponding angles
    thetas = np.linspace(mesh1.xStart / mesh1.zStart, mesh1.xFin / mesh1.zStart, num_points)
    # Return the theta values and the corresponding intensity array
    #thetas = np.linspace(meshSp.xStart / meshSp.zStart, meshSp.xFin / meshSp.zStart, num_points)

    
    return thetas, intensity_profile

# Calculate angular distributions
thetas, intensity_s = calculate_angular_distribution(wfr, 1)  # s-polarization
thetas, intensity_p = calculate_angular_distribution(wfr, 2)  # p-polarization
intensity_total = intensity_s + intensity_p

print('status11')


# Calculate beam spot size

def fwhm(projection, extent):
    """Calculate the Full Width at Half Maximum (FWHM) for a given projection."""
    half_max = max(projection) / 2.0
    indices = np.where(projection >= half_max)[0]
    return (indices[-1] - indices[0]) * (extent[1] - extent[0]) / len(projection)

# E
def calculate_beam_spot_size(intensity, x_extent, y_extent):
    """Estimate beam spot size (FWHM) based on the spatial intensity distribution."""
    x_projection = np.sum(intensity, axis=0)
    y_projection = np.sum(intensity, axis=1)
    
    x_fwhm = fwhm(x_projection, x_extent)
    y_fwhm = fwhm(y_projection, y_extent)
    
    print(f'Horizontal Beam Spot Size (FWHM): {x_fwhm} mm')
    print(f'Vertical Beam Spot Size (FWHM): {y_fwhm} mm')
    return x_fwhm, y_fwhm


# Beam spot size (FWHM)
x_extent = [wfr.mesh.xStart, wfr.mesh.xFin]
y_extent = [wfr.mesh.yStart, wfr.mesh.yFin]
beam_spot_sizes = calculate_beam_spot_size(intensity, x_extent, y_extent)

'''
def calculate_angular_distribution(wfrSp, num_points=1000):
    gamma = wfrSp.partBeam.partStatMom1.gamma
    theta_max = 2 / gamma  # Reduce to 2 times the characteristic angle
    
    # Create a 2D mesh for observation
    thetas = np.linspace(-theta_max, theta_max, num_points)
    meshSp = deepcopy(wfrSp.mesh)
    meshSp.xStart = -theta_max * meshSp.zStart
    meshSp.xFin = theta_max * meshSp.zStart
    meshSp.nx = num_points
    meshSp.yStart = meshSp.yFin = 0
    meshSp.ny = 1
    
    # Calculate intensity
    arSp = array('f', [0] * num_points)
    srwl.CalcIntFromElecField(arSp, wfrSp, 6, 0, 3, meshSp.eStart, 0, 0)
    
    return thetas, np.array(arSp)

# Calculate angular distributions
thetas, intensity_total = calculate_angular_distribution(wfrSp)

# Calculate s and p polarizations
arSp_s = array('f', [0] * len(thetas))
arSp_p = array('f', [0] * len(thetas))
srwl.CalcIntFromElecField(arSp_s, wfrSp, 6, 1, 3, wfrSp.mesh.eStart, 0, 0)
srwl.CalcIntFromElecField(arSp_p, wfrSp, 6, 2, 3, wfrSp.mesh.eStart, 0, 0)
intensity_s = np.array(arSp_s)
intensity_p = np.array(arSp_p)


def calculate_angular_distribution(wfrSp, meshSp, pol, num_points=100):
    arSp = np.zeros(num_points)  # Array to store the calculated intensity
    
    gamma = wfrSp.partBeam.partStatMom1.gamma
    theta_max = 5 / gamma  # Extend to 5 times the characteristic angle
    
    # Define the mesh range: theta corresponds to x (horizontal direction)
    meshSp.xStart = -theta_max * meshSp.zStart  # Start angle (horizontal)
    meshSp.xFin = theta_max * meshSp.zStart     # End angle (horizontal)
    meshSp.nx = num_points                      # Number of horizontal observation points
    
    # Vertical direction (assuming no variation)
    meshSp.yStart = meshSp.yFin = 0             # Single point in the vertical direction
    meshSp.ny = 1                               # No variation in vertical direction
    
    # Now calculate the intensity over the horizontal mesh
    #srwl.CalcIntFromElecField(arSp, wfrSp, 6, pol, 0, meshSp.eStart, 0, 0)  # 6: intensity, pol: polarization, 3: vs horizontal angle
    
    # Return the theta values and the corresponding intensity array
    thetas = np.linspace(meshSp.xStart / meshSp.zStart, meshSp.xFin / meshSp.zStart, num_points)
    return thetas, arSp

# Now to can call the function for s- and p-polarization as before
thetas, intensity_s = calculate_angular_distribution(wfrSp, wfrSp.mesh, 1)  # s-polarization
thetas, intensity_p = calculate_angular_distribution(wfrSp, wfrSp.mesh, 2)  # p-polarization
intensity_total = intensity_s + intensity_p

# Plot the results
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(thetas * 1e3, intensity_s / np.max(intensity_s))
plt.title('S-polarization')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.subplot(132)
plt.plot(thetas * 1e3, intensity_p / np.max(intensity_p))
plt.title('P-polarization')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.subplot(133)
plt.plot(thetas * 1e3, intensity_total / np.max(intensity_total))
plt.title('Total Intensity')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.tight_layout()
plt.show()


# Propagation distance (to observation plane)
prop_distance = 20.0  # meters (adjust as needed)

# Observation plane grid (vertical scan)
obs_grid = SRWLRadMesh(
    _xStart=-0.002, _xFin=0.002,  # Horizontal range
    _yStart=-0.002, _yFin=0.002,  # Vertical range
    _zStart=prop_distance,        # Distance to observation plane
    _ne=1000,                     # Number of photon energy points (set if needed)
    _nx=10, _ny=10              # Grid resolution
)

polar_pi = SRWLRadMesh(_xStart=obs_grid.xStart, _xFin=obs_grid.xFin, _nx=obs_grid.nx, _ne=obs_grid.ne, _ny=obs_grid.ny, _yStart=obs_grid.yStart, _yFin=obs_grid.yFin, _zStart=obs_grid.zStart)
srwl.CalcIntFromElecField(polar_pi, wfr, 6, 0, 1)  # 6: Pi-polarized intensity
intensity_pi = np.array(polar_pi)

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(thetas * 1e3, intensity_s / np.max(intensity_s))
plt.title('S-polarization')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.subplot(132)
plt.plot(thetas * 1e3, intensity_p / np.max(intensity_p))
plt.title('P-polarization')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.subplot(133)
plt.plot(thetas * 1e3, intensity_total / np.max(intensity_total))
plt.title('Total Intensity')
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')

plt.tight_layout()
plt.show()

print('status12')

def extract_intensity(wfr, polarization, intensity_type):
    nx, ny = wfr.mesh.nx, wfr.mesh.ny
    intensity = np.zeros(nx * ny, dtype='float')
    srwl.CalcIntFromElecField(intensity, wfr, polarization, intensity_type, 0)
    return intensity.reshape((ny, nx))


# Calculate pi-polarized intensity (vertical, p-polarization)
pi_polarized_intensity = extract_intensity(wfr, 1, 0)  # 1 for p-polarization

# Calculate sigma-polarized intensity (horizontal, s-polarization)
sigma_polarized_intensity = extract_intensity(wfr, 0, 0)  # 0 for s-polarization

# Combined intensity
combined_intensity = pi_polarized_intensity + sigma_polarized_intensity

# Plotting the combined intensity profile
plt.imshow(combined_intensity, extent=[wfr.mesh.xStart, wfr.mesh.xFin, wfr.mesh.yStart, wfr.mesh.yFin])
plt.colorbar()
plt.title("Combined Pi- and Sigma-Polarized OSR Intensity")
plt.xlabel("Horizontal Position (mm)")
plt.ylabel("Vertical Position (mm)")
plt.show()
'''

# Angular distribution
print('status13')
# Define the parameters for angular distribution calculation
horizontal_angle_range = 1e-3  # Horizontal angular range in rad (±0.001 rad)
vertical_angle_range = 1e-3  # Vertical angular range in rad (±0.001 rad)
num_points_h = 100  # Number of points in the horizontal direction
num_points_v = 100  # Number of points in the vertical direction

# Define a new mesh for angular distribution
#wfr_ang = srwlib.SRWLWfr()
wfr_ang = SRWLWfr()
wfr_ang.allocate(1, num_points_h, num_points_v)  # Single energy point, with angular distribution
wfr_ang.mesh.zStart = 2 #10.0  # Distance to the observation plane in meters (adjust as necessary)

# Define the angular ranges for horizontal (x) and vertical (y) directions
wfr_ang.mesh.xStart = -horizontal_angle_range  # Start angle in horizontal (radians)
wfr_ang.mesh.xFin = horizontal_angle_range  # End angle in horizontal (radians)
wfr_ang.mesh.yStart = -vertical_angle_range  # Start angle in vertical (radians)
wfr_ang.mesh.yFin = vertical_angle_range  # End angle in vertical (radians)

# Perform the SRW calculation for angular distribution of intensity
#srwl.CalcElecFieldSR(wfr_ang, 0, magFldCnt, arPrecSR) #arPrecParSR)
'''
# Extract the angular intensity data
angular_intensity = np.array(wfr_ang.arS).reshape((num_points_v, num_points_h))

# Generate the angular arrays (horizontal and vertical angles)
horizontal_angles = np.linspace(wfr_ang.mesh.xStart, wfr_ang.mesh.xFin, num_points_h)
vertical_angles = np.linspace(wfr_ang.mesh.yStart, wfr_ang.mesh.yFin, num_points_v)

# Plot the angular intensity distribution as a 2D heatmap
plt.imshow(angular_intensity, extent=[horizontal_angles.min(), horizontal_angles.max(), vertical_angles.min(), vertical_angles.max()],
           origin='lower', aspect='auto', cmap='plasma')
plt.colorbar(label='Angular Intensity (a.u.)')
plt.title('Angular Distribution of OSR Intensity')
plt.xlabel('Horizontal Angle (rad)')
plt.ylabel('Vertical Angle (rad)')
plt.grid(True)
plt.show()
'''
'''
def calculate_angular_distribution(wfr, mesh, num_points=1000):
    arI = np.zeros(num_points)
    theta_max = 1 / wfr.partBeam.partStatMom1.gamma  # Characteristic angle
    thetas = np.linspace(-5*theta_max, 5*theta_max, num_points)
    
    for i, theta in enumerate(thetas):
        mesh.xStart = mesh.xFin = theta * mesh.zStart
        srwl.CalcIntFromElecField(arI[i:i+1], wfr, 6, 0, 3, mesh.eStart, 0, 0)
    
    return thetas, arI

# Calculate angular distribution
thetas, intensity_angular = calculate_angular_distribution(wfr, wfr.mesh)

# Plot angular distribution
plt.figure(figsize=(10, 6))
plt.plot(thetas * 1e3, intensity_angular / np.max(intensity_angular))
plt.xlabel('Angle (mrad)')
plt.ylabel('Normalized Intensity')
plt.title('Angular Distribution of Synchrotron Radiation')
plt.grid(True)
plt.show()
'''

#***************************************************************************************************

# Attempt of writing the extracted field info into a ZBF file
# Function to write ZBF file
def write_ZBF1(nx, ny, dx, dy, isPol, lam, ExR, ExI, EyR, EyI, gamma, theta, filename):
    is_polarized = isPol  # use 1 for a polarized beam
    unittype = 0  # 3 for m, 0 for mm
    wavelength = lam  # in meters
    
    #rayleigh = (gamma * gamma * lam) / (2 * np.pi)
    ficl = 0.0  # unused for this ZBF
    #position = 0.0
    #waist = (gamma * lam) / (2 * np.pi)
    
    # Correct definitions for Rayleigh length and beam waist
    waist = lam / (np.pi * theta) #w0 i.e. Beam waist remains as given
    rayleigh = (np.pi * waist**2) / lam  # Rayleigh length zR
    position = 0.0
    
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
    print("All done!\n\n\n")
    print(f"Beam Waist (w0): {waist} ")
    print(f"rayleight length: {rayleigh} ")

# Extracting wavefront data from SRW (example)
# Assuming wfr is your SRW wavefront object
nx = wfr.mesh.nx
ny = wfr.mesh.ny
dx = (wfr.mesh.xFin - wfr.mesh.xStart) / (nx - 1)
dy = (wfr.mesh.yFin - wfr.  mesh.yStart) / (ny - 1)

print(f"n_x4: {wfr.mesh.nx}, n_y4: {wfr.mesh.ny}")

#freq = 3e15 # PetaHz
#sourceE = srwl_uti_ph_en_conv(freq, 'Hz', 'eV')
#dis = 1.678 #Distance from geometrical source point to lens [m]
#lam = (1.242E-6)/sourceE     #λ(m)= hc/E(eV)

#freq = 0.0605 #3e15 # in PetaHz
#sourceE = srwl_uti_ph_en_conv(freq, 'Hz', 'eV')
#sourceE = 2.5 # in eV
#print('energy = ', sourceE) # photon energy in eV
#dis = 1 #Distance from geometrical source point to lens [m]
#lam = (1.242E-6)/sourceE      #λ(m)= hc/E(eV)
lam = 0.00000055 # in meters; so the value is= 550 nm ; previously was:4.95936e-7
print('wavelength = ', lam) # wavelength in meters

beamE = 220
mass = 0.511
#pol = 0
gamma = beamE/mass
theta = 1 / gamma  # Divergence angle for relativistic electron beam


#lam = 1.24e-10  # Wavelength in meters (example value, adjust as needed)
#gamma = 1.0  # Example gamma value (adjust as needed)

# Extracting real and imaginary parts of Ex and Ey
ExR = np.array(wfr.arEx[::2]).reshape((ny, nx))
ExI = np.array(wfr.arEx[1::2]).reshape((ny, nx))
EyR = np.array(wfr.arEy[::2]).reshape((ny, nx))
EyI = np.array(wfr.arEy[1::2]).reshape((ny, nx))

# Determine if the beam is polarized
isPol = 1 if np.any(EyR) or np.any(EyI) else 0

# Call the function to write the ZBF file single/multi-E
#filename = "data_test_beamline_CLEAR/output_single_electron_intensity.zbf"
#write_ZBF(nx, ny, dx, dy, isPol, lam, ExR, ExI, EyR, EyI, gamma, theta, filename) # for single-electron case
filename = "data_test_beamline_CLEAR/output_multi_electron_intensity.zbf"
write_ZBF1(
    nx, ny, dx, dy, isPol, lam, 
    Ex_real_multi, Ex_imag_multi, 
    Ey_real_multi, Ey_imag_multi, 
    gamma, theta, filename
)

print("Final nx: ", wfr.mesh.nx)
print("Final ny: ", wfr.mesh.ny)
print("mesh1 nx: ", mesh1.nx)


def write_ZBF_intensity(nx, ny, dx, dy, intensity, filename):
    """
    Write intensity data to ZBF file format.
    """
    unittype = 0  # 3 for m, 0 for mm
    if unittype == 0:  # Convert to mm
        dx *= 1000
        dy *= 1000

    print(f"Writing intensity data to ZBF file: {filename}...")
    with open(filename, "wb") as out:
        # Write the format version number
        out.write((0).to_bytes(4, 'little'))  # Version number

        # Write the beam nx size
        out.write(nx.to_bytes(4, 'little'))

        # Write the beam ny size
        out.write(ny.to_bytes(4, 'little'))

        # Write the is_polarized flag (0 for intensity only)
        out.write((0).to_bytes(4, 'little'))  # Not polarized

        # Write out the current units
        out.write(unittype.to_bytes(4, 'little'))

        # 4 unused integers
        for _ in range(4):
            out.write((0).to_bytes(4, 'little'))

        # dx and dy
        out.write(struct.pack('d', dx))
        out.write(struct.pack('d', dy))

        # 8 unused doubles
        for _ in range(8):
            out.write(struct.pack('d', 0.0))

        # Write the intensity data
        #intensity.astype(np.float64).tofile(out) # for converting array.array into a NumPy array
        # Instead Write the intensity data directly from array.array
        intensity.tofile(out)

    print(f"Intensity ZBF file saved: {filename}")


# Perform the multi-electron intensity calculation
#arI1m = np.zeros(mesh1.nx * mesh1.ny, dtype='float')
arI1m = array('f', [0] * mesh1.nx * mesh1.ny)
srwl.CalcIntFromElecField(
    arI1m,  # Output array for intensity
    wfr,    # Wavefront object
    6,      # Calculation type (6: intensity, convolution method)
    1,      # Multi-electron mode (1: convolution)
    3,      # Polarization (3: total)
    mesh1.eStart,  # Start photon energy
    0,      # Horizontal position
    0       # Vertical position
)

#arI1m = array('f', [0] * mesh1.nx * mesh1.ny)  # "Flat" array to take 2D single-electron intensity data (vs X & Y)
#srwl.CalcIntFromElecField(arI1m, wfr, 6, 1, 3, mesh1.eStart, 0, 0) #Calculating multi-electron intensity vs X & Y using convolution method (assuming it to be valid!)

# Reshape the calculated intensity array for plotting and saving
#arI1m_reshaped = np.array(arI1m).reshape((mesh1.ny, mesh1.nx))

# Plot the intensity
uti_plot2d1d(arI1m, [mesh1.xStart, mesh1.xFin, mesh1.nx], [mesh1.yStart, mesh1.yFin, mesh1.ny],
             labels=('Horizontal position', 'Vertical position', 'Multi-E Intensity in Image Plane'),
             units=unitsIntPlot)
#uti_plot2d1d(arI1m, [-0.0005, 0.0005, 450], [-0.0005, 0.0005, 450],
 #            labels=('Horizontal position', 'Vertical position', 'Multi-E Intensity in Image Plane'),
  #           units=unitsIntPlot)
uti_plot_show()

'''
# Convert the 'array.array' object to a NumPy array
arI1m_np = np.array(arI1m, dtype=np.float64)
# Reshape the calculated intensity array for plotting and saving
arI1m_reshaped = arI1m_np.reshape((mesh1.ny, mesh1.nx))
'''
# Save the intensity directly to ZBF format
filename_multi_intensity = "data_test_beamline_CLEAR/old_multi_electron_intensity.zbf"
dx = (mesh1.xFin - mesh1.xStart) / mesh1.nx
dy = (mesh1.yFin - mesh1.yStart) / mesh1.ny

write_ZBF_intensity(mesh1.nx, mesh1.ny, dx, dy, arI1m, filename_multi_intensity)



# cropped zbf file (512x512 out of a larger grid resol.)
'''
def write_ZBF(data, mesh, filename):
    """
    Writes the given intensity data into a Zemax-compatible .zbf file.

    Parameters:
    - data: numpy array, intensity data to save.
    - mesh: SRWLRadMesh, the mesh defining the spatial grid.
    - filename: str, the name of the .zbf file to save.
    """
    # Convert the 2D data into a flat 1D array (Zemax expects this format)
    data_flat = np.array(data).ravel()

    # Open the file for binary writing
    with open(filename, 'wb') as f:
        # Write the header (16 bytes) according to Zemax .zbf format specification
        f.write(np.array([mesh.nx, mesh.ny], dtype=np.uint32).tobytes())  # Grid size
        f.write(np.array([mesh.xStart, mesh.xFin], dtype=np.float64).tobytes())  # X range
        f.write(np.array([mesh.yStart, mesh.yFin], dtype=np.float64).tobytes())  # Y range
        
        # Write the intensity data
        f.write(data_flat.astype(np.float32).tobytes())  # Zemax requires float32 intensity values

    print(f"File saved as: {filename}")


# As `wfr` is the SRW wavefront object with 900x900 grid
nx, ny = 900, 900  # Original grid size
nx_new, ny_new = 512, 512  # Target grid size

# Central indices for cropping
start_x = (wfr.mesh.nx - nx_new) // 2
start_y = (wfr.mesh.ny - ny_new) // 2
end_x = start_x + nx_new
end_y = start_y + ny_new

print("start_x: ", start_x)
print("end_y: ", end_y)

arI1m = array('f', [0] * mesh1.nx * mesh1.ny)
srwl.CalcIntFromElecField(
    arI1m,  # Output array for intensity
    wfr,    # Wavefront object
    6,      # Calculation type (6: intensity, convolution method)
    1,      # Multi-electron mode (1: convolution)
    3,      # Polarization (3: total)
    mesh1.eStart,  # Start photon energy
    0,      # Horizontal position
    0       # Vertical position
)

# Reshape the wavefront intensity for easier slicing
#arI1m = np.array(wfr.arI1m).reshape((wfr.mesh.ny, wfr.mesh.nx))  # Assuming `arI` contains the intensity array

# Convert `arI1m` to a NumPy array and reshape it
arI1m_np = np.array(arI1m).reshape((mesh1.ny, mesh1.nx))  # Convert and reshape to 2D

# Extract the central 512x512 portion
arI_cropped = arI1m_np[start_y:end_y, start_x:end_x]

# Update wavefront mesh for the cropped region
mesh = deepcopy(wfr.mesh)
mesh.nx, mesh.ny = nx_new, ny_new  # Update grid size
mesh.xStart += start_x * (mesh.xFin - mesh.xStart) / wfr.mesh.nx  # Update xStart
mesh.xFin = mesh.xStart + nx_new * (mesh.xFin - mesh.xStart) / wfr.mesh.nx  # Update xFin
mesh.yStart += start_y * (mesh.yFin - mesh.yStart) / wfr.mesh.ny  # Update yStart
mesh.yFin = mesh.yStart + ny_new * (mesh.yFin - mesh.yStart) / wfr.mesh.ny  # Update yFin

# Save the cropped data to a new zbf file using the write_ZBF function
filename = "data_test_beamline_CLEAR/output_cropped_512x512.zbf"
write_ZBF(arI_cropped, mesh, filename)
'''

def write_ZBF2(nx, ny, dx, dy, isPol, lam, ExR, ExI, EyR, EyI, gamma, theta, filename):
    is_polarized = isPol  # use 1 for a polarized beam
    unittype = 0  # 3 for m, 0 for mm
    wavelength = lam  # in meters
    
    #rayleigh = (gamma * gamma * lam) / (2 * np.pi)
    ficl = 0.0  # unused for this ZBF
    #position = 0.0
    #waist = (gamma * lam) / (2 * np.pi)
    
    # Correct definitions for Rayleigh length and beam waist
    waist = lam / (np.pi * theta) #w0 i.e. Beam waist remains as given
    rayleigh = (np.pi * waist**2) / lam  # Rayleigh length zR
    position = 0.0
    
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
    print("All done!\n\n\n")
    print(f"Beam Waist (w0): {waist} ")
    print(f"rayleight length: {rayleigh} ")

# Extracting wavefront data from SRW (example)
nx = wfr.mesh.nx
ny = wfr.mesh.ny
dx = (wfr.mesh.xFin - wfr.mesh.xStart) / (nx - 1)
dy = (wfr.mesh.yFin - wfr.  mesh.yStart) / (ny - 1)

#print(f"n_x4: {wfr.mesh.nx}, n_y4: {wfr.mesh.ny}")

#nx, ny = 900, 900  # Original grid size
nx_new, ny_new = 512, 512  # Target grid size

# Central indices for cropping
start_x = (wfr.mesh.nx - nx_new) // 2
start_y = (wfr.mesh.ny - ny_new) // 2
end_x = start_x + nx_new
end_y = start_y + ny_new


lam = 0.00000055 # in meters; so the value is= 550 nm ; previously was:4.95936e-7
print('wavelength = ', lam) # wavelength in meters

beamE = 220
mass = 0.511
#pol = 0
gamma = beamE/mass
theta = 1 / gamma  # Divergence angle for relativistic electron beam

# Extracting real and imaginary parts of Ex and Ey has been done previously already!
print("Shape of full multi-electron field (Ex_real_multi) before reshaping:", Ex_real_multi.shape)

# Reshape the multi-electron fields to match the grid
Ex_real_multi = Ex_real_multi.reshape((mesh1.ny, mesh1.nx))
Ex_imag_multi = Ex_imag_multi.reshape((mesh1.ny, mesh1.nx))
Ey_real_multi = Ey_real_multi.reshape((mesh1.ny, mesh1.nx))
Ey_imag_multi = Ey_imag_multi.reshape((mesh1.ny, mesh1.nx))

print("Shape of full multi-electron field (Ex_real_multi) after reshaping:", Ex_real_multi.shape)

# Perform the 512x512 extraction
ExR_cropped = Ex_real_multi[start_y:end_y, start_x:end_x]
ExI_cropped = Ex_imag_multi[start_y:end_y, start_x:end_x]
EyR_cropped = Ey_real_multi[start_y:end_y, start_x:end_x]
EyI_cropped = Ey_imag_multi[start_y:end_y, start_x:end_x]

print("Shape of cropped multi-electron field (ExR_cropped):", ExR_cropped.shape)

# Update the grid size and sampling intervals for the cropped data
nx_cropped = end_x - start_x  # This will be 512
ny_cropped = end_y - start_y  # This will be 512
dx_cropped = dx  # Keep the original step size
dy_cropped = dy  # Keep the original step size

# Determine if the beam is polarized
isPol = 1 if np.any(EyR) or np.any(EyI) else 0

# Call the function to write the ZBF file single/multi-E
filename = "data_test_beamline_CLEAR/output_cropped_512x512_100muradxydivergence.zbf"
write_ZBF2(
    nx_cropped, ny_cropped, dx_cropped, dy_cropped, isPol, lam, 
    ExR_cropped, ExI_cropped,
    EyR_cropped, EyI_cropped, 
    gamma, theta, filename
)

print("SuperFinal nx: ", wfr.mesh.nx)
print("SuperFinal ny: ", wfr.mesh.ny)
