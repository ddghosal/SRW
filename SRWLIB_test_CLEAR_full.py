# -*- coding: utf-8 -*-
#############################################################################
# SRWLIB test code: Simulating synchrotron radiation for the full CLEAR beamline-
# consisting of 11 quadrupole magnets, 4 corrector magnets, and 2 dipole magnets
# v 0.01
#############################################################################

from __future__ import print_function  # Python 2.7 compatibility
from srwlib import *
from uti_plot import *  # required for plotting
import time

print('SRWLIB test code: Simulating Synchrotron Radiation for CLEAR Beamline- @BHB400 dipole')
#print('Consisting of three quadrupole magnets, one corrector magnet, and a dipole magnet')

print('status1')

# ***********Data Folder and File Names
strExDataFolderName = 'data_test_fullbeamline_CLEAR' # test data sub-folder name
strSpecOutFileName = 'ex_beamline_res_spec.dat'  # file name for output SR spectrum vs photon energy data
strIntOutFileName = 'ex_beamline_res_int.dat'  # file name for output intensity vs horizontal position data

# ***********Quadrupole Magnets
B_quad = 0.3  # Quadrupole magnetic field [T]
L_quad = 2.0  # Quadrupole magnet length [m]
#L_quad = 0.2  # Quadrupole length [m] - new

quad1 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad2 = SRWLMagFldM(-B_quad, 1, 'n', L_quad)
quad3 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad4 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad5 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad6 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad7 = SRWLMagFldM(-B_quad, 1, 'n', L_quad)
quad8 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad9 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad10 = SRWLMagFldM(B_quad, 1, 'n', L_quad)
quad11 = SRWLMagFldM(B_quad, 1, 'n', L_quad)

print('status2')

# ***********Corrector Magnet
B_corr = 0.2  # Corrector magnet magnetic field [T]
L_corr = 1.0  # Corrector magnet length [m]
#L_corr = 0.1  # Corrector length [m] - new

corr1 = SRWLMagFldM(B_corr, 1, 'n', L_corr)
corr2 = SRWLMagFldM(B_corr, 1, 'n', L_corr)
corr3 = SRWLMagFldM(B_corr, 1, 'n', L_corr)
corr4 = SRWLMagFldM(B_corr, 1, 'n', L_corr)

# ***********Dipole Magnet
B_dipole = 0.4  # Dipole magnetic field [T]
L_dipole = 4.0  # Dipole magnet length [m]
#L_dipole = 0.5  # Dipole length [m] - new

dipole1 = SRWLMagFldM(B_dipole, 1, 'n', L_dipole)
dipole2 = SRWLMagFldM(B_dipole, 1, 'n', L_dipole)

print('status3')


#magFldCnt = SRWLMagFldC([dipole], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D
magFldCnt = SRWLMagFldC([quad1, quad2, quad3, quad4, quad5, quad6, quad7, quad8, quad9, quad10, quad11,
                         corr1, corr2, corr3, corr4,
                         dipole1, dipole2], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print('status4')

#BM = SRWLMagFldM(B, 1, 'n', LeffBM)
#magFldCnt = SRWLMagFldC([BM], [0], [0], [0]) #Container of magnetic field elements and their positions in 3D
    
# ***********Electron Beam
eBeam = SRWLPartBeam()
eBeam.Iavg = 0.5  # Average current [A]
# 1st order statistical moments:
eBeam.partStatMom1.x = 0.  # Initial horizontal position of central trajectory [m]
eBeam.partStatMom1.y = 0.  # Initial vertical position of central trajectory [m]
eBeam.partStatMom1.z = 0.  # Initial longitudinal position of central trajectory [m]
eBeam.partStatMom1.xp = 0.  # Initial horizontal angle of central trajectory [rad]
eBeam.partStatMom1.yp = 0.  # Initial vertical angle of central trajectory [rad]
eBeam.partStatMom1.gamma = 3. / 0.51099890221e-03  # Relative energy
# 2nd order statistical moments:
eBeam.arStatMom2[0] = (127.346e-06)**2  # <(x-x0)^2> [m^2]
eBeam.arStatMom2[1] = -10.85e-09  # <(x-x0)*(x'-x'0)> [m]
eBeam.arStatMom2[2] = (92.3093e-06)**2  # <(x'-x'0)^2>
eBeam.arStatMom2[3] = (13.4164e-06)**2  # <(y-y0)^2>
eBeam.arStatMom2[4] = 0.0072e-09  # <(y-y0)*(y'-y'0)> [m]
eBeam.arStatMom2[5] = (0.8022e-06)**2  # <(y'-y'0)^2>
eBeam.arStatMom2[10] = (0.89e-03)**2  # <(E-E0)^2>/E0^2

# ***********Radiation Sampling for the On-Axis SR Spectrum
wfrSp = SRWLWfr()  # Wavefront structure (placeholder for data to be calculated)
wfrSp.allocate(500, 1, 1)  # Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)
wfrSp.mesh.zStart = 5.  # Longitudinal position for initial wavefront [m]

wfrSp.mesh.eStart = 0.1  # Initial photon energy [eV]
wfrSp.mesh.eFin = 10000.  # Final photon energy [eV]

wfrSp.mesh.xStart = 0.  # Initial horizontal position [m]
wfrSp.mesh.xFin = wfrSp.mesh.xStart  # Final horizontal position [m]
wfrSp.mesh.yStart = 0.  # Initial vertical position [m]
wfrSp.mesh.yFin = 0.  # Final vertical position [m]

wfrSp.partBeam = eBeam  # e-beam data is contained inside the wavefront struct

# ***********Radiation Sampling for the Initial Wavefront (before first optical element)
wfr = SRWLWfr()  # Wavefront structure (placeholder for data to be calculated)
wfr.allocate(1, 10, 10)  # Numbers of points vs photon energy, horizontal and vertical positions (the last two will be modified in the process of calculation)

distSrcLens = 5.  # sample/example Distance from geometrical source point to lens [m]
wfr.mesh.zStart = distSrcLens  # Longitudinal position for initial wavefront [m]

wfr.mesh.eStart = 0.123984  # Initial photon energy [eV]
wfr.mesh.eFin = wfr.mesh.eStart  # Final photon energy [eV]

horAng = 0.03  # Horizontal angle [rad]
wfr.mesh.xStart = -0.5 * horAng * distSrcLens  # Initial horizontal position [m]
wfr.mesh.xFin = 0.5 * horAng * distSrcLens  # Final horizontal position [m]
verAng = 0.02  # Vertical angle [rad]
wfr.mesh.yStart = -0.5 * verAng * distSrcLens  # Initial vertical position [m]
wfr.mesh.yFin = 0.5 * verAng * distSrcLens  # Final vertical position [m]

wfr.partBeam = eBeam  # e-beam data is contained inside the wavefront struct

# ***********Optical Elements and their Corresponding Propagation Parameters
distLensImg = distSrcLens  # Distance from lens to image plane
#focLen = wfr.mesh.zStart * distLensImg / (distSrcLens + distLensImg)
#optLens = SRWLOptL(_Fx=focLen, _Fy=focLen) #Thin lens
#optDrift = SRWLOptD(distLensImg) #Drift space from lens to image plane

print('status5')

optDrift1 = SRWLOptD(distLensImg)
optDrift2 = SRWLOptD(distLensImg)
optDrift3 = SRWLOptD(distLensImg)
optDrift4 = SRWLOptD(distLensImg)
optDrift5 = SRWLOptD(distLensImg)
optDrift6 = SRWLOptD(distLensImg)

optQuad1 = SRWLOptL(L_quad)
optQuad2 = SRWLOptL(L_quad)
optQuad3 = SRWLOptL(L_quad)
optQuad4 = SRWLOptL(L_quad)
optQuad5 = SRWLOptL(L_quad)
optQuad6 = SRWLOptL(L_quad)
optQuad7 = SRWLOptL(L_quad)
optQuad8 = SRWLOptL(L_quad)
optQuad9 = SRWLOptL(L_quad)
optQuad10 = SRWLOptL(L_quad)
optQuad11 = SRWLOptL(L_quad)

optCorr1 = SRWLOptL(L_corr)
optCorr2 = SRWLOptL(L_corr)
optCorr3 = SRWLOptL(L_corr)
optCorr4 = SRWLOptL(L_corr)

optDipole1 = SRWLOptL(L_dipole)
optDipole2 = SRWLOptL(L_dipole)

#Propagation paramaters (SRW specific)
#                [0][1][2] [3][4] [5] [6] [7] [8]
#propagParLens =  [1, 1, 1., 0, 0, 1., 2., 1., 2., 0, 0, 0]
propagParDrift1 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift2 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift3 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift4 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift5 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDrift6 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

propagParQuad1 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad2 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad3 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad4 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad5 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad6 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad7 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad8 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad9 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad10 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]
propagParQuad11 =  [1, 1, 1., 1, 0, 1., 2., 1., 2., 0, 0, 0]

propagParCorr1 =  [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParCorr2 =  [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParCorr3 =  [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParCorr4 =  [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

propagParDipole1 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
propagParDipole2 = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]

# Define beamline elements
beamline_elements = [optQuad1, optQuad2, optQuad3, optDrift1, optCorr1, optDipole1, 
                     optQuad4, optQuad5, optQuad6, optDrift2, optCorr2, optDrift3, 
                     optQuad7, optQuad8, optQuad9, optDrift4, optCorr3, optDrift5, optCorr4, optDrift6, 
                     optQuad10, optQuad11, optDipole2]

propagation_parameters = [propagParQuad1, propagParQuad2, propagParQuad3, propagParDrift1, propagParCorr1, propagParDipole1,
                          propagParQuad4, propagParQuad5, propagParQuad6, propagParDrift2, propagParCorr2, propagParDrift3,
                          propagParQuad7, propagParQuad8, propagParQuad9, propagParDrift4, propagParCorr3, propagParDrift5, propagParCorr4, propagParDrift6,
                          propagParQuad10, propagParQuad11, propagParDipole2]

#beamline - Container of optical elements (together with their corresponding wavefront propagation parameters / instructions)
#optBL = SRWLOptC([optQuad1, optQuad2, optQuad3, optCorr, optDipole])
print('status6')

#optBL = SRWLOptC(beamline_elements, propagation_parameters)
#alternatively without the propagation params.
#optBL = SRWLOptC([optQuad1, optQuad2, optQuad3, optQuad4, optQuad5, optQuad6, optQuad7, optQuad8, optQuad9, optQuad10, optQuad11, 
                  #optCorr1, optCorr2, optCorr3, optCorr4, 
                  #optDipole1, optDipole2])
optBL = SRWLOptC([optQuad1, optQuad2, optQuad3, optCorr1, optDipole1,
                  optQuad4, optQuad5, optQuad6, optCorr2,
                  optQuad7, optQuad8, optQuad9, optCorr3, optCorr4, 
                  optQuad10, optQuad11, optDipole2])

#optBL = SRWLOptC([
    #optLens, propagParLens,  # Lens
    #optDrift, propagParDrift,  # Drift space after Lens
    #optQuad1, propagParQuad,  # Quadrupole 1
    #optDrift, propagParDrift,  # Drift space after Quadrupole 1
    #optQuad2, propagParQuad,  # Quadrupole 2
    #optDrift, propagParDrift,  # Drift space after Quadrupole 2
    #optQuad3, propagParQuad,  # Quadrupole 3
    #optDrift, propagParDrift,  # Drift space after Quadrupole 3
    #optQuad4, propagParQuad,  # Quadrupole 4
    #optDrift, propagParDrift,  # Drift space after Quadrupole 4
    #optQuad5, propagParQuad,  # Quadrupole 5
    #optDrift, propagParDrift,  # Drift space after Quadrupole 5
    #optQuad6, propagParQuad,  # Quadrupole 6
    #optDrift, propagParDrift,  # Drift space after Quadrupole 6
    #optQuad7, propagParQuad,  # Quadrupole 7
    #optDrift, propagParDrift,  # Drift space after Quadrupole 7
    #optQuad8, propagParQuad,  # Quadrupole 8
    #optDrift, propagParDrift,  # Drift space after Quadrupole 8
    #optQuad9, propagParQuad,  # Quadrupole 9
    #optDrift, propagParDrift,  # Drift space after Quadrupole 9
    #optQuad10, propagParQuad,  # Quadrupole 10
    #optDrift, propagParDrift,  # Drift space after Quadrupole 10
    #optQuad11, propagParQuad,  # Quadrupole 11
    #optDrift, propagParDrift,  # Drift space after Quadrupole 11
    #optCorr1, propagParCorr,  # Corrector 1
    #optDrift, propagParDrift,  # Drift space after Corrector 1
    #optCorr2, propagParCorr,  # Corrector 2
    #optDrift, propagParDrift,  # Drift space after Corrector 2
    #optCorr3, propagParCorr,  # Corrector 3
    #optDrift, propagParDrift,  # Drift space after Corrector 3
    #optCorr4, propagParCorr,  # Corrector 4
    #optDrift, propagParDrift,  # Drift space after Corrector 4
    #optDipole1, propagParDipole,  # Dipole 1
    #optDrift, propagParDrift,  # Drift space after Dipole 1
    #optDipole2, propagParDipole  # Dipole 2
#])#, unitIsSI=1)

print('status7')

# ***********BM SR Calculation
# Precision parameters
meth = 2  # SR calculation method: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"
relPrec = 0.005  # Relative precision
zStartInteg = 0  # Longitudinal position to start integration (effective if < zEndInteg)
zEndInteg = 0  # Longitudinal position to finish integration (effective if > zStartInteg)
npTraj = 20000  # Number of points for trajectory calculation
useTermin = 1  # Use "terminating terms" (i.e. asymptotic expansions at zStartInteg and zEndInteg) or not (1 or 0 respectively)

print('status8')

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

print('status9')

print('   Performing initial electric field wavefront calculation ... ', end='')
t0 = time.time()
sampFactNxNyForProp = 0.8  # Sampling factor for adjusting nx, ny (effective if > 0)
arPrecSR = [meth, relPrec, zStartInteg, zEndInteg, npTraj, useTermin, sampFactNxNyForProp]
srwl.CalcElecFieldSR(wfr, 0, magFldCnt, arPrecSR)  # Calculating electric field
print('done in', round(time.time() - t0), 's')

print('status10')

print('   Extracting intensity and saving it to a file ... ', end='')
t0 = time.time()
mesh0 = deepcopy(wfr.mesh)
arI0 = array('f', [0] * mesh0.nx * mesh0.ny)  # "Flat" array to take 2D intensity data (vs X & Y)
srwl.CalcIntFromElecField(arI0, wfr, 6, 0, 3, mesh0.eStart, 0, 0)  # Extracting intensity vs horizontal and vertical positions
srwl_uti_save_intens_ascii(arI0, mesh0, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName))
print('done in', round(time.time() - t0), 's')

print('status11')

# ***********Wavefront Propagation
print('   Simulating single-electron electric field wavefront propagation ... ', end='')
t0 = time.time()
srwl.PropagElecField(wfr, optBL)
print('done in', round(time.time() - t0), 's')

print('status12')

# ***********Extracting Intensity from Calculated Electric Field and Saving it to File
print('   Extracting intensity from calculated electric field and saving it to file ... ', end='')
t0 = time.time()
mesh1 = deepcopy(wfr.mesh)
arI1 = array('f', [0] * mesh1.nx * mesh1.ny)  # "Flat" array to take 2D single-electron intensity data (vs X & Y)
srwl.CalcIntFromElecField(arI1, wfr, 6, 0, 3, mesh1.eStart, 0, 0)  # Extracting single-electron intensity vs X & Y
srwl_uti_save_intens_ascii(arI1, mesh1, os.path.join(os.getcwd(), strExDataFolderName, strIntOutFileName))
print('done in', round(time.time() - t0), 's')

print('status13')

# ***********Plotting the Calculation Results
uti_plot1d(arSp, [meshSp.eStart, meshSp.eFin, meshSp.ne], labels=('Photon Energy', 'Spectral Intensity', 'On-Axis SR Intensity Spectrum'), units=['eV', 'ph/s/0.1%bw/mm^2'])
unitsIntPlot = ['m', 'm', 'ph/s/.1%bw/mm^2']
uti_plot2d1d(arI0, [mesh0.xStart, mesh0.xFin, mesh0.nx], [mesh0.yStart, mesh0.yFin, mesh0.ny], labels=('Horizontal position', 'Vertical position', 'Intensity Before Lens'), units=unitsIntPlot)
uti_plot2d1d(arI1, [mesh1.xStart, mesh1.xFin, mesh1.nx], [mesh1.yStart, mesh1.yFin, mesh1.ny], labels=('Horizontal position', 'Vertical position', 'Single-E Intensity in Image Plane'), units=unitsIntPlot)
uti_plot_show()  # show all graphs (blocks script execution; close all graph windows to proceed)

print('status14')
