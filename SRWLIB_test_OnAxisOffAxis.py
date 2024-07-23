# -*- coding: utf-8 -*-
#############################################################################
# SRWLIB test code: Simulating ON-AXIS and OFF-AXIS synchrotron radiation for
# the partial and/or full CLEAR beamline and compare results
# v 0.01
#############################################################################

"""
Created on Thu Jun 27 10:47:53 2024

@author: quasar-debdeep
"""

from __future__ import print_function  
import os
from copy import deepcopy
from srwlib import *
from uti_plot import *
import numpy as np
import matplotlib.pyplot as plt

def save_initial_mesh(wfr):
    return deepcopy(wfr.mesh)

def restore_mesh(wfr, initial_mesh):
    wfr.mesh = deepcopy(initial_mesh)

def simulate_wavefront(wfr, magFldCnt, optBL, filename, strExDataFolderName):
    srwl.CalcElecFieldSR(wfr, 0, magFldCnt, [2, 0.01, 0, 0, 20000, 1, 0.8])
    mesh = deepcopy(wfr.mesh)
    arI = array('f', [0] * mesh.nx * mesh.ny)
    srwl.CalcIntFromElecField(arI, wfr, 6, 0, 3, mesh.eStart, 0, 0)
    srwl_uti_save_intens_ascii(arI, mesh, os.path.join(os.getcwd(), strExDataFolderName, filename))
    return arI, mesh

# Define constants
strExDataFolderName = 'data_example13_bunch'
strIntOutFileName_on_axis = 'bunch_res_int_on_axis.dat'
strIntOutFileName_off_axis = 'bunch_res_int_off_axis.dat'
grid_size_x = 75e-3
grid_size_y = 75e-3
nx_new = 2048
ny_new = 2048
distSrcLens = 5.0

# Define electron beam
eBeam = SRWLPartBeam()
eBeam.Iavg = 500  # Total current for n electrons
eBeam.partStatMom1.x = 0.0
eBeam.partStatMom1.y = 0.0
eBeam.partStatMom1.z = 0.0
eBeam.partStatMom1.xp = 0.0
eBeam.partStatMom1.yp = 0.0
eBeam.partStatMom1.gamma = 3. / 0.51099890221e-03

# Define magnetic field and wavefront
B = 0.4
LeffBM = 4.0
BM = SRWLMagFldM(B, 1, 'n', LeffBM)
magFldCnt = SRWLMagFldC([BM], [0], [0], [0])

# Initialize wavefront
wfr = SRWLWfr()
wfr.allocate(1, 10, 10)
wfr.mesh.zStart = distSrcLens
wfr.mesh.eStart = 0.123984
wfr.mesh.eFin = wfr.mesh.eStart
wfr.mesh.xStart = -grid_size_x / 2
wfr.mesh.xFin = grid_size_x / 2
wfr.mesh.yStart = -grid_size_y / 2
wfr.mesh.yFin = grid_size_y / 2
wfr.mesh.nx = nx_new
wfr.mesh.ny = ny_new 
wfr.partBeam = eBeam

# Save initial mesh
initial_mesh = save_initial_mesh(wfr)

# Define optical elements and beamline
distLensImg = distSrcLens
focLen = wfr.mesh.zStart * distLensImg / (distSrcLens + distLensImg)
optLens = SRWLOptL(_Fx=focLen, _Fy=focLen)
optDrift = SRWLOptD(distLensImg)
propagParLens = [1, 1, 1., 0, 0, 1., 2., 1., 2., 0, 0, 0]
propagParDrift = [1, 1, 1., 0, 0, 1., 1., 1., 1., 0, 0, 0]
optBL = SRWLOptC([optLens, optDrift], [propagParLens, propagParDrift])

# Simulate on-axis wavefront
print('Simulating on-axis wavefront...')
arI_on_axis, mesh_on_axis = simulate_wavefront(wfr, magFldCnt, optBL, strIntOutFileName_on_axis, strExDataFolderName)

# Restore initial mesh
restore_mesh(wfr, initial_mesh)

# Introduce horizontal dispersion (off-axis)
print('Simulating off-axis wavefront...')
dispersion_angle = 0.1  # in radian, adjust as needed
eBeam.partStatMom1.xp = dispersion_angle  # Introduce angular spread for dispersion
wfr.partBeam = eBeam

arI_off_axis, mesh_off_axis = simulate_wavefront(wfr, magFldCnt, optBL, strIntOutFileName_off_axis, strExDataFolderName)

# Plot results
uti_plot2d1d(arI_on_axis, [mesh_on_axis.xStart, mesh_on_axis.xFin, mesh_on_axis.nx],
             [mesh_on_axis.yStart, mesh_on_axis.yFin, mesh_on_axis.ny],
             labels=('Horizontal position', 'Vertical position', 'On-Axis Intensity'), units=['m', 'm', 'ph/s/.1%bw/mm^2'])
uti_plot2d1d(arI_off_axis, [mesh_off_axis.xStart, mesh_off_axis.xFin, mesh_off_axis.nx],
             [mesh_off_axis.yStart, mesh_off_axis.yFin, mesh_off_axis.ny],
             labels=('Horizontal position', 'Vertical position', 'Off-Axis Intensity with Dispersion'), units=['m', 'm', 'ph/s/.1%bw/mm^2'])
uti_plot_show()

print("Final nx: ", wfr.mesh.nx)
print("Final ny: ", wfr.mesh.ny)
