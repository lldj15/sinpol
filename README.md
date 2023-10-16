# sinpol
This python package provides a tool to model the neutron transmission  Bragg edge of pollycrystalline materials starting from a single to an ensemble crystal.  The model would allow the use of energy-resolved neutron transmission for the analysis of single crystals, polycrystals, andoligocrystals (materials composed by a relatively small number of crystals).


**Main functionality:** given lattice structure of a material, number of grains in distribution, mosaic distribution, grain size distribution, and optionally a texture model,
calculate neutron transmisison spectrum as a function of neutron wavelength.
## Features
* Calculation of neutron transmission spectrum of single crystal as a function of crystal structure, neutron energy, temerature, mosaic distribution, and crystal orientation. Accounts for various contributions to neutron scattering including, for example, diffraction and inelastic scattering (using incoherent approximation)
* Modeling of texture:
  -3D Gaussian Distribution
* Modeling of effect o strain:
  - Voigt model
  - Reuss model
  - Eshelby-Kroner model
  - Hill Model
* Flexible design to allow future extension to texture and peak profile models
* Allow easy fitting to measured Bragg Edge data

