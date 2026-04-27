---
title: "Sinpol: A Tool for Neutron Transmission Simulation of Single Crystals and Polycrystals"
tags:
  - neutron imaging
  - Bragg-edge imaging
  - time-of-flight
  - crystallography
  - polycrystals
  - Python
authors:
  - name: "Dessieux L. L."
    orcid: ""
    affiliation: 1
    corresponding: true
    email: "dessieuxll@ornl.gov"
affiliations:
  - name: "Neutron Scattering Division, Oak Ridge National Laboratory"
    index: 1
date: "May 2025"
bibliography: paper.bib
---

# Summary

Neutron transmission imaging has emerged as a powerful non-destructive technique for probing the internal structure of crystalline materials. Unlike traditional attenuation imaging, energy-resolved transmission techniques—such as time-of-flight (TOF) and Bragg-edge imaging—exploit the coherent interaction of neutrons with crystallographic planes to reveal orientation, phase, and strain information [@tremsin; @Santis; @wor]. These methods are particularly effective for both single crystals, where distinct transmission features arise from well-defined lattice orientations, and multi-grain polycrystalline samples, where overlapping Bragg edges reflect aggregate texture and strain distributions [@kard; @sato].

In crystalline materials, Bragg scattering leads to characteristic features in the transmission spectrum when specific lattice spacings satisfy the Bragg condition. The spectra of single crystals or multi-grain samples showcase unique Bragg dips at specific wavelengths, where Bragg's law is applicable to particular crystallographic planes [@santis1]. This phenomenon allows for the identification of crystal orientations and the assessment of microstructures at the local grain level [@dess; @mor]. In contrast, TOF neutron transmission spectra obtained from non-textured polycrystalline (powder) samples exhibit sudden intensity increases referred to as Bragg edges at neutron wavelengths that exceed the Bragg condition for coherent scattering related to that lattice spacing [@lehmann2010]. In cases where grain orientations are not randomly distributed, TOF neutron transmission spectra display deformed Bragg edges that reflect these favored crystallographic orientations [@dess1]. In polycrystalline or textured samples, the observed spectrum is a superposition of effects from grains with different orientations and spatial arrangements [@wit]. Modeling these spectra is essential for accurately interpreting measured data, especially in the presence of complex geometries, preferred orientations, or evolving microstructures during in situ measurements [@vog].

Neutron transmission modeling enables researchers to simulate the expected energy-dependent transmission for both idealized and real-world microstructures. Forward models that incorporate crystallographic symmetry, sample orientation, and instrumental resolution are indispensable for linking spectral features to underlying structural parameters. In the case of single crystals, such models support orientation indexing and misorientation analysis [@dess; @mor]. For multi-grain systems, they facilitate texture quantification, strain reconstruction, and validation of orientation distribution functions (ODFs) derived from neutron imaging or diffraction [@Hil; @koc].

# Statement of need

Accurate interpretation of energy-resolved neutron transmission measurements requires forward models that capture crystallographic effects, sample orientation distributions, and experimental resolution. This need is particularly acute for (i) single crystals, where narrow orientation-dependent Bragg dips encode lattice orientation and misorientation, and (ii) polycrystalline or textured materials, where Bragg edges can be broadened or deformed by preferred orientation, strain, and multi-grain superposition effects.

To address this need, we present **Sinpol**, an open-source Python library for modeling neutron transmission spectra of both single-crystal and polycrystalline materials. Sinpol supports arbitrary crystal orientations and energy-resolved transmission calculations, including Bragg-edge position and shape prediction. It provides routines to compute the total cross section governing neutron attenuation in crystalline solids within a semi-empirical framework that accounts for crystal structure, neutron energy, temperature, and orientation. The model also incorporates parasitic Bragg scattering by treating the mosaic spread of the crystal and its alignment with the neutron beam as critical parameters.

Sinpol enables users to specify ranges or distributions of crystal orientations, supporting customized powder or pseudo-powder ensembles. This capability supports simulation of non-uniformities such as texture and strain, improving analysis of neutron interactions in real-world materials. The software is designed to interface with experimental TOF and Bragg-edge imaging data, providing tools for simulating spectra, benchmarking measurements, and validating inversion routines. By enabling accurate and reproducible spectral modeling, Sinpol supports neutron imaging workflows ranging from static characterization to operando studies.

# Acknowledgements

This work is sponsored by the Laboratory Directed Research and Development Program of Oak Ridge National Laboratory, managed by UT-Battelle LLC, for DOE. Part of this research is supported by the U.S. Department of Energy, Office of Science, Office of Basic Energy Sciences, User Facilities under contract number DE-AC05-00OR22725.

The author thanks Dr. Alexandru Stoica and Dr. Philip Bingham for valuable discussions and guidance in deriving the physics used for the calculations in this code.

# Notice of Copyright

This manuscript has been authored by UT-Battelle, LLC under Contract No. DE-AC05-00OR22725 with the U.S. Department of Energy. The United States Government retains and the publisher, by accepting the article for publication, acknowledges that the United States Government retains a non-exclusive, paid-up, irrevocable, worldwide license to publish or reproduce the published form of this manuscript, or allow others to do so, for United States Government purposes. The Department of Energy will provide public access to these results of federally sponsored research in accordance with the DOE Public Access Plan (http://energy.gov/downloads/doe-public-access-plan).

# References
