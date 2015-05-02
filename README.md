This program calculates the potential energy, kinetic energy, and forces on a group of particles around a 
specified, particle-specific time step. This allows, for instance, the averaging of energies relative to
the desorption of molecules from a liquid slab. The kinetic energy and forces are separated into tangential
and normal components where the normal direction is assumed to along the simulation box's Z axis. Potential 
energy and force calulations can be separated into contributions from individual atom-types by selecting
appropriate groups from the .ndx index file. In addition to plotting the energies and forces as a function of 
time around the specified event, the same properties are assigned to spatial bins along the Z-axis to allow 
one to plot them relative to, for instance, the Gibbs' dividing surface.

THIS PROGRAM HAS NOT BEEN THOROUGHLY TESTED!!!

Current limitations include:
* Input files must be in the formats from the GROMACS simulation package with coordinates in .xtc format
  and velocities in .trr format.
* Energy and force calculations are limited to Lennard-Jones interactions with Lorentz-Berthelot mixing rules.
* Some simulation-specific properties (e.g., the cutoff distance for energy calculations) are currently
  hard-coded.

The following libraries are required:
* The Boost program_options library.
* The Armadillo matrix library.
* The xdrfile library for reading GROMACS trajectory files.
