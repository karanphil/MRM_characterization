# MRM_characterization

DESCRIPTION:

This repository contains all the code and scripts used to produce results and
figures for "Crossing-fiber robust characterization of the orientation dependence
of magnetization transfer measures", submitted to Magnetic Resonance in Medicine.

The main script used to characterize the orientation dependence of any measure is
scripts/scil_characterize_orientation_dependence.py. This will save the results as
npz files, which can then be used with the various visualization scripts to
produce figures. It also has the option to directly save the plotted results for
each measures individually. It can also save the orientation information.

The bash scripts give examples of how to characterize multiple subjects or bundles
with one script. However, all the paths are hardcoded and will not apply directly
to another setup/data.

The visualization scripts were used to produce the figures of the paper. However,
they are pretty messy and longer than they should be, but they work just fine.

INSTALLATION AND USAGE:

While it is not mandatory, we strongly suggest to install the scilpy library in order
to have all the dependencies necessary. First, clone the scilpy repository
(https://github.com/scilus/scilpy), and follow the instructions for installation,
also available on this github page. It is now also possible to install scilpy directly
with "pip install scilpy".
The only missing package should be cmcrameri for colormaps, which can be
installed with "pip install cmcrameri".


If you have any concerns while using this repository, don't hesitate to contact me
at philippe.karan@usherbrooke.ca.