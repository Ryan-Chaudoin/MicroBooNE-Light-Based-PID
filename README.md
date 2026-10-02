# MicroBooNE-Light-Based-PID

This is the folder for some cleaned up MicroBooNE light yield analysis and PID. In the scripts folder, you can find the scripts that contain all the processing and physics definitions for the analysis:
- mc_selection_report.py: prints out a summary of the purity and efficiency for MC proton and muon selections
- physics.py: contains all the definitions for stopping power, liquid argon parameters, etc for light yield simulation and functions for simulation, as well as simulation of a bragg profile.
- reconstruction.py: contains pmt positions, detector fiducial volume, ntuples needed for reconstruction, and functions for processing the light library and the light reconstruction
- selection.py: everything related to selecting the proton and muon samples
- waveforms.py: deals with getting the prompt and late light fractions from found waveforms

In the notebooks folder, there are notebooks that make use of all of these functions and show the results.
