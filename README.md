# Evaluating Radial Distribution Functions on a GPU

In this repository is a code, `compute_rdf.py`, that computes the O-O, O-H, and H-H intermolecular radial distribution functions (RDFs) for a liquid water.
It requires an XYZ trajectory file, `water_traj.xyz`, which you can find [here](https://github.com/Chem284Materials/rdf_data/blob/main/water_traj.xyz) (click the "Download raw file" button).

## Task 1 - Port to CUDA

Using PyCUDA, port the `accumulate` function to run on GPUs.
Use CUDA events to obtain accurate measurements of the kernel execution walltime.
Make intelligent use of shared memory.
Do not use CUDA streams (we'll introduce streams in the next task).

In addition to measuring the kernel execution time, also measure the time required to read the `water_traj.xyz` data file and the total time.
Report your timings and the GPU you ran the timings on.

Place the `water_rdfs.png` file you generate in this repository.

**Hint:** You will need to be careful about integer overflow during you accumulation operation.
You may use `unsigned long long` integers where appropriate.

## Task 2 - Use Streams

Modify your code so that the data corresponding to each trajectory frame is evaluated by a different CUDA stream.

In addition to measuring the kernel execution time, also measure the time required to read the `water_traj.xyz` data file and the total time.
Report your timings and the GPU you ran the timings on.

## Task 3 - Use Streams While Reading the Trajectory File

Modify your code so that upon reading the data for a trajectory frame, you immediately launch a CUDA stream to perform the accumulation calculation.
In other words, you should have the GPU performing accumulation on previously read frames concurrently with the process of reading from `water_traj.xyz`.

Measure and report your total walltime.
