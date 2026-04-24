# Evaluating Radial Distribution Functions on a GPU

In this repository is a code, `compute_rdf.py`, that computes the O-O, O-H, and H-H intermolecular radial distribution functions (RDFs) for a liquid water.
It requires an XYZ trajectory file, `water_traj.xyz`, which you can find [here](https://github.com/Chem284Materials/rdf_data/blob/main/water_traj.xyz) (click the "Download raw file" button).

## Task 1 - Port to CUDA

Using PyCUDA, port the `accumulate` function to run on GPUs.
Use CUDA events to obtain accurate measurements of the kernel execution walltime.
Make intelligent use of shared memory.

In addition to measuring the kernel execution time, also measure the time required to read the `water_traj.xyz` data file and the total time.
Report your timings and the GPU you ran the timings on.

Place the `water_rdfs.png` file you generate in this repository.

**Hints:** You may find that the [atomicAdd](https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/#atomicadd) function is useful.
You should keep in mind that, much like the case of atomic operations with OpenMP, atomic operations in CUDA have a performance impact.
In addition, you will need to be careful about integer overflow during your accumulation operation.
You may use integer types larger than `int32` where appropriate.

## Task 2 - Use Streams While Reading the Trajectory File

Modify your code so that upon reading the data for a trajectory frame, you immediately execute a CUDA stream to perform the accumulation calculation for that frame.
In other words, utilize the GPU to perform accumulation on previously read frames concurrently with the process of reading from `water_traj.xyz`.
One way to implement this is to have a fixed number of streams that you rotate through; after you've assigned frames to all the streams, you assign the next frame to the first stream.

Measure and report your timings.
Provide your code for this task **in addition** to your code for Task 1 (add it to the repository as a separate file).

## Answers
