# ntua-cv-lab

This used to be a private repository for the version control of the lab projects of the Computer Vision course of ECE NTUA

In order for the code to be properly run, the scripts have to executed in an environment with opencv, numpy, scikit-learn, tqdm, and matplotlib. For jupyter support, jupyter and nb_conda_kernels should also be included. Following is a command to create a virtual environment in conda.

```
conda create -n cv_lab1_env python=3.7 opencv=3.4.2 numpy scikit-learn tqdm matplotlib jupyter nb_conda_kernels

conda activate cv_lab1_env
```

Numba (a just in time compiler for python) should also be installed in order for the box_detection scripts to be fast.

```
conda install numba
```
