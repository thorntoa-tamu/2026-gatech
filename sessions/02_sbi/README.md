# ML4FP SBI Tutorial

This tutorial covers the basic steps in an NSBI analysis. 

Networks are trained using the CARL teachnique outlined in [1907.10621](https://arxiv.org/abs/1907.10621) and mimics the analysis framework of [2412.01600](https://arxiv.org/abs/2412.01600v1). There are four Jupyter notebooks in `tutorials` which walk through:

1. Looking at various kinematics to be used for training
2. Preparing the CARL training framework
3. Calibrating learned density ratio estimate
4. Perfroming a likelihood scan on toy data using the learned density ratio estimate

This exercise also includes a rudimentary training framework whose CLI command is outlined in tutorial notebook 2. 

## Running the Turorial

This tutorial was written and designed to be executed on the [NERSC Jupyter Hub](https://jupyter.nersc.gov/hub/login?next=%2Fhub%2Fhome). All data used in these notbooks can be found in `/global/cfs/cdirs/ntrain6/NSBIData`. To install the necessary modules, open a virtual terminal on the [NERC Jupyter Hub](https://jupyter.nersc.gov/hub/login?next=%2Fhub%2Fhome) and run 

```sh
. setupTutorialEnv.sh
```

Now launch each notebook using the `pytorch-2.6.0` kernal. 