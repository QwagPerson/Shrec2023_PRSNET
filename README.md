# My PRS-Net adaptation for the Shrec2023 dataset

This code is kinda messy looking back. To train a new model you can use the main.py file
that implements the Pytorch Lighting API and to evaluate you must predict using the predict_main script
and evaluate using the evaluation script contained in the evaluation folder.

Experiments configs are stored in the configs folder.
Dataset folder contains the code to create the voxel_dataset datamodule.
model contains the implementation of prsnet
setup contains the code used to precompute the voxel representations of the point clouds.
test contains some test code i used to make sure the code didnt have bugs.

experiment_gridsearch.py allows us to run a grid search of parameters on the PRS-Net model.
At the end i didnt use this because it was too costly in terms of computations.

exploracion_modelo.py allows us to explore the predictions of the model and compare it to the
ground truths.

plot_results.ipynb it a notebook that creates some plots of the different results.

In case of any questions my email is gsantelicesn2 (at) gmail (dot) com