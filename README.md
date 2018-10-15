# Freesound Audio Tagging

https://www.kaggle.com/c/freesound-audio-tagging

`1d_final_launch`:
Training 10 1d models, saving predictions

`2d_final_launch`:
Training 10 2d models, saving predictions

`ensembling_submit` - calculates the geometric mean of the predictions made on the previous two
steps.

Network architectures templates for experiments: `archs_2.py.`

Template to conduct experiments on choosing network architecture - `pipeline_general.ipynb`

## Evaluation
MAP@3

Result: 0.91
