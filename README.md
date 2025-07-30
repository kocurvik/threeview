# Practical solutions to the relative pose of three calibrated cameras

This repo contains code for paper "Practical solutions to the relative pose of three calibrated cameras" (CVPR 2025 - [paper link](https://openaccess.thecvf.com/content/CVPR2025/papers/Tzamos_Practical_Solutions_to_the_Relative_Pose_of_Three_Calibrated_Cameras_CVPR_2025_paper.pdf))
## Installation

Create an environment with pytorch and packaged from `requirements.txt`.

Install [PoseLib fork with implemented estimators in branch threeview](https://github.com/kocurvik/PoseLib) into the environment:
```shell
git clone https://github.com/kocurvik/PoseLib
git cd PoseLib
git checkout threeview
pip install .
```

Before running the python scripts make sure that the repo is in your python path (e.g. `export PYTHONPATH=/path/to/repo/threeview`)

## Data preparation

Triplets can be made using script `dataset/prepare_im.py` from a dataset with a Colmap reconstruction available.

You can also download the triplets already extracted for [Phototourism and Aachen](https://doi.org/10.5281/zenodo.16603086).

## Evaluation

To perform the evaluation on real data run for each scene:
```
python eval.py -nw 64 triplets-features_superpoint_noresize_2048-LG /path/to/scene_folder_with_triplets
```


