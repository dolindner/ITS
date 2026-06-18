# Zero-Shot Test-Time Canonicalization
using Out-of-Distribution Scoring

![Summary](./res/img/summary.svg)

This is the official repository of the paper:
"[Zero-Shot Test-Time Canonicalization
using Out-of-Distribution Scoring]
published at the ECML 2026.


## Abstract
Pretrained vision models often misclassify inputs that are rotated, scaled, or sheared, even though these affine transformations leave the object class unchanged. 
Robustness is usually restored either by building equivariance into the architecture or by retraining with augmentation,
both of which require changing or retraining the model. 
Test-time canonicalization instead leaves the classifier untouched.
It undoes the transformation of each input, mapping it to a canonical form near the training distribution before classification. 
Existing canonicalizers, however, rely on a narrow set of logit-based energy scores and bespoke search procedures,
leaving the design space of scoring functions and optimizers unexplored. 
We reframe canonicalization as out-of-distribution (OOD) detection, which lets any OOD score serve as the energy minimized over transformations. 
Across benchmarks ranging from handwritten characters and sketches to natural images and 3D point clouds, 
we systematically evaluate around twenty OOD scores and nine search algorithms,
finding that distance-based scores paired with random search and local refinement perform best overall. 
Because canonicalizing an already-aligned input can hurt accuracy, 
we add a gated mechanism that transforms an input only when its OOD score indicates this is needed, 
preserving most in-distribution accuracy while retaining the robustness gains on transformed inputs.


## Installation
First, clone this repository.
Once inside the repository folder (`cd ITS`), you install it by running
```
pip install . 
```
Note that this installs `torch` for `Python >= 3.8` and `CUDA == 11.8`. 
You can check your current `CUDA` version using `nvcc --version`.


A complete list of all requirements can be found [here](./requirements.txt).
You can also install from ITS using
```
python -m pip install -r requirements.txt
```
Some datasets have to be manually downloaded for the TU Berlin dataset. Unfortunately the original 
download link is down at the time of writing. We require the sketches_matlab.zip file from the 
dataset from https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_matlab.zip.
Alternativly you may need to request the file from the authors, use a mirror or cached version of the file(like wayback machine).
The sketches.mat file should be put under experimenent_files/data/tu_berlin/sketches_matlab.
<!---
Maybe
https://beta.hyper.ai/en/datasets/16424
-->
For the SI-Score dataset can be downloaded as described in https://github.com/google-research/si-score.
We only require the subset with only rotation transform. The folders named after the imagenet classes
can then be put under experimenent_files/data/si_score/rotation. The imagenet subset for fitting
OOD detectors requires huggingface for dowloading the init method of the dataset class provides
a way to download it.

For the Vector Neuron Comparison https://github.com/FlyingGiraffe/vnn needs to be downloaded and put
under external in its subfolder vnn.

Pretrained models for reproducibility can be found under:
https://huggingface.co/dlindner/ITSModels
In addition there are .yaml files that include the best found hyperparameters for the search comparison
as well as the comparison of OOD detectors.




## Example Notebooks

![Example](./res/img/algorithm_example.svg)

Under the [examples](./examples) folder you find `jupyter notebooks` that help you getting started.
Under the [paper_experiment](examples/paper_experiment) subfolder there is a collection of all experiments that were run 
for the Zero-Shot Test-Time Canonicalization
using Out-of-Distribution Scoring paper.


## Bibtex
If you find our work interesting, please cite us.
```
@inproceedings{Schmidt2024,
  title={Tilt your Head: Activating the Hidden Spatial-Invariance of Classifiers},
  author={Johann Schmidt and Sebastian Stober},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2024}
}
#TODO
```