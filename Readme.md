![alttext](https://github.com/DPR-Sanchez/neural-net-models/blob/dev/Neur_Icon_64.png)

## Neur
A neural network training and utilization tool

#### Dependencies

* [Neupy](https://github.com/itdxer/neupy)
* [Pandas 0.24.0](https://pandas.pydata.org/pandas-docs/stable/whatsnew/index.html#version-0-24)
* [Numpy](https://github.com/numpy/numpy)
* [Matplotlib](https://github.com/matplotlib/matplotlib)
* [dill](https://github.com/uqfoundation/dill)
* [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI)

#### Installing

* Clone this repository
```bash
git clone https://github.com/DPR-Sanchez/neural-net-models.git
```
* Install dependencies
```bash
pip install -r requirements.txt
```
* Run the program
```bash
python GUI.py
```

#### Running on Docker
If you are on a Linux distribution that uses X Server, you can run this program
from within a docker container by following these steps:

* Grant execution permissions to the shell scripts
```bash
chmod +x build-neur start-neur
```
* Build the docker image
```bash
./build-neur
```
* Run the docker image
```bash
./start-neur
```

#### Getting Started:
![alttext](https://github.com/DPR-Sanchez/neural-net-models/blob/dev/training_main_window_screen_shot.png)

On launch, choose either General Training to start making your own, or take a look at
Deepwatch, our demonstration program that predicts a win or loss in a game of overwatch.
Note: as of right now, the training set is outdated and won't work with the new hero,
Sigma.

###### General Training:
![alttext](https://github.com/DPR-Sanchez/neural-net-models/blob/dev/training_general_training_window_screen_shot.png)

###### Deepwatch:
![alttext](https://github.com/DPR-Sanchez/neural-net-models/blob/dev/training_deepwatch_window_screen_shot.png)
