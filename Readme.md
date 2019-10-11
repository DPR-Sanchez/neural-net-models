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
<!-- If you are on a Linux distribution that uses X Server, you can run this program
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
``` -->

##### Linux (with X Server)
* This repository comes with a shell script that will automatically build and run the container
    * `./start-neur cpu` to run the CPU only version
    * `./start-neur gpu` to run with GPU support
* As long as only one version is installed, it will automatically infer which version you want to run

#### Windows 
* In order to run the GUI interface, you will need VcXsrv (X Server for Windows)
    * You can install it with Chocolatey by issuing the following command: `choco install vcxsrv`
    * Alternatively, you can download it [here](https://sourceforge.net/projects/vcxsrv/)
* From a PowerShell prompt, run the command `.\start-neur.ps1`
    * Depending on your execution policy, you may need to issue the command: `Unblock-File -Path .\start-neur.ps1` to allow it to be run

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
