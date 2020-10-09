# Multi-element microscope optimization by a learned sensing network with composite physical layers

## Getting started
### Git clone

### Installing Dependencies
```bash
conda env create -f environment.yml
```

### Download data
Download the data from [here](https://figshare.com/s/3b06e6546a0f2d898444) and move it to project directory
<!---https://figshare.com/articles/dataset/pupil-and-illumination/12777542-->

## Experiement
Digital-only optimization
```bash
python pupil_experiment --data "center" --is_pupil_train 0 --is_illu 0
```

Pupil optimization
```bash
python pupil_experiment --data "center" --is_pupil_train 1 --is_illu 0
```

Illumination optimization
```bash
python pupil_experiment --data "illumination" --is_pupil_train 0 --is_illu 1
```

Pupil and Illumination optimization
```bash
python pupil_experiment --data "illumination" --is_pupil_train 1 --is_illu 1
```
