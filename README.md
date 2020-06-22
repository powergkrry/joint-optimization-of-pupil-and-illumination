# joint-optimization-of-pupil-and-illumination

## Getting started
### Git clone

### Installing Dependencies
```bash
conda env create -f environment.yml
```
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
