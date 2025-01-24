# BrainChop

BrainChop is a lightweight tool for brain segmentation that runs on pretty much everything.

---

## Installation

You can install BrainChop using pip:

```
pip install brainchop
```

## Usage

To use BrainChop, run the following command:

```
brainchop input.nii.gz -o output.nii.gz
```

Where:
- `input.nii.gz` is your input NIfTI file
- `output.nii.gz` is the desired output file name

## Export model to WebGPU

1. Get the tinygrad repo to be able to use `export_model` (`export_model` is not yet in core tinygrad. The fork fetched also has f16 support)
`./setup_tinygrad.sh`
<br>

2. Use `EXPORT=1` to export the model, and pass in the model, which you want to select (`tissue_fast` in this example):
`EXPORT=1 WEBGPU=1 PYTHONPATH=".:./tinygrad" python3 brainchop/main.py -m tissue_fast '/path/to/volume.nii'`

## Requirements

- Python 3.6+
- tinygrad : our tiny and portable (but powerful) ML inference engine
- numpy : basic tensor operations
- nibabel : to read nifti files
- requests : to download models

## License

This project is licensed under the MIT License.
