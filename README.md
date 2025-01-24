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

Get the tinygrad repo to be able to use `export_model` (`export_model` is not yet in core tinygrad. The fork fetched also has f16 support)

`./setup_tinygrad.sh`

Use `EXPORT=1 WEBGPU=1` to export the model, and pass in the model, which you want to select (`tissue_fast` in this example):

`EXPORT=1 WEBGPU=1 PYTHONPATH=".:./tinygrad" python3 brainchop/main.py -m tissue_fast '/path/to/volume.nii'`

The output of the export step will be a `net.js` file with the WebGPU kernels, and wrapper code to run the model, and a `net.safetensors` file which stores the weights and biases in [safetensors](https://github.com/huggingface/safetensors) format.

To use the exported model:

```JavaScript
import tissue_fast from './net_tissue_fast.js'

const getDevice = async () => {
     if (!navigator.gpu) return false;
     const requiredLimits = {};
     /*
     Your console will report if this value is enough or not,
     you can request a larger value if needed, but you can't exceed 
     the maximum buffer size supported by your browsers.
     */
     const maxBufferSize = 2013265920;
     requiredLimits.maxStorageBufferBindingSize = maxBufferSize;
     requiredLimits.maxBufferSize = maxBufferSize;
     const adapter = await navigator.gpu.requestAdapter();
    return await adapter.requestDevice({
        requiredLimits: requiredLimits,
        requiredFeatures: ["shader-f16"]
    });
};
const device = await getDevice();
const tissuefastSession = await tissue_fast.load(device, "./net_tissue_fast.safetensors");
const results = await tissuefastSession(img32);
```

## Requirements

- Python 3.6+
- tinygrad : our tiny and portable (but powerful) ML inference engine
- numpy : basic tensor operations
- nibabel : to read nifti files
- requests : to download models

## License

This project is licensed under the MIT License.
