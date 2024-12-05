import os
import sys
import argparse
import subprocess
from nibabel import save, Nifti1Image
from tinygrad import Tensor
import numpy as np
from brainchop.model import meshnet
from brainchop.niimath import conform, inverse_conform, bwlabel

from .utils import update_models, list_available_models, find_model_files, AVAILABLE_MODELS


def main():
    default_model = next(iter(AVAILABLE_MODELS.keys()))
    parser = argparse.ArgumentParser(description="BrainChop: portable brain segmentation tool")
    parser.add_argument("input", nargs="?", help="Input NIfTI file path")
    parser.add_argument("-l", "--list", action="store_true", help="List available models")
    parser.add_argument("-i", "--inverse_conform", action="store_true", help="Perform inverse conformation into original image space")
    parser.add_argument("-u", "--update", action="store_true", help="Update the model listing")
    parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
    parser.add_argument("-m", "--model", default="",
                        help=f"Name of segmentation model, default: {default_model}")
    args = parser.parse_args()

    # Early interrupt options
    if args.update:     update_models();            return
    if args.list:       list_available_models();    return
    if not args.input:  parser.print_help();        return

    # Convert input and output paths to absolute paths
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Find model files
    json_file, bin_file = find_model_files(args.model)
    if not json_file or not bin_file:
        print("Error: Unable to locate or download the required model files.")
        sys.exit(1)

    # Verify all required files exist
    for file_path in [args.input, json_file, bin_file]:
        if not os.path.isfile(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)

    try:
        # Apply _conform to the loaded image
        img, _affine, _header = conform(args.input)
        tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
        t = Tensor(tensor.astype(np.float16))
        out_tensor = meshnet(json_file, bin_file, t)

        # model raw output
        save(Nifti1Image(out_tensor, img.affine, img.header), args.output)

        # connected component mask
        bwlabel(args.output)

        # (optional) inverse transform
        if args.inverse_conform:
            inverse_conform(args.input, args.output)

        print(f"Output saved as {args.output}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
    finally:
        subprocess.run(["rm", "conformed.nii.gz"])

if __name__ == "__main__":
    main()
