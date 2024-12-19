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

def find_model_in_path(model_path: str) -> tuple[str | None, str | None]:
    """Find model.json and model.bin files in the given path.
    
    Args:
        model_path: Directory path to search for model files
        
    Returns:
        Tuple of (json_path, bin_path) or (None, None) if not found
    """
    try:
        model_path = os.path.abspath(model_path)
        if os.path.isfile(model_path):
            # If path is a file, check if it's json/bin and look for counterpart
            dirname = os.path.dirname(model_path)
            basename = os.path.basename(model_path)
            if basename.endswith('.json'):
                bin_path = os.path.join(dirname, 'model.bin')
                return model_path, bin_path if os.path.exists(bin_path) else None
            elif basename.endswith('.bin'):
                json_path = os.path.join(dirname, 'model.json')
                return json_path if os.path.exists(json_path) else None, model_path
        else:
            # Search directory for model files
            json_path = os.path.join(model_path, 'model.json')
            bin_path = os.path.join(model_path, 'model.bin')
            
            json_exists = os.path.isfile(json_path)
            bin_exists = os.path.isfile(bin_path)
            
            if json_exists and bin_exists:
                return json_path, bin_path
                
        return None, None
    except Exception as e:
        print(f"Error finding model files: {str(e)}")
        return None, None

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
    parser.add_argument("-c", "--custom", type=str, 
                        help="Path to custom model directory or model file (will look for model.json and model.bin)")
    
    args = parser.parse_args()

    # Early interrupt options
    if args.update:     update_models();            return
    if args.list:       list_available_models();    return
    if not args.input:  parser.print_help();        return

    # Convert input and output paths to absolute paths
    args.input = os.path.abspath(args.input)
    args.output = os.path.abspath(args.output)

    # Handle model file paths
    if args.custom:
        json_file, bin_file = find_model_in_path(args.custom)
        if not json_file or not bin_file:
            print(f"Error: Could not find model.json and model.bin in {args.custom}")
            sys.exit(1)
    else:
        # Find built-in model files
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
        conform_result = conform(args.input)
        img = conform_result[0]  # First element is the image
        
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
