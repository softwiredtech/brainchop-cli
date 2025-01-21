# shell.nix
let
  # Import nixpkgs
  pkgs = import <nixpkgs> {};

  # Create a Python environment with all required dependencies
  pythonEnv = pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
    pip
    setuptools
    pybind11
    tinygrad
    requests
    nibabel
  ]);

in pkgs.mkShell {
  packages = [
    pythonEnv
    # Add system-level dependencies here (if needed)
    pkgs.stdenv.cc  # Example: Add a C compiler if needed
  ];

  # Use shellHook to install your local package in editable mode
  shellHook = ''
    # Create a writable directory for editable installations
    export PYTHONUSERBASE=$(mktemp -d)

    # Install the package in editable mode to the writable directory
    pip install --prefix="$PYTHONUSERBASE" --editable .

    # Add the user site-packages to PYTHONPATH
    export PYTHONPATH="$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH"
  '';
}
