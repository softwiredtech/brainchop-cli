 #!/bin/bash

git clone https://github.com/tinygrad/tinygrad.git
cd tinygrad
echo "from .export_model import *" > ./extra/__init__.py
echo "Tinygrad setup completed successfully."
