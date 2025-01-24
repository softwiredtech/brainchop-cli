 #!/bin/bash

git clone https://github.com/wpmed92/tinygrad
cd tinygrad
git fetch
git checkout dawn-python
echo "from .export_model import *" > ./extra/__init__.py
echo "Tinygrad setup completed successfully."
