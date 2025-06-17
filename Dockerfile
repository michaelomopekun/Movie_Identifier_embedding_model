# Install all other dependencies
RUN pip install -r requirements.txt

# Then install torch (CPU-only) manually
RUN pip install torch==2.2.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
