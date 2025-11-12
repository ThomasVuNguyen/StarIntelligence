# Pretraining arc lm

git clone https://github.com/huggingface/nanotron
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
cd nanotron
pip install -e .
pip install datasets transformers datatrove[io] numba wandb
pip install wheel ninja triton flash-attn --no-build-isolation
huggingface-cli login
wandb login

cd ..