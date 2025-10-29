


python3.11 -m venv ".venv"
source ".venv/bin/activate"

pip install --upgrade pip
pip install ipykernel
ipython kernel install --user --name="venv"


pip install dapla-toolbelt
pip install scikit-learn
pip install matplotlib