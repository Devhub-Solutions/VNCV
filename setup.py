import os
import urllib.request
import zipfile
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

def download_and_extract_weights():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vncv", "weights")
    
    # Check if key models already exist to avoid re-downloading
    essential_files = ["model_encoder.onnx", "model_decoder.onnx", "vocab.json"]
    if all(os.path.exists(os.path.join(weights_dir, f)) for f in essential_files):
        print("[VNCV Setup] Weights already exist, skipping download.")
        return

    url = "https://github.com/Devhub-Solutions/VNCV/releases/download/vocab/vocab.zip"
    os.makedirs(weights_dir, exist_ok=True)
    zip_path = os.path.join(weights_dir, "vocab.zip")
    
    try:
        print(f"[VNCV Setup] Downloading weights from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"[VNCV Setup] Extracting to {weights_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
            
        os.remove(zip_path)
        print("[VNCV Setup] Weights installed successfully.")
    except Exception as e:
        print(f"[VNCV Setup] ERROR: Failed to download weights: {e}")
        # We don't exit(1) here to allow build to continue for dependencies, 
        # but user will get error at runtime.

class CustomBuildPy(build_py):
    def run(self):
        download_and_extract_weights()
        super().run()

class CustomDevelop(develop):
    def run(self):
        download_and_extract_weights()
        super().run()

setup(
    cmdclass={
        'build_py': CustomBuildPy,
        'develop': CustomDevelop,
    }
)
