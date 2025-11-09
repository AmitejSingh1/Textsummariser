"""
Safe installation script for transformers library
This script helps install transformers without TensorFlow dependencies
"""

import subprocess
import sys

def install_packages():
    print("=" * 60)
    print("Installing Transformers (CPU-only, no TensorFlow)")
    print("=" * 60)
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "transformers",
        "sentencepiece"
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + package.split())
            print(f"✅ Successfully installed {package.split()[0]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package.split()[0]}: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ Installation complete!")
    print("=" * 60)
    print("\nPlease restart your Streamlit app to use abstractive methods.")
    return True

if __name__ == "__main__":
    install_packages()

