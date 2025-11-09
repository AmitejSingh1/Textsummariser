"""
Fix script for transformers import issues
This script helps fix common dependency compatibility problems
"""

import subprocess
import sys

def fix_dependencies():
    print("=" * 60)
    print("Fixing Transformers Dependencies")
    print("=" * 60)
    
    print("\nStep 1: Updating NumPy...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"])
        print("[OK] NumPy updated")
    except Exception as e:
        print(f"[WARN] NumPy update failed: {e}")
    
    print("\nStep 2: Uninstalling conflicting packages...")
    packages_to_remove = ["jax", "jaxlib", "tensorflow"]
    for pkg in packages_to_remove:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[OK] Removed {pkg}")
        except:
            print(f"[INFO] {pkg} not installed or already removed")
    
    print("\nStep 3: Installing compatible versions...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "numpy>=1.26.0", "transformers", "torch", "sentencepiece"])
        print("[OK] Dependencies installed")
    except Exception as e:
        print(f"[ERROR] Installation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[OK] Fix complete! Please restart your Streamlit app.")
    print("=" * 60)
    
    print("\nTesting import...")
    try:
        from transformers import pipeline
        print("[OK] Transformers pipeline imported successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Import still failing: {e}")
        print("\nTips:")
        print("   1. Restart your terminal/command prompt")
        print("   2. Create a fresh virtual environment")
        print("   3. Use extractive methods only (they work great!)")
        return False

if __name__ == "__main__":
    fix_dependencies()

