#!/usr/bin/env python3
"""
Fix pytorch-forecasting import compatibility issues
Replaces lightning.pytorch imports with pytorch_lightning
"""

import os
import sys
import subprocess
from pathlib import Path

def find_pytorch_forecasting_path():
    """Find the pytorch-forecasting installation path"""
    try:
        import pytorch_forecasting
        return Path(pytorch_forecasting.__file__).parent
    except ImportError:
        print("❌ pytorch-forecasting not found. Please install it first:")
        print("pip install pytorch-forecasting")
        sys.exit(1)

def fix_imports_in_file(file_path):
    """Fix lightning.pytorch imports in a single file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the problematic imports
        original_content = content
        content = content.replace('lightning.pytorch', 'pytorch_lightning')
        
        if content != original_content:
            # Create backup
            backup_path = str(file_path) + '.backup'
            with open(backup_path, 'w') as f:
                f.write(original_content)
            
            # Write fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            
            return True
        return False
    except Exception as e:
        print(f"⚠️ Could not fix {file_path}: {e}")
        return False

def main():
    print("🔧 Fixing pytorch-forecasting import compatibility...")
    
    # Find pytorch-forecasting installation
    pf_path = find_pytorch_forecasting_path()
    print(f"📍 Found pytorch-forecasting at: {pf_path}")
    
    # Files that need fixing
    files_to_fix = [
        pf_path / "models" / "base_model.py",
        pf_path / "utils" / "_utils.py",
        pf_path / "models" / "temporal_fusion_transformer" / "tuning.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        if file_path.exists():
            print(f"🔄 Fixing {file_path.name}...")
            if fix_imports_in_file(file_path):
                print(f"✅ Fixed {file_path.name}")
                fixed_count += 1
            else:
                print(f"ℹ️ No changes needed in {file_path.name}")
        else:
            print(f"⚠️ File not found: {file_path}")
    
    print(f"\n🎉 Fix completed! Modified {fixed_count} files.")
    
    # Test the fix
    try:
        import pytorch_lightning as pl
        from pytorch_forecasting import TemporalFusionTransformer
        
        is_lightning_module = issubclass(TemporalFusionTransformer, pl.LightningModule)
        
        if is_lightning_module:
            print("✅ TFT is now properly recognized as LightningModule!")
        else:
            print("❌ Fix may not have worked. TFT still not recognized as LightningModule.")
            
    except Exception as e:
        print(f"❌ Error testing fix: {e}")

if __name__ == "__main__":
    main() 