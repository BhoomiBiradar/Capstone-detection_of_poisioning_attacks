"""
Migration script to move existing files to backend structure
Run this once to reorganize your project
"""
import os
import shutil
from pathlib import Path

def migrate_files():
    """Move existing files to backend structure."""
    base_dir = Path(".")
    
    # Create backend directories
    backend_dirs = [
        "backend/models",
        "backend/detectors",
        "backend/feedback",
        "backend/data",
        "backend/static/outputs",
        "backend/utils/attacks"
    ]
    
    for dir_path in backend_dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    # Move files
    moves = [
        ("models/", "backend/models/"),
        ("detectors/", "backend/detectors/"),
        ("feedback/", "backend/feedback/"),
        ("evaluation_metrics.py", "backend/evaluation_metrics.py"),
    ]
    
    for src, dst in moves:
        src_path = Path(src)
        dst_path = Path(dst)
        
        if src_path.exists():
            if src_path.is_dir():
                # Copy directory contents
                for item in src_path.iterdir():
                    if item.is_file() and not item.name.startswith("__"):
                        final_dst = dst_path / item.name
                        shutil.copy2(item, final_dst)
                        print(f"✓ Moved {item} → {final_dst}")
            else:
                # Copy file - check if dst is a directory or full file path
                if dst_path.suffix:  # dst is a file path (has extension)
                    final_dst = dst_path
                else:  # dst is a directory
                    final_dst = dst_path / src_path.name
                shutil.copy2(src_path, final_dst)
                print(f"✓ Moved {src_path} → {final_dst}")
        else:
            print(f"⚠ {src_path} not found, skipping...")
    
    # Create __init__.py files
    init_files = [
        "backend/__init__.py",
        "backend/models/__init__.py",
        "backend/detectors/__init__.py",
        "backend/feedback/__init__.py",
        "backend/utils/__init__.py",
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✓ Created {init_file}")
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Update imports in your files to use 'backend.' prefix")
    print("2. Run: python backend/api.py to start the backend")
    print("3. Run: streamlit run frontend/streamlit_app.py to start frontend")

if __name__ == "__main__":
    print("Starting migration...")
    migrate_files()

