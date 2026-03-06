import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Script to run all pytest tests for the DroneLocalization project.
    It automatically adds the project root to PYTHONPATH and runs pytest.
    """
    # Get the project root directory (one level up from the scripts directory)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    print(f"==================================================")
    print(f"🚀 Running DroneLocalization Test Suite")
    print(f"📂 Project Root: {project_root}")
    print(f"==================================================\n")

    # Set the working directory to the project root
    os.chdir(project_root)

    # Use the virtual environment python if it exists, otherwise use system python
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    python_exec = str(venv_python) if venv_python.exists() else sys.executable

    # Define the pytest command with coverage
    cmd = [
        python_exec,
        "-m",
        "pytest",
        "tests",
        "--cov=src",
        "--cov-report=term-missing",
        "-v"
    ]

    print(f"Executing command: {' '.join(cmd)}\n")

    try:
        # Run pytest
        result = subprocess.run(cmd, check=False)
        
        print("\n==================================================")
        if result.returncode == 0:
            print("✅ All tests passed successfully!")
            sys.exit(0)
        else:
            print(f"❌ Tests failed with exit code: {result.returncode}")
            sys.exit(result.returncode)
            
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
