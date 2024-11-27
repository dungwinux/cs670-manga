import subprocess

# Read requirements file
with open('requirements.conda.txt') as f:
    packages = f.readlines()

# Try installing each package individually
for package in packages:
    package = package.strip()
    if package:
        print(f"Attempting to install: {package}")
        result = subprocess.run(["pip", "install", package], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Skipping {package}: {result.stderr}")
        else:
            print(f"Installed {package} successfully")
