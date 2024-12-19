import subprocess

subprocess.call(["conda", "install", "--channel", "conda-forge", "pyvips"])

with open('requirements.txt') as f:
    lines = f.readlines()
    for line in lines:
        package_name = line.strip()
        if not package_name.startswith("#"):
            print(f"Installing {package_name}")
            subprocess.call(["pip", "install", package_name])

