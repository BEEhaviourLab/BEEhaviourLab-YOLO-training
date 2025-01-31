from setuptools import setup, find_packages

setup(
    name="beeyolo",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'ultralytics',
        'pandas',
        'matplotlib',
        'seaborn',
        'opencv-python',
        'ipykernel',
        'torch',
        'torchvision',
        'torchaudio',
        'numpy',
    ]
)