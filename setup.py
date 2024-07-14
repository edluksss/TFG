from setuptools import setup, find_packages

setup(
    name='pnebulae_torch',  # Aquí va el nombre de tu biblioteca
    version='0.1',  # Aquí va la versión de tu biblioteca
    packages=find_packages(), # Encuentra las carpetas con código fuente
    package_dir={'': '.'}
)
