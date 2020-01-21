from setuptools import setup, find_packages
setup(
    name="cyphy2cad_postprocess",
    version="0.1.1",
    package_dir={"": "src"},
    packages=["cyphy2cad_postprocess"],
    install_requires=["numpy"]
)