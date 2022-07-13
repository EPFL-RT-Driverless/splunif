from setuptools import setup, find_packages

setup(
    name="splunif",
    version="1.0.0",
    description="A very simple package that constructs 2D uniform splines, whose continuous parameters correspond to the arc length of the spline.",
    url="https://github.com/EPFL-RT-Driverless/splunif",
    author="Tudor Oancea",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Private :: Do Not Upload",
    ],
    packages=find_packages(include=["splunif"]),
    install_requires=["numpy", "scipy"],
)
