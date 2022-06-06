from os.path import join
from setuptools import setup
import sys

assert sys.version_info > (3, 6, 0), "Only support Python 3.6 and above."

setup(
    name="bvn",
    py_modules=["rl"],
    install_requires=[
        "cloudpickle",
        "gym[atari,box2d,classic_control]",
        "ipython",
        "joblib",
        "matplotlib",
        "mpi4py",
        "numpy",
        "pandas",
        "pytest",
        "psutil",
        "scipy",
        "torch>=1.5.1",
        "tqdm",
        "params_proto",
        "termcolor",
        "jaynes",
        "requests",
        "ml_logger",
        "pyyaml",
        "sklearn",
        "networkx",
        "graph_search",
        "sparse_graphs"
    ],
    description="A collection of clean implementation of reinforcement learning algorithms",
    author="Annoymous",
    license="MIT",
)
