from os.path import join
from setuptools import setup
import sys

assert sys.version_info > (3, 6, 0), "Only support Python 3.6 and above."

setup(
    name="bvn",
    py_modules=["rl"],
    install_requires=[
        "wandb",
        "cloudpickle",
        "opencv-python",
        "gym[atari,box2d,classic_control]==0.20.0",
        "mujoco-py==2.0.2.8",
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
        "sparse_graphs",
        "graphene==2.1.3",
        "graphql-core==2.1",
        "graphql-relay==0.4.5",
        "graphql-server-core==1.1.1"
    ],
    description="A collection of clean implementation of reinforcement learning algorithms",
    author="Annoymous",
    license="MIT",
)
