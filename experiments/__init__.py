from os.path import dirname, join

from ml_logger import RUN, instr
from termcolor import colored

assert instr  # single-entry for the instrumentation thunk factory
RUN.project = "bvn"  # Specify the project name
RUN.script_root = dirname(__file__)  # specify that this is the script root.
print(colored('set', 'blue'), colored("RUN.script_root", "yellow"), colored('to', 'blue'),
      RUN.script_root)