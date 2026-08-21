import site
import sys
import json
import os

# NumPy 2.0 compat: wandb 0.16.6 uses np.float_/np.complex_ (removed in NumPy 2).
# This alias must be set BEFORE any wandb import (including in spawned service processes).
import numpy as _np
if not hasattr(_np, "float_"):
    _np.float_ = _np.float64
if not hasattr(_np, "complex_"):
    _np.complex_ = _np.complex128

# Keep wandb quiet + non-interactive: no TTY prompt rendering, no progress bar
# spamming stderr with tcsetattr errors when stdout is not a TTY.
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_MODE", "online")
os.environ.setdefault("WANDB_PROGRAM", "carnage")

wandb_run = None

# Takes in the python executable path, the three wandb init strings, and optionally the current run ID
# Returns the ID of the run (either newly created or resumed)
def init(py_exec_path, project, group, name, id = None):

	global wandb_run
	
	# Fix the path of our interpreter so wandb doesn't run RLGym_PPO instead of Python
	# Very strange fix for a very strange problem
	sys.executable = py_exec_path
	
	try:
		site_packages_dir = os.path.join(os.path.join(os.path.dirname(py_exec_path), "Lib"), "site-packages")
		sys.path.append(site_packages_dir)
		site.addsitedir(site_packages_dir)
		import wandb
	except Exception as e:
		raise Exception(f"""
			FAILED to import wandb! Make sure GigaLearnCPP isn't using the wrong Python installation.
			This installation's site packages: {site.getsitepackages()}
			Exception: {repr(e)}"""
		)
	
	print("Calling wandb.init()...")
	if not (id is None) and len(id) > 0:
		wandb_run = wandb.init(project = project, group = group, name = name, id = id, resume = "allow")
	else:
		wandb_run = wandb.init(project = project, group = group, name = name)
	return wandb_run.id

def add_metrics(metrics):
	global wandb_run
	if wandb_run is None:
		return
	try:
		wandb_run.log(metrics)
	except Exception as e:
		print(f"[metric_receiver] wandb.log failed (ignored): {e!r}")