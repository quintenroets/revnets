import cli
import numpy as np

from .config import get_num_devices
from .path import Path


def launch_experiments(devices: list = None):
    if devices is None:
        devices = list(range(get_num_devices()))

    config_names = [
        path.relative_to(Path.config_assets).with_suffix("")
        for path in Path.active_config.rglob("*.yaml")
    ]
    num_experiments = len(config_names)
    num_devices = len(devices)
    num_per_device = num_experiments // num_devices
    device_numbers = [num_per_device] * num_devices
    for i in range(num_experiments % num_devices):
        device_numbers[i] += 1
    experiment_end_indices = np.cumsum(device_numbers)

    start = 0
    for i, device_idx in enumerate(devices):
        stop = experiment_end_indices[i]
        device_config_names = [config_names[i] for i in range(start, stop)]
        device_command = f"export CUDA_VISIBLE_DEVICES={device_idx}; "
        device_command += "; ".join(
            f"mtl_labels --config-name {name}" for name in device_config_names
        )
        tmux_command = f"tmux new-session -s session{i+1} '{device_command}'"
        cli.run(tmux_command, shell=True, wait=False)
        start = stop
