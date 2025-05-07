import os
import sys
from sumolib import checkBinary


def set_sumo(gui, config_file, **kwargs):
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    if not gui:
        sumo_binary = checkBinary('sumo')
    else:
        sumo_binary = checkBinary('sumo-gui')
    # note that the boolean values in sumo are "true" or "false"
    # see https://sumo.dlr.de/docs/Basics/Notation.html#referenced_data_types
    sumo_cmd = [sumo_binary]
    if config_file is not None:
        sumo_cmd += ["-c", config_file]
    for key, value in kwargs.items():
        option = "--" + key.replace("_", "-")
        value = str(value).lower()
        sumo_cmd += [option, value]
    if gui:
        sumo_cmd += ["--quit-on-end", "true"]
    return sumo_cmd
