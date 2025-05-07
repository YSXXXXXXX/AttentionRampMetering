import traci
import numpy as np
from typing import List
from Env.detectorGroup import E1Group, E2Group


class OnRamp(object):
    dimension = 9

    def __init__(self, e1_group_list: List[E1Group], e2_group_list: List[E2Group]):
        self.e1_group_list = e1_group_list
        self.e2_group_list = e2_group_list
        self.length_list = np.array([detector.length for detector in self.e2_group_list])
        self.total_length = np.sum(self.length_list)

        self.enter_detector_state = None
        self.exit_detector_state = None

        self.ramp_mean_occupancy = 0
        self.ramp_max_jam_num = 0
        self.ramp_max_vehicle_num = 0

    def update(self):
        self.enter_detector_state = self.e1_group_list[0].get_state()
        self.exit_detector_state = self.e1_group_list[-1].get_state()
        e2_state_list = np.array([detector.get_state() for detector in self.e2_group_list])
        # max occupancy of on-ramp is 1/on-ramp length * \Sigma max_occupancy_i * length_i
        self.ramp_mean_occupancy = np.matmul(e2_state_list[:, 0], self.length_list.T) / self.total_length
        # max vehicle num is \Sigma max_num_i
        self.ramp_max_vehicle_num = np.sum(e2_state_list[:, 1])
        self.ramp_max_jam_num = np.sum(e2_state_list[:, 2])

    def get_state(self):
        return np.concatenate((self.enter_detector_state, self.exit_detector_state,
                               [self.ramp_mean_occupancy, self.ramp_max_vehicle_num, self.ramp_max_jam_num]),
                              dtype=np.float32)


class Observation(object):
    def __init__(self, mainline_e1_detector: List = None, mainline_e2_detector: List = None,
                 on_ramp: List = None):
        if mainline_e1_detector is None:
            self.mainline_e1_detector = []
        else:
            self.mainline_e1_detector = mainline_e1_detector
        if mainline_e2_detector is None:
            self.mainline_e2_detector = []
        else:
            self.mainline_e2_detector = mainline_e2_detector
        if on_ramp is None:
            self.onRamp = []
        else:
            self.onRamp = on_ramp
        self.dimension_info = {"e1": E1Group.dimension * len(self.mainline_e1_detector),
                               "e2": E2Group.dimension * len(self.mainline_e2_detector),
                               "ramp": OnRamp.dimension * len(self.onRamp)}

    def bound_on_ramp(self, on_ramp: OnRamp):
        self.onRamp.append(on_ramp)
        self.dimension_info["ramp"] += OnRamp.dimension

    def bound_mainstream_e1_detector(self, detector: E1Group):
        self.mainline_e1_detector.append(detector)
        self.dimension_info["e1"] += E1Group.dimension

    def bound_mainstream_e2_detector(self, detector: E2Group):
        self.mainline_e2_detector.append(detector)
        self.dimension_info["e2"] += E2Group.dimension

    @property
    def dimensions(self):
        return sum(self.dimension_info.values())

    @property
    def max_state(self):
        return np.inf

    @property
    def min_state(self):
        return float(0)

    def get_traffic_state(self) -> dict:
        state = {"e1": np.zeros((len(self.mainline_e1_detector), E1Group.dimension), dtype=np.float32),
                 "e2": np.zeros((len(self.mainline_e2_detector), E2Group.dimension), dtype=np.float32),
                 "ramp": np.zeros((len(self.onRamp), OnRamp.dimension), dtype=np.float32)}
        for idx, e1 in enumerate(self.mainline_e1_detector):
            state["e1"][idx] = e1.get_state()
        for idx, e2 in enumerate(self.mainline_e2_detector):
            state["e2"][idx] = e2.get_state()
        for idx, ramp in enumerate(self.onRamp):
            state["ramp"][idx] = ramp.get_state()
        # for key, value in state.items():
        #     state[key] = value.flatten()
        return state


class NormalTL(object):
    def __init__(self, traffic_light_id, control_step_length: int, yellow: int):
        self.name = traffic_light_id
        self.control_step_length = control_step_length
        self.yellow = yellow

        self.switch_step = 0
        self.switch_to_phase = 0
        self.current_step = 0
        self.last_cycle_total_green = 0

    @property
    def max_action(self):
        return self.control_step_length

    @property
    def min_action(self):
        return int(0)

    def init_logic(self):
        control_phase = [traci.trafficlight.Phase(0, "") for _ in range(3)]
        current_phase = traci.trafficlight.getAllProgramLogics(self.name)[0].phases
        for phase in current_phase:
            if "r" in phase.state or "s" in phase.state:
                control_phase[2].state = phase.state
                # set phase duration to a big number
                control_phase[2].duration = 3600
            elif "y" in phase.state:
                control_phase[1].state = phase.state
                control_phase[1].duration = self.yellow
            else:
                control_phase[0].state = phase.state
                # set phase duration to a big number
                control_phase[0].duration = 3600
        # traffic logic type is set to 0 (static); currentPhaseIndex is set to 0
        control_logic = traci.trafficlight.Logic(self.name, 0, 0, control_phase)
        traci.trafficlight.setProgramLogic(self.name, control_logic)

    def apply_action(self, total_green: int):
        self.current_step = 0
        if total_green == 0:
            # normal/all red -> all red
            if self.last_cycle_total_green < self.control_step_length:
                traci.trafficlight.setPhase(self.name, 2)
                self.switch_step = 0
            # all green -> all red
            else:
                traci.trafficlight.setPhase(self.name, 1)
                self.switch_step = self.yellow
            self.switch_to_phase = 0
        else:
            traci.trafficlight.setPhase(self.name, 0)
            self.switch_step = total_green
            if not self.yellow == 0:
                self.switch_to_phase = 1
            # do not have yellow phase
            else:
                self.switch_to_phase = 2
        self.last_cycle_total_green = total_green

    def step(self):
        self.current_step += 1
        if self.current_step == self.switch_step:
            traci.trafficlight.setPhase(self.name, self.switch_to_phase)


class OneCarPerGreenTL(object):
    def __init__(self, traffic_light_id, control_step_length: int, one_car_pass_time: int = 2, yellow: int = 1):
        self.name = traffic_light_id
        self.control_step_length = control_step_length
        self.one_car_pass_time = one_car_pass_time
        self.yellow = yellow
        self.slice = self.one_car_pass_time + self.yellow
        self.max_pass = int(self.control_step_length / self.slice)

        self.scheme_list = self.get_schemes()

        self.current_scheme = 0
        self.current_step = 0

    @property
    def max_action(self):
        return self.max_pass

    @property
    def min_action(self):
        return int(0)

    def init_logic(self):
        control_phase = [traci.trafficlight.Phase(0, "") for _ in range(3)]
        current_phase = traci.trafficlight.getAllProgramLogics(self.name)[0].phases
        for phase in current_phase:
            if "r" in phase.state or "s" in phase.state:
                control_phase[2].state = phase.state
                control_phase[2].duration = self.slice
            elif "y" in phase.state:
                control_phase[1].state = phase.state
                control_phase[1].duration = self.yellow
            else:
                control_phase[0].state = phase.state
                control_phase[0].duration = self.one_car_pass_time
        # traffic logic type is set to 0 (static); currentPhaseIndex is set to 0
        control_logic = traci.trafficlight.Logic(self.name, 0, 0, control_phase)
        traci.trafficlight.setProgramLogic(self.name, control_logic)

    def get_schemes(self) -> list:
        scheme_list = []
        for red_num in range(self.max_pass + 1):
            intervals = np.diff(np.linspace(0, self.max_pass - red_num, red_num + 2, dtype=int))
            accumulate_intervals = np.cumsum(intervals).tolist()
            red_idx = np.zeros(self.max_pass, dtype=int)
            for idx, accumulate_interval in enumerate(accumulate_intervals[:-1]):
                red_idx[idx + accumulate_interval] = 2
            scheme_list.append(red_idx.tolist())
        return scheme_list

    def apply_action(self, pass_num: int):
        self.current_scheme = self.max_pass - pass_num
        self.current_step = 0
        traci.trafficlight.setPhase(self.name, self.scheme_list[self.current_scheme][0])

    def all_green(self):
        self.current_step = self.control_step_length  # disable step method
        traci.trafficlight.setPhase(self.name, 0)
        # traci.trafficlight.setPhaseDuration(self.name, self.control_steps - self.yellow)
        traci.trafficlight.setPhaseDuration(self.name, self.control_step_length)

    def step(self):
        self.current_step += 1
        if self.current_step >= self.control_step_length:
            return
        if self.current_step % self.slice != 0:
            return
        else:
            cycle_idx = int(self.current_step // self.slice)
            if self.scheme_list[self.current_scheme][cycle_idx] == self.scheme_list[self.current_scheme][cycle_idx - 1]:
                traci.trafficlight.setPhase(self.name, self.scheme_list[self.current_scheme][cycle_idx])
