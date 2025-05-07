from abc import ABC

import gymnasium as gym
import networkx as nx
import pandas as pd
import traci
import sumolib
import numpy as np
import traci.constants as tc
from collections import defaultdict
# from gym.utils import seeding

import Env.quickStart as qs
from Env.sumoStart import set_sumo
from Env.parseDetectors import build_detector_group
from Env.detectorGroup import E1Group, E2Group
from Env.trafficElement import OnRamp, OneCarPerGreenTL, NormalTL


class MultiRampEnv(gym.Env, ABC):

    _state = {"mainline": ["density", "accumulate_veh_num"],
              "onramp": ["density", "accumulate_vehicle_num"]}

    def __init__(self, ctrl_step_length: int, simulation_precision: float, total_step: int, warmup: int, port: int,
                 gui: bool, terminate_when_unhealthy: bool, network_filename: str, detector_frame: pd.DataFrame,
                 detector_filename: str, sumocfg_filename: str, **kwargs):
        self.num_ramps = qs.num_of_ramps
        # total number of SUMO simulation steps
        self.total_step = total_step
        # step length for each control action
        self.ctrl_step_length = ctrl_step_length
        # real world time that one SUMO simulation step represents
        self.simulation_precision = simulation_precision
        self.warmup = warmup
        self.gui = gui
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.sumocfg = sumocfg_filename
        self.port = port
        self.connection_label = "init_connection_{}".format(str(self.port))
        self.kwargs = kwargs
        # note that the first policy is applied at step 0
        self.max_control_step = 1 + (self.total_step - 1) // self.ctrl_step_length
        self.current_control_step = 0
        self.current_step = 0
        # ---------- initialize detectors ----------
        self.e1Detector, self.e2Detector = build_detector_group(detector_filename)
        # ---------- initialize network ----------
        self.net = nx.DiGraph()
        self.link2detector = defaultdict(list)  # {link1: [E1, E1, E1], ...} in detector: [:-1], out detector: [1:]
        self.edge2node = defaultdict(tuple)  # {edge_id: (from_node_id, to_node_id), ...}
        # aggregate traffic state for every model_step_length
        self.aggregate_traffic_state = defaultdict(dict)  # {link1: {num vehicle: [], speed: [], flow: []}, ...}
        self.build_network(network_filename, self.e1Detector, detector_frame)
        # ---------- initialize traffic lights ----------
        self.tls = []
        for i in range(self.num_ramps):
            if qs.tl_type[i] == "OneCarPerGreen":
                self.tls.append(
                    OneCarPerGreenTL(qs.tl_name[i], self.ctrl_step_length, qs.tl_yellow[i], qs.tl_pass[i]))
            elif qs.tl_type[i] == "Normal":
                self.tls.append(NormalTL(qs.tl_name[i], self.ctrl_step_length, qs.tl_yellow[i]))
            else:
                raise ValueError("No such traffic light class")
        # build action space
        self.action_space = gym.spaces.Box(0., 1., (self.num_ramps,))
        # ---------- initialize on-ramps ----------
        self.ramps = []
        for i in range(self.num_ramps):
            e1 = [self.e1Detector[idx] for idx in qs.ramp_e1Detector[i]]
            e2 = [self.e2Detector[idx] for idx in qs.ramp_e2Detector[i]]
            self.ramps.append(OnRamp(e1, e2))
        # ---------- initialize agent observation ----------
        observation_dim = len(self._state["mainline"]) * len(qs.mainline_observed_segment)  # mainline dim
        observation_dim += len(self._state["onramp"]) * len(qs.onramp_observed_segment)  # on-ramp dim
        # observation_dim += qs.num_of_ramps  # last control step action dim
        self.observation_space = gym.spaces.Box(0., np.inf, (observation_dim,))  # add action dimension
        self.num_mainline_cell, self.num_ramp_cell = len(qs.mainline_observed_segment), len(qs.onramp_observed_segment)
        # ---------- initialize recorder ----------
        self.action_record = np.zeros((self.max_control_step, self.num_ramps), dtype=np.float32)
        self.running_vehicles = np.zeros(self.total_step)
        self.pending_vehicles = np.zeros(self.total_step)
        self.throughput = np.zeros(self.total_step)
        # the default policy is all green
        self.null_policy = np.ones(self.num_ramps, dtype=np.float32)

    # def seed(self, seed=None):
    #     # this method has been removed in gymnasium
    #     self.np_random, seed = seeding.np_random(seed)
    #     return [seed]

    def set_aggregate_parameters(self, aggregate_length):
        # calculate the average values across aggregate_length
        from functools import partial
        self.refresh_aggregate_traffic_state = partial(self.refresh_aggregate_traffic_state, aggregate_length=aggregate_length)

    def reset(self, seed=None, options=None):
        super(MultiRampEnv, self).reset(seed=seed)
        self.current_control_step = 0
        self.current_step = 0
        sumo_cmd = set_sumo(self.gui, self.sumocfg, step_length=self.simulation_precision, **self.kwargs)
        traci.start(sumo_cmd, port=self.port, label=self.connection_label)
        traci.main.switch(label=self.connection_label)
        for tl in self.tls:
            tl.init_logic()
        self.subscribe()
        self.reset_recorder()
        self.reset_aggregate_traffic_state()
        self.reset_detector()
        observations, _, _, _, info = self.step(self.null_policy)  # at least execute once
        while not self.can_start():
            observations, _, _, _, info = self.step(self.null_policy)
        return observations, info

    def step(self, action: np.ndarray):
        traci.main.switch(label=self.connection_label)
        self.action_record[self.current_control_step] = action
        action = np.round(action * self.ctrl_step_length, 0)
        self.apply_action(action)
        target_step = min(self.ctrl_step_length * (self.current_control_step + 1), self.total_step)
        while self.current_step < target_step:
            self.inner_step()
            self.current_step += 1
        # update on ramp state after a control step
        for i in range(self.num_ramps):
            self.ramps[i].update()
        self.refresh_aggregate_traffic_state()
        observation = self.get_observation()
        reward = self.compute_reward()
        attach = self.get_detector_state()
        info = {"step_tts": self.get_tts(self.current_step - self.ctrl_step_length, self.current_step), "occ": attach["ramp"][:, 6]}
        # deal with terminated and truncated
        terminated = self.terminate_when_unhealthy and self.is_unhealthy()
        truncated = self.current_step >= self.total_step
        if terminated or truncated:
            traci.close()
        else:
            # reset the counters in E1/E2 detector
            self.reset_detector()
        self.current_control_step += 1
        return observation, reward, terminated, truncated, info
    
    def close(self):
        if self.connection_label in traci.main._connections:
            traci.main.switch(label=self.connection_label)
            traci.close()
        else:
            return

    def can_start(self):
        if self.throughput[self.current_step - 1] > 0 and self.current_step >= self.warmup:
            return True
        else:
            return False

    def is_unhealthy(self):
        for i in range(self.num_ramps):
            if self.ramps[i].ramp_max_vehicle_num < 0.9 * qs.ramp_max_queue_length[i]:
                return False
        return True

    def reset_recorder(self):
        self.action_record = np.zeros((self.max_control_step, self.num_ramps), dtype=np.float32)
        self.running_vehicles = np.zeros(self.total_step)
        self.pending_vehicles = np.zeros(self.total_step)
        self.throughput = np.zeros(self.total_step)

    def reset_detector(self):
        for e1 in self.e1Detector.values():
            e1.reset()
        for e2 in self.e2Detector.values():
            e2.reset()
    
    def reset_aggregate_traffic_state(self):
        aggregate_traffic_state = {}
        for link, state in self.aggregate_traffic_state.items():
            aggregate_traffic_state[link] = {}
            for key, value in state.items():
                aggregate_traffic_state[link][key] = np.zeros_like(value)
        self.aggregate_traffic_state = aggregate_traffic_state

    def subscribe(self):
        global_subscribe = [tc.VAR_ARRIVED_VEHICLES_NUMBER, tc.VAR_DEPARTED_VEHICLES_NUMBER, tc.VAR_PENDING_VEHICLES]
        traci.simulation.subscribe(global_subscribe)
        for e1 in self.e1Detector.values():
            e1.subscribe()
        for e2 in self.e2Detector.values():
            e2.subscribe()

    def inner_step(self):
        traci.simulationStep()
        result = traci.simulation.getSubscriptionResults()
        delta = result[tc.VAR_DEPARTED_VEHICLES_NUMBER] - result[tc.VAR_ARRIVED_VEHICLES_NUMBER]
        # when the current step is 0, self.running_vehicles[-1] is 0
        self.running_vehicles[self.current_step] = self.running_vehicles[self.current_step - 1] + delta
        self.pending_vehicles[self.current_step] = len(result[tc.VAR_PENDING_VEHICLES])
        for e1 in self.e1Detector.values():
            e1.step()
        for e2 in self.e2Detector.values():
            e2.step()
        for tl in self.tls:
            tl.step()
        throughput = 0
        for idx in qs.throughput_detector_idx:
            throughput += self.e1Detector[idx].get_accumulated_veh_num()
        self.throughput[self.current_step] = throughput

    def apply_action(self, action):
        for idx, a in enumerate(action):
            self.tls[idx].apply_action(a)

    def get_observation(self) -> np.ndarray:
        mainline_ob = np.zeros((len(qs.mainline_observed_segment), len(self._state["mainline"])), dtype=np.float32)
        ramp_ob = np.zeros((len(qs.onramp_observed_segment), len(self._state["onramp"])), dtype=np.float32)
        for i, (link_name, segment_idx) in enumerate(qs.mainline_observed_segment):
            mainline_ob[i][0] = self.aggregate_traffic_state[link_name]["density"][segment_idx]
            mainline_ob[i][1] = self.aggregate_traffic_state[link_name]["ctrl_step_flow"][segment_idx]
        for i, (link_name, segment_idx) in enumerate(qs.onramp_observed_segment):
            ramp_ob[i][0] = self.aggregate_traffic_state[link_name]["density"][segment_idx]
            ramp_ob[i][1] = self.aggregate_traffic_state[link_name]["ctrl_step_flow"][segment_idx]
            # ramp_ob[i][2] = self.aggregate_traffic_state[link_name]["w"][segment_idx]
        # last_action = self.action_record[self.current_control_step]
        # return np.concatenate((mainline_ob.flatten(), ramp_ob.flatten(), last_action.flatten()))
        return np.concatenate((mainline_ob, ramp_ob)).flatten()

    def get_detector_state(self):
        detector_state = {"e1": np.zeros((len(self.e1Detector), E1Group.dimension), dtype=np.float32),
                          "e2": np.zeros((len(self.e2Detector), E2Group.dimension), dtype=np.float32),
                          "ramp": np.zeros((qs.num_of_ramps, OnRamp.dimension), dtype=np.float32)}
        for idx, e1 in enumerate(self.e1Detector.values()):
            detector_state["e1"][idx] = e1.get_state()
        for idx, e2 in enumerate(self.e2Detector.values()):
            detector_state["e2"][idx] = e2.get_state()
        for idx, ramp in enumerate(self.ramps):
            detector_state["ramp"][idx] = ramp.get_state()
        return detector_state

    def refresh_aggregate_traffic_state(self, aggregate_length):
        for link_id in self.link2detector.keys():
            detector_num = len(self.link2detector[link_id])
            if detector_num >= 2:
                # exclude the self-loop (Virtual Origin)
                assert detector_num == self.net.edges[self.edge2node[link_id]]["segment_num"] + 1
                speed, model_step_speed = np.zeros(detector_num, dtype=np.float32), np.zeros(detector_num, dtype=np.float32)
                accumulated_veh = np.zeros(detector_num, dtype=int)
                model_step_accumulated_veh = np.zeros(detector_num, dtype=int)
                for i, detector in enumerate(self.link2detector[link_id]):
                    # get last ctrl step speed, return in m/s
                    speed[i] = detector.get_mean_speed()
                    # get last model step speed, return in m/s
                    model_step_speed[i] = detector.get_mean_speed(-1 * aggregate_length)
                    # get last model step's accumulate vehicle, to calculate the outflow
                    model_step_accumulated_veh[i] = detector.get_accumulated_veh_num(-1 * aggregate_length)
                    # get the whole control step's accumulate vehicle. to calculate the density
                    accumulated_veh[i] = detector.get_accumulated_veh_num()
                # flow is in veh/h, record outflow, index begins with 1 (the out detector of first segment)
                self.aggregate_traffic_state[link_id]["flow"] = model_step_accumulated_veh[1:] / aggregate_length * 3600
                self.aggregate_traffic_state[link_id]["ctrl_step_flow"] = accumulated_veh[1:] / self.ctrl_step_length * 3600
                length, lane_num = self.net.edges[self.edge2node[link_id]]["length"], self.net.edges[self.edge2node[link_id]]["lane_num"]
                # density is in veh/km
                self.aggregate_traffic_state[link_id]["density"] += (accumulated_veh[:-1] - accumulated_veh[1:]) / (length * lane_num)
                self.aggregate_traffic_state[link_id]["w"] += accumulated_veh[:-1] - accumulated_veh[1:]
                # speed is in km/h
                self.aggregate_traffic_state[link_id]["speed"] = model_step_speed[1:] * 3.6
                self.aggregate_traffic_state[link_id]["ctrl_step_speed"] = speed[1:] * 3.6

    def compute_reward(self):
        sub_tts = self.get_tts(self.current_step - self.ctrl_step_length, self.current_step) / self.ctrl_step_length
        # Damn reward function...
        # After so many failures...
        # Just keep it **simple** and **stupid** in ramp metering scenario.
        reward = 1.5 * (2050 - sub_tts) / 134
        return reward

    def get_tts(self, start, end):
        # return in second
        return np.sum(self.running_vehicles[start: end]) + np.sum(self.pending_vehicles[start: end])

    def build_network(self, network_file, detector_entity, detector_frame):
        net = sumolib.net.readNet(network_file)
        detector_frame = detector_frame[(detector_frame["detector"] == "E1") | (detector_frame["detector"] == "e1")]
        edge_ids = set(detector_frame["edge"])  # only process the edges that have detectors
        sorted_detector = detector_frame.groupby("edge", group_keys=False).apply(lambda x: x.sort_values(by="pos"))
        for edge_id in edge_ids:
            edge = net.getEdge(edge_id)
            edge_detector = sorted_detector[sorted_detector["edge"] == edge_id]
            from_node_id, to_node_id = edge.getFromNode().getID(), edge.getToNode().getID()
            # add edge for the network
            self.net.add_edge(from_node_id, to_node_id)
            self.edge2node[edge_id] = (from_node_id, to_node_id)
            num_segment = len(edge_detector) - 1  # have detectors on both sides
            self.net.edges[(from_node_id, to_node_id)]["segment_num"] = num_segment
            self.net.edges[(from_node_id, to_node_id)]["lane_num"] = edge.getLaneNumber()
            self.net.edges[(from_node_id, to_node_id)]["length"] = np.zeros(num_segment)
            # -------- init the traffic state record dict ----------
            self.aggregate_traffic_state[edge_id]["density"] = np.zeros(num_segment, dtype=np.float32)
            self.aggregate_traffic_state[edge_id]["speed"] = np.zeros(num_segment, dtype=np.float32)
            self.aggregate_traffic_state[edge_id]["flow"] = np.zeros(num_segment, dtype=np.float32)
            self.aggregate_traffic_state[edge_id]["w"] = np.zeros(num_segment, dtype=np.float32)  # number of vehicles
            # init the segment in detector and out detector
            for i, (_, detector) in enumerate(edge_detector.iterrows()):
                self.link2detector[edge_id].append(detector_entity[str(detector["id"])])
                if i < self.net.edges[(from_node_id, to_node_id)]["segment_num"]:
                    self.net.edges[(from_node_id, to_node_id)]["length"][i] = detector["pos"]  # record start pos here
                if i > 0:
                    length = detector["pos"] - self.net.edges[(from_node_id, to_node_id)]["length"][i - 1]
                    length /= 1000  # in km
                    self.net.edges[(from_node_id, to_node_id)]["length"][i - 1] = length
        del net

    @staticmethod
    def get_flatten_observation(ob_dict):
        observation = np.array([], dtype=np.float32)
        for value in ob_dict.values():
            observation = np.append(observation, value)  # `append` method can flatten the vector
        return observation
