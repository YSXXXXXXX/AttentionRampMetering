import traci
import numpy as np
from abc import ABC
import traci.constants as tc


class DetectorGroup(object):
    def __init__(self, detector_id: str, idx, pos, edge):
        if type(detector_id) == str:
            self.detector_id = []
            self.detector_id.append(detector_id)

        self.detector_num = len(self.detector_id)
        self.idx = idx
        self.pos = pos
        self.edge = edge

    def append(self, detector_id):
        self.detector_id.append(detector_id)
        self.detector_num += 1

    def reset(self):
        raise NotImplementedError("please override this method")

    def subscribe(self):
        raise NotImplementedError("please override this method")

    def step(self):
        raise NotImplementedError("please override this method")

    def filter(self):
        raise NotImplementedError("please override this method")

    def get_state(self, from_step, to_step):
        raise NotImplementedError("please override this method")


class E1Group(DetectorGroup, ABC):
    dimension = 3

    def __init__(self, detector_id: str, idx, pos, edge):
        super(E1Group, self).__init__(detector_id, idx, pos, edge)
        self.mean_speed = []
        self.mean_occupancy = []
        self.accumulated_vehicle_num = []
        self.last_step_vehicle_id: set = set()

    def append(self, detector_id):
        super(E1Group, self).append(detector_id)

    def reset(self):
        self.mean_speed = []
        self.mean_occupancy = []
        self.accumulated_vehicle_num = []
        # do not reset vehicle id record
        # self.last_step_vehicle_id = set()

    def subscribe(self):
        subscribe = [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_OCCUPANCY, tc.LAST_STEP_VEHICLE_ID_LIST]
        for detectorID in self.detector_id:
            traci.inductionloop.subscribe(detectorID, subscribe)

    def step(self):
        arrive_vehicle_id, curr_veh, curr_occ, curr_speed = [], [], [], []
        for detectorID in self.detector_id:
            result = traci.inductionloop.getSubscriptionResults(detectorID)
            arrive_vehicle_id += list(result[tc.LAST_STEP_VEHICLE_ID_LIST])
            curr_speed.append(result[tc.LAST_STEP_MEAN_SPEED])
            # note that the mean speed return -1 when no vehicle passed
            if result[tc.LAST_STEP_MEAN_SPEED] >= 0.:
                curr_veh.append(len(set(result[tc.LAST_STEP_VEHICLE_ID_LIST]) - self.last_step_vehicle_id))
                curr_occ.append(result[tc.LAST_STEP_OCCUPANCY])
            else:
                # no vehicle passed in last simulation step
                curr_veh.append(0)
                curr_occ.append(0.)
        accumulated_vehicle_num = len(set(arrive_vehicle_id) - self.last_step_vehicle_id)
        self.last_step_vehicle_id = set(arrive_vehicle_id)
        self.accumulated_vehicle_num.append(accumulated_vehicle_num)
        # calculate mean speed among lanes, if no vehicle passes, return -1
        # self.mean_speed.append(np.average(curr_speed, weights=curr_veh) if sum(curr_veh) > 0 else -1.)
        self.mean_speed.append(np.mean(curr_speed, where=(np.array(curr_veh) > 0)) if sum(curr_veh) > 0 else -1.)
        self.mean_occupancy.append(np.average(curr_occ))

    def get_state(self, from_step=None, to_step=None):
        # the mean traffic state of [from_step, to_step)
        mean_speed = self.get_mean_speed(from_step, to_step)
        mean_occupancy = self.get_mean_occupancy(from_step, to_step)
        accumulated_vehicle_num = self.get_accumulated_veh_num(from_step, to_step)
        return np.array([mean_speed, mean_occupancy, accumulated_vehicle_num], dtype=np.float32)

    def get_mean_speed(self, from_step=None, to_step=None):
        from_step, to_step = self.get_step(from_step, to_step)
        speed_array = np.array(self.mean_speed[from_step: to_step], dtype=np.float32)
        if (speed_array >= 0.).any():
            # exclude the time that no vehicle passes
            if (speed_array <= 0.).all():
                # all passing vehicles speed is zero (only -1 and 0. in speed array)
                return 0.
            return self.harmonic_mean(np.array(speed_array), where=(speed_array > 0.))  # exclude zero speed
        else:
            # no vehicle passes during from_step to to_step, return free flow speed
            return traci.lane.getMaxSpeed("{}_0".format(self.edge))

    def get_mean_occupancy(self, from_step=None, to_step=None):
        from_step, to_step = self.get_step(from_step, to_step)
        return np.mean(self.mean_occupancy[from_step: to_step])

    def get_accumulated_veh_num(self, from_step=None, to_step=None):
        from_step, to_step = self.get_step(from_step, to_step)
        return sum(self.accumulated_vehicle_num[from_step: to_step])

    def get_step(self, from_step=None, to_step=None):
        from_step = from_step % len(self.mean_speed) if from_step is not None else 0
        to_step = to_step % len(self.mean_speed) if to_step is not None else len(self.mean_speed)
        return from_step, to_step

    @staticmethod
    def harmonic_mean(x: np.ndarray, where):
        return 1 / np.mean(1 / x[where])


class E2Group(DetectorGroup, ABC):
    dimension = 3

    def __init__(self, detector_id: str, idx, pos, edge, length):
        super(E2Group, self).__init__(detector_id, idx, pos, edge)
        self.jam_vehicle_num = []
        self.vehicle_num = []
        self.mean_occupancy = []
        self.length = length

    def reset(self):
        # the jam length during the statics period
        self.jam_vehicle_num = []
        # the vehicle length during the statics period
        self.vehicle_num = []
        # the mean occupancy during the statics period
        self.mean_occupancy = []

    def subscribe(self):
        subscribe = [tc.JAM_LENGTH_VEHICLE, tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_OCCUPANCY]
        for detectorID in self.detector_id:
            traci.lanearea.subscribe(detectorID, subscribe)

    def step(self):
        curr_jam_num, curr_vehicle_num, curr_occ = 0, 0, []
        for detectorID in self.detector_id:
            result = traci.lanearea.getSubscriptionResults(detectorID)
            curr_jam_num += result[tc.JAM_LENGTH_VEHICLE]
            curr_vehicle_num += result[tc.LAST_STEP_VEHICLE_NUMBER]
            curr_occ.append(result[tc.LAST_STEP_OCCUPANCY])
        self.jam_vehicle_num.append(curr_jam_num)
        self.vehicle_num.append(curr_vehicle_num)
        self.mean_occupancy.append(np.mean(curr_occ))

    def get_state(self, from_step=None, to_step=None):
        # the mean traffic state [from_step, to_step)
        from_step, to_step = self.get_step(from_step, to_step)
        mean_occupancy = np.mean(self.mean_occupancy[from_step: to_step])
        max_vehicle_num = np.max(self.vehicle_num[from_step: to_step])
        max_jam_vehicle_num = np.max(self.jam_vehicle_num[from_step: to_step])
        return np.array([mean_occupancy, max_vehicle_num, max_jam_vehicle_num], dtype=np.float32)

    def get_step(self, from_step=None, to_step=None):
        from_step = from_step % len(self.mean_occupancy) if from_step is not None else 0
        to_step = to_step % len(self.mean_occupancy) if to_step is not None else len(self.mean_occupancy)
        return from_step, to_step
