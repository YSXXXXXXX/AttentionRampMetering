# This python file records the necessary constants used in multi-ramp environment.
# The order of the list should obey the driving direction.

# number of agents
num_of_agents = 1
# number of ramps
num_of_ramps = 4
# the name of each ramp
ramps_name = ["ONR0_1", "ONR1_1", "ONR2_1", "ONR3_1"]

# the capacity of mainline (veh/h)
mainline_capacity = 8100
# the capacity of on-ramp (veh/h)
ramp_capacity = [2000, 2100, 1800, 2100]
# the speed limit of mainline (m/s)
mainline_max_speed = 27.78
# the critical occupancy of mainline (%)
mainline_critical_occupancy = [14.5, 12.0, 11.0, 12.0]
# the speed limit of on-ramp (m/s)
ramp_max_speed = 13.89
# the maximum number of cars that on-ramp can contain
ramp_max_queue_length = [54, 54, 31, 61]
# ramp_max_queue_length = [106, 112, 60, 142]
# the maximum occupancy of on-ramp
ramp_max_occupancy = [72., 72., 72., 72.]
# ramp_max_occupancy = [55., 72., 72., 72.]

# the name of traffic lights
tl_name = ["ON0_1", "ON1_1", "ON2_1", "ON3_1"]
# tl_name = ["N1", "N5", "N7", "N10"]
# the class of traffic lights, "OneCarPerGreen" or "Normal"
tl_type = ["Normal", "Normal", "Normal", "Normal"]
# yellow light duration of each traffic light
tl_yellow = [1., 1., 1., 1.]
# one car pass duration of each traffic light
tl_pass = [2., 2., 2., 2.]
# -------- Multi-agent --------
# the index of e1 detectors on each ramp
ramp_e1Detector = [["32", "35"], ["36", "39"], ["40", "43"], ["44", "47"]]
# the index of e2 detectors on each ramp
# ramp_e2Detector = [["0", "1"], ["2", "3"], ["4", "5"], ["6", "7"]]
ramp_e2Detector = [["0"], ["2"], ["4"], ["6"]]
# the index of e1 detectors on each agent observation
ob_e1Detector = ["1", "2", "4", "6", "13", "14", "16", "18", "19", "21", "23", "25", "26", "28", "30"]
# the index of e2 detectors on each agent observation
ob_e2Detector = []
# the index of e1 detector that used to measure the throughput
throughput_detector_idx = ["31", "49", "51", "53", "55"]
# the mainline link name and segment number that can be observed by the RL agent [(link name, segment number), ...]
mainline_observed_segment = [("M1", 0), ("M1", 1), ("M2", 0), ("M3", 0),
                             ("M5", 0), ("M5", 1), ("M6", 0),
                             ("M7", 0), ("M7", 1), ("M8", 0), ("M9", 0),
                             ("M9", 2), ("M9", 3), ("M10", 0), ("M11", 0)]
# the on-ramp link name and segment number that can be observed by the RL agent [(link name, segment number), ...]
onramp_observed_segment = [("ONR0_1", 0), ("ONR1_1", 0), ("ONR2_1", 0), ("ONR3_1", 0)]
