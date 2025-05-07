from collections import OrderedDict
import xml.etree.ElementTree as eTree

from Env.detectorGroup import E1Group, E2Group


def build_detector_group(detector_filename):
    parser = eTree.parse(detector_filename)
    root = parser.getroot()
    e1_detector, e2_detector = OrderedDict(), OrderedDict()
    last_idx = ""
    for detector in root.findall("inductionLoop"):
        current_idx = detector.get("idx")
        if not current_idx == last_idx:
            temp = E1Group(detector.get("id"), current_idx, float(detector.get("pos")),
                           detector.get("lane")[::-1].split("_", 1)[-1][::-1])  # edge name + "_" + lane index
            e1_detector[current_idx] = temp
            last_idx = current_idx
        else:
            e1_detector[current_idx].append(detector.get("id"))
    last_idx = ""
    for detector in root.findall("laneAreaDetector"):
        current_idx = detector.get("idx")
        if not current_idx == last_idx:
            temp = E2Group(detector.get("id"), current_idx, float(detector.get("pos")),
                           detector.get("lane")[::-1].split("_", 1)[-1][::-1], float(detector.get("length")))
            e2_detector[current_idx] = temp
            last_idx = current_idx
        else:
            e2_detector[current_idx].append(detector.get("id"))
    return e1_detector, e2_detector
