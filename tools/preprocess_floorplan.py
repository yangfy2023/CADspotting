import numpy as np
import xml.etree.ElementTree as ET
from svgpathtools import parse_path, Arc, Line
from typing import Union, List
from scipy.optimize import minimize, fsolve, brentq
from scipy.integrate import quad
import math
import re
import os
import json
from tqdm import tqdm
import concurrent.futures
import argparse

DEFAULT_SAMPLE_DISTANCE = 140 / 980
LABEL_NUM = 35


class MyEllipse:
    def __init__(self, cx, cy, rx, ry):
        self.cx = float(cx)
        self.cy = float(cy)
        self.rx = float(rx)
        self.ry = float(ry)

    def length(self, t0=0, t1=1, error=1e-12, min_depth=5):
        assert 0 <= t0 <= 1 and 0 <= t1 <= 1
        theta_0 = t0 * np.pi * 2
        theta_1 = t1 * np.pi * 2
        integrand = lambda theta: np.sqrt((self.rx * np.sin(theta)) ** 2 + (self.ry * np.cos(theta)) ** 2)
        length, _ = quad(integrand, theta_0, theta_1)
        return length

    def point(self, t):
        assert 0 <= t <= 1
        theta = t * np.pi * 2
        x = self.cx + self.rx * np.cos(theta)
        y = self.cy + self.ry * np.sin(theta)
        return x, y


class MyCircle:
    def __init__(self, cx, cy, r):
        self.cx = float(cx)
        self.cy = float(cy)
        self.r = float(r)

    def length(self, t0=0, t1=1):
        assert 0 <= t0 <= 1 and 0 <= t1 <= 1
        theta_0 = t0 * np.pi * 2
        theta_1 = t1 * np.pi * 2
        length = self.r * abs(theta_1 - theta_0)
        return length

    def point(self, t):
        assert 0 <= t <= 1
        theta = t * np.pi * 2
        x = self.cx + self.r * np.cos(theta)
        y = self.cy + self.r * np.sin(theta)
        return x, y


def _get_complex_point(obj: Union[Arc, Line], t):
    complex_coord = obj.point(t)
    return (complex_coord.real, complex_coord.imag)


def _objective(t, obj, s):
    return (obj.length(0, t) - s) ** 2


# ? 为什么**2
# * 因为minimize是求最小值


def _objective_root(t, obj, s):
    return obj.length(0, t) - s


def _sample_points(obj: Union[Line, Arc, MyEllipse, MyCircle], d_samp) -> List[tuple]:
    length = obj.length()
    if d_samp > length:
        return []
    # TODO: how to sample points with obj length < d?
    # assert d <= length, f"Sampling distance must be less than or equal to the object's length. While happen to {obj.__class__.__name__} and length is {obj.length()}"

    sample_num = int(length // d_samp)
    results = []

    if obj.__class__.__name__ in ("Line", "MyCircle"):
        results = [i * d_samp / length for i in range(sample_num)]
        if obj.__class__.__name__ == "Line":
            points = [_get_complex_point(obj, t) for t in results]
        elif obj.__class__.__name__ == "MyCircle":
            points = [obj.point(t) for t in results]

    elif obj.__class__.__name__ in ("Arc", "MyEllipse"):
        s_i = [d_samp * i for i in range(sample_num)]
        for s in s_i:
            try:
                solution = brentq(_objective_root, 0, 1, args=(obj, s))
                results.append(solution)
            except ValueError as e:
                print(f"Error: {e}")

        if obj.__class__.__name__ == "Arc":
            points = [_get_complex_point(obj, t) for t in results]
        elif obj.__class__.__name__ == "MyEllipse":
            points = [obj.point(t) for t in results]

    else:
        raise TypeError(f"Unsupported type for sampling: {obj.__class__.__name__}")

    return points


def _rotate_point(point, tsfm_rotate):
    x, y = point
    angle_degrees, cx, cy = tsfm_rotate
    angle_radians = math.radians(angle_degrees)
    x_rotated = (x - cx) * math.cos(angle_radians) - (y - cy) * math.sin(angle_radians) + cx
    y_rotated = (x - cx) * math.sin(angle_radians) + (y - cy) * math.cos(angle_radians) + cy

    return (x_rotated, y_rotated)


def _parse_rotate_tsfm(tsfm_str):
    match = re.match(r"rotate\(([^)]+)\)", tsfm_str)
    if match:
        values = [float(val) for val in match.group(1).split(",")]
        return tuple(values)
    else:
        raise ValueError("Invalid rotate transform string: ", tsfm_str)


def get_json_from_svg(svg_path, d_samp=DEFAULT_SAMPLE_DISTANCE):
    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = root.tag[:-3]
    coords = []
    semanticIds = []
    instanceIds = []
    elementIds = []
    rgbs = []
    elementlengths = []
    elementId = int(0)
    reverse_json = {}
    element_added = False
    for path in root.iter(ns + "path"):
        for obj in parse_path(path.attrib["d"]):
            assert obj.__class__ == Line or obj.__class__ == Arc
            element_added = False
            semanticId = int(path.attrib["semanticId"]) - 1 if "semanticId" in path.attrib else LABEL_NUM
            instanceId = int(path.attrib["instanceId"]) if "instanceId" in path.attrib else -1
            rgb = path.attrib["stroke"].strip("rgb()").split(",")
            pathid = path.attrib.get("id", None)
            elementlength = obj.length()
            if elementlength >= 10000:  #! 人为筛选
                continue
            sampled_points = _sample_points(obj, d_samp)
            for point in sampled_points:
                if all(0 <= coord <= 140 for coord in point):
                    coords.append(point)
                    semanticIds.append(semanticId)
                    instanceIds.append(instanceId)
                    elementIds.append(elementId)
                    elementlengths.append(elementlength)
                    rgbs.append(rgb)
                    element_added = True
            if element_added:
                if obj.__class__ == Line:
                    reverse_json[elementId] = {
                        "type": "Line",
                        "start_point": (obj.start.real, obj.start.imag),
                        "end_point": (obj.end.real, obj.end.imag),
                    }
                elif obj.__class__ == Arc:
                    reverse_json[elementId] = {
                        "type": "Arc",
                        "start_point": (obj.start.real, obj.start.imag),
                        "end_point": (obj.end.real, obj.end.imag),
                        "rx": obj.radius.real,
                        "ry": obj.radius.imag,
                        "rotation": obj.rotation,
                        "large_arc": int(obj.large_arc),
                        "sweep": int(obj.sweep),
                    }
                elementId += 1
    for ellipse in root.iter(ns + "ellipse"):
        semanticId = int(ellipse.attrib["semanticId"]) - 1 if "semanticId" in ellipse.attrib else LABEL_NUM
        instanceId = int(ellipse.attrib["instanceId"]) if "instanceId" in ellipse.attrib else -1
        rgb = ellipse.attrib["stroke"].strip("rgb()").split(",")
        my_ellipse = MyEllipse(ellipse.attrib["cx"], ellipse.attrib["cy"], ellipse.attrib["rx"], ellipse.attrib["ry"])
        rotate_tsfm = _parse_rotate_tsfm(ellipse.attrib["transform"])
        elementlength = my_ellipse.length()
        sampled_points = _sample_points(my_ellipse, d_samp)
        element_added = False
        for point in sampled_points:
            rotated_point = _rotate_point(point, rotate_tsfm)
            if all(0 <= coord <= 140 for coord in rotated_point):
                coords.append(rotated_point)
                semanticIds.append(semanticId)
                instanceIds.append(instanceId)
                elementIds.append(elementId)
                elementlengths.append(elementlength)
                rgbs.append(rgb)
                element_added = True
        if element_added:
            reverse_json[elementId] = {
                "type": "Ellipse",
                "cx": ellipse.attrib["cx"],
                "cy": ellipse.attrib["cy"],
                "rx": ellipse.attrib["rx"],
                "ry": ellipse.attrib["ry"],
                "transform": ellipse.attrib["transform"],
            }
            elementId += 1
    for circle in root.iter(ns + "circle"):
        semanticId = int(circle.attrib["semanticId"]) - 1 if "semanticId" in circle.attrib else LABEL_NUM
        instanceId = int(circle.attrib["instanceId"]) if "instanceId" in circle.attrib else -1
        rgb = circle.attrib["stroke"].strip("rgb()").split(",")
        my_circle = MyCircle(circle.attrib["cx"], circle.attrib["cy"], circle.attrib["r"])
        elementlength = my_circle.length()
        sampled_points = _sample_points(my_circle, d_samp)
        element_added = False
        for point in sampled_points:
            if all(0 <= coord <= 140 for coord in point):
                coords.append(point)
                semanticIds.append(semanticId)
                instanceIds.append(instanceId)
                elementIds.append(elementId)
                elementlengths.append(elementlength)
                rgbs.append(rgb)
                element_added = True
        if element_added:
            reverse_json[elementId] = {
                "type": "Circle",
                "cx": circle.attrib["cx"],
                "cy": circle.attrib["cy"],
                "r": circle.attrib["r"],
            }
            elementId += 1

    temp_json = {
        "coords": coords,
        "semanticIds": semanticIds,
        "instanceIds": instanceIds,
        "elementIds": elementIds,
        "elementlengths": elementlengths,
        "rgbs": rgbs,
    }
    assert len(coords) == len(semanticIds) == len(instanceIds) == len(elementIds)
    if len(coords) > 0:
        return temp_json, reverse_json
    else:
        return None, None


def get_svg_files(directory):
    svg_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".svg"):
                svg_files.append(file)
    return svg_files


def compute_bboxes_optimized(element_ids, coords):
    element_ids = np.asarray(element_ids)
    coords = np.asarray(coords)

    unique_element_ids, counts = np.unique(element_ids, return_counts=True)

    sort_indices = np.argsort(element_ids)
    sorted_coords = coords[sort_indices]

    split_coords = np.split(sorted_coords, np.cumsum(counts)[:-1])

    bboxes = np.array(
        [
            (c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max())  # xmin  # ymin  # xmax  # ymax
            for c in split_coords
        ]
    )

    return bboxes


def process_svg(svg_name, raw_dataset_path, npy_path, json_path, d_samp=DEFAULT_SAMPLE_DISTANCE):
    svg_filepath = os.path.join(raw_dataset_path, svg_name)
    json_data, reverse_json = get_json_from_svg(svg_filepath, d_samp)
    if json_data:
        coords = np.array(json_data["coords"])
        semanticIds = np.array(json_data["semanticIds"])
        instanceIds = np.array(json_data["instanceIds"])
        elementIds = np.array(json_data["elementIds"])
        elementlengths = np.array(json_data["elementlengths"])
        bboxes = compute_bboxes_optimized(elementIds, coords)
        rgbs = np.array(json_data["rgbs"], dtype=int)

        with open(json_path, "w") as json_file:
            json.dump(reverse_json, json_file)

        np.savez(
            npy_path,
            coords=coords,
            semanticIds=semanticIds,
            instanceIds=instanceIds,
            elementIds=elementIds,
            elementlengths=elementlengths,
            rgbs=rgbs,
            bboxes=bboxes,
        )
    # with open(json_filepath, 'w') as f:
    #     json.dump(json_data, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument(
        "--datasets", type=str, help="Comma-separated list of dataset names, separated by ,", required=True
    )
    parser.add_argument("--raw_base_path", type=str, help="path of the svg_gt", required=True)
    parser.add_argument("--d_samp", type=float, help="distance for sampling points", required=True)

    args = parser.parse_args()
    return args


def main():
    # datasets = ['test','train','val']
    args = parse_args()
    datasets = args.datasets.split(",")
    datasets = [dataset.strip() for dataset in datasets]
    svg_path = "svg_gt"
    npy_root_name = "npz_gt"
    json_root_name = "reverse_json"
    raw_base_path = args.raw_base_path
    d_samp = args.d_samp

    for dataset in datasets:
        raw_dataset_path = os.path.join(raw_base_path, dataset, svg_path)
        svg_names = get_svg_files(raw_dataset_path)
        npy_root_dir = os.path.join(raw_base_path, dataset, npy_root_name)
        os.makedirs(npy_root_dir, exist_ok=True)
        json_root_dir = os.path.join(raw_base_path, dataset, json_root_name)
        os.makedirs(json_root_dir, exist_ok=True)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for svg_name in svg_names:
                npy_path = os.path.join(npy_root_dir, os.path.splitext(svg_name)[0]) + ".npz"
                json_path = os.path.join(json_root_dir, os.path.splitext(svg_name)[0]) + ".json"
                if not os.path.exists(npy_path):
                    futures.append(
                        executor.submit(process_svg, svg_name, raw_dataset_path, npy_path, json_path, d_samp)
                    )

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {dataset} dataset"
            ):
                future.result()


def debugger():
    datasets = ["temp"]
    svg_path = "svg_gt"
    npy_root_name = "npz_gt"
    json_root_name = "reverse_json"
    raw_base_path = os.path.join(os.path.dirname(__file__), "..", "data", "floorplan_raw")

    for dataset in datasets:
        raw_dataset_path = os.path.join(raw_base_path, dataset, svg_path)
        svg_names = get_svg_files(raw_dataset_path)
        npy_root_dir = os.path.join(raw_base_path, dataset + "_color", npy_root_name)
        os.makedirs(npy_root_dir, exist_ok=True)
        json_root_dir = os.path.join(raw_base_path, dataset + "_color", json_root_name)
        os.makedirs(json_root_dir, exist_ok=True)
        print(npy_root_dir)
        print(svg_names)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for svg_name in svg_names:
                npy_path = os.path.join(npy_root_dir, os.path.splitext(svg_name)[0]) + ".npz"
                json_path = os.path.join(json_root_dir, os.path.splitext(svg_name)[0]) + ".json"
                if not os.path.exists(npy_path):
                    futures.append(executor.submit(process_svg, svg_name, raw_dataset_path, npy_path, json_path))

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {dataset} dataset"
            ):
                future.result()


if __name__ == "__main__":
    main()
    # debugger()
