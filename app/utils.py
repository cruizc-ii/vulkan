from __future__ import annotations
from enum import Enum
from pathlib import Path
import yaml
import json
import os
import base64
import io
from typing import Any
from dataclasses import asdict
import pandas as pd
import numpy as np

ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
DESIGN_DIR = MODELS_DIR / "design"
HAZARD_DIR = MODELS_DIR / "hazard"
COMPARE_DIR = MODELS_DIR / "compare"
RESULTS_DIR = ROOT_DIR / "results"
ASSETS_PATH = "https://vulkan1.s3.amazonaws.com/"
GRAVITY = 9.81
METERS_TO_FEET = 1 / 0.3048


def find_files(
    folder: Path,
    files_only=True,
    ignore_hidden=True,
    only_yml=True,
    only_csv=False,
    return_sorted=True,
):
    all_files_or_dirs = os.listdir(folder)
    filters = []
    if files_only:
        filters.append(lambda f: os.path.isfile(os.path.join(folder, f)))

    if ignore_hidden:
        filters.append(lambda f: not f.startswith("."))

    if only_yml:
        filters.append(lambda f: ".yml" in f)

    if only_csv:
        filters.append(lambda f: ".csv" in f)

    files = list(
        filter(
            lambda file_or_dir: all(fil(file_or_dir) for fil in filters),
            all_files_or_dirs,
        )
    )
    if return_sorted:
        files = sorted(files)

    return files


class AnalysisTypes(Enum):
    K = "K"
    GRAVITY = "gravity"
    STATIC = "static"
    MODAL = "modal"
    PUSHOVER = "pushover"
    TIMEHISTORY = "timehistory"
    IDA = "ida"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class SummaryEDP(Enum):
    """these are the summarized versions
    or some computed value from .csvs
    to show on the UI or precompute for asset loss computation"""

    peak_drifts = "peak_drifts"
    peak_floor_accels = "pfa"
    peak_floor_vels = "pfv"
    # residual_drifts = "residual_drifts"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class EDP(Enum):
    disp_env = "disp_env"
    disp = "disp"
    drift = "drift"
    vel_env = "vel_env"
    axial = "P"
    shear = "V"
    moment = "M"
    storey_shears = "storey_shears"
    storey_moments = "storey_moments"
    roof_displacement = "roof_disp"
    overturning_moments = "overturning_moments"
    rotations = "rotations"
    rotations_env = "rotations_env"
    spring_moment_rotation_th = "spring_moment_rotation_th"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


class IM(Enum):
    accel = "acc"
    vel = "vel"
    disp = "disp"
    pseudo_accel = "Sa"
    pseudo_vel = "Sv"


IDAResultsDataFrame = pd.DataFrame
AccelHazardSeries = pd.Series
ResultsDataFrame = pd.DataFrame
SummaryResultsDataFrame = pd.DataFrame

OPENSEES_EDPs: tuple = ("disp", "vel", "accel", "reaction")
OPENSEES_ELEMENT_EDPs: tuple = (
    "localForce",
)  # "globalForce", might be better to know that 1-P, 2-V, 3-M always no matter the orientation
OPENSEES_REACTION_EDPs = [
    (EDP.shear.value, 1),
    (EDP.axial.value, 2),
    (EDP.moment.value, 3),
]


class YamlMixin:
    """
    does not serialize private methods
    i.e. those that start with _
    such as '_risk' or  '__df'
    TODO; doesn't work with defaultdict :(.. json.loads(json.dumps(o))
    TODO@project:utils
         include more filters for different serialization procedures
         write-only, read-only etc.
         numpy array tolist()
         !python/object/apply:numpy.core.multiarray.scalar
    """

    @classmethod
    def numpy_dict_factory(cls, data: dict) -> dict:
        res = {}
        for k, v in data.items():
            if isinstance(v, (np.ndarray)):
                v = v.tolist()
            elif isinstance(v, (np.integer)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = v.item()
            elif isinstance(v, (np.generic)):
                v = v.item()
            res[k] = v
        return res

    def read_only_dict_factory(self, data: list[tuple[str, Any]]):
        result = {}
        for k, v in data:
            if k.startswith("_"):
                continue
            elif isinstance(v, (np.ndarray)):
                v = v.tolist()
                print(f"field {k} is numpy.ndarray {v} {type(v)}, should be list")
            elif isinstance(v, (np.integer)):
                v = int(v)
                print(f"field {k} is {type(v)}, should be native int !")
            elif isinstance(v, (np.floating,)):
                v = v.item()
                print(f"field {k} is {type(v)}, should be native float !")
            elif isinstance(v, (np.generic)):
                v = v.item()
                print(f"field {k} is {type(v)}, generic should be python native !")
            result[k] = v
        return result

    @classmethod
    def from_file(cls, filepath):
        with open(filepath) as f:
            doc = yaml.safe_load(f)
        return cls(**doc)

    def to_file(self, filepath):
        data = self.to_dict
        with open(filepath, "w") as file:
            yaml.dump(data, file)

    @property
    def to_json(self):
        return json.dumps(self.to_dict)

    @property
    def to_dict(self):
        return asdict(self, dict_factory=self.read_only_dict_factory)

    @classmethod
    def multiple_to_file(cls, filepath, name, *instances):
        documents = []
        for inst in instances:
            doc = inst.to_dict
            documents.append(doc)
        os.makedirs(filepath, exist_ok=True)
        filepath = os.path.join(filepath, name + ".yml")
        with open(filepath, "w") as file:
            yaml.dump_all(documents, file)


class NamedYamlMixin(YamlMixin):
    @property
    def name_yml(self) -> str:
        return f"{self.name}.yml"

    def to_file(self, folder: Path) -> None:
        filepath = folder / self.name_yml
        return super().to_file(filepath)

    def delete(self, folder: Path) -> None:
        filepath = folder / self.name_yml
        # folder = DESIGN_DIR / name
        try:
            Path.unlink(filepath)
            # shutil.rmtree(folder)
        except (FileNotFoundError, OSError) as e:
            print(e)


def save_file(name, content, directory):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    with open(os.path.join(directory, name), "wb") as fp:
        fp.write(base64.decodebytes(data))


def decode_hazard_csv(content, normalize=1) -> pd.DataFrame:
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep="\s+", names=["x", "y"])
    df.x = df.x / normalize
    return df


def UploadComponent(id, msg: str = "Upload design criterion."):
    # return html.Div(
    #     [
    #         dcc.Upload(
    #             id=id,
    #             children=html.Div([msg]),
    #             style={
    #                 "width": "100%",
    #                 "height": "60px",
    #                 "lineHeight": "60px",
    #                 "borderWidth": "1px",
    #                 "borderStyle": "dashed",
    #                 "borderRadius": "5px",
    #                 "textAlign": "center",
    #                 "margin": "10px",
    #             },
    #             multiple=False,
    #         ),
    #     ],
    #     style={"max-width": "500px"},
    # )
    pass


def eigenvectors_similar(a: np.ndarray, b: np.ndarray, rtol=1e-3) -> bool:
    """A, B are similar in an eigenvector sense
    if their columns are similar when multiplied by -1 or +1"""
    a = np.array(a)
    b = np.array(b)
    for ix, (cola, colb) in enumerate(zip(a.T, b.T)):
        eq = np.allclose(cola, colb, rtol=rtol)
        opp = np.allclose(-cola, colb, rtol=rtol)
        if not (eq or opp):
            print(f"col {ix} is not similar!")
            return False
    return True
