from __future__ import annotations
from app.utils import SummaryEDP, YamlMixin, ROOT_DIR, find_files
from app.fem import ElasticBeamColumn, FiniteElementModel, Node
from app.assets import Asset, AssetFactory, AssetNotFoundException, RiskAsset
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import random
import yaml

UNIT_DEFINITIONS_FOLDER = ROOT_DIR / "models" / "units"
OCCUPANCY_MODELS_PATH = ROOT_DIR / "models" / "occupancy"

UNITS_YML = find_files(UNIT_DEFINITIONS_FOLDER, only_yml=True)
LOADED_UNITS = {}


@dataclass
class Unit:
    name: str
    dx: float = 0
    assets: Optional[list[str]] = None

    def __str__(self):
        return f"Unit({self.name} dx={self.dx})"


class UnitNotFoundException(Exception):
    pass


class InvalidUnitFileDefinition(Exception):
    pass


@dataclass
class Group:
    name: str = None
    units: list[str] = None
    _units: list[Unit] = None
    priority: float = 0
    pct: float = 0
    dx: float = 0
    dy: float = 1
    x: float = 0
    unique: bool = False
    floor: int = None
    placed: bool = False
    tried: bool = False
    sticky: bool = False
    closest_node_id: int = None

    def __str__(self) -> str:
        return f"Group({self.name} dx={self.dx})"

    @property
    def x_end(self) -> float:
        return self.x + self.dx

    @property
    def x_start(self) -> float:
        return self.x

    def __post_init__(self):
        self._units = []
        dx = 0
        errors = 0
        for name in self.units:
            try:
                while name not in LOADED_UNITS:
                    try:
                        unit_file = UNITS_YML.pop()
                        filepath = UNIT_DEFINITIONS_FOLDER / unit_file
                        with open(filepath) as f:
                            more_units = yaml.safe_load(f)
                            # try and see if it's
                            if isinstance(more_units, list):
                                for ix, u in enumerate(more_units):
                                    unit_name = u.pop("name", f"{unit_file}-{ix}")
                                    LOADED_UNITS[unit_name] = u
                            elif isinstance(more_units, dict):
                                for ix, (name, data) in enumerate(more_units.items()):
                                    LOADED_UNITS[name] = data
                            else:
                                print(
                                    f"File {unit_file} contains an unexpected format."
                                )
                                raise InvalidUnitFileDefinition
                    except IndexError:
                        print(f"Group.__post_init__: unit '{name}' not found!")
                        raise UnitNotFoundException
                unit_data = LOADED_UNITS[name]
                dx += unit_data.get("dx", 0)
                # merged_unit_data = LOADED_UNITS[name] | unit_data
                unit = Unit(name=name, **unit_data)
                self._units.append(unit)
            except (UnitNotFoundException, InvalidUnitFileDefinition) as e:
                print(f"Group.__post_init__: encountered an error", e)
                errors += 1
                if errors > 30:
                    print(
                        f"Group.__post_init__: exceeded maximum number of errors loading files, try checking unit definitons"
                    )
                    break
                continue
        self.dx = dx


@dataclass
class FilledBucket:
    start: float = 0
    end: float = 0

    @property
    def block(self) -> list[float]:
        return [self.start, self.end]

    @property
    def dx(self) -> float:
        return self.end - self.start


@dataclass
class StoreySpace:
    floor: int = 0
    length: float = 0
    fixed_nodes: list[Node] = None
    beams: list[ElasticBeamColumn] = None
    buckets: list[FilledBucket] = field(default_factory=list)

    def collocate_group(self, g: Group) -> bool:
        dxs, buckets = self.get_sizes_and_buckets()
        if any(dxs >= g.dx):
            chosen_block_ix = np.where(dxs >= g.dx)[0][0]
            start = buckets[chosen_block_ix][0]
            end = start + g.dx
            bucket = FilledBucket(start=start, end=end)
            self.buckets.append(bucket)
            g.floor = self.floor
            g.x = float(start)
            if g.sticky:
                g.closest_node_id = self._get_closest_node_id(g.x)
            return True
        return False

    def _get_closest_node_id(self, x: float) -> int:
        """fixated to the left (doesn't take dx into consideration)"""
        if self.fixed_nodes:
            closest_node = sorted(self.fixed_nodes, key=lambda n: abs(n.x - x))[0]
            node_id = closest_node.id
        else:
            closest_beam = sorted(self.beams, key=lambda b: abs(b.bay * b.length - x))[
                0
            ]
            beam_middlepoint = (2 * closest_beam.bay - 1) * closest_beam.length / 2
            closest_node = (
                closest_beam.i if (beam_middlepoint - x) > 0 else closest_beam.j
            )
            node_id = closest_node
        return node_id

    def get_sizes_and_buckets(self) -> tuple[np.ndarray, np.ndarray]:
        buckets = np.array([b.block for b in self.buckets])
        buckets = np.append(buckets, [self.length, 0])
        buckets = np.roll(buckets, 1)
        buckets = np.resize(buckets, (len(self.buckets) + 1, 2))
        dxs = np.diff(buckets)
        buckets = np.delete(buckets, np.where(dxs == [0.0]), 0)
        dxs = np.delete(dxs, np.where(dxs == [0.0]))
        return dxs, buckets

    def get_empty_spaces(self) -> np.ndarray:
        emtpy_spaces, buckets = self.get_sizes_and_buckets()
        return emtpy_spaces

    def has_space_left(self) -> bool:
        return sum([b.dx for b in self.buckets]) < self.length


@dataclass
class OccupancyModel(YamlMixin):
    groups: Optional[dict[str, dict]] = None
    _groups: Optional[dict[str, Group]] = None
    spaces: Optional[list[StoreySpace]] = None

    def __post_init__(self):
        self._groups = {}
        for ix, data in enumerate(self.groups):
            self._groups[data.get("name", ix)] = Group(**data)

    def lengths_by_storey(
        self, beams_by_storey: list[list[ElasticBeamColumn]]
    ) -> list[list[float]]:
        return [sum([b.length for b in st]) for st in beams_by_storey]

    @property
    def sorted_groups(self) -> list[Group]:
        from operator import itemgetter, attrgetter

        groups = filter(lambda g: 1 >= g.pct >= 0, self._groups.values())
        groups = list(sorted(groups, key=attrgetter("pct", "priority")))
        return groups

    def was_group_placed(self, g: Group, spaces: list[StoreySpace]) -> bool:
        was_placed = False
        g.tried = True
        for space in spaces[g.floor - 1 :]:
            was_placed = space.collocate_group(g)
            if was_placed:
                g.placed = True
                break
        return was_placed

    def _get_spaces_left(self) -> np.ndarray:
        return np.array([s.get_empty_spaces() for s in self.spaces]).flatten()

    def _has_space_left(self) -> bool:
        return any([s.has_space_left() for s in self.spaces])

    def _build_groups(self, fem: FiniteElementModel) -> list[Group]:
        placed_groups = []
        spaces = [StoreySpace(floor=1, fixed_nodes=fem.fixed_nodes, length=fem.length)]
        spaces += [
            StoreySpace(beams=beams, length=fem.length, floor=floor)
            for floor, beams in enumerate(fem.beams_by_storey[:-1], 2)
        ]
        self.spaces = spaces

        WIDTH = fem.length
        TOTAL_LENGTH = fem.total_length
        for g in self.sorted_groups:
            floor = int(((TOTAL_LENGTH - WIDTH) * g.pct) // WIDTH + 1)
            g.floor = floor

        remaining_groups = list(self.sorted_groups)
        has_space_left = True
        while has_space_left and len(remaining_groups) > 0:
            g = remaining_groups.pop(0)
            was_placed = self.was_group_placed(g, spaces)
            if was_placed:
                placed_groups.append(g)
            has_space_left = self._has_space_left()

        already_placed_non_unique = list(filter(lambda g: not g.unique, placed_groups))
        from itertools import cycle

        complete_pass = len(already_placed_non_unique)
        nothing_more_fits = False
        c = 0
        if len(already_placed_non_unique) > 0:
            smallest_group = list(
                sorted(already_placed_non_unique, key=lambda g: g.dx)
            )[0]
            smallest_dx = smallest_group.dx
            for g in cycle(already_placed_non_unique):
                c += 1
                g = Group(**asdict(g))
                was_placed = self.was_group_placed(g, spaces)
                if was_placed:
                    placed_groups.append(g)
                if c % complete_pass == 0:
                    spaces_left = self._get_spaces_left()
                    nothing_more_fits = not any(smallest_dx <= spaces_left)
                    if nothing_more_fits:
                        break

        from operator import itemgetter, attrgetter

        placed_groups = sorted(placed_groups, key=attrgetter("floor", "x"))
        return placed_groups

    def _generate_assets_from_groups(
        self, gs: list[Group], fem: FiniteElementModel
    ) -> list[Asset]:
        assets = []
        summary_edps = SummaryEDP.list()
        try:
            for g in gs:
                num_assets = sum([len(u.assets) for u in g._units])
                xs = np.linspace(g.x_start, g.x_end, num_assets)
                group_assets = []
                for unit in g._units:
                    for asset_str in unit.assets:
                        group_assets.append(asset_str)
                for asset_str, x in zip(group_assets, xs):
                    asset: RiskAsset = AssetFactory(
                        floor=g.floor, name=asset_str, x=float(x)
                    )
                    DEPTH_FACTOR = fem.depth / g.dy
                    asset.net_worth = DEPTH_FACTOR * asset.net_worth
                    assets.append(asset)
                    if asset.edp not in summary_edps:
                        asset.node = g.closest_node_id
        except AssetNotFoundException as e:
            print(f'_generate_assets_from_groups: asset "{asset_str}" not found', e)
        return assets

    def build(self, fem: FiniteElementModel) -> list[Asset]:
        groups = self._build_groups(fem)
        assets = self._generate_assets_from_groups(groups, fem)
        return assets

    def random_build(self, fem: FiniteElementModel) -> list[Asset]:
        assets = []
        summary_edps = SummaryEDP.list()
        for group in self._groups.values():
            for unit in group._units.values():
                for asset_name in unit.assets:
                    st = np.random.randint(1, fem.num_modes + 1)
                    try:
                        asset: RiskAsset = AssetFactory(floor=st, name=asset_name)
                        if asset.edp not in summary_edps:
                            nd = random.choice(fem.nodes)
                            asset.x = nd.x
                            asset.node = nd.id
                        assets.append(asset)
                    except AssetNotFoundException:
                        print(f"Asset {asset_name} not found!")
                        continue
        return assets


@dataclass
class BuildingOccupancy:
    fem: FiniteElementModel
    model_str: str
    _model: Optional[OccupancyModel] = None

    # DEFAULT: str = "MidRisePrivateOffice.yml" ## TOO SLOW for HUGE buildings
    DEFAULT: str = "EmptyOccupancy.yml"

    @classmethod
    def options(cls) -> dict:
        options = find_files(OCCUPANCY_MODELS_PATH, only_yml=True)
        return options

    def __post_init__(self):
        options = find_files(OCCUPANCY_MODELS_PATH, only_yml=True)
        if ".yml" not in self.model_str:
            self.model_str = self.model_str + ".yml"
        if self.model_str not in options:
            raise Exception(f"Invalid model str {self.model_str} for BuildingOccupancy")
        self._model = OccupancyModel.from_file(OCCUPANCY_MODELS_PATH / self.model_str)

    def build(self) -> list[Asset]:
        assets = self._model.build(fem=self.fem)
        return assets
