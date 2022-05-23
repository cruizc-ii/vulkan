from .utils import NamedYamlMixin
from dataclasses import dataclass, field
from abc import ABC
from app.utils import GRAVITY
from typing import Optional


@dataclass
class BuildingSpecification(ABC, NamedYamlMixin):
    """
    2d rectangular frame
    - no set-backs & no holes
    - fixed base columns, uniform loads by storey.
    - masses per storey lumped at leftmost node
    """

    name: str
    storeys: list[float] = field(default_factory=lambda: [1.0])
    bays: list[float] = field(default_factory=lambda: [1.0])
    masses: list[float] = field(default_factory=lambda: [1.0])
    damping: float = 0.05
    num_frames: int = 1
    Ecols: Optional[list[float]] = None
    Icols: Optional[list[float]] = None
    Ebeams: Optional[list[float]] = None
    Ibeams: Optional[list[float]] = None
    uniform_beam_loads_by_mass: Optional[list[float]] = None
    # design_criteria: list[str] = field(
    #     default_factory=DesignCriterionFactory.default_criteria
    # )
    design_criteria: Optional[list[str]] = None
    fems: Optional[list[str]] = None

    design_spectra: dict[dict[str, str], str] = field(default_factory=dict)

    _design_criteria: Optional[list["DesignCriterion"]] = None
    _design_spectra: Optional[list["Spectra"]] = None
    _adjacency: Optional[list[tuple[float, int, dict[str, object]]]] = None

    # _DEFAULT_RESULTS_PATH: Path = DESIGN_DIR
    columns: Optional[list[float]] = None
    floors: Optional[list[float]] = None
    height: Optional[float] = None
    width: Optional[float] = None

    num_storeys: Optional[float] = None
    num_floors: Optional[float] = None
    num_bays: Optional[float] = None
    num_cols: Optional[float] = None

    occupancy: Optional[str] = None
    # fems: list[FiniteElementModel] = field(default_factory=list)

    def __pre_init__(self) -> None:
        # self._design_criteria = [
        #     DesignCriterionFactory(c) for c in self.design_criteria
        # ]
        self.num_storeys = len(self.storeys)
        self.num_floors = self.num_storeys + 1
        self.num_bays = len(self.bays)
        self.num_cols = self.num_bays + 1
        self.floors = [0.0] + self.storeys
        self.columns = [0.0] + self.bays
        self.height = sum(self.storeys)
        self.width = sum(self.bays)

        # if self.occupancy is None:
        #     from api.occupancy import BuildingOccupancy

        #     self.occupancy = BuildingOccupancy.DEFAULT_MODEL

        if len(self.masses) == 1:
            self.masses = [self.masses[0] for _ in range(self.num_storeys)]

        self.uniform_beam_loads_by_mass = [
            GRAVITY * mass / self.width for mass in self.masses
        ]

        #     self._design_spectra = {
        #         criteria: Spectra(**data) for criteria, data in self.design_spectra.items()
        #     }
        self.__set_up__()

    def __post_init__(self):
        self.__pre_init__()
        pass

    #     self.__pre_init__()
    #     if all([isinstance(f, dict) for f in self.fems]):
    #         self.fems = [FEMFactory(**data) for data in self.fems]

    def __set_up__(self):
        columns, colIDs = {}, {}
        beams, beamIDs = {}, {}
        nodes, fixed_nodes = {}, []
        _adjacency = []
        masses_by_storey = {}
        eID, nodeID = -1, -1
        y = 0
        for iy, st in enumerate(self.floors):
            y += st
            x = 0
            for ix, bay in enumerate(self.columns):
                x += bay
                nodeID += 1
                nodes[nodeID] = dict(
                    x=x, y=y, mass=None, floor=iy + 1, storey=iy, bay=ix, column=ix + 1
                )
                if iy > 0:
                    # for floors above ground level, create the columns
                    eID += 1
                    col_coords = (
                        ix,
                        iy,
                    )  # given in a natural coordinate system (col, floor)
                    i, j = nodeID - self.num_cols, nodeID
                    col = dict(
                        id=eID,
                        i=i,
                        j=j,
                        coords=col_coords,
                        # element_type=ElementTypes.COLUMN.value,
                        element_type="column",
                        storey=iy,
                        floor=iy,
                        bay=ix,
                        column=ix + 1,
                        E=self.Ecols,
                        Ix=self.Ibeams,
                        length=st,
                    )
                    _adjacency.append((i, j, col))
                    columns[col_coords] = col
                    colIDs[eID] = col
                    if ix > 0:
                        # if we are not in the first column, create the beam.
                        eID += 1
                        beam_coords = (ix, iy)
                        i = nodeID - 1
                        j = nodeID
                        beam = dict(
                            id=eID,
                            i=i,
                            j=j,
                            coords=beam_coords,
                            # element_type=ElementTypes.BEAM.value,
                            element_type="beam",
                            storey=iy,
                            floor=iy + 1,
                            bay=ix,
                            E=self.Ebeams,
                            Ix=self.Ibeams,
                            length=bay,
                        )
                        _adjacency.append((i, j, beam))
                        beams[beam_coords] = beam
                        beamIDs[eID] = beam
                    else:
                        # for the first column, add the storey mass to the nodeID id.
                        mass = self.masses[iy - 1]
                        masses_by_storey[nodeID] = mass
                        nodes[nodeID]["mass"] = mass

                else:
                    # for ground floor, add the fixed nodeID coordinates.
                    nodes[nodeID]["fixed"] = True
                    fixed_nodes.append(nodeID)

        self._adjacency = _adjacency
        self.nodes = nodes
        self.fixed_nodes = fixed_nodes

    # @property
    # def fem(self):
    #     return self.fems[-1]

    @property
    def total_mass(self):
        return sum(self.masses)

    @property
    def weight(self):
        return f"{self.total_mass * GRAVITY:.1f}"

    # def force_design(
    #     self, results_dir: Path = None, seed_class: FiniteElementModel = PlainFEM
    # ) -> None:
    #     fem = seed_class.from_spec(self)
    #     self.fems = self.design(
    #         results_dir=results_dir, criteria=self._design_criteria, fem=fem
    #     )

    # def design(
    #     self,
    #     results_dir,
    #     criteria: list[DesignCriterion] = None,
    #     fem: FiniteElementModel = None,
    #     *args,
    #     **kwargs,
    # ) -> list[FiniteElementModel]:
    #     if results_dir is None:
    #         results_dir = self._DEFAULT_RESULTS_PATH / f"{self.name}"
    #     else:
    #         results_dir = results_dir / f"{self.name}"

    #     if criteria is None:
    #         criteria = self._design_criteria

    #     if fem is None:
    #         fem = self.fem

    #     fems = []
    #     for index, criterion in enumerate(criteria):
    #         criterion: DesignCriterion = criterion(specification=self, fem=fem)
    #         filepath = results_dir / str(index)
    #         fem = criterion.run(results_dir=filepath, *args, **kwargs)
    #         fems.append(fem)

    #     return fems

    def cytoscape(self) -> tuple[list[dict], list[dict]]:
        # ASSET_Y_OFFSET = 25
        SCALE = 100
        elems = [
            {
                "data": {
                    "id": id,
                    "label": f"{id}",
                },
                "classes": "fixed" if "fixed" in node else "free",
                "position": {"x": SCALE * node["x"], "y": SCALE * node["y"]},
            }
            for id, node in self.nodes.items()
        ]

        elems += [
            {
                "data": {
                    "source": i,
                    "target": j,
                    "type": e["element_type"],
                    "tag": e["id"],
                    "r": 2 * SCALE * 0.1,
                }
            }
            for i, j, e in self._adjacency
        ]
        # elems += [
        #     {
        #         "data": {
        #             "id": np.random.random(),
        #             "label": content.name,
        #             "url": FLASK_ASSETS_PATH + content.icon if content.icon else None,
        #         },
        #         "classes": "contents" if content.icon else None,
        #         "position": {
        #             # "x": (SCALE * (content.x + 2 * np.random.random()))
        #             # if content.x is not None
        #             # else SCALE * np.random.random() * self.length,
        #             "x": SCALE * content.x if content.x is not None else 0,
        #             "y": -SCALE * self.leftmost_nodes[content.floor - 1].y
        #             - ASSET_Y_OFFSET,
        #         },
        #     }
        #     for content in self.contents
        #     if not content.hidden
        # ]
        # elems += [
        #     {
        #         "data": {
        #             "id": np.random.random(),
        #             "label": asset.name,
        #             "url": FLASK_ASSETS_PATH + asset.icon if asset.icon else None,
        #         },
        #         "classes": "nonstructural" if asset.icon else None,
        #         "position": {
        #             "x": SCALE * asset.x if asset.x is not None else 0,
        #             # else SCALE * np.random.random() * self.length,
        #             "y": -SCALE * self.leftmost_nodes[asset.floor - 1].y
        #             - ASSET_Y_OFFSET,
        #         },
        #     }
        #     for asset in self.nonstructural_elements
        #     if not asset.hidden
        # ]
        # add floor line
        # elems += [
        #     {
        #         "data": {"id": -2, },
        #         "position": {"x": -400, "y": 0},
        #     },
        #     {
        #         "data": {"id": -1, },
        #         "position": {"x": 400, "y": 0},
        #     },
        # ]
        # elems += [
        #     {"data": {"source": -2, "target": -1, "k": 1.0 }}
        # ]
        style = [
            {"selector": "edge", "style": {"label": "data(tag)"}},
            {"selector": "edge", "style": {"width": "data(r)"}},  # width of elements
            # {
            #     "selector": ".contents",
            #     "style": {
            #         "background-image": "data(url)",
            #         "background-fit": "cover cover",
            #         "width": 50,
            #         "height": 50,
            #         # "background-opacity": 0.1,
            #         # 'background-fit': 'cover',
            #     },
            # },  # contents
            # {
            #     "selector": ".nonstructural",
            #     "style": {
            #         "background-image": "data(url)",
            #         "background-fit": "cover cover",
            #         "width": 50,
            #         "height": 50,
            #         # 'background-fit': 'cover',
            #     },
            # },  # contents
            {"selector": "node", "style": {"label": "data(label)"}},
            {
                "selector": ".fixed",
                "style": {"shape": "rectangle", "background-color": "#444242"},
            },
        ]
        print(elems)
        return elems, style


@dataclass
class ReinforcedConcreteFrame(BuildingSpecification):
    fc: float = 30e3
    Ec: float = 30e6
    fy: float = 420e3
    Es: float = 200e6
    Ecols: Optional[float] = 30e6
    Ebeams: Optional[float] = 30e6
    Icols: Optional[float] = 0.0015
    Ibeams: Optional[float] = 0.0015

    def __post_init__(self):
        self.Ec = 4.4e6 * (self.fc / 1e3) ** 0.5
        return super().__post_init__()
