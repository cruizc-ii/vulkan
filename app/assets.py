from __future__ import annotations
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field
from app.utils import (
    AccelHazardSeries,
    IDAResultsDataFrame,
    YamlMixin,
    find_files,
)
from abc import abstractmethod, ABC
from pathlib import Path
from scipy.stats import lognorm
from collections import Counter
from typing import Optional

LOSS_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "loss"
RISK_MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "risk"
LossResultsDataFrame = pd.DataFrame  # accels vs loss values


class Lognormal:
    """
    scipy.stats lognorm is a bit confusing.

    stats.pdfdistributions have a shift (loc) and scale parameters

    A common parametrization for a lognormal random variable Y
    is in terms of the mean, mu, and standard deviation, sigma,
    of the unique normally pdfributed random variable X such that exp(X) = Y.
    This parametrization corresponds to setting s = sigma and scale = exp(mu).

    a useful way to think about it is that the mean/median is the 50th percentile
    so when plotted, it should correspond numerically to the value given
    i.e. median drift of 0.10, when plotted look for the 50th percentile.

    and that the std of the logarithm of the values is numerically close to the
    coefficient of variation (Ordaz) so it should be in the numerical range
    of about 0 to around 1.0
    """

    def __init__(self, median, std_log):
        rv = lognorm(std_log, scale=median)
        domain = np.linspace(rv.ppf(0.001), rv.ppf(0.999), 100)
        df = pd.DataFrame(rv.cdf(domain), index=domain)
        self.rv = rv
        self.df = df


@dataclass
class LognormalRisk(YamlMixin):
    name: str = None
    category: str = None
    edp: str = None
    damage_states: str = None
    net_worth: float = None
    icon: str = None
    vulnerabilities: list[Lognormal] = field(default_factory=list)
    frag_dict: dict = None
    vuln_dict: dict = None
    frag_pdf_df: pd.DataFrame = None
    frag_cdf_df: pd.DataFrame = None
    vuln_pdf_df: pd.DataFrame = None
    vuln_cdf_df: pd.DataFrame = None
    vuln_means: np.ndarray = None
    init: bool = False
    hidden: bool = False
    rugged: bool = False

    def validate(self):
        med = 0
        for ds in self.damage_states:
            new_med = float(ds["median"])
            if new_med < med or new_med < 0:
                raise Exception(
                    "Medians must be monotonically increasing and positive!"
                )
            med = new_med

    def __lazy_post_init__(self) -> None:
        self.validate()
        self.frag_dict = {}
        frag_lns = []
        self.vuln_dict = {}
        vuln_lns = []
        self.vuln_means = []
        if self.rugged:
            return
        for ds in self.damage_states:
            name = ds["name"]
            median, std_log = ds["median"], ds["std"]
            frag_ln = Lognormal(float(median), float(std_log))
            self.frag_dict[name] = frag_ln
            frag_lns.append(frag_ln)

            vulnerability = ds["vulnerability"]
            median_vuln, std_log_vuln = (
                vulnerability["median"],
                vulnerability["std"],
            )
            vuln_ln = Lognormal(float(median_vuln), float(std_log_vuln))
            self.vuln_dict[name] = vuln_ln
            vuln_lns.append(vuln_ln)

            self.vuln_means.append(vuln_ln.rv.mean())

        self.vuln_means = np.array(self.vuln_means)

        DOMAIN_POINTS = 100
        frag_domain = np.linspace(
            frag_lns[0].rv.ppf(0.001), frag_lns[-1].rv.ppf(0.999), DOMAIN_POINTS
        )

        frag_pdf_data = {
            self.damage_states[ix]["name"]: ln.rv.pdf(frag_domain)
            for ix, ln in enumerate(frag_lns)
        }

        frag_cdf_data = {
            self.damage_states[ix]["name"]: ln.rv.cdf(frag_domain)
            for ix, ln in enumerate(frag_lns)
        }

        self.frag_pdf_df = pd.DataFrame(frag_pdf_data, index=frag_domain)
        self.frag_cdf_df = pd.DataFrame(frag_cdf_data, index=frag_domain)

        vuln_domain = np.linspace(
            vuln_lns[0].rv.ppf(0.001), vuln_lns[-1].rv.ppf(0.999), DOMAIN_POINTS
        )

        vuln_pdf_data = {
            self.damage_states[ix]["name"]: ln.rv.pdf(vuln_domain)
            for ix, ln in enumerate(vuln_lns)
        }

        vuln_cdf_data = {
            self.damage_states[ix]["name"]: ln.rv.cdf(vuln_domain)
            for ix, ln in enumerate(vuln_lns)
        }
        self.vuln_pdf_df = pd.DataFrame(vuln_pdf_data, index=vuln_domain)
        self.vuln_cdf_df = pd.DataFrame(vuln_cdf_data, index=vuln_domain)
        self.init = True

    def fragility_figure(self):
        if not self.init:
            self.__lazy_post_init__()
        fig = self.frag_cdf_df.plot()
        fig.update_layout(
            title="Fragility",
            xaxis_title=self.edp,
            yaxis_title="P(ds >= Ds | edp)",
        )
        return fig

    def vulnerability_figure(self):
        if not self.init:
            self.__lazy_post_init__()
        fig = self.vuln_cdf_df.plot()
        fig.update_layout(
            title="Vulnerability functions",
            xaxis_title="L_j",
            yaxis_title="P(l <= L_j | ds_i)",
        )
        return fig

    def expected_loss(self, x: float) -> float:
        if not self.init:
            self.__lazy_post_init__()
        vuln_means: np.ndarray = self.vuln_means
        ix = self.frag_cdf_df.index.get_loc(x, method="nearest")
        cdfs = self.frag_cdf_df.iloc[ix].to_numpy()
        probs = np.flip(np.diff(np.flip(cdfs), prepend=0))
        expected_losses = vuln_means * probs
        return expected_losses.sum()

    def losses(self, xs: list[float]) -> list[float]:
        return self.simulate_losses(
            xs,
        )

    def loss(self, x: float) -> float:
        ds = self.simulate_damage_state(x)
        return self.simulate_loss_from_damage_state(ds)

    def simulate_damage_state(self, x: float) -> str:
        if not self.init:
            self.__lazy_post_init__()
        r = random.random()
        for ds in reversed(self.damage_states):
            name = ds["name"]
            prob = self.frag_dict[name].rv.cdf(x)
            if r <= prob:
                return name
        return self.damage_states[0]["name"]

    def simulate_loss_from_damage_state(self, ds: str) -> float:
        return self.vuln_dict[ds].rv.rvs()

    def simulate_loss(self, x: float) -> float:
        ds = self.simulate_damage_state(x)
        return self.simulate_loss_from_damage_state(ds)

    def simulate_damage_states(self, xs: list[float]) -> list[str]:
        return [self.simulate_damage_state(x) for x in xs]

    def simulate_losses_from_damage_states(self, states: list[str]) -> list[float]:
        return [self.simulate_loss_from_damage_state(state) for state in states]

    def simulate_losses(self, xs: list[float]) -> list[float]:
        states = self.simulate_damage_states(xs)
        return self.simulate_losses_from_damage_states(states)


class RiskModelFactory:
    models = {
        "lognormal": LognormalRisk,
        # "normal": NormalRisk
    }

    def __new__(cls, name):
        return cls.models["lognormal"].from_file(RISK_MODELS_DIR / f"{name}.yml")


@dataclass
class Asset(ABC):
    """
    assets are what matters to stakeholders,
    what generate loss,
    they are the elements 'at risk'
    the responsiblity of an asset is to give one of the three D's:
    dollar, downtime or deaths.
    """

    net_worth: float = None
    category: str = None
    name: str = None
    node: int = None
    edp: str = None
    floor: int = None
    x: float = None
    y: float = None
    icon: str = None
    hidden: bool = None
    rugged: bool = None

    @abstractmethod
    def dollars(
        self,
        *,
        strana_results_df: IDAResultsDataFrame,
    ) -> np.ndarray:
        """
        this method reduces strana_results_df indexed by accels
        into a loss_results dataframe which is ONLY accels vs loss
        """
        pass


@dataclass
class RiskAsset(YamlMixin, Asset):
    _risk: Optional[
        LognormalRisk
    ] = None  # done this way to be able to have NormalRisk or GammaRisk using different functions..
    # though it is bad.. it should be agnostic to the class of the _risk... think about this more.

    def __post_init__(self):
        self._risk: LognormalRisk = RiskModelFactory(self.name)
        if self.name is None:
            self.name = self._risk.name
        if self.category is None:
            self.category = self._risk.category
        if self.net_worth is None:
            self.net_worth = self._risk.net_worth
        if self.icon is None:
            self.icon = self._risk.icon
        if self.hidden is None:
            self.hidden = self._risk.hidden
        if self.rugged is None:
            self.rugged = self._risk.rugged
        if self.edp is None:
            self.edp = self._risk.edp

    def dollars(
        self,
        *,
        strana_results_df: IDAResultsDataFrame,
    ) -> np.ndarray:
        print(f"Processing {self.name=} {self.node=} {self.rugged=}")
        strana_results_df["collapse_losses"] = (
            strana_results_df["collapse"]
            .apply(lambda r: self.net_worth if r else 0)
            .values
        )
        if self.rugged:
            strana_results_df["losses"] = (
                strana_results_df["collapse"].apply(lambda r: 0).values
            )
        elif self.node is not None:
            strana_results_df = self.dollars_for_node(
                strana_results_df=strana_results_df
            )
        else:
            strana_results_df = self.dollars_for_storey(
                strana_results_df=strana_results_df
            )
        # print(strana_results_df[['collapse_losses', 'losses']].head())
        losses = strana_results_df[["collapse_losses", "losses"]].apply(max, axis=1)
        return losses

    def dollars_for_node(self, *, strana_results_df: IDAResultsDataFrame) -> np.ndarray:
        from app.strana import StructuralResultView

        paths = strana_results_df["path"].values
        xs = []
        views = {}
        for path in paths:
            if not isinstance(path, str) and np.any(np.isnan(path)):
                x = 1e9
                xs.append(x)
                continue
            x = views.get(path)
            if not x:
                view = StructuralResultView.from_file(Path(path))
                x = view.view_result_by_edp_and_node(edp=self._risk.edp, node=self.node)
                views[path] = x
            xs.append(x)
        losses = self.net_worth * np.array(self._risk.losses(xs))
        strana_results_df["losses"] = losses
        return strana_results_df

    def dollars_for_storey(
        self, *, strana_results_df: IDAResultsDataFrame
    ) -> np.ndarray:
        """
        collapse is taken implicitly as a disproportionate response
        the asset must return complete loss to a big value of edp
        """

        def get_edp_by_floor(row: list[float]) -> float:
            try:
                edp = row[self.floor - 1]
            except IndexError as e:
                print(
                    f"get_edp_by_floor.IndexError {self.name=} {self.edp=} {self.floor=}"
                )
                edp = 0
            except TypeError as e:
                if np.any(np.isnan(row)):
                    return 1e9
                else:
                    raise e
            return edp

        summary_edp = self._risk.edp
        xs = strana_results_df[summary_edp].apply(get_edp_by_floor).values
        losses = self.net_worth * np.array(self._risk.losses(xs))
        strana_results_df["losses"] = losses
        return strana_results_df

    @property
    def fragility_figure(self):
        return self._risk.fragility_figure()

    @property
    def vulnerability_figure(self):
        return self._risk.vulnerability_figure()


class AssetNotFoundException(Exception):
    pass


@dataclass
class AssetFactory:
    @classmethod
    @property
    def model_options(self) -> dict:
        files = find_files(RiskModelFactory, only_yml=True)
        return [{"label": f, "value": f} for f in files]

    def __new__(cls, **data) -> RiskAsset:
        try:
            asset = RiskAsset(**data)
        except FileNotFoundError:
            raise AssetNotFoundException
        return asset
