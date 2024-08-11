from __future__ import annotations
import time
import numpy as np
import pandas as pd
import plotly.express as px
from app.strana import IDA, SummaryEDP
from random import randrange, uniform
from dataclasses import dataclass, asdict, field
from human_id import generate_id as uuid
from plotly.graph_objects import Figure, Scattergl, Scatter, Bar, Line
from app.assets import Asset, LossResultsDataFrame
from app.hazard import HazardCurve, TimeHistorySeries
from app.utils import (
    EDP,
    NamedYamlMixin,
    IDAResultsDataFrame,
    YamlMixin,
    LossModelsResultsDataFrame,
    ScatterResultsDataFrame,
)
from pathlib import Path
from app.utils import ROOT_DIR
from functools import reduce
from collections import defaultdict


LOSS_DIR = ROOT_DIR / "models" / "loss"
LOSS_CSV_RESULTS_DIR = ROOT_DIR / "models" / "loss_csvs"
RATE_CSV_RESULTS_DIR = ROOT_DIR / "models" / "rate_csvs"


class IDANotFoundException(FileNotFoundError):
    pass


@dataclass
class Loss:
    name: str | None = None
    src: str | None = None
    net_worth: float | None = None
    average_annual_loss: float | None = None
    average_annual_loss_pct: float | None = None
    expected_loss: float | None = None
    expected_loss_pct: float | None = None
    std_loss: float | None = None
    _loss_df: LossResultsDataFrame | None = None
    _ida_results_df: IDAResultsDataFrame | None = None
    rate_src: str | None = None
    _rate_df: pd.DataFrame | None = None
    scatter_src: str | None = None
    _srcs_dfs_cache: dict = field(default_factory=dict)
    _scatter_df: ScatterResultsDataFrame | None = None
    _RATE_NUM_BINS: int = 10
    _csv_name: str = ""
    _scatter_csv_name: str = ""
    _collapse_mask_csv_name: str = ""
    _collapse_mask_df: pd.DataFrame | None = None
    _collapse_rate_df: pd.DataFrame | None = None

    def __post_init__(self):
        if self.src:
            self._loss_df = pd.read_csv(self.src)
            self._loss_df = self._loss_df.set_index(["intensity", "freq"])
        if self.rate_src:
            self._rate_df = pd.read_csv(self.rate_src, index_col=0)

    def save_df(self, df: pd.DataFrame, folder: Path, name: str = "") -> str:
        simID = uuid()
        src = str((folder / f"{name}-{simID}.csv").resolve())
        df.to_csv(src)
        return src

    def stats(self):
        return (
            self.average_annual_loss,
            self.expected_loss,
            self.std_loss,
            # self.sum_losses,
            self.expected_loss_pct,
            self.average_annual_loss_pct,
            self.net_worth,
        )

    def _get_and_set_loss_statistics(self):
        df = self._loss_df.copy(deep=True)
        freq = df.index.get_level_values("freq")
        lambda0 = sum(freq)
        df["aal"] = df["mean"] * freq.values
        aal = float(df["aal"].sum())
        mean = aal / lambda0
        self.expected_loss = mean
        self.average_annual_loss = aal
        self.average_annual_loss_pct = self.average_annual_loss / self.net_worth * 100
        self.expected_loss_pct = self.expected_loss / self.net_worth * 100
        return self.stats()

    def _compute_rate_losses(self) -> pd.DataFrame:
        self._loss_linspace = np.linspace(0, self.net_worth, self._RATE_NUM_BINS)
        df = self._loss_df.copy(deep=True)
        df = df[df.columns.difference(["mean", "aal", "std"])]
        record_columns = df.columns
        df["num"] = df[record_columns].count(axis=1).values
        columns = df.columns
        freq = df.index.get_level_values("freq")
        for loss in self._loss_linspace:
            df[loss] = (
                df[df[record_columns] > loss].count(axis=1).values
                * freq.values
                / df["num"]
            )
        self._rate_df = df[
            df.columns.difference(columns)
        ].sum()  # difference messes up column order, but since we are summing it doesnt matter

        self.rate_src = self.save_df(
            self._rate_df, RATE_CSV_RESULTS_DIR, name=self._csv_name
        )
        return self._rate_df

    def aggregated_expected_loss_and_variance_fig(self, df: LossModelsResultsDataFrame):
        # start, stop, num = df.index.min(), df.index.max(), 10
        # binspace = np.linspace(start=start, stop=stop, num=num)
        # bins = pd.cut(df.index, binspace)
        # df = df.groupby(bins).agg(["mean", "std"]).fillna(0)
        # df.columns = [
        #     "_".join([str(c) for c in col]) if type(col) is tuple else col
        #     for col in df.columns.values
        # ]  # this is because groupy and .agg makes them a multiindex and plotly does not support plotting this..
        # df.index = df.index.astype(
        #     "str"
        # )  # sometimes they are ints so we have to convert them to strings..
        fig = df.plot()
        fig.update_layout(
            xaxis_title="accel (g)",
            yaxis_title="Loss $",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            title_font_size=24,
            # xaxis_type="log",
            # yaxis_type="log",
        )
        return fig

    def deaggregate_rate_of_exceedance(
        self, df: LossModelsResultsDataFrame, key: str
    ) -> pd.DataFrame:
        """
        loss_results_df has either a column or index called freq
        and has columns for the loss of each record at each intensity
        ---
        record              78_JA250489EW.csv  87_BL250489EW.csv      mean
        intensity freq
        0.081549  0.054995           0.057233           0.059654  0.058443
        0.182521  0.043240           0.084003           0.085006  0.084505
        0.283492  0.008582           8.812020           0.164348  4.488184
        ---
        """
        if key == "collapse":
            return self._collapse_rate_df

        # make k-LossModelResultsDataFrame, then use the mask trick
        scatter_df = self._scatter_df
        values = scatter_df[key].values.flatten()
        values = set(values)
        dfs = {}
        for v in values:
            filtered_df = scatter_df[scatter_df[key] == v]
            t = pd.pivot_table(
                filtered_df,
                index=["freq", "intensity"],
                columns="record",
                values="loss",
                aggfunc="sum",
            )
            t = t.sort_index(ascending=False)
            dfs[v] = t

        df = self._loss_df.copy(deep=True)
        df = df[df.columns.difference(["mean", "aal", "std"])]
        record_columns = df.columns
        num_records = len(record_columns)
        freq = df.index.get_level_values("freq")
        df_records = df[record_columns]
        new_dfs = defaultdict(list)

        freq = df_records.index.get_level_values("freq").values
        self._loss_linspace = np.linspace(0, self.net_worth, self._RATE_NUM_BINS)
        for loss in self._loss_linspace:
            gt_mask = df_records > loss
            total_df = df_records[gt_mask]
            count = total_df.count(axis=1)
            n = count.sum()
            val = count * freq / num_records
            val = val.sum()
            for v in values:
                _df = dfs[v]
                pct_df = _df[gt_mask]
                pct = (pct_df / total_df).sum().sum() / n
                new_dfs[v].append(pct * val)
        ndf = pd.DataFrame(new_dfs, index=self._loss_linspace)
        return ndf

    def _rate_of_exceedance_for_loss_df(self, df: LossResultsDataFrame) -> pd.DataFrame:
        df = df[df.columns.difference(["mean", "aal", "std"])]
        columns = df.columns
        num_columns = len(columns)
        freq = df.index.get_level_values("freq")
        for loss in self._loss_linspace:
            df[loss] = (
                df[df[columns] > loss].count(axis=1).values * freq.values / num_columns
            )
        rate_df = df[df.columns.difference(columns)].sum()
        return rate_df

    def multiple_rates_of_exceedance_fig(
        self, df: LossModelsResultsDataFrame, key: str
    ):
        vdf = self.deaggregate_rate_of_exceedance(df, key)
        vdf["total"] = vdf.sum(axis=1)
        fig = vdf.plot()
        fig.update_layout(
            xaxis_title="$",
            yaxis_title="v(L)",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            xaxis_type="log",
            yaxis_type="log",
            # title_font_size=24,
            yaxis_range=[-5, -1],
            # margin=dict(l=0, r=0, b=0),
            # title_text=f"rates of exceedance",
        )
        return fig

    # fig = Figure()
    # for col in df.columns:
    #         arr = df[col]
    #         v_df = HazardCurve.rate_of_exceedance(array=arr, years=years, name=col)[
    #             [col, "v"]
    #         ]
    #         trace = Scattergl(
    #             x=v_df[col],
    #             y=v_df.v,
    #             name=col,
    #         )
    #         fig.add_trace(trace)

    #     return fig


@dataclass
class LossModel(YamlMixin, Loss):
    _asset: Asset | None = None
    category: str | None = None
    floor: str | None = None
    _views_by_path: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self._asset is not None:
            self.category = self._asset.category
            self.floor = self._asset.floor
            self.net_worth = self._asset.net_worth
            self.name = self._asset.name
            tic = time.perf_counter()
            self._compute_losses()
            self._get_and_set_loss_statistics()
            self._compute_rate_losses()
            toc = time.perf_counter()
            # print(f"Computed losses in {toc - tic:0.1f} seconds {self.name}")

    def _compute_losses(self) -> LossResultsDataFrame:
        self._csv_name = f"_src-{self.name}-{self.floor}"
        self._scatter_csv_name = f"_scatter-{self.name}-{self.floor}"
        df = self._ida_results_df
        df["loss"] = self._asset.dollars(
            strana_results_df=self._ida_results_df, views_by_path=self._views_by_path
        )
        df["name"] = self._asset.name
        df["category"] = self._asset.category
        df["floor"] = self._asset.floor
        self._scatter_df: ScatterResultsDataFrame = df.copy(deep=True)
        self.scatter_src = self.save_df(
            df, LOSS_CSV_RESULTS_DIR, name=self._scatter_csv_name
        )
        df: LossResultsDataFrame = df.pivot(
            index=["intensity", "freq"], columns="record", values="loss"
        )
        df["mean"] = df.mean(axis=1)
        self._loss_df = df
        self.src = self.save_df(
            self._loss_df, LOSS_CSV_RESULTS_DIR, name=self._csv_name
        )
        return df


@dataclass
class LossAggregator(NamedYamlMixin, Loss):
    ida_model_path: str | None = None
    loss_models: list["LossModel"] | None = None
    _assets: list[Asset] = field(default_factory=list)
    collapse_mask_src: str = ""
    collapse_rate_src: str = ""

    def __post_init__(self):
        super().__post_init__()
        try:
            self._ida = IDA.from_file(self.ida_model_path)
            self.net_worth = self._ida._design.fem.total_net_worth
            self._hazard = self._ida._hazard
            self._assets = self._ida._design.fem.assets
            self._ida_results_df = pd.DataFrame.from_dict(self._ida.results)
            self._csv_name = "__total"
            self._scatter_csv_name = "__total_scatter"
            self._collapse_mask_csv_name = "__collapse_mask"
            self._collapse_rate_csv_name = "__collapse_rate"
        except FileNotFoundError:
            raise IDANotFoundException

        if self.scatter_src:
            self._scatter_df = pd.read_csv(self.scatter_src, index_col=0)

        if self.collapse_mask_src:
            self._collapse_mask_df = pd.read_csv(self.collapse_mask_src, index_col=0)

        if self.collapse_rate_src:
            self._collapse_rate_df = pd.read_csv(self.collapse_rate_src, index_col=0)

    @property
    def asset_records(self) -> list[dict]:
        return [a.to_dict for a in self._assets]

    @property
    def assets_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.asset_records)

    @property
    def loss_models_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.loss_models)

    def run(self) -> None:
        tic = time.perf_counter()
        self._create_loss_models()
        self._compute_losses()
        self._get_and_set_scatter_df()
        self._get_and_set_loss_statistics()
        toc = time.perf_counter()
        print(f"Ran loss in {toc - tic:0.4f} s.")

    def _create_loss_models(self) -> list[LossModel]:
        views_by_path = {}  # cache views because disk i/o is slow
        self.loss_models = [
            LossModel(
                _asset=asset,
                _ida_results_df=self._ida_results_df,
                _views_by_path=views_by_path,
            )
            for asset in self._assets
            if asset
        ]
        return self.loss_models

    def _compute_losses(self) -> LossResultsDataFrame:
        loss_dfs = [lm._loss_df for lm in self.loss_models]
        self._loss_df = self.sum_columns_for_similar_dfs(loss_dfs)
        self.src = self.save_df(
            self._loss_df, LOSS_CSV_RESULTS_DIR, name=self._csv_name
        )
        self._compute_rate_losses()
        return self._loss_df

    def _get_and_set_scatter_df(self) -> ScatterResultsDataFrame:
        df: ScatterResultsDataFrame = self._concat_scatter_dfs()
        self._save_collapse_mask_df(df)
        self._disaggregate_collapse_rate_losses()
        self.scatter_src = self.save_df(
            df, LOSS_CSV_RESULTS_DIR, name=self._scatter_csv_name
        )
        self._scatter_df = df
        return df

    def _disaggregate_collapse_rate_losses(self) -> pd.DataFrame:
        """
        handle it individually, since it is a k-class event that is independent of the loss value
        """
        df = self._loss_df.copy(deep=True)
        df = df[df.columns.difference(["mean", "aal", "std"])]
        record_columns = df.columns
        df["num"] = df[record_columns].count(axis=1).values
        freq = df.index.get_level_values("freq")
        cdf = self._collapse_mask_df
        collapse_types = set(cdf.values.flatten())
        df_records = df[record_columns]
        new_df = defaultdict(list)

        for loss in self._loss_linspace:
            gt_mask = df_records > loss
            collapse_masked = cdf[gt_mask]
            for key in collapse_types:
                key_mask = collapse_masked == key
                df_for_key: pd.DataFrame = df_records[key_mask]
                v = df_for_key.count(axis=1) * freq.values / df.num
                new_df[key].append(v.sum())
        ndf = pd.DataFrame(new_df, index=self._loss_linspace)

        self.collapse_rate_src = self.save_df(
            ndf, LOSS_CSV_RESULTS_DIR, self._collapse_rate_csv_name
        )
        return ndf

    def _save_collapse_mask_df(self, scatter_df: ScatterResultsDataFrame):
        df: LossResultsDataFrame = self._ida_results_df.pivot(
            index=["intensity", "freq"], columns="record", values="collapse"
        )
        self._collapse_mask_df = df
        self.collapse_mask_src = self.save_df(
            df, LOSS_CSV_RESULTS_DIR, name=self._collapse_mask_csv_name
        )

    def _concat_scatter_dfs(self) -> pd.DataFrame:
        scatter_dfs = [lm._scatter_df for lm in self.loss_models]
        df = scatter_dfs[0]
        for _df in scatter_dfs[1:]:
            df = pd.concat([df, _df])
        return df

    def sum_columns_for_similar_dfs(
        self, dfs: list[pd.DataFrame], columns=True
    ) -> pd.DataFrame:
        """algebraic sums dfs that have the same 'schema' across cells, ignoring index"""
        df = dfs[0]
        # if index in df.columns:
        #     df = df.set_index(index)
        for _df in dfs[1:]:
            # if index in df.columns:
            #     _df = _df.set_index(index)
            df = df.add(_df)
        # df = df.reset_index()
        return df

    def aggregate_src_df(
        self, df: LossModelsResultsDataFrame, key: str
    ) -> pd.DataFrame:
        """
        we need to load the df for each asset
        first, get all csvs from src and put the numpy array of losses in a col
        now groupby the key and construct a new dataframe
        """
        # # this does not work because the summary groups by intensity, so the record variability is lost, we must go one level deeper
        # df["collapse"] = df[["scatter_src"]].applymap(
        #     lambda src: pd.read_csv(
        #         src,
        #         usecols=["collapse"],
        #     )
        #     .squeeze("columns")
        #     .to_numpy()
        # )
        if key == "collapse":
            dfs = []

            def deaggregate_collapse(src: str):
                df: LossResultsDataFrame = pd.read_csv(src)
                num_records = len(set(df["record"].values))
                df = (
                    pd.pivot_table(
                        df,
                        index="intensity",
                        columns="collapse",
                        values="loss",
                        aggfunc=sum,
                    )
                    / num_records
                )
                df = df.fillna(0)
                dfs.append(df)
                return df

            df["collapse"] = df[["scatter_src"]].applymap(deaggregate_collapse)
            df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
        else:
            df["asset"] = df[["src"]].applymap(
                lambda src: pd.read_csv(
                    src,
                    usecols=["mean"],
                )
                .squeeze("columns")
                .to_numpy()
            )
            grouped = df.groupby(key)[["asset"]].agg(np.sum)
            data = {
                **grouped.to_dict()["asset"],
            }
            df = pd.DataFrame(
                data, index=self._loss_df.index.get_level_values("intensity")
            )
            df["total"] = df.sum(axis=1)
        return df

    @staticmethod
    def filter_src_df(
        df: LossModelsResultsDataFrame,
        category_filter: list[str],
        name_filter: list[str],
        storey_filter: list[str],
    ) -> pd.DataFrame:
        df = df[
            (df["category"].isin(category_filter))
            & (df["name"].isin(name_filter))
            & (df["floor"].isin(storey_filter))
        ]
        return df

    def scatter_fig(
        self, category_filter=None, name_filter=None, floor_filter=None
    ) -> Figure:
        df = self._scatter_df
        df = self.filter_src_df(df, category_filter, name_filter, floor_filter)
        fig = px.scatter(
            df,
            x="intensity",
            y="loss",
            color="name",
            # size='loss',
            # hover_data=['petal_width']
        )
        return fig

    # def _init_dfs(self):
    #     """
    #     samples from the strana results df using the simulated accels from the hazard and interpolating
    #     constructs an expected_results_df sampling from uniform linspace and interpolating
    #     """
    #     df = self._ida_results_df.sample(frac=1)  # shuffle the dataframe.
    #     df = df.drop_duplicates(
    #         "intensity_str"
    #     )  # effectively samples records for a given intensity, dropping the duplicate rows
    #     df = df.set_index("intensity")
    #     df2 = df.copy(deep=True)
    #     accels = self._accels_series.values
    #     merged_index = df.index.union(accels).sort_values()
    #     df = df.sort_index().reindex(merged_index, method="nearest")
    #     self._sampled_strana_results_df = df.loc[accels]
    #     self.save_df(self._sampled_strana_results_df, SAMPLED_STRANA_CSV_RESULTS_DIR)

    #     low, high = min(df.index), max(df.index)
    #     uniform_linspace = np.linspace(
    #         start=low, stop=high, num=self._NUM_SAMPLES_EXPECTED_LOSS
    #     )
    #     uniform_merged_index = df2.index.union(uniform_linspace).sort_values()
    #     df2 = df2.sort_index().reindex(uniform_merged_index, method="nearest")
    #     self._expected_strana_results_df = df2.loc[uniform_linspace]
    #     self.save_df(self._expected_strana_results_df, EXPECTED_STRANA_CSV_RESULTS_DIR)

    # def _simulate_accels(self) -> TimeHistorySeries:
    #     self._accels_series = self._hazard.get_simulated_timehistory(
    #         num_simulations=self.num_simulations
    #     )
    #     self.num_years = float(self._accels_series.index[-1])
    #     self.accels_src = self.save_df(self._accels_series, ACCELS_CSV_RESULTS_DIR)
    #     return self._accels_series

    # def deaggregate_rate_of_exceedance(
    # """ this doesnt work because it treats each case individually, and so for big losses, there are not events which have l>L because we deaggregated first
    # what you have to do instead is count l>L, THEN look at each case how it is deaggregated... """
    #     self, df: LossModelsResultsDataFrame, key: str
    # ) -> pd.DataFrame:
    #     if key == "collapse":
    #         dfs = []
    #         rates = {}

    #         def deaggregate_collapse(src: str):
    #             df: LossResultsDataFrame = pd.read_csv(src)
    #             num_records = len(set(df["record"].values))
    #             df = (
    #                 pd.pivot_table(
    #                     df,
    #                     index=["intensity", "freq"],
    #                     columns="collapse",
    #                     values="loss",
    #                     aggfunc=sum,
    #                 )
    #                 / num_records
    #             )
    #             df = df.fillna(0)
    #             dfs.append(df)
    #             return df

    #         df["collapse"] = df[["scatter_src"]].applymap(deaggregate_collapse)
    #         df = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
    #         for col in df.columns:
    #             rates[col] = self._rate_of_exceedance_for_loss_df(df[[col]])
    #         index = rates[col].index
    #         vdf = pd.DataFrame(rates, index=index)

    #         return vdf
    #     else:
    #         rates = {}
    #         keys = df[key].unique()
    #         for k in keys:
    #             srcs = df[df[key].isin([k])]["src"].values
    #             sum_df = None
    #             for src in srcs:
    #                 adf = pd.read_csv(src)
    #                 adf = adf.set_index(["intensity", "freq"])
    #                 if sum_df is None:
    #                     sum_df = adf
    #                 else:
    #                     sum_df = sum_df.add(adf)
    #             rates[k] = self._rate_of_exceedance_for_loss_df(sum_df)

    #         index = rates[k].index

    #         vdf = pd.DataFrame(rates, index=index)
    #         return vdf

    def expected_loss_and_variance_fig(self, normalization: float = 1.0):
        df = self._loss_df
        df = df * 1.0 / normalization
        df["std"] = df[df.columns.difference(["mean"])].std(axis=1, ddof=0)
        df = df.reset_index()
        fig = px.line(df, x="intensity", y=["mean", "std"], markers=True)
        fig.update_layout(
            xaxis_title="accel (g)",
            yaxis_title="Loss $",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            legend_font=dict(size=12),
            title_font_size=24,
        )
        return fig

    def rate_fig(self, normalization: float = 1.0, log_axes: bool = True):
        df = self._rate_df
        df.index = df.index * 1.0 / normalization
        fig = px.line(df, markers=True)
        if log_axes:
            fig.update_layout(
                # xaxis_type="log",
                yaxis_type="log",
            )
        fig.update_layout(
            # df.values.max()
            # xaxis_range=[-2, 3],
            # yaxis_range=[-5, 0],
            xaxis_title="$",
            yaxis_title="v($) [1/yr]",
        )
        return fig

    @property
    def summary(self) -> dict:
        df = self._rate_df
        _rate_x = []
        _rate_y = []
        if df is not None:
            _rate_x = df.index.astype(float).tolist()
            _rate_y = df.values.astype(float).tolist()
        return {
            "AAL $": self.average_annual_loss,
            "EL $": self.expected_loss,
            "std L $": self.std_loss,
            # self.sum_losses,
            "EL %": self.expected_loss_pct,
            "AAL %": self.average_annual_loss_pct,
            "net worth $": self.net_worth,
            "_rate_x": _rate_x,
            "_rate_y": _rate_y,
        }
