from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import plotly.express as px
from app.assets import LOSS_MODELS_DIR
from app.concrete import RectangularConcreteColumn
from app.criteria import DesignCriterionFactory
from app.design import ReinforcedConcreteFrame
from app.hazard import (
    RECORDS_DIR,
    Hazard,
    HazardCurves,
    Record,
)
from app.strana import (
    STRANA_DIR,
    IDA,
    HazardNotFoundException,
    SpecNotFoundException,
    StructuralResultView,
)
from app.loss import LossAggregator, IDANotFoundException, LossModel
from app.utils import (
    DESIGN_DIR,
    HAZARD_DIR,
    RESULTS_DIR,
    find_files,
    COMPARE_DIR,
    LossModelsResultsDataFrame,
)
from app.occupancy import BuildingOccupancy
from app.compare import IDACompare
from st_cytoscape import cytoscape
from streamlit import session_state as state
from functools import partial


HOME = Path.home()
FIGURES_PATH = (
    HOME / "Dropbox/phd_thesis/figures/compare"
)  # TODO: remove when wrapping up


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_MODULE = 1
padding_top = 2

st.markdown(
    f"""
    <style>
        .appview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)  # remove unnecessary padding at top

if "module" not in state:
    state.module = DEFAULT_MODULE
    state.design_abspath = ""
    state.hazard_abspath = ""
    state.ida_abspath = ""
    state.loss_abspath = ""
    state.compare_abspath = ""

if "first_render" not in state:
    state.first_render = True


def switch_module(delta: int):
    state.module = state.module + delta


def goto_module(mod: int):
    state.module = mod


title = (
    "design",
    "hazard",
    "structural analysis",
    "loss",
    "compare",
)

left, right = st.columns(2)
if state.module != 1:
    prev = left.button("back", on_click=switch_module, args=[-1])
if state.module != 5:
    next = right.button("next", on_click=switch_module, args=[1])

with st.sidebar:
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.button("des", on_click=partial(goto_module, 1))
    b2.button("haz", on_click=partial(goto_module, 2))
    b3.button("ida", on_click=partial(goto_module, 3))
    b4.button("loss", on_click=partial(goto_module, 4))
    b5.button("com", on_click=partial(goto_module, 5))
    st.header(title[state.module - 1])

    if state.module == 1:
        state.design_abspath
        state.hazard_abspath
        buildings = find_files(DESIGN_DIR)
        file = st.selectbox("select a building", options=buildings)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "",
        )
        if state.design_abspath and not state.first_render:
            design = ReinforcedConcreteFrame.from_file(state.design_abspath)
        elif file:
            state.design_abspath = DESIGN_DIR / file
            design = ReinforcedConcreteFrame.from_file(state.design_abspath)
        else:
            design = ReinforcedConcreteFrame(name=name)

        col1, col2 = st.columns(2)
        damping = col1.number_input(
            r"% damping",
            value=int(design.damping * 100) or 5,
            min_value=0,
            step=1,
            max_value=100,
            format="%d",
            help="number of identical perpendicular frames",
        )
        fc = col2.number_input(
            r"f'c MPa",
            value=int(design.fc / 1000) or 30,
            min_value=10,
            step=1,
            max_value=100,
            format="%d",
            help="number of identical perpendicular frames",
        )
        storeys_input = st.text_input(
            "storeys",
            value=",".join([str(s) for s in design.storeys]) or "3,",
            help="heights in meters separated by comma",
        )
        storeys = [float(s) for s in storeys_input.split(",")]
        np.array(storeys).cumsum().tolist()
        cumstoreys = "sum: " + ",".join(
            [f"{b:.2f}" for b in np.array(storeys).cumsum().tolist()]
        )
        cumstoreys
        bays_input = st.text_input(
            "bays",
            value=",".join([str(s) for s in design.bays]) or "5",
            help="widths in meters separated by comma",
        )
        bays = [float(s) for s in bays_input.split(",")]
        cumbays = "sum: " + ",".join(
            [f"{b:.2f}" for b in np.array(bays).cumsum().tolist()]
        )
        cumbays
        num_frames = st.number_input(
            "num frames",
            min_value=2,
            step=1,
            max_value=len(bays) + 1,
            value=min(design.num_frames, len(bays) + 1),
            format="%d",
            help="number of identical perpendicular frames, they will follow bay spacing.",
        )
        occupancy = st.selectbox(
            "occupancy class",
            BuildingOccupancy.options(),
            index=BuildingOccupancy.options().index(
                design.occupancy or BuildingOccupancy.DEFAULT
            ),
        )
        design_criteria = st.selectbox(
            "design criteria",
            DesignCriterionFactory.public_options(),
            index=DesignCriterionFactory.public_options().index(
                design.design_criteria[0] or DesignCriterionFactory.DEFAULT
            ),
            # this [0] is bad design. the idea was to have multiple criteria, this is too complex in the ui.
            # FEMs and Criteria should be singletons instead of lists
        )

        left, right = st.columns(2)
        delete = left.button("🗑️", help="delete this building")
        if delete:
            with st.spinner("deleting building"):
                time.sleep(2)
                design.delete(DESIGN_DIR)
                design = None
                state.design_abspath = None
                st.success("delete successful")

        run_design = right.button("design", help="run a design")
        params_missing = not name or len(storeys) == 0 or len(bays) == 0
        if run_design and params_missing:
            st.error("params missing or incorrect")
        elif run_design:
            with st.spinner("running design"):
                design = ReinforcedConcreteFrame(
                    name=name,
                    storeys=storeys,
                    bays=bays,
                    num_frames=int(num_frames),
                    fc=fc * 1000,
                    damping=damping / 100,
                    occupancy=occupancy,
                    design_criteria=[design_criteria],
                )
                design.force_design(DESIGN_DIR)
                design.fem.pushover(DESIGN_DIR / design.name)
                design.to_file(DESIGN_DIR)
                state.design_abspath = DESIGN_DIR / design.name_yml

            st.success("design successful")

        elements = None
        if design and design.fems:
            elements, stylesheet = design.fem.cytoscape()

    if state.module == 2:
        state.design_abspath
        state.hazard_abspath
        hazards = find_files(HAZARD_DIR)
        hazard = Hazard(
            name="sample_hazard",
        )
        file = st.selectbox("select a hazard", options=hazards)
        if file:
            state.hazard_abspath = HAZARD_DIR / file
            hazard = Hazard.from_file(HAZARD_DIR / file)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "",
            help="add or remove records to save",
        )
        hazard.name = name
        # hazard._curve.html(st)
        # curve_type = st.selectbox(
        #     "select a curve type",
        #     options=HazardCurveFactory.options(),
        #     index=HazardCurveFactory.options().index(hazard._curve.name),
        # )
        sites = find_files(RECORDS_DIR / "hazards", suffix=".gra")
        file = st.selectbox("select hazard file", options=sites)
        path = RECORDS_DIR / "hazards" / file
        hazards = HazardCurves(path)
        site_names = list(hazards.dfs.keys())
        try:

            index = site_names.index(hazard.site)
            site_name = st.selectbox(
                "select hazard sites", options=site_names, index=index
            )
            site = hazards.dfs[site_name]
            periods = list(site.keys())
            design = ReinforcedConcreteFrame.from_file(state.design_abspath)
            T = design.fem.period
            period = sorted(periods, key=lambda t: abs(t - T))[0]

            # select_manually = st.checkbox("select period manually")
            # if select_manually:
            #     period = st.selectbox("periods", periods)
            # hazard.period = period

            curve = site[period]
            hazard.curve = curve
            hazard._curve = curve
            hazard.site = site_name
        except ValueError:
            pass
        hazard.to_file(HAZARD_DIR)

        st.subheader(f"Records ({len(hazard.records)})")
        record_files = find_files(RECORDS_DIR, only_yml=False, only_csv=True)
        record_files = ["add a record"] + record_files
        record_name = st.selectbox("add a record", options=record_files)
        num_recs = st.number_input("num records to sample", 1, 100, 5, step=1)
        if record_name != "add a record":
            record_path = str((RECORDS_DIR / record_name).resolve())
            record = Record(record_path)
            hazard.add_record(record)
            hazard.to_file(HAZARD_DIR)
        left, middle, right = st.columns(3)
        go = left.button(f"sample {int(num_recs)}", help="grab at random")

        if go:
            with st.spinner("sampling..."):
                time.sleep(1)
                untouched = [r for r in record_files if r not in hazard.record_names]
                samples = random.sample(untouched, int(num_recs))
                for path in samples:
                    if path == "add a record":
                        continue
                    record_path = str((RECORDS_DIR / path).resolve())
                    record = Record(record_path)
                    hazard.add_record(record)
            hazard.to_file(HAZARD_DIR)
            st.success("sampled records successfully")

        remove_all = right.button("🗑️ rm all", help="remove all records")
        if remove_all:
            with st.spinner("removing..."):
                time.sleep(2)
                hazard.records = []
            hazard.to_file(HAZARD_DIR)
            st.success("removed all records successfully")

        add_all = middle.button("add all", help="add all records")
        if add_all:
            with st.spinner("adding..."):
                time.sleep(2)
                for path in record_files:
                    if path == "add a record":
                        continue
                    record_path = str((RECORDS_DIR / path).resolve())
                    record = Record(record_path)
                    hazard.add_record(record)
            hazard.to_file(HAZARD_DIR)
            st.success("added all records successfully")
        selected_ix = None
        for ix, r in enumerate(hazard.records):
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(r.name)
                view_record = c2.button("view", key=f"record{ix}")
                if view_record:
                    selected_ix = ix
                rm = c3.button("🗑️", key=f"rm_record{ix}")
                if rm:
                    rec = hazard.records[ix]
                    record_path = str((RECORDS_DIR / rec.name).resolve())
                    hazard.remove_record(record_path)
                    hazard.to_file(HAZARD_DIR)
                    st.success("record removed")

            state.first_render = False

    if state.module == 3:
        "design:", state.design_abspath

        stranas = find_files(STRANA_DIR)
        design_missing = False
        hazard_missing = False
        selected_ix = None
        file = st.selectbox("select an analysis", options=stranas)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "default ida",
            help="click run to save",
        )
        # "file:", file
        try:
            if file:
                ida = IDA.from_file(STRANA_DIR / file)
                state.ida_abspath = STRANA_DIR / file
                ida.design_abspath = str(state.design_abspath) or ida.design_abspath
                ida.hazard_abspath = str(state.hazard_abspath) or ida.hazard_abspath
            else:
                ida = IDA(
                    name="default-ida",
                    design_abspath=str(state.design_abspath),
                    hazard_abspath=str(state.hazard_abspath),
                )
                ida.name = name
                file = ida.name_yml
        except HazardNotFoundException as e:
            e
            print(e)
            hazard_missing = True
        except SpecNotFoundException as e:
            e
            print(e)
            design_missing = True

        if file and not (design_missing or hazard_missing):
            "hazard:", state.hazard_abspath
            run = st.button("run IDA", help="runs an evenly spaced grid")
            with st.expander("manual ida"):
                haz_spaced = st.button(
                    "run for hazard points", help="uses the hazard points only"
                )
                even_spaced = st.button("even grid", help="run an evenly spaced grid")
                elastic_spaced = st.button(
                    "use Say", help="run some points until Say, then fill the rest"
                )
                delete = st.button("🗑️", help="delete this analysis")
                # st.warning(
                #     "BETA: this might give wrong loss results for Sa>a in hazard"
                # )
                # start = st.number_input(
                #     r"start Sa (g)",
                #     value=ida.start,
                #     min_value=0.0,
                #     step=ida.step,
                #     max_value=10.0,
                #     format="%g",
                #     help="starting Sa (g)",
                # )
                # stop = st.number_input(
                #     r"stop Sa (g)",
                #     value=ida.stop,
                #     min_value=0.0,
                #     step=ida.step,
                #     max_value=10.0,
                #     format="%g",
                #     help="stop Sa (g)",
                # )
                # step = st.number_input(
                #     r"step Sa (g)",
                #     value=ida.step,
                #     min_value=0.005,
                #     step=0.1,
                #     max_value=10.0,
                #     format="%g",
                #     help="step Sa (g)",
                # )
            if delete:
                with st.spinner("deleting analysis"):
                    time.sleep(1)
                    ida.delete(STRANA_DIR)
                    ida = None
                    state.ida_abspath = None
                    st.success("delete successful")

            if run or haz_spaced or even_spaced or elastic_spaced:
                with st.spinner("running..."):
                    time.sleep(1)
                    ida = IDA(
                        **{
                            **ida.to_dict,
                            # "start": start,
                            # "stop": stop,
                            # "step": step,
                            "name": name,
                            "hazard_spaced": haz_spaced,
                            "evenly_spaced": even_spaced or run,
                            "elastically_spaced": elastic_spaced,
                        },
                    )
                    ida.run_parallel(results_dir=RESULTS_DIR)
                    state.ida_abspath = STRANA_DIR / ida.name_yml
                    ida.to_file(STRANA_DIR)

                st.success("analysis successful")

            if ida.results:
                for ix, r in enumerate(ida.results):
                    with st.container():
                        c1, c2 = st.columns([5, 1])
                        coll = "💀" if r["collapse"] != "none" else ""
                        c1.write(f'{r["record"]} - {r["intensity"]:.4f}g - {coll}')
                        view_result = c2.button("view", key=f"record{ix}")
                        if view_result:
                            selected_ix = ix

    if state.module == 4:
        "ida path:", state.ida_abspath
        losses = find_files(LOSS_MODELS_DIR)
        ida_missing = False
        file = st.selectbox("select a loss file", options=losses)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "default loss",
            help="click run to save",
        )
        loss = None
        "loss abspath", state.loss_abspath
        try:
            if file:
                loss = LossAggregator.from_file(LOSS_MODELS_DIR / file)
                state.loss_abspath = LOSS_MODELS_DIR / file
                loss.ida_model_path = state.ida_abspath
            elif state.loss_abspath:
                loss = LossAggregator.from_file(state.loss_abspath)
            else:
                loss = LossAggregator(name=name, ida_model_path=str(state.ida_abspath))
                loss.name = name
        except IDANotFoundException as e:
            e
            print(e)
            ida_missing = True

        left, right = st.columns(2)
        go = left.button("run", help="perform loss computation")
        if go:
            with st.spinner("running..."):
                loss_dict = {**loss.to_dict} if loss else {}
                loss = LossAggregator(
                    **{
                        **loss_dict,
                        "name": name,
                        "ida_model_path": str(state.ida_abspath),
                    }
                )
                loss.run()
                loss.to_file(LOSS_MODELS_DIR)
                state.loss_abspath = LOSS_MODELS_DIR / loss.name_yml
            st.success("success")

        rm = right.button("🗑️", help="delete")
        if rm:
            with st.spinner("deleting..."):
                time.sleep(1)
                loss.delete(LOSS_MODELS_DIR)
                loss = None
            st.success("delete successful")

        if loss is not None:
            if not ida_missing:
                design = loss._ida._design
                st.header("Design")
                st.text(f"{design.name}")
                st.text(f"St {design.num_storeys} bays {design.num_bays}")
                st.text(f"{design.design_criteria}")
                st.text(f"{design.occupancy.split('.')[0]}")
                st.metric("Net worth", design.fem.readable_total_net_worth)

            assets = loss.to_dict["loss_models"] or []
            filtered_assets = assets
            fig = design.fem.assets_pie_fig

            fig.update_layout(height=300, width=300)
            st.plotly_chart(fig, theme=None)
            st.header("Filter")

            all_categories = sorted(list(set([lm["category"] for lm in assets])))
            selected_categories = st.multiselect(
                "Category", options=all_categories, default=all_categories
            )
            all_floors = sorted(list(set([lm["floor"] for lm in assets])))
            selected_floors = st.multiselect(
                "Floor", options=all_floors, default=all_floors
            )

            all_names = sorted(list(set([lm["name"] for lm in assets])))
            selected_names = st.multiselect(
                "Name", options=all_names, default=all_names
            )

            st.header("View")
            selected_ix = None
            view_all = st.button("view all", help="view stats for building")
            view_asset = None
            for ix, a in enumerate(filtered_assets):
                with st.container():
                    c1, c2 = st.columns([5, 1])
                    c1.write(f'{a["name"]}-{a["category"]}-{a["floor"]}')
                    view_asset = c2.button("view", key=f"asset{ix}")
                    if view_asset:
                        selected_ix = ix

    if state.module == 5:
        "hazard abspath: ", state.hazard_abspath
        compares = find_files(COMPARE_DIR)
        compare_file = st.selectbox("select a compare file", options=compares)
        name = st.text_input(
            "give it a name",
            value=compare_file.split(".")[0] if compare_file else "default compare",
            help="click run to save",
        )
        hazard_abspath = str(state.hazard_abspath)
        hazard_missing = False
        try:
            if compare_file:
                compare = IDACompare.from_file(COMPARE_DIR / compare_file)
            else:
                compare = IDACompare(name=name, hazard_abspath=hazard_abspath)
        except HazardNotFoundException as e:
            print("You must select a hazard first", e)
            hazard_missing = True
            raise e

        compare.name = name
        compare.hazard_abspath = hazard_abspath
        hazard = None
        hazards = find_files(HAZARD_DIR)
        hazard_file = st.selectbox("select a hazard", options=hazards)
        if hazard_file:
            state.hazard_abspath = HAZARD_DIR / hazard_file
            hazard = Hazard.from_file(HAZARD_DIR / hazard_file)

        "num records:", len(hazard.records)
        left, right = st.columns(2)
        discount_factor = left.number_input(
            "discount factor", min_value=0.0, max_value=1.0, value=0.05, step=0.01
        )
        secondary_loss = right.number_input(
            "secondary losses", min_value=0.0, max_value=100.0, value=0.0, step=1.0
        )
        # not sure why this floats below every element
        # left, right = st.columns(2)
        # logx = left.checkbox("log x", value=True)
        # logy = right.checkbox("log y", value=True)
        # if hazard:
        #     hazard_fig = hazard.rate_figure(logx=logx, logy=logy)
        # st.plotly_chart(hazard_fig, use_container_width=True, theme=None)

        design_files = find_files(DESIGN_DIR, only_yml=True)
        design_files = ["add a design"] + design_files
        design_name = st.selectbox("add a design", options=design_files)

        if design_name and not state.first_render and design_name != "add a design":
            design_path = str((DESIGN_DIR / design_name).resolve())
            compare.add_design(design_path)
            compare.to_file(COMPARE_DIR)

        for ix, comp in enumerate(compare.comparisons):
            _c1, _c2, _c3, _c4, _c5, _c6 = st.columns([2, 1, 1, 1, 1, 1])
            with st.container():
                _c1.write(comp.design_abspath.split("/")[-1])
                go_po = _c2.button(
                    "po", help="perform pushover", key=f"po-compare-design-{ix}"
                )
                go_strana = _c3.button(
                    "ida", help="perform ida", key=f"ida-compare-design-{ix}"
                )
                go_loss = _c4.button(
                    "loss",
                    help="perform loss computation",
                    key=f"loss-compare-design-{ix}",
                )
                go_complete = _c5.button(
                    "all", help="perform ida then loss", key=f"all-compare-design-{ix}"
                )
                rm = _c6.button("🗑️", key=f"rm-compare-design-{ix}")
                if go_po:
                    comp.run(pushover=True)
                if go_strana:
                    comp.run(strana=True)
                if go_loss:
                    comp.run(loss=True)
                if go_complete:
                    comp.run(strana=True, loss=True)
                if rm:
                    compare.remove_design(comp.design_abspath)
                if any([go_po, go_strana, go_loss, go_complete, rm]):
                    compare.to_file(COMPARE_DIR)
                    st.success("success")

        c0, c1, c2, c3, c4 = st.columns(5)
        po = c0.button("run po", help="perform pushover comparison")
        go = c1.button("run ida", help="perform ida comparison")
        loss = c2.button("run loss", help="perform loss computation ")
        complete = c3.button("run all", help="run ida then loss on all models")
        rm = c4.button("🗑️", help="delete")
        if rm:
            with st.spinner("deleting..."):
                time.sleep(1)
                compare.delete(COMPARE_DIR)
            st.success("delete successful")

        if compare and hazard and not hazard_missing:
            if any(
                [
                    po,
                    go,
                    loss,
                    complete,
                ]
            ):
                compare_dict = compare.to_dict
                compare = IDACompare(
                    **{
                        **compare_dict,
                        "name": name,
                        "hazard_abspath": hazard_abspath,
                    }
                )
                with st.spinner("running..."):
                    if po:
                        compare.run(pushover=True)
                    if go:
                        compare.run(strana=True)
                    if loss:
                        compare.run(loss=True)
                    if complete:
                        compare.run(strana=True, loss=True)

                compare.to_file(COMPARE_DIR)
                st.success("success")

        state.first_render = False

if state.module == 1:
    if elements is None or design is None or design.fems is None:
        st.header("Create a design")
    else:
        selected = cytoscape(
            elements=elements,
            stylesheet=stylesheet,
            layout={
                "name": "preset",
            },  # this layout respects nodes' (x,y)
            height="600px",
            user_panning_enabled=True,
            max_zoom=1,
            min_zoom=0.1,
            key="fem",
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Net worth", design.fem.readable_total_net_worth)
        col2.metric("fundamental period", f"{design.fem.period:.2f} s")
        col3.metric("total area", f"{design.total_area:.1f} m²")
        col4.metric(
            "cost per unit area", f"{design.fem.cost_per_unit_area:.1f} k USD/m²"
        )
        col5.metric("weight", f"{design.weight_str} kN")

        with st.expander("costs"):
            col1, col2, col3 = st.columns(3)
            col2.metric("Structural", f"$ {design.fem.elements_net_worth:.0f} k ")
            col1.metric(
                "Nonstructural", f"$ {design.fem.nonstructural_net_worth:.0f} k "
            )
            col3.metric("Contents", f"$ {design.fem.contents_net_worth:.0f} k ")
            fig = design.fem.assets_pie_fig
            st.header(f"summary {design.name}")
            st.plotly_chart(fig, theme=None)
            asset_records = [a.to_dict for a in design.fem.assets]
            df = pd.DataFrame.from_records(asset_records)
            columns = "name category edp net_worth hidden bay storey floor rugged x node".split(
                " "
            )
            df = df[columns]
            df2 = df[df.category == "structural"].groupby("name").sum()
            df2["name"] = df2.index
            structural_pie_fig = px.pie(
                df2, names="name", values="net_worth", height=400
            )
            st.header("structural")
            st.plotly_chart(structural_pie_fig, theme=None)

            df3 = df[df.category == "nonstructural"].groupby("name").sum()
            df3["name"] = df3.index
            nonstructural_pie_fig = px.pie(
                df3, names="name", values="net_worth", height=400
            )
            st.header("non structural")
            st.plotly_chart(nonstructural_pie_fig, theme=None)

            df4 = df[df.category == "contents"].groupby("name").sum()
            df4["name"] = df4.index
            contents_pie_fig = px.pie(df4, names="name", values="net_worth", height=400)
            st.header("contents")
            st.plotly_chart(contents_pie_fig, theme=None)
            df

        # with st.expander("summary"):
        #     st.dataframe(pd.DataFrame([design.fem.pushover_stats()]))

        # with st.expander("assets"):
        #     pass

        with st.expander("eigen"):
            eigen, storeys = design.fem.eigen_df
            "modes"
            st.dataframe(eigen)
            "storeys"
            st.dataframe(storeys)

        with st.expander("capacity"):
            # path = DESIGN_DIR / design.name
            # fig, nfig = design.fem.pushover_figs(path)
            fig, nfig = design.fem.pushover_figs()
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig, theme=None)
            col2.plotly_chart(nfig, theme=None)
            stats = design.fem.pushover_stats()
            design_c_error = stats["design_error"]
            c_design = stats["c_design"]
            design_period_error = stats["period_error"]
            period = stats["period [s]"]
            miranda_period = stats["miranda period [s]"]
            cs = stats["cs"]
            Vy_design = stats["Vy_design"]
            Vy = stats["Vy"]
            Vy_error = stats["Vy_error"]
            uy = stats["uy"]
            drift_y = stats["drift_y"]
            ductility = stats["ductility"]

            col1, col2 = st.columns(2)
            col1.header("Design values")
            col1.metric(
                label="period T0",
                value=miranda_period,
            )
            col1.metric(
                label="seismic coeff Cs",
                value=c_design,
            )
            col1.metric(
                label="Vy base shear",
                value=Vy_design,
            )

            col2.header("Empirical (measured)")
            col2.metric(label="period T0", value=period, delta=design_period_error)
            col2.metric(label="Vy base shear", value=Vy, delta=Vy_error)
            col2.metric(label="Say_g, cs (g)", value=cs, delta=design_c_error)
            col2.metric(
                label="drift yield",
                value=drift_y,
            )
            col2.metric(
                label="roof disp yield",
                value=uy,
            )

            col2.metric(label="ductility", value=f"{ductility:.2f}")

            st.subheader("Springs timehistory (Moments)")
            fig = design.fem._pushover_view.generate_springs_visual_timehistory_fig(
                design, thinner=1
            )
            st.plotly_chart(fig, theme=None)

        with st.expander("design details"):
            # st.header("Moments and shears")
            # st.dataframe(design.fem.extras)

            def color_survived(val):
                if val > 0 and val < 1:
                    color = "yellow"
                elif val <= 0:
                    color = f"opacity: 1%;"
                elif val > 1.0 and val < 10.0:
                    color = "green"
                elif val > 10:
                    color = "silver"
                else:
                    color = "white"
                return f"background-color: {color}"

            st.subheader("Column/beam ratios")
            sdf = design.fem.structural_elements_breakdown()
            properties = sdf.columns.to_list()
            DEFAULT_PROPERTY = "s"
            My_index = properties.index(DEFAULT_PROPERTY)
            key = st.selectbox("property", options=properties, index=My_index)
            df = design.fem.column_beam_ratios(key=key)

            df = df.replace(0, np.nan)
            styler = df.style.format("{:.2f}", na_rep="-")
            sty = styler.applymap(color_survived)
            st.dataframe(sty)

            c1, c2 = st.columns(2)

            view = design.fem._pushover_view
            cols_moments = view.view_column_springs_moments()
            beams_moments = view.view_beam_springs_moments()

            c1.header("Columns backbone")
            options = list(range(len(design.fem.springs_columns)))
            ix = c1.selectbox("index", options=options, index=0)
            #
            col = design.fem.springs_columns[ix]
            base_col = col
            fig = base_col.moment_rotation_member_figure()
            c1.plotly_chart(fig, theme=None)

            c1.subheader("Columns modified backbone")
            fig = base_col.moment_rotation_figure()
            c1.plotly_chart(fig, theme=None)

            c1.dataframe(col.to_dict)
            fig = cols_moments[ix + 1].plot()
            c1.plotly_chart(fig)

            c2.header("Beams backbone")
            options = list(range(len(design.fem.springs_beams)))
            ix = c2.selectbox("index", options=options, index=0)
            #
            beam = design.fem.springs_beams[ix]
            fig = beam.moment_rotation_member_figure()
            c2.plotly_chart(fig, theme=None)

            c2.subheader("Beams modified backbone")
            fig = beam.moment_rotation_figure()
            c2.plotly_chart(fig, theme=None)

            c2.dataframe(beam.to_dict)

            fig = beams_moments[ix + 1].plot()
            c2.plotly_chart(fig)

        with st.expander("Element properties"):
            desired_columns = "name model type storey bay My Vy Mc b h radius phi_y phi_y2 phi_y_fardis phi_y_fardis2 theta_y theta_y_fardis  theta_cap_cyclic theta_pc_cyclic theta_u_cyclic Ks Ks_original Ke Ke_Ks_ratio edp p s Ix Iy Ig Ic ".split(
                " "
            )
            desired_columns = [c for c in desired_columns if c in sdf.columns.to_list()]
            sorted_unique_columns = sorted(
                list(set(sdf.columns.tolist()) - set(desired_columns))
            )
            columns = desired_columns + sorted_unique_columns
            sdf = sdf[columns]
            sdf = sdf.sort_values(["storey", "bay"])
            st.dataframe(sdf, height=1000)
            st.subheader("ratios")
            df = pd.DataFrame()
            st.dataframe(df)

if state.module == 2:
    left, mid, right = st.columns(3)
    normalize_g = left.checkbox("normalize (g)", value=True)
    logx = mid.checkbox("log x", value=True)
    logy = right.checkbox("log y", value=True)
    if hazard.curve is not None:
        fig = hazard.rate_figure(normalize_g=normalize_g, logx=logx, logy=logy)
        # st.text(hazard.hazard_spaced_intensities_for_idas())
        # st.text(hazard.evenly_spaced_intensities_for_idas())
        # st.text(
        #     hazard.elastically_spaced_intensities_for_idas(
        #         design.fem.extras["c_design"]
        #     )
        # )
        st.plotly_chart(fig, theme=None)
        left, mid, right = st.columns(3)
        hazard_spaced = left.checkbox(label="hazard")
        evenly = mid.checkbox(label="even spacing")
        elastic = right.checkbox(label="elastic")
        fig = hazard.freq_figure(hazard=hazard, evenly=evenly, elastic=elastic)
        st.plotly_chart(fig, theme=None)
        if len(hazard.records) > 0:
            record = (
                hazard.records[selected_ix]
                if selected_ix is not None
                else hazard.records[0]
            )
            st.plotly_chart(record.figure(normalize_g=normalize_g), theme=None)
            st.plotly_chart(record.spectra(normalize_g=normalize_g), theme=None)

if state.module == 3:
    if design_missing:
        st.warning("Please select a design")
    if hazard_missing:
        st.warning("Please select a hazard")
    if not (design_missing or hazard_missing) and ida and ida.results:
        normalize = st.checkbox("Normalize")
        if normalize:
            fig = ida.view_normalized_ida_curves()
        else:
            fig = ida.view_ida_curves()
        st.plotly_chart(fig, container_width=True, theme=None)

        fig = ida.view_median_curves()
        st.plotly_chart(fig, container_width=True, theme=None)

        st.dataframe(pd.DataFrame.from_records(ida.results), height=800)

        if selected_ix is not None:
            instance_path = ida.results[selected_ix]["path"]
            view = StructuralResultView.from_file(instance_path)
            st.subheader("Springs timehistory (Moments)")
            fig = view.generate_springs_visual_timehistory_fig(ida._design)
            st.plotly_chart(fig, theme=None)
            figures = view.timehistory_figures
            # findme
            for fig in figures:
                st.plotly_chart(fig, theme=None)
            # cols_moments = view.view_spring_moment_rotation_th(ele_type="COLUMN")
            # beams_moments = view.view_spring_moment_rotation_th(ele_type="BEAM")

            c1, c2 = st.columns(2)
            design = ida._design
            c1.header("Columns hysteresis")
            options = design.fem.springs_columns[:5]
            for ix, _ in enumerate(options, start=1):
                col: RectangularConcreteColumn = design.fem.springs_columns[ix - 1]
                df = view.view_column_spring_moment_rotation_th(ix=ix)
                DS = col.park_ang_kunnath_DS(df)
                fig = view.view_column_spring_moment_rotation_fig(ix=ix, DS=DS)
                c1.plotly_chart(fig, theme=None)

            c2.header("Beams hysteresis")
            options = design.fem.springs_beams[:5]
            for ix, _ in enumerate(options, start=1):
                beam: RectangularConcreteColumn = design.fem.springs_beams[ix - 1]
                df = view.view_beam_spring_moment_rotation_th(ix=ix)
                DS = beam.park_ang_kunnath_DS(df)
                fig = view.view_beam_spring_moment_rotation_fig(ix=ix, DS=DS)
                c2.plotly_chart(fig, theme=None)

    else:
        st.header("run analysis to see results")

if state.module == 4:
    if ida_missing:
        st.warning("Please select an analysis")
    normalization = 1.0
    left, right = st.columns(2)
    normalize = left.checkbox("Normalize")
    # WIP normalization
    asset: LossAggregator = loss
    # if not view_all or selected_ix is None:
    #     st.header("please run loss")
    if view_asset or selected_ix is not None:
        model_dict = asset.loss_models[selected_ix]
        asset = LossModel(**model_dict, _ida_results_df=asset._ida_results_df)

    if normalize:
        normalization = asset.net_worth

    normalize_wrt_building = right.checkbox("Normalize to building cost", True)

    if normalize_wrt_building:
        normalization = loss.net_worth
    (
        average_annual_loss,
        expected_loss,
        std_loss,
        # sum_losses,
        expected_loss_pct,
        average_annual_loss_pct,
        net_worth,
    ) = asset.stats()
    if loss and average_annual_loss is not None:
        one, two, three, four = st.columns(4)
        one.metric("AAL", f"{average_annual_loss or 0:.4f}")
        two.metric("AAL %", f"{average_annual_loss_pct or 0:.3f}")
        three.metric("E[L]", f"{expected_loss or 0:.4f}")
        four.metric("E[L] %", f"{expected_loss_pct or 0:.3f}")
        st.header("Building" if view_all or selected_ix is None else asset.name)
        agg_key_1 = None
        if selected_ix is None:
            agg_key_1 = st.selectbox(
                "Deaggregate",
                options=["", "name", "category", "floor", "collapse"],
                format_func=lambda x: "Select an option" if x == "" else x,
                help="display results broken down by",
            )
        if not agg_key_1:
            fig = asset.rate_fig(normalization=normalization)
            one.plotly_chart(fig, theme=None)
            fig = asset.expected_loss_and_variance_fig(normalization=normalization)
            two.plotly_chart(fig, theme=None)
            if selected_ix is None:
                fig = asset.scatter_fig(
                    category_filter=selected_categories,
                    name_filter=selected_names,
                    floor_filter=selected_floors,
                )
                three.plotly_chart(fig, theme=None)
        elif agg_key_1 and selected_ix is None:
            df: LossModelsResultsDataFrame = loss.loss_models_df
            df = loss.filter_src_df(
                df,
                category_filter=selected_categories,
                name_filter=selected_names,
                storey_filter=selected_floors,
            )
            fig = loss.multiple_rates_of_exceedance_fig(df, key=agg_key_1)
            st.plotly_chart(fig, theme=None)
            df = loss.aggregate_src_df(df, key=agg_key_1)
            df = df * 1.0 / normalization
            fig = loss.aggregated_expected_loss_and_variance_fig(df)
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, theme=None)
            fig = asset.scatter_fig(
                category_filter=selected_categories,
                name_filter=selected_names,
                floor_filter=selected_floors,
            )
            st.plotly_chart(fig, theme=None)
            if agg_key_1 != "collapse":
                agg_key_2 = st.selectbox(
                    "2nd deaggregator",
                    options=[
                        k
                        for k in ["", "name", "category", "floor", "collapse"]
                        if k != agg_key_1
                    ],
                    format_func=lambda x: "Select an option" if x == "" else x,
                    help="display heatmap broken down by agg1 x agg2",
                )
                if agg_key_2:
                    df2 = pd.DataFrame.copy(loss.loss_models_df, deep=True)
                    df2 = pd.pivot_table(
                        df2,
                        values="expected_loss",
                        index=agg_key_2,
                        columns=agg_key_1,
                        aggfunc=sum,
                    )
                    df2 = df2 * 1.0 / normalization
                    fig = px.imshow(df2)
                    st.plotly_chart(fig, theme=None)

        st.header("Loss frequency details")
        df = loss._loss_df
        df = df.reset_index()
        df["freq*mean"] = df["freq"] * df["mean"]
        df = df.append(df.sum(numeric_only=True), ignore_index=True)
        st.dataframe(df)

if state.module == 5:
    if hazard_missing:
        st.warning("Please select a hazard")

    units = st.button("use units")
    if not units:
        pushover_fig = compare.normalized_pushover_figs
        ida_fig = compare.normalized_ida_figs
        rate_fig = compare.rate_figs
        risk_fig = compare.risk_figs
    else:
        pushover_fig = compare.pushover_figs
        ida_fig = compare.ida_figs
        rate_fig = compare.rate_figs
        risk_fig = compare.risk_figs

    st.subheader("Pushover")
    st.plotly_chart(pushover_fig, theme=None)
    st.subheader("IDA")
    st.plotly_chart(ida_fig, theme=None)
    st.subheader("Rate of exceedance of $")
    st.plotly_chart(rate_fig, theme=None)
    st.subheader("Risk $")
    st.plotly_chart(risk_fig, theme=None)

    print_figs = st.button("print figures")
    if print_figs:
        pushover_fig.write_image(FIGURES_PATH / "pushovers.png", engine="kaleido")
        ida_fig.write_image(FIGURES_PATH / "idas.png", engine="kaleido")
        rate_fig.write_image(FIGURES_PATH / "roels.png", engine="kaleido")
        risk_fig.write_image(FIGURES_PATH / "risks.png", engine="kaleido")

    df = compare.summary_df
    st.subheader("Summary")
    sdf = df
    # st.text(df.columns)
    desired_columns = [
        "design name",
        "AAL %",
        "AAL $",
        "cs [1]",
        "ductility",
        "net worth [$]",
        "elements net worth [$]",
    ]
    sdf = sdf[desired_columns]
    st.dataframe(sdf)

    for ix, comp in enumerate(compare.comparisons):
        design = comp._design_model
        with st.expander(label=str(ix), expanded=True):
            if not comp.summary.get("design name"):
                st.warning("Could not find design.")
            if not comp.strana_abspath:
                st.warning("No IDA")
            if not comp.loss_abspath:
                st.warning("No loss")
            st.header(comp.summary.get("design name"))
            col1, col2, col3, col4 = st.columns(4)
            initial_cost = comp.summary.get("net worth $")
            aal = comp.summary.get("AAL $")
            if aal is not None and initial_cost is not None:
                pvl = (1 + secondary_loss) * aal / discount_factor
                risk = initial_cost + pvl
                col1.metric("Initial cost $", f"{initial_cost:.1f}")
                col2.metric("AAL $", f"{aal:.1f}")
                col3.metric("PVL $", f"{pvl:.1f}")
                col4.metric("Risk $", f"{risk:.1f}")
                st.dataframe(comp.summary_df)

    st.header("Compare point values")
    stat = st.selectbox("select", options=sdf.columns, index=2)
    if stat:
        fig = px.bar(sdf, x="design name", y=stat)
        st.plotly_chart(fig, theme=None)
