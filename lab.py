import streamlit as st
from app.assets import LOSS_MODELS_DIR
from app.criteria import DesignCriterionFactory
from app.design import ReinforcedConcreteFrame
from app.hazard import RECORDS_DIR, Hazard, HazardCurveFactory, Record
from app.strana import (
    STRANA_DIR,
    IDA,
    HazardNotFoundException,
    SpecNotFoundException,
    StructuralResultView,
)
from app.loss import LossAggregator, IDANotFoundException, LossModel
from app.utils import DESIGN_DIR, MODELS_DIR, HAZARD_DIR, RESULTS_DIR, find_files
from app.occupancy import BuildingOccupancy
import pandas as pd
import numpy as np
import time
from st_cytoscape import cytoscape
import random


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


padding_top = 2
DEFAULT_MODULE = 3

st.markdown(
    f"""
    <style>
        .appview-container .main .block-container{{
            padding-top: {padding_top}rem;
        }}
    </style>""",
    unsafe_allow_html=True,
)  # remove unnecessary padding at top

if "module" not in st.session_state:
    st.session_state.module = DEFAULT_MODULE
    st.session_state.design_abspath = ""
    st.session_state.hazard_abspath = ""
    st.session_state.ida_abspath = ""
    st.session_state.loss_abspath = ""

if "first_render" not in st.session_state:
    st.session_state.first_render = True


def switch_module(delta: int):
    st.session_state.module = st.session_state.module + delta


title = (
    "design",
    "hazard",
    "structural analysis",
    "loss",
)

left, right = st.columns(2)
if st.session_state.module != 1:
    prev = left.button("back", on_click=switch_module, args=[-1])
if st.session_state.module != 4:
    next = right.button("next", on_click=switch_module, args=[1])

with st.sidebar:
    st.header(title[st.session_state.module - 1])
    if st.session_state.module == 1:
        buildings = find_files(DESIGN_DIR)
        design = ReinforcedConcreteFrame(
            name="default",
            storeys=[3.5, 3.0],
            bays=[5.0],
        )
        file = st.selectbox("select a building", options=buildings)
        if file:
            st.session_state.design_abspath = DESIGN_DIR / file
            design = ReinforcedConcreteFrame.from_file(DESIGN_DIR / file)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "",
        )
        num_frames = st.number_input(
            "num frames",
            min_value=1,
            step=1,
            max_value=100,
            format="%d",
            help="number of identical perpendicular frames",
        )
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
            value=",".join([str(s) for s in design.storeys]) or "3",
            help="heights in meters separated by comma",
        )
        storeys = [float(s) for s in storeys_input.split(",")]
        np.array(storeys).cumsum().tolist()
        cumstoreys = "sum: " + ",".join(
            [str(b) for b in np.array(storeys).cumsum().tolist()]
        )
        cumstoreys
        bays_input = st.text_input(
            "bays",
            value=",".join([str(s) for s in design.bays]) or "5",
            help="widths in meters separated by comma",
        )
        bays = [float(s) for s in bays_input.split(",")]
        cumbays = "sum: " + ",".join([str(b) for b in np.array(bays).cumsum().tolist()])
        cumbays
        occupancy = st.selectbox(
            "occupancy class",
            BuildingOccupancy.options(),
            index=BuildingOccupancy.options().index(
                design.occupancy or BuildingOccupancy.DEFAULT
            ),
        )

        design_criteria = st.selectbox(
            "design criteria",
            DesignCriterionFactory.options(),
            index=DesignCriterionFactory.options().index(
                design.design_criteria[0] or DesignCriterionFactory.DEFAULT
            )
            # this [0] is bad design. the idea was to have multiple criteria, this is too complex in the ui. rather we must have the criteria combined in code.
        )

        left, right = st.columns(2)
        delete = left.button("🗑️", help="delete this building")
        if delete:
            with st.spinner("deleting building"):
                time.sleep(2)
                design.delete(DESIGN_DIR)
                design = None
                st.success("design successful")

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
                design.force_design(DESIGN_DIR, pushover=True)
                design.to_file(DESIGN_DIR)
            st.success("design successful")

        if design:
            elements, stylesheet = design.fem.cytoscape()

    if st.session_state.module == 2:
        hazards = find_files(HAZARD_DIR)
        hazard = Hazard(
            name="sample_hazard",
        )
        file = st.selectbox("select a hazard", options=hazards)
        if file:
            st.session_state.hazard_abspath = HAZARD_DIR / file
            hazard = Hazard.from_file(HAZARD_DIR / file)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "",
            help="to save just add or remove records",
        )
        hazard.name = name
        hazard._curve.html(st)
        curve_type = st.selectbox(
            "select a curve type",
            options=HazardCurveFactory.options(),
            index=HazardCurveFactory.options().index(hazard._curve.name),
        )
        st.subheader(f"Records ({len(hazard.records)})")
        record_files = find_files(RECORDS_DIR, only_yml=False, only_csv=True)
        record_name = st.selectbox("add a record", options=record_files)
        if record_name and not st.session_state.first_render:
            record_path = str((RECORDS_DIR / record_name).resolve())
            record = Record(record_path)
            hazard.add_record(record)
            hazard.to_file(HAZARD_DIR)
        left, middle, right = st.columns(3)
        sample = left.button("sample 5", help="grab 5 at random")
        if sample:
            with st.spinner("sampling..."):
                time.sleep(1)
                untouched = [r for r in record_files if r not in hazard.record_names]
                samples = random.sample(untouched, 5)
                for path in samples:
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
                rm = c3.button("🗑️", key=f"record{ix}")
                if rm:
                    rec = hazard.records[ix]
                    record_path = str((RECORDS_DIR / rec.name).resolve())
                    hazard.remove_record(record_path)
                    hazard.to_file(HAZARD_DIR)
                    st.success("record removed")

            st.session_state.first_render = False

    if st.session_state.module == 3:
        stranas = find_files(STRANA_DIR)
        design_missing = False
        hazard_missing = False
        selected_ix = None
        file = st.selectbox("select an analysis", options=stranas)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "default ida",
            help="to save just run",
        )
        try:
            ida = IDA(
                name="default ida",
                design_abspath=str(st.session_state.design_abspath),
                hazard_abspath=str(st.session_state.hazard_abspath),
            )
            ida.name = name
        except HazardNotFoundException:
            ida = IDA(name="default_ida", design_abspath=None, hazard_abspath=None)
            hazard_missing = True
        except SpecNotFoundException:
            ida = IDA(name="default_ida", design_abspath=None, hazard_abspath=None)
            design_missing = True

        if file:
            ida = IDA.from_file(STRANA_DIR / file)
            st.session_state.ida_abspath = STRANA_DIR / file

        if file or not (design_missing or hazard_missing):
            start = st.number_input(
                r"start Sa (g)",
                value=ida.start,
                min_value=0.0,
                step=ida.step,
                max_value=10.0,
                format="%g",
                help="starting Sa (g)",
            )
            stop = st.number_input(
                r"stop Sa (g)",
                value=ida.stop,
                min_value=0.0,
                step=ida.step,
                max_value=10.0,
                format="%g",
                help="stop Sa (g)",
            )
            step = st.number_input(
                r"step Sa (g)",
                value=ida.step,
                min_value=0.005,
                step=0.1,
                max_value=10.0,
                format="%g",
                help="step Sa (g)",
            )
            run = st.button("run IDA", help="run with chosen Sa")
            standard = st.button(
                "run standard", help="means that Sa are chosen for you"
            )
            if run or standard:
                with st.spinner("running..."):
                    time.sleep(1)
                    ida = IDA(
                        **{
                            **ida.to_dict,
                            "start": start,
                            "stop": stop,
                            "step": step,
                            "name": name,
                            "standard": standard,
                        },
                    )
                    ida.run_parallel(results_dir=RESULTS_DIR)
                    ida.to_file(STRANA_DIR)

                st.success("analysis successful")

            for ix, r in enumerate(ida.results):
                with st.container():
                    c1, c2 = st.columns([5, 1])
                    c1.write(f'{r["record"]} - {r["intensity"]:.4f}g')
                    view_result = c2.button("view", key=f"record{ix}")
                    if view_result:
                        selected_ix = ix

    if st.session_state.module == 4:
        losses = find_files(LOSS_MODELS_DIR)
        ida_missing = False
        file = st.selectbox("select a loss file", options=losses)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "default loss",
            help="to save just run",
        )
        try:
            loss = LossAggregator(
                name=name, ida_model_path=(st.session_state.ida_abspath)
            )
            loss.name = name
        except IDANotFoundException:
            loss = LossAggregator(name=name)
            ida_missing = True

        if file:
            loss = LossAggregator.from_file(LOSS_MODELS_DIR / file)
            st.session_state.loss_abspath = LOSS_MODELS_DIR / file

        if loss:
            left, right = st.columns(2)
            sample = left.button("run", help="perform loss computation")
            if sample:
                with st.spinner("running..."):
                    loss = LossAggregator(
                        **{
                            **loss.to_dict,
                            "name": name,
                            "ida_model_path": str(st.session_state.ida_abspath),
                        },
                    )
                    loss.run()
                    loss.to_file(LOSS_MODELS_DIR)
                st.success("success")

            rm = right.button("🗑️", help="delete")
            if rm:
                with st.spinner("removing..."):
                    time.sleep(1)
                    loss.delete(LOSS_MODELS_DIR)
                    loss = None
                st.success("delete successful")
            design = loss._ida._design
            st.header("Design")
            st.text(f"{design.name}")
            st.text(f"St {design.num_storeys} bays {design.num_bays}")
            st.text(f"{design.design_criteria}")
            st.text(f"{design.occupancy.split('.')[0]}")
            st.metric("$ Net worth", 3414)

            assets = loss.to_dict["loss_models"]
            filtered_assets = assets
            fig = design.fem.assets_pie_fig

            fig.update_layout(height=300, width=300)
            st.plotly_chart(fig)
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
            for ix, a in enumerate(filtered_assets):
                with st.container():
                    c1, c2 = st.columns([5, 1])
                    c1.write(f'{a["name"]}-{a["category"]}-{a["floor"]}')
                    view_asset = c2.button("view", key=f"asset{ix}")
                    if view_asset:
                        selected_ix = ix

if st.session_state.module == 1:
    if design:
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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Net worth", f"$ {design.fem.total_net_worth:.0f} k USD")
        col2.metric("fundamental period", f"{design.fem.periods[0]:.2f} s")
        col3.metric("height", f"{design.height:.1f} m")
        col4.metric("width", f"{design.width:.1f} m")

        with st.expander("costs"):
            col1, col2, col3 = st.columns(3)
            col2.metric("Structural", f"$ {design.fem.elements_net_worth:.0f} k ")
            col1.metric(
                "Nonstructural", f"$ {design.fem.nonstructural_net_worth:.0f} k "
            )
            col3.metric("Contents", f"$ {design.fem.contents_net_worth:.0f} k ")
            fig = design.fem.assets_pie_fig
            st.plotly_chart(fig)

        with st.expander("assets"):
            pass

        with st.expander("eigen"):
            st.dataframe(design.fem.eigen_df)

        with st.expander("capacity"):
            path = DESIGN_DIR / design.name
            fig, nfig = design.fem.pushover_figs(path)
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig)
            col2.plotly_chart(nfig)
            df = pd.DataFrame(design.fem.pushover_stats, index=[0])
            st.table(df)


if st.session_state.module == 2:
    left, right = st.columns(2)
    logx = left.checkbox("log x", value=True)
    logy = right.checkbox("log y", value=True)
    if hazard:
        fig = hazard.rate_figure(logx=logx, logy=logy)
        st.plotly_chart(fig)
        if selected_ix is not None:
            record = hazard.records[selected_ix]
            st.plotly_chart(record.figure)
            st.plotly_chart(record.spectra)
        elif len(hazard.records) > 0:
            record = hazard.records[0]
            st.plotly_chart(record.figure)
            st.plotly_chart(record.spectra)


if st.session_state.module == 3:
    if design_missing:
        st.warning("Please select a design")
    if hazard_missing:
        st.warning("Please select a hazard")
    if ida:
        fig = ida.view_ida_curves()
        # todo@carlo width 85% parent container
        fig.update_layout(width=1000)
        st.plotly_chart(fig)
        if selected_ix is not None:
            filepath = ida.results[selected_ix]["path"]
            view = StructuralResultView.from_file(filepath)
            figures = view.timehistory_figures
            for fig in figures:
                st.plotly_chart(fig)

if st.session_state.module == 4:
    if ida_missing:
        st.warning("Please select a design")
    normalization = 1.0
    left, right = st.columns(2)
    normalize = left.checkbox("Normalize")
    # WIP normalization
    asset = loss
    if view_all or selected_ix is None:
        "success"
    elif view_asset or selected_ix is not None:
        model_dict = asset.loss_models[selected_ix]
        asset = LossModel(**model_dict, _ida_results_df=asset._ida_results_df)

    if normalize:
        st.write("Great!")
        normalization = asset.net_worth

    normalize_wrt_building = right.checkbox("Normalize to building cost")

    if normalize_wrt_building:
        st.write("Great!")
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
    one, two, three, four = st.columns(4)
    one.metric("AAL", f"{average_annual_loss:.4f}")
    two.metric("AAL %", f"{average_annual_loss_pct:.3f}")
    three.metric("E[L]", f"{expected_loss:.4f}")
    four.metric("E[L] %", f"{expected_loss_pct:.3f}")
    st.header("Building" if view_all or selected_ix is None else asset.name)
    if asset:
        fig = asset.rate_fig(normalization=normalization)
        st.plotly_chart(fig)
        fig = asset.expected_loss_and_variance_fig(normalization=normalization)
        st.plotly_chart(fig)
        fig = asset.scatter_fig(
            category_filter=selected_categories,
            name_filter=selected_names,
            floor_filter=selected_floors,
        )
        st.plotly_chart(fig)
