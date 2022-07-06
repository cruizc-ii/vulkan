import streamlit as st
from app.criteria import DesignCriterion, DesignCriterionFactory
from app.design import ReinforcedConcreteFrame
from app.hazard import RECORDS_DIR, Hazard, HazardCurveFactory, Record
from app.utils import find_files
from pathlib import Path
from st_cytoscape import cytoscape
import time
import shutil
from app.utils import ROOT_DIR, DESIGN_DIR, MODELS_DIR, HAZARD_DIR
import pandas as pd
import numpy as np
from app.occupancy import BuildingOccupancy


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


padding_top = 2
DEFAULT_MODULE = 2

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
                    num_frames=num_frames,
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
            hazard = Hazard.from_file(HAZARD_DIR / file)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "",
            help="to save just add or remove records",
        )
        hazard.name = name
        hazard.curve.html(st)
        curve_type = st.selectbox(
            "select a curve type",
            options=HazardCurveFactory.options(),
            index=HazardCurveFactory.options().index(hazard.curve.name),
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
                import random

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
