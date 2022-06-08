import streamlit as st
from app.design import ReinforcedConcreteFrame
from app.utils import find_files
from pathlib import Path
from st_cytoscape import cytoscape
import time
import shutil
from app.utils import ROOT_DIR, DESIGN_DIR, MODELS_DIR
import pandas as pd


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

if "module" not in st.session_state:
    st.session_state.module = 1


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

layout = {"name": "preset"}  # this layout respects nodes' (x,y)
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
        storeys_input = st.text_input(
            "storeys",
            value=",".join([str(s) for s in design.storeys]) or "3",
            help="heights in meters separated by comma",
        )
        storeys = [float(s) for s in storeys_input.split(",")]
        bays_input = st.text_input(
            "bays",
            value=",".join([str(s) for s in design.bays]) or "5",
            help="widths in meters separated by comma",
        )
        bays = [float(s) for s in bays_input.split(",")]

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
            print("design")
            with st.spinner("running design"):
                design = ReinforcedConcreteFrame(
                    name=name,
                    storeys=storeys,
                    bays=bays,
                )
                time.sleep(2)
                design.force_design(DESIGN_DIR, pushover=True)
                design.fem.pushover_abs_path
                design.to_file(DESIGN_DIR)
            st.success("design successful")

        elements, stylesheet = design.fem.cytoscape()


if st.session_state.module == 1:
    print("selected")
    selected = cytoscape(
        elements=elements,
        stylesheet=stylesheet,
        # stylesheet={},
        layout=layout,
        user_panning_enabled=True,
        max_zoom=1,
        min_zoom=0.1,
        key="fem",
    )

    st.metric("Net worth", f"$ {design.fem.total_net_worth:.0f} k USD")

    with st.expander("costs"):
        col1, col2, col3 = st.columns(3)
        col2.metric("Structural", f"$ {design.fem.elements_net_worth:.0f} k ")
        col1.metric("Nonstructural", f"$ {design.fem.nonstructural_net_worth:.0f} k ")
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
