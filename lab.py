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
from app.utils import DESIGN_DIR, HAZARD_DIR, RESULTS_DIR, find_files, COMPARE_DIR
from app.occupancy import BuildingOccupancy
from app.compare import IDACompare
import pandas as pd
import numpy as np
import time
from st_cytoscape import cytoscape
import plotly.express as px
import random
from streamlit import session_state as state
from functools import partial


st.set_page_config(
    page_title="vulkan",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULT_MODULE = 5
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
    b3.button("str", on_click=partial(goto_module, 3))
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
        if file:
            state.design_abspath = DESIGN_DIR / file
            design = ReinforcedConcreteFrame.from_file(state.design_abspath)
        else:
            design = ReinforcedConcreteFrame(name=name)

        num_frames = st.number_input(
            "num frames",
            min_value=1,
            step=1,
            max_value=50,
            value=design.num_frames,
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
            value=",".join([str(s) for s in design.storeys]) or "3,",
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
        if record_name and not state.first_render:
            record_path = str((RECORDS_DIR / record_name).resolve())
            record = Record(record_path)
            hazard.add_record(record)
            hazard.to_file(HAZARD_DIR)
        left, middle, right = st.columns(3)
        go = left.button("sample 5", help="grab 5 at random")
        if go:
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

            state.first_render = False

    if state.module == 3:
        state.design_abspath
        state.hazard_abspath

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
        "file:", file
        try:
            if file:
                ida = IDA.from_file(STRANA_DIR / file)
                state.ida_abspath = STRANA_DIR / file
                ida.design_abspath = str(state.design_abspath) or ida.design_abspath
                ida.hazard_abspath = str(state.hazard_abspath) or ida.hazard_abspath
            else:
                ida = IDA(
                    name="default ida",
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
            "design:", state.design_abspath
            "hazard:", state.hazard_abspath
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
            delete = st.button("🗑️", help="delete this analysis")
            if delete:
                with st.spinner("deleting analysis"):
                    time.sleep(2)
                    ida.delete(STRANA_DIR)
                    ida = None
                    state.ida_abspath = None
                    st.success("delete successful")

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

            if ida.results:
                for ix, r in enumerate(ida.results):
                    with st.container():
                        c1, c2 = st.columns([5, 1])
                        c1.write(f'{r["record"]} - {r["intensity"]:.4f}g')
                        view_result = c2.button("view", key=f"record{ix}")
                        if view_result:
                            selected_ix = ix

    if state.module == 4:
        state.ida_abspath
        losses = find_files(LOSS_MODELS_DIR)
        ida_missing = False
        file = st.selectbox("select a loss file", options=losses)
        name = st.text_input(
            "give it a name",
            value=file.split(".")[0] if file else "default loss",
            help="to save just run",
        )
        loss = None
        try:
            if file:
                loss = LossAggregator.from_file(LOSS_MODELS_DIR / file)
                state.loss_abspath = LOSS_MODELS_DIR / file
                loss.ida_model_path = state.ida_abspath
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
            st.success("success")

        rm = right.button("🗑️", help="delete")
        if rm:
            with st.spinner("deleting..."):
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

        assets = loss.to_dict["loss_models"] or []
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
        selected_names = st.multiselect("Name", options=all_names, default=all_names)

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

    if state.module == 5:
        "hazard abspath: ", state.hazard_abspath
        compares = find_files(COMPARE_DIR)
        compare_file = st.selectbox("select a compare file", options=compares)
        name = st.text_input(
            "give it a name",
            value=compare_file.split(".")[0] if compare_file else "default compare",
            help="to save just run",
        )
        hazard_abspath = str(state.hazard_abspath)
        hazard_missing = False
        try:
            if compare_file:
                compare = IDACompare.from_file(COMPARE_DIR / compare_file)
            else:
                compare = IDACompare(name=name, hazard_abspath=hazard_abspath)
        except HazardNotFoundException as e:
            print(e)
            hazard_missing = True

        compare.name = name
        compare.hazard_abspath = hazard_abspath
        hazard = None
        hazards = find_files(HAZARD_DIR)
        hazard_file = st.selectbox("select a hazard", options=hazards)
        if hazard_file:
            state.hazard_abspath = HAZARD_DIR / hazard_file
            hazard = Hazard.from_file(HAZARD_DIR / hazard_file)

        "num records:", len(hazard.records)
        # not sure why this floats like crazy below every element
        # left, right = st.columns(2)
        # logx = left.checkbox("log x", value=True)
        # logy = right.checkbox("log y", value=True)
        # if hazard:
        #     hazard_fig = hazard.rate_figure(logx=logx, logy=logy)
        # st.plotly_chart(hazard_fig, use_container_width=True)

        design_files = find_files(DESIGN_DIR, only_yml=True)
        design_name = st.selectbox("add a design", options=design_files)
        if design_name and not state.first_render:
            design_path = str((DESIGN_DIR / design_name).resolve())
            compare.add_design(design_path)
            compare.to_file(COMPARE_DIR)

        for ix, comp in enumerate(compare.comparisons):
            _c1, _c2, _c3, _c4, _c5 = st.columns([2, 1, 1, 1, 1])
            with st.container():
                _c1.write(comp.design_abspath.split("/")[-1])
                go_strana = _c2.button(
                    "ida", help="perform ida", key=f"compare-design-{ix}"
                )
                go_loss = _c3.button(
                    "loss", help="perform loss computation", key=f"compare-design-{ix}"
                )
                go_complete = _c4.button(
                    "all", help="perform ida then loss", key=f"compare-design-{ix}"
                )
                rm = _c5.button("🗑️", key=f"compare-design-{ix}")
                if go_strana:
                    comp.run(strana=True)
                if go_loss:
                    comp.run(loss=True)
                if go_complete:
                    comp.run(strana=True, loss=True)
                if rm:
                    compare.remove_design(comp.design_abspath)
                if any([go_strana, go_loss, go_complete, rm]):
                    compare.to_file(COMPARE_DIR)
                    st.success("success")

        c1, c2, c3, c4 = st.columns(4)
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
    if design and design.fems:
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

        with st.expander("summary"):
            st.dataframe(pd.DataFrame([design.summary]))

        with st.expander("assets"):
            pass

        with st.expander("eigen"):
            eigen, storeys = design.fem.eigen_df
            "modes"
            st.dataframe(eigen)
            "storeys"
            st.dataframe(storeys)

        with st.expander("capacity"):
            # path = DESIGN_DIR / design.name
            # fig, nfig = design.fem.pushover_figs(path)
            fig, nfig = design.fem.pushover_figs
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig)
            col2.plotly_chart(nfig)
            design_error = design.fem.pushover_stats["design_error"]
            c_design = design.fem.pushover_stats["c_design"]
            st.metric(label="Design Cs", value=c_design, delta=design_error)
            df = pd.DataFrame(design.fem.pushover_stats, index=[0])
            st.table(df)
            st.dataframe(design.fem.extras)


if state.module == 2:
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

if state.module == 3:
    if design_missing:
        st.warning("Please select a design")
    if hazard_missing:
        st.warning("Please select a hazard")
    if not (design_missing or hazard_missing) and ida and ida.results:
        fig = ida.view_ida_curves()
        # todo@carlo width 85% parent container
        # fig.update_layout(width=640 * 3, height=640,)
        st.plotly_chart(fig, container_width=True)
        st.dataframe(pd.DataFrame.from_records(ida.stats), height=800)
        if selected_ix is not None:
            filepath = ida.results[selected_ix]["path"]
            view = StructuralResultView.from_file(filepath)
            figures = view.timehistory_figures
            for fig in figures:
                st.plotly_chart(fig)
    else:
        st.header("run analysis to see results")

if state.module == 4:
    if ida_missing:
        st.warning("Please select an analysis")
    normalization = 1.0
    left, right = st.columns(2)
    normalize = left.checkbox("Normalize")
    # WIP normalization
    asset = loss
    # if not view_all or selected_ix is None:
    #     st.header("please run loss")
    if view_asset or selected_ix is not None:
        model_dict = asset.loss_models[selected_ix]
        asset = LossModel(**model_dict, _ida_results_df=asset._ida_results_df)

    if normalize:
        normalization = asset.net_worth

    normalize_wrt_building = right.checkbox("Normalize to building cost")

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
    if average_annual_loss:
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
            st.plotly_chart(fig)
            fig = asset.expected_loss_and_variance_fig(normalization=normalization)
            st.plotly_chart(fig)
            if selected_ix is None:
                fig = asset.scatter_fig(
                    category_filter=selected_categories,
                    name_filter=selected_names,
                    floor_filter=selected_floors,
                )
                st.plotly_chart(fig)
        elif agg_key_1 and selected_ix is None:
            df = loss.loss_models_df
            df = loss.filter_src_df(
                df,
                category_filter=selected_categories,
                name_filter=selected_names,
                storey_filter=selected_floors,
            )
            fig = loss.multiple_rates_of_exceedance_fig(df, key=agg_key_1)
            st.plotly_chart(fig)

            df = loss.aggregate_src_df(df, key=agg_key_1)
            df = df * 1.0 / normalization
            fig = loss.aggregated_expected_loss_and_variance_fig(df)
            st.plotly_chart(fig)

            fig = asset.scatter_fig(
                category_filter=selected_categories,
                name_filter=selected_names,
                floor_filter=selected_floors,
            )
            st.plotly_chart(fig)
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
                st.plotly_chart(fig)

if state.module == 5:
    if hazard_missing:
        st.warning("Please select a hazard")

    units = st.button("use units")
    if not units:
        norm_pushover_figs = compare.normalized_pushover_figs
        st.plotly_chart(norm_pushover_figs)

        norm_ida_figs = compare.normalized_ida_figs
        st.plotly_chart(norm_ida_figs)

        df = compare.summary_df
        st.dataframe(df)
    else:
        pushover_fig = compare.pushover_figs
        st.plotly_chart(pushover_fig)

        ida_fig = compare.ida_figs
        st.plotly_chart(ida_fig)

        df = compare.summary_df
        st.dataframe(df)

    stat = st.selectbox("select", options=df.columns)
    if stat:
        fig = df[stat].plot()
        st.plotly_chart(fig)

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
            col1.metric("Net worth $", comp.summary.get("net worth $"))
            col2.metric("period s", comp.summary.get("period [s]"))
            st.dataframe(comp.summary_df)
