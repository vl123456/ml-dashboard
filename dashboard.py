import multiprocessing
import os
import sys
from contextlib import contextmanager

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, \
    RegressionExplainer

count_cpu = multiprocessing.cpu_count()

b = False


@contextmanager
def silence_stdout():
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


def launch_dashboard_api(
        x, y, estimator, problem,
        experiment_name, processing_pipeline=None, port=8050, host="0.0.0.0"):
    if len(x) > 1000:
        print(
            "Warning : you can not run shap analysis witn test > 1000 samples")
        idx = np.random.choice(range(len(x)), size=100)
        x = x.iloc[idx]
        y = y.iloc[idx]
    if processing_pipeline is None:
        x_proj = x
    else:
        try:
            if "feature_preproc" in str(
                    processing_pipeline[-1]) and "passthrough" not in str(
                processing_pipeline[-1].choice.preprocessor):
                x_col = x.columns[
                    processing_pipeline[-1].choice.preprocessor.get_support()]
            else:
                x_col = x.columns

            if b:
                # FIXME : everything is here to extract feature name
                #  but we need to map things correctly
                dp = processing_pipeline.steps[0][1]
                ohe = dp.column_transformer.transformers[0][1].steps[-1][1]
                prep = ohe.choice.get_preprocessor()
                x_col = prep.get_feature_names(x.columns)

            x_proj = pd.DataFrame(processing_pipeline.transform(x.to_numpy()),
                                  index=x.index)

        except (ValueError, AttributeError):
            x_col = range(
                processing_pipeline.transform(x.to_numpy()).shape[1])
            x_proj = pd.DataFrame(processing_pipeline.transform(x.to_numpy()),
                                  index=x.index,
                                  columns=x_col)

    if problem == "classification":
        explainer = ClassifierExplainer(
            estimator, x_proj
            , y)
    elif problem == "regression":
        explainer = RegressionExplainer(
            estimator,
            x_proj, y

        )
    else:
        raise ("Problem %s unknown" % problem)
    args_tab = dict(
        importances=True,
        model_summary=True,
        contributions=True,
        whatif=False,
        shap_dependence=False,
        shap_interaction=False,
        decision_trees=False)
    args_component = dict(
        # TODO : classification stats tab:
        hide_precision=True,
        hide_classification=True,
        hide_prauc=True,
        hide_liftcurve=True,
        hide_rocauc=False,
        hide_cumprecision=True,
        # TODO  regression stats tab:
        # TODO  hide_modelsummary=True,
        hide_predsvsactual=True, hide_residuals=True,
        hide_regvscol=True,
        # individual predictions tab:
        hide_predindexselector=True,
        hide_predictionsummary=True,
        hide_contributiongraph=False,
        hide_pdp=True,
        # TODO  shap interactions tab:
        hide_interactionsummary=True, hide_interactiondependence=True,
        # decisiontrees tab:
        hide_treeindexselector=True, hide_treesgraph=True,
        hide_treepathgraph=True, )
    args_opt = dict(no_permutations=True)
    with silence_stdout():
        db = ExplainerDashboard(
            explainer, title=experiment_name,
            **args_tab, **args_component,
            **args_opt,
            bootstrap=dbc.themes.CERULEAN,
            n_jobs=int(0.8 * count_cpu),

        )
    db.run(port=port, mode="external", host=host)


def launch_dashboard_api_from_expe(model, experiment_name: str):
    launch_dashboard_api(
        x=model.data_instance.X.iloc[model.data_instance.test_idx],
        y=model.data_instance.y.iloc[model.data_instance.test_idx],
        estimator=model.final_pipeline.steps[-1][1].estimator,
        processing_pipeline=model.processing_pipeline,
        problem=model.data_instance.problem,
        experiment_name=experiment_name
    )
