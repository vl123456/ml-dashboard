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

from explainerdashboard import ExplainerDashboard
from explainerdashboard.custom import *


class CustomModelTab(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Model Summary")
        self.precision = PrecisionComponent(explainer,
                                title='Precision',
                                hide_subtitle=True, hide_footer=True,
                                hide_selector=True,
                                cutoff=None)
        self.shap_summary = ShapSummaryComponent(explainer,
                                title='Impact',
                                hide_subtitle=True, hide_selector=True,
                                hide_depth=True, depth=8,
                                hide_cats=True, cats=True)
        self.shap_dependence = ShapDependenceComponent(explainer,
                                title='Dependence',
                                hide_subtitle=True, hide_selector=True,
                                hide_cats=True, cats=True,
                                hide_index=True,
                                col='Fare', color_col="PassengerClass")
        self.connector = ShapSummaryDependenceConnector(
                self.shap_summary, self.shap_dependence)

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                     html.H3("Model Performance"),
                    html.Div("As you can see on the right, the model performs quite well."),
                    html.Div("The higher the predicted probability of survival predicted by "
                            "the model on the basis of learning from examples in the training set"
                            ", the higher is the actual percentage of passengers surviving in "
                            "the test set"),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.precision.layout()
                ], style=dict(margin=30))
            ]),
            dbc.Row([
                dbc.Col([
                    self.shap_summary.layout()
                ], style=dict(margin=30)),
                dbc.Col([
                    html.H3("Feature Importances"),
                    html.Div("On the left you can check out for yourself which parameters were the most important."),
                    html.Div(f"Clearly {self.explainer.columns_ranked_by_shap()[0]} was the most important"
                            f", followed by {self.explainer.columns_ranked_by_shap()[1]}"
                            f" and {self.explainer.columns_ranked_by_shap()[2]}."),
                    html.Div("If you select 'detailed' you can see the impact of that variable on "
                            "each individual prediction. With 'aggregate' you see the average impact size "
                            "of that variable on the final prediction."),
                    html.Div("With the detailed view you can clearly see that the the large impact from Sex "
                            "stems both from males having a much lower chance of survival and females a much "
                            "higher chance.")
                ], width=4, style=dict(margin=30)),
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Feature dependence"),
                    html.Div("In the plot to the right you can see that the higher the cost "
                            "of the fare that passengers paid, the higher the chance of survival. "
                            "Probably the people with more expensive tickets were in higher up cabins, "
                            "and were more likely to make it to a lifeboat."),
                    html.Div("When you color the impacts by PassengerClass, you can clearly see that "
                            "the more expensive tickets were mostly 1st class, and the cheaper tickets "
                            "mostly 3rd class."),
                    html.Div("On the right you can check out for yourself how different features impacted "
                            "the model output."),
                ], width=4, style=dict(margin=30)),
                dbc.Col([
                    self.shap_dependence.layout()
                ], style=dict(margin=30)),
            ])
        ])

class CustomPredictionsTab(ExplainerComponent):
    def __init__(self, explainer):
        super().__init__(explainer, title="Predictions")

        self.index = ClassifierRandomIndexComponent(explainer,
                                                    hide_title=True, hide_index=False,
                                                    hide_slider=True, hide_labels=True,
                                                    hide_pred_or_perc=True,
                                                    hide_selector=True, hide_button=False)

        self.contributions = ShapContributionsGraphComponent(explainer,
                                                            hide_title=True, hide_index=True,
                                                            hide_depth=True, hide_sort=True,
                                                            hide_orientation=True, hide_cats=True,
                                                            hide_selector=True,
                                                            sort='importance')

        self.trees = DecisionTreesComponent(explainer,
                                            hide_title=True, hide_index=True,
                                            hide_highlight=True, hide_selector=True)


        self.connector = IndexConnector(self.index, [self.contributions, self.trees])

        self.register_components()

    def layout(self):
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H3("Enter name:"),
                    self.index.layout()
                ])
            ]),
            dbc.Row([
                dbc.Col([
                    html.H3("Contributions to prediction:"),
                    self.contributions.layout()
                ]),

            ]),
            dbc.Row([

                dbc.Col([
                    html.H3("Every tree in the Random Forest:"),
                    self.trees.layout()
                ]),
            ])
        ])

ExplainerDashboard(explainer, [CustomModelTab, CustomPredictionsTab], 
                        title='Titanic Explainer', header_hide_selector=True,
                        bootstrap=FLATLY).run()

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
