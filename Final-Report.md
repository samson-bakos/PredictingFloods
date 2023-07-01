Predicting the Largest Floods in Canadian Rivers
================
Marian Agyby, Samson Bakos, Spencer Gerlach, Zilong Yi
2023-06-28

- <a href="#executive-summary" id="toc-executive-summary">Executive
  Summary</a>
- <a href="#list-of-acronyms" id="toc-list-of-acronyms">List of
  Acronyms</a>
- <a href="#introduction" id="toc-introduction">Introduction</a>
- <a href="#project-deliverables" id="toc-project-deliverables">Project
  Deliverables</a>
- <a href="#machine-learning-for-rffa"
  id="toc-machine-learning-for-rffa">Machine Learning for RFFA</a>
  - <a href="#data-cleaning-and-prep" id="toc-data-cleaning-and-prep">Data
    Cleaning and Prep</a>
  - <a href="#report-clustering" id="toc-report-clustering">Unsupervised
    Clustering</a>
  - <a href="#index-flood-method" id="toc-index-flood-method">Index Flood
    Method</a>
    - <a href="#background" id="toc-background">Background</a>
    - <a href="#ifm-selection" id="toc-ifm-selection">Index Flood Model: Model
      Selection and Justification</a>
    - <a href="#ifm-growth-curves" id="toc-ifm-growth-curves">Growth
      Curves</a>
    - <a href="#ifm-quantile-pred" id="toc-ifm-quantile-pred">River Flow
      Quantile Prediction</a>
  - <a href="#direct-quantile-method" id="toc-direct-quantile-method">Direct
    Quantile Method</a>
    - <a href="#background-1" id="toc-background-1">Background</a>
    - <a href="#model-selection" id="toc-model-selection">Model Selection</a>
    - <a href="#river-flow-quantile-prediction"
      id="toc-river-flow-quantile-prediction">River Flow Quantile
      Prediction</a>
- <a href="#limitations" id="toc-limitations">Limitations</a>
- <a href="#future-improvements" id="toc-future-improvements">Future
  Improvements</a>
- <a href="#conclusions" id="toc-conclusions">Conclusions</a>
- <a href="#appendices" id="toc-appendices">Appendices</a>
  - <a href="#appendix-ifm" id="toc-appendix-ifm">Appendix: Supporting
    Analysis for Index Flood Method</a>
    - <a href="#appendix-ifm-preprocessing"
      id="toc-appendix-ifm-preprocessing">IFM Data Preprocessing</a>
    - <a href="#appendix-ifm-eval-metric"
      id="toc-appendix-ifm-eval-metric">IFM Evaluation Metric Selection</a>
    - <a href="#example-model-predictions"
      id="toc-example-model-predictions">Example Model Predictions</a>
    - <a href="#additional-figures" id="toc-additional-figures">Additional
      Figures</a>
  - <a href="#appendix-sa-dqm" id="toc-appendix-sa-dqm">Appendix: Supporting
    Analysis for Direct Quantile Prediction</a>
    - <a href="#appendix-dqm-preprocessing"
      id="toc-appendix-dqm-preprocessing">DQM Data Preprocessing</a>
    - <a href="#appendix-coef-importance"
      id="toc-appendix-coef-importance">Coefficient importance</a>
    - <a href="#appendix-max-out" id="toc-appendix-max-out">“Max-out”
      Behavior</a>
    - <a href="#appendix-decreasing-portion"
      id="toc-appendix-decreasing-portion">Decreasing portion</a>
    - <a href="#prediction-concerns" id="toc-prediction-concerns">Prediction
      Concerns</a>
  - <a href="#appendix-choosing-ifm-or-dqm"
    id="toc-appendix-choosing-ifm-or-dqm">Appendix: Choosing IFM or DQM</a>
  - <a href="#appendix-clustering" id="toc-appendix-clustering">Appendix:
    Regional Homogeneity, Clustering and H-Index</a>
    - <a href="#cluster-interpretation"
      id="toc-cluster-interpretation">Cluster Interpretation</a>
    - <a href="#criterion-standard-deviation-of-residual-ratios"
      id="toc-criterion-standard-deviation-of-residual-ratios">Criterion:
      Standard Deviation of Residual Ratios</a>
    - <a href="#criterion-distribution-of-variance-ratios"
      id="toc-criterion-distribution-of-variance-ratios">Criterion:
      Distribution of Variance Ratios</a>
    - <a
      href="#criterion-physicalprincipal-component-analysis-pca-interpretability"
      id="toc-criterion-physicalprincipal-component-analysis-pca-interpretability">Criterion:
      Physical/Principal Component Analysis (PCA) Interpretability</a>
    - <a href="#criterion-heterogeneity-index-hosking-and-wallis-1997"
      id="toc-criterion-heterogeneity-index-hosking-and-wallis-1997">Criterion:
      Heterogeneity Index (Hosking and Wallis, 1997)</a>
    - <a href="#classifying-clusters"
      id="toc-classifying-clusters">Classifying Clusters</a>
    - <a href="#clustering-recommendations-and-limitations"
      id="toc-clustering-recommendations-and-limitations">Clustering
      Recommendations and Limitations</a>
  - <a href="#appendix-expanded-limitations-and-future-improvements"
    id="toc-appendix-expanded-limitations-and-future-improvements">Appendix:
    Expanded Limitations and Future Improvements</a>
    - <a href="#appendix-limitations" id="toc-appendix-limitations">Expanded
      Limitations</a>
    - <a href="#appendix-improvements" id="toc-appendix-improvements">Expanded
      Future Improvements</a>
- <a href="#references" id="toc-references">References</a>

# Executive Summary

This project focuses on the prediction of flood magnitudes in Canadian
rivers. While the flows of many rivers are actively monitored, many more
are not. Estimates of maximum flow in these unmonitored rivers are
necessary to guide engineering decisions in the surrounding areas.
Existing methods for flood magnitude prediction have not been updated to
keep pace with modern statistical techniques, and may provide less
reliable estimations of flood magnitude. As ML becomes more ubiquitous
and sophisticated, it is becoming clear that traditional RFFA methods
could benefit from integrating these modern ML and statistical
techniques. This project updates the existing methodology used by BGC,
outlined by the United States Geological Survey (USGS), using modernized
machine learning (ML) and applied statistics techniques.

The University of British Columbia Master of Data Science (UBC MDS, or
MDS) team pursued two main approaches to complete this objective: The
Index Flood Method (IFM) and Direct Quantile Method (DQM). The IFM
combined unsupervised (hierarchical) clustering, supervised ML
predictions of the index flood, and statistical distribution fitting to
predict high river flow quantiles. The DQM used supervised ML quantile
regression models to predict high flow quantiles directly. Both
approaches present statistical and empirical improvements over existing
methods.

The ML approaches have some limitations, such as the existence of an
overprediction bias at the highest quantiles for IFM and DQM, and the
need to modify the current clustering approach to increase the
reasonability of statistical assumptions for IFM. However, the MDS team
clearly documents the potential benefits of adopting the outlined
approach, as well as suggesting next steps to address limitations and
improve the two approaches.

This document provides clear proof-of-concept and justification for the
application of ML in the prediction of flood magnitudes. The MDS team
has provide pre-trained models that can be deployed with minor
adjustments, and the MDS team recommends the adoption of the outlined
methodology to improve the accuracy of BGC’s flood magnitude prediction.

# List of Acronyms

| Acronym |              Meaning              |
|:-------:|:---------------------------------:|
|   AEP   |   Annual Exceedance Probability   |
|   AMS   |       Annual Maxima Series        |
|   DQM   |      Direct Quantile Method       |
|   FFA   |     Flood Frequency Analysis      |
|   IFM   |        Index Flood Method         |
|  L-CV   |    L-Coefficient of Variation     |
|  MAPE   |  Mean Absolute Percentage Error   |
|   MDS   |      Master of Data Science       |
|   ML    |         Machine Learning          |
|   MLE   |   Maximum Likelihood Estimation   |
|   MSE   |        Mean Squared Error         |
|   PCA   |   Principal Component Analysis    |
|  RFFA   | Regional Flood Frequency Analysis |
|  RMSE   |      Root Mean Squared Error      |
|   UBC   |  University of British Columbia   |
|  USGS   |  United States Geological Survey  |

# Introduction

River floods pose a significant threat to various aspects of society,
including urban safety, agriculture, and transportation. To mitigate the
effects of floods, engineers rely on estimations of the magnitude of
different flood severities through a process called Flood Frequency
Analysis (*Hydrology of Floods in Canada* 1989) which quantifies “flood
severities” as quantiles of annual maximum river flows.

A Flood Frequency Analysis (FFA) estimates quantiles of annual maximum
river flows by fitting a probability distribution to annual maximum flow
values (*Hydrology of Floods in Canada* 1989). These annual maxima are
obtained from data measured daily or sub-daily by a river flow gauge.
The distribution of maximum flows is used to determine the annual
exceedance probability (AEP), which is the complement of a given
quantile level ($\text{AEP} = 1 - \text{quantile level}$), or the return
period, which is the inverse of the AEP
($\text{return period} = 1 / \text{AEP}$). The 0.5% AEP (i.e. a return
period of 200 years) is often called the “design flood” and is typically
considered the standard for engineering decisions (EGBC 2018).

FFAs are conducted on individual, monitored rivers with available flow
data. To extend this analysis to unmonitored rivers, analysts use a
Regional Flood Frequency Analysis (RFFA). RFFA is a type of FFA that
groups similar rivers together such that all their flow data can be
combined. RFFA uses combined data from monitored rivers to estimate flow
quantiles in similar unmonitored rivers (*Hydrology of Floods in Canada*
1989).

However, existing methods of RFFA have not been updated to keep pace
with modern statistical techniques, and may provide less reliable
estimations of flood magnitude. As ML becomes more ubiquitous and
sophisticated, it is becoming clear that traditional RFFA methods could
benefit from integrating these modern ML and statistical techniques.

The goal of this project was to employ modern ML algorithms and
refinements to the statistical methodology of RFFA to enhance existing
approaches and create more accurate predictions for high return period
floods, specifically the 0.5% AEP design flood. Success was evaluated
empirically by demonstrating improved quantile scores (a specific method
to evaluate quantile predictions) on the design flood (and other high
quantiles) for holdout gauges from the ML models compared to the current
BGC approach baseline. Ongoing success will also be based on buy-in and
feedback from BGC’s engineers and hydrologists.

# Project Deliverables

There are two primary deliverables presented by the MDS team to BGC:

1.  GitLab repository containing a `python` script
    (`flood_predictor.py`) that generates river flow quantile
    predictions for new watersheds, and supporting analysis collected in
    Jupyter Notebooks.
2.  Report document that justifies a transition to ML-based RFFA based
    on the analysis completed by the MDS team in May-June 2023.

The script is intended to inform improvements to existing software, and
the functions and code within the script and notebooks is intended to
become the basis for a final production-ready product.

This report and its appendices are intended to provide justification for
the transition to an ML-based approach for river flow quantile
prediction. As a result, the report includes analysis that supports this
transition. The contents of the report, and external content located in
the project repository should be considered important reference material
in any future decision to finalize a transition to ML-based quantile
predictions.

If time had permitted, the MDS team would have included the following
items in the project scope:

- Test the integration of the script deliverable into BGC’s GitLab
  environment, ensuring the scripts could be run locally via the GitLab
  repository.
- Automate the ML model re-training process via reproducible command
  line scripting tools, such as GNU Make `makefile`.
- Create a companion dashboard to provide context to predicted values
  via:
  - visualizing the IFM growth curve
  - displaying SHAP force plots explaining the prediction
  - check for outlier predictions via PCA plots

These features could be implemented in Cambio by repurposing or
expanding exploratory code in the repository notebooks.

# Machine Learning for RFFA

The project had four major stages:

1.  Data Prep and Cleaning
2.  Unsupervised Clustering
3.  ML-based Quantile Flow Prediction for IFM
4.  ML-based Quantile Flow Prediction for DQM

The flowchart below describes the project development process,
simplified in some areas for ease of visual understanding. The figure is
an abstraction of technical coding processes completed with the python
and R programming languages, and contained within Jupyter Notebooks and
`.R` and `.py` scripting files.

The important components of the figure will be explained in the upcoming
report sections.

![Figure 1: Flowchart of Project Development
Process](img/MDS-floods-process.png)

## Data Cleaning and Prep

The first stage of the project was to convert the raw input data into a
format that is suitable for a supervised machine learning application.
There were three major external datasets used in the project:

1.  **Annual Maxima Series (AMS) data**: Instantaneous maximum annual
    river flow values for multiple years at various flow gauges across
    the prairies, boreal plains, and boreal shield regions of Canada.
    This data was the “target” column of the supervised ML process.
2.  **Watershed Characteristics**: Various hydrological features of
    watersheds. This data was the “features” part of the supervised ML
    process, and was used to predict the “target” i.e. the instantaneous
    maximum annual flow.
3.  **Canadian Geo-regions (`.shp`)**: Geospatial representation of
    Canada’s geo-regions. Used to assign correct georegion labels
    (e.g. prairies, boreal plain, boreal shield) to gauges across
    Canada.

The raw and cleaned datasets can be viewed in the `data/` subdirectory
of the project repository.

The data cleaning, prep work, and exploratory data analysis can be found
in the `.ipynb` notebooks in the `reports/EDA` subdirectory of the
project repository, and more explanation on steps undertaken can be
found in the [Appendix: IFM Data
Preprocessing](#appendix-ifm-preprocessing) section.

A significant outcome of the data preparation stage was the creation of
“training” and “testing” data splits for the ML models. The full dataset
is split into two groups, the larger split (approx. 70%), training data,
was used to learn relationships between the features and target outcome.
The smaller split (approx. 30%), testing data, was used to evaluate how
closely model predictions matched actual flow values.

## Unsupervised Clustering

The unsupervised clustering process built homogenous groups within the
data to improve adherence statistical assumptions and ultimately improve
the accuracy of predictions. A detailed description can be found in
section the [Appendix: Regional Homogeneity, Clustering and
H-Index](#appendix-clustering) section.

## Index Flood Method

### Background

IFM for RFFA relies on the combination of two components:

1.  **Index Flood Prediction**: The predicted mean of logged
    instantaneous maximum annual flows. The prediction is based on the
    learned relationship between the watershed characteristics and
    observed flows of monitored rivers.

2.  **Growth Curve**: The relationship between river flow frequency and
    residual ratios. Residual ratios are the observed flow quantile
    divided by the observed geometric mean of flow (see equation below).
    The curve itself is a statistical distribution fit to the flow
    frequency versus residual ratio data.

$$\text{residual ratio quantile} = \frac{\text{observed flow quantile}}{\text{observed geomean flow}}$$

These two components are implemented within the current industry
methodology as such:

1.  The index flood prediction is generated via a standard
    un-regularized linear regression on all watershed characteristics,
    with watershed area transformed to log base-10.

2.  A combined (or “pooled”) statistical distribution is created for
    similar river groups. To do this, l-moments are calculated for each
    gauge, averaged, then a distribution is fit to the averaged
    l-moments. Multiple statistical distributions are fit to determine
    the best one. A list of candidate distributions and an illustrative
    figure is shown later in the report, in the [Growth
    Curves](#ifm-growth-curves) section.

The MDS team improved RFFA IFM methodology via the following changes:

1.  **Index Flood Prediction with Machine Learning**:

- **Random Forest Model**: Replaced linear regression with a modern ML
  algorithm, specifically a Random Forest Regressor. Random Forests
  learn complicated, non-linear relationships between the features and
  the target flows, and are more resilient to high dimensionality and
  co-linearity.

- **Training and evaluation criteria**: The objective of linear
  regression is to minimize the squared error. Rivers with larger flows
  tend to have large prediction errors, which are then squared.
  Minimizing squared error causes the model to favor reducing error on
  large rivers, often at the expense of accuracy for small rivers. To
  address this, the MDS team used an alternative training and evaluation
  criteria, discussed in the [Evaluation
  Metrics](#ifm-selection-metrics) section. It is worth noting that
  transforming instantaneous maximum annual flows to log base-10 did
  also partially alleviate this issue, but it did not fully resolve it.

2.  **Improved Growth Curves**: Replaced the method of l-moments with
    distributions directly fit to predefined sub-sections of the data
    using Maximum Likelihood Estimation (MLE). Fitting to larger
    sub-sections of the data reduces the potential sample penalty
    incurred when making estimations based on small, gauge-level
    samples.

### Index Flood Model: Model Selection and Justification

Model selection for this project was based on two differing but
important definitions of a “model”:

1.  The type of machine learning model to use: E.g. Random Forest
    Regressor.

2.  The data used to create the model:

- The model is trained on full undivided training data (one model), OR
- The model is trained on disaggregated sections of the training data
  separately (one model per data sub-section).

#### Evaluation Metrics

Making model selection decisions was based on multiple numeric
evaluation metrics and visual evaluation strategies. Due to the unique
structure of the river data and the default algorithms in ML model
training software, the MDS team used a combination of custom and
pre-existing numeric evaluation metrics. ML model performance was
evaluated using Weighted Root Mean Squared Error (wRMSE), R-squared, and
Mean Absolute Percentage Error (MAPE). wRMSE required a custom
implementation, which is described in the [Appendix: IFM Evaluation
Metric Selection](#appendix-ifm-eval-metric) section.

ML models were also evaluated visually, using figures such as Prediction
vs. Observed plots. These plots show the predicted value on one axis,
and the observed value on the other axis. A diagonal line is plotted on
the figure to represent predictions that are equal to observed values.
Other important numerical evaluation criteria, such as quantile scores,
are plotted visually to aid decision making.

#### Model Selection: Model Type

Model type was selected during the index flood prediction stage.
Multiple supervised learning models were evaluated, with Random Forest
Regressor being identified as the best-performer.

Other models tested include:

- **Baseline linear model**: Linear regression model with unscaled
  numerical watershed characteristics, and log base-10 watershed area.
  No regularization to prevent overfitting. This baseline model is
  considered a proxy for the current BGC process, and is used as a basis
  for comparison.
- **Regularized linear regression models**: Lasso (L1-regularization)
  and Ridge (L2-regularization).
- **Ensemble models**: Random Forest regression, XGBoost regression, and
  LightGBM regression models were all explored.

Neural network models were not explored due to a prioritization of model
explainability. However, future improvements to this study may wish to
explore neural networks for their ability to learn complex relationships
and implement custom loss functions.

Random Forest Regressor was chosen as the best model based on superior
numerical scoring metrics, and low spread on the Prediction vs. Actual
plots compared to other models. Numerical data is presented in the table
below, and Predicted vs. Actual plots can be viewed in the
`reports/ML-exploration/ML-approach-1_sg_v2.ipynb` Jupyter Notebook in
all sub-sections titled `Predicted vs. Actual Figures`.

**Table 1: Performance of Candidate Index Flood Models on Test Data**

|       Model       | Weighted RMSE | R-squared | Mean Absolute Percentage Error (%) |
|:-----------------:|:-------------:|:---------:|:----------------------------------:|
| Linear (Baseline) |     10.33     |   0.44    |                404                 |
|  Linear (Lasso)   |     24.25     |   0.67    |                144                 |
|  Linear (Ridge)   |     24.22     |   0.65    |                162                 |
|   Random Forest   |     8.92      |   0.85    |                124                 |
|      XGBoost      |     2.60      |   0.82    |                153                 |
|     LightGBM      |     27.00     |   0.66    |                265                 |

<span style="font-size: smaller;">Data reference:
`reports/ML-exploration/ML-approach-1_sg_v2.ipynb`</span>

#### Model Selection: Training Data Structure

The choice of whether to dis-aggregate training data, and therefore the
choice of how many models to train, can affect the accuracy of
predictions. Training one model on overall data is beneficial due to the
larger size of training data, but training a separate model for each
geo-region (for example), may improve internal prediction accuracy
within regions.

A random forest regression model was scored on test data to inform this
decision. First, one random forest model was trained on the full train
dataset. Second, the model based on full training data was compared to
two sets of models, trained on the following sub-sections of the
training data:

- Disaggregated by georegion (3 separate models): one model per
  georegion in the training data (prairies, boreal plain, and boreal
  shield).
- Disaggregated by cluster (4 separate models): one model per latent
  homogeneous cluster in the training data, as discovered via the
  hierarchical clustering process (cluster 1, cluster 2, cluster 3,
  cluster 4).

To compare disaggregated models to the overall model trained on all
data, the overall model was scored on disaggregated test data. The
figure below gives a visual explanation of this process. In the table
below, the `Pair for Comparison` column is used to define how to pair
the rows of the table to make comparisons between approaches. E.g. the
`A-A` comparison would compare the overall model trained on full train
data scored on test data filtered for prairies to the disaggregated
model trained on train data filtered for prairies scored on test data
filtered for prairies.

![Table 2: List to Explain Data Structure
Comparisons](img/ifm-model-selec-agg-methods.png)

Numerical and visual results showed that models disaggregated by
georegion produced the best performance on unseen test data. Detailed
results can be found in the
`proj-floods/reports/ML-exploration/ML-approach-1_sg_v2.ipynb` Jupyter
Notebook, in the section
`Numeric Evaluation Method 1: Score overall model on different subsets of train data`.

#### Index Flood Model: Results and Evaluation

The figure below is a Predicted vs. Actual plot for index flood
predictions. It shows evidence that the index flood predictions from the
ML approach were much closer to the actual index flood values, compared
to a baseline linear model (proxy for the conventional IFM prediction
method).

![Figure 2: Gauge vs. Predicted Regional Index Flood by Random Forest
Model](../ML-exploration/ma_plots/true_mean_vs_pred_combined_Random_Forest.png)

<span style="font-size: smaller;">Caption: River Flow Index Flood
Prediction from Random Forest ML Model plotted against Observed River
Index Flood, Colored by Gauge Georegion. Each point represents a gauge.
Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`</span>

![Figure 3: Gauge vs. Predicted Regional Index Flood by Linear Baseline
Model](../ML-exploration/ma_plots/true_mean_vs_pred_combined_Linear_Baseline.png)

<span style="font-size: smaller;">Caption: River Flow Index Flood
Prediction from Linear Baseline Model plotted against Observed River
Flow Index Flood, Colored by Gauge Georegion. Each point represents a
gauge. Data reference:
`reports/ML-exploration/ML-approach1-ma.ipynb`</span>

### Growth Curves

Growth curves provide the essential scaling factor required to convert
the index flood prediction to a desired quantile flow magnitude.

To create a growth curve, a statistical distribution is chosen from a
list of candidate distributions based on how well it fits AEP vs
residual ratio data. The list below is a collection of the most common
distributions used in industry:

- Gumbel
- Log-Normal
- Log-Pearson Type 3
- Normal
- Pearson Type 3

The best distributional parameters for each were selected by Maximum
Likelihood Estimation (MLE) / maximum goodness-of-fit estimation (MGE).
Mean quantile losses were then used to choose the most suitable
distribution. Other methods could have been used to choose the best
distribution; this is discussed in the [Appendix: Expanded
Limitations](#appendix-limitations-distselect) section.

#### Disaggregated Statistical Distributions

Statistical distributions were fit to homogenous sub-sections of the
data, rather than at the individual gauge level or to overall data. The
data was disaggregated by both geo-region (prairies, boreal plain,
boreal shield) and hierarchical cluster (cluster 1-4). This provided the
most homogenous groups, based on findings from the clustering analysis
(refer to [Appendix: Regional Homogeneity, Clustering and
H-Index](#appendix-clustering) for a detailed description of
clustering).

> Note: Trained ML models were disaggregated by geo-region only, while
> the statistical distributions were disaggregated by both geo-region
> and cluster. The objective of the ML model is to create accurate index
> flood predictions, while the objective of creating growth curves is to
> use the most homogenous groups to address homogeneity requirements.

Fitting on disaggregated data resulted in 60 distributions
($\text{3 geo-regions} \times \text{4 clusters} \times \text{5 candidate distributions}$).
However, there are no gauges in Boreal Shield and Cluster 1, meaning
only 55 distributions are created
($[(\text{3 geo-regions} \times \text{4 clusters)}-1] \times \text{5 candidate distributions}$).

Each distribution describes the relationship between the AEP (or
quantile level) and the residual ratio (flow quantile / index flood) for
each data sub-section.

#### Log Pearson Type III

Based on mean quantile loss, the Log Pearson Type III distribution was
the most suitable distribution across all sub-groups. Evidence of this
is shown in the figure below. The Log Pearson Type III distribution
performs best (low quantile loss) across most quantile levels; most
importantly it performs best at the largest quantiles (furthest left on
x-axis).

![Figure 4: Mean Quantile Loss of Candidate Statistical
Distributions](img/distribution-justification.png)

<span style="font-size: smaller;">Caption: Mean quantile loss of five
candidate distributions. Mean quantile loss is calculated across cluster
and region to get an overall mean. Log Pearson Type III shows the lowest
mean quantile loss across the most quantile levels, most importantly
high quantiles (furthest left on x-axis). X-axis labels could not be
fixed due to custom functions in `reports/Dist_ratio/scale_gumbel.R`
that override label settings. Data reference:
`reports/Dist_ratio/distribution_justification/distribution-justifications.R`</span>

A numerical equivalent of the figure above can be viewed in the
`reports/Dist_ratio/distribution_justification/distribution_justification.csv`
file in the project repository.

To integrate the Log Pearson Type III distribution into the ML pipeline,
a data table was created to relate each quantile level between 50% and
99.8% with a corresponding residual ratio for each of the 11 data
sub-sections. These values were then used as scale-up factors for the
index flood prediction to obtain a river flow quantile prediction.

### River Flow Quantile Prediction

Residual ratios act as scaling ratios for the index flood prediction,
and are used to convert the index flood into a quantile prediction. Once
the index flood prediction model was optimized, and the residual ratio
relationships were developed for each subsection of the data, they are
combined to create a river flow quantile prediction
($\text{River Flow Quantile} = 10^{\text{Index Flood Prediction}} \times \text{Residual Ratio from Growth Curve}$).
Note that it is necessary to exponentiate the Index Flood prediction as
it is generated by the ML model on the $log_{10}$ scale.

#### Methodology

For a given new watershed to be predicted, the methodology follows these
steps:

1.  Assign the watershed to its most likely geo-region (via spatial
    join), and cluster (via hierarchical clustering).

2.  Get index flood prediction from Random Forest Model.

3.  Based on geo-region and cluster combination, choose the
    corresponding growth curve.

4.  For each quantile of interest, multiply the index flood prediction
    by the corresponding residual ratio to yield quantile flow
    predictions.

For example, for a new watershed in Boreal Plains and Cluster 3:

$$\text{Quantile Flow}_{\text{new watershed, 0.995 quantile}} = \text{Residual Ratio}_{\text{0.995 quantile}} * 10^{\text{ML index flood prediction}_{\text{new watershed}}}$$

$$= (\frac{\text{Quantile Flow}_{\text{boreal plain, cluster 3, 0.995 quantile}}}{\text{Geomean Flow}_{\text{boreal plain, cluster 3}}}) * 10^{\text{ML Index Flood Prediction}_{\text{new watershed}}}$$

#### Performance Evaluation

The plots below illustrate the quantile loss scores for river flow
quantile predictions across various different quantile levels, with
smaller scores indicating better performance. The figure includes the
following models:

- A conventional FFA process used to estimate the river flow quantile
  prediction when flow data is available. This is the benchmark for
  comparison.
- A proxy for the baseline RFFA methods used to estimate river flow
  quantiles in the absence of river flow data.
- The optimal ML model river flow quantile prediction in the absence of
  river flow data (RFFA), generated from the Random Forest Index Flood
  model.

![Figure 5: Mean Quantile Losses for River Flow Quantile Prediction
Methods](../ML-exploration/ma_plots/mean_quantile_scores_baseline_vs_our_model.png)

<span style="font-size: smaller;">Caption: Mean quantile losses for
river flow quantile prediction methods. Lower scores are better. The ML
model is closer to the FFA benchmark for quantile losses than the
conventional RFFA approach (baseline). This indicates the ML model
performs better than the conventional approach at predicting river flow
quantiles. Data reference:
`reports/ML-exploration/ML-approach1-ma.ipynb`</span>

The figure shows that the ML model losses are closer to the FFA
benchmark than the RFFA baseline. This confirms that the trained ML
model outperforms the baseline based on quantile score. A raw,
disaggregated version of this figure showing gauge-level quantile losses
is available in the [Appendix: Gauge-level Quantile Loss of IFM Quantile
Flow Predictions](#appendix-ifm-gauge-qloss) section. Overall, there is
strong evidence to suggest the ML model outperforms the baseline model
at predicting high quantiles, including the 0.5% AEP.

## Direct Quantile Method

### Background

DQM for RFFA relies on a model that can predict river flow quantiles
directly. The model uses monitored watersheds, similar to the
unmonitored watershed in question, to predict quantile river flow
directly. A significant component of this process is the determination
of what constitutes a “similar watershed”.

The DQM approach is implemented within the current industry methodology
as such:

- Given an unmonitored watershed of interest, select monitored
  watersheds with similar characteristics to the reference watersheds.
  This process relies on manual intervention by a hydrologist to group
  similar watersheds, though unsupervised (hierarchical) clustering is
  considered in complex situations.
- Fit selected watersheds with frequency-magnitude curves using standard
  FFA methodology (method of L-moments).
- For each desired quantile of interest in the unmonitored river, fit a
  linear regression with log(area) as a regressor and the quantile river
  flow of interest as the response. This linear regression can then be
  used to predict river flow quantiles for unmonitored watersheds.

The primary drawback to this method is the need for manual intervention.
This requires domain knowledge to define and locate similar watersheds,
and may also require localized expertise of specific hydrological
regions. This encourages watershed selection based on similarity of
features under the assumption it will improve downstream predictions.

The ML version of DQM instead uses all data to directly generate
quantile predictions based on learned relationships between watershed
characteristics and river flow quantiles. All gauges are used to create
the model, not just gauges from similar watersheds. Watersheds with
similar features should receive similar predictions due to the complex
relationships learned by the ML model. This removes the requirement of
manual intervention, but requires trust in the ML algorithm to learn the
proper relationships.

### Model Selection

To implement DQM with ML, the MDS team used quantile regression
algorithms available from various python libraries.

Candidate models included:

- Linear Quantile Regression
- Tree based ensemble models, such as:
  - Random Forest Quantile Regression
  - LightGBM Quantile Regression

The off-the-shelf implementations of the Linear Quantile Regression and
LightGBM models required fitting one model per quantile level.
Therefore, creating a range of quantile predictions per gauge required a
combination of multiple models. This was not necessary for the Random
Forest Quantile regression model, as a single model is able to learn
multiple pre-specified quantile levels.

The loss function for quantile regression models minimizes quantile
loss, where `q` is the non-exceedance probability of interest:

$L = \begin{cases} (1-q)|\hat{y}_q - y| & \text{if }y \lt \hat{y}_q \\ q|\hat{y}_q - y| & \text{if }y \ge \hat{y}_q \end{cases}$

This loss function is linear with respect to prediction error
($(\hat{y}_q - y)$). Therefore, this loss function suffers a similar
error scale issue to the least squares loss function in the IFM index
flood prediction (see [Appendix: IFM Evaluation Metric
Selection](#appendix-ifm-eval-metric)). However, in DQM this
relationship is linear rather than quadratic as in IFM. Log-scaling flow
was deemed sufficient to address the scale issue in DQM. As a result, a
refit error metric was not required.

The Linear Quantile Regression model was chosen from the list of
candidate models based on quantile loss score at high quantiles, as well
as increased interpretability.

The figure below shows the mean quantile scores across all gauges for
Linear Quantile Regression, Random Forest Quantile Regression, and
LightGBM Quantile Regression.

![Figure 6: Mean Quantile Loss Scores of Candidate DQM ML
Models](img/DQM-selection.png)

<span style="font-size: smaller;">Caption: Mean quantile loss scores for
DQM ML models. Smaller loss scores are better. Random Forest and
LightGBM models outperform Linear model at low quantiles, but perform
worse at high quantiles. Linear is marginally better than LightGBM at
the highest quantiles. Data reference:
`reports/Approach2/quantile-overall.ipynb`</span>

While the Random Forest and LightGBM models outperformed the Linear
model at low and mid-range quantiles, the Linear model slightly
outperforms LightGBM at the highest quantile levels (furthest left on
x-axis). The highest quantiles are most important (e.g. design flood at
0.5% AEP), so the linear quantile regression model was chosen as the
best model.

### River Flow Quantile Prediction

The plots below illustrate the quantile loss scores for river flow
quantile predictions across various different quantile levels, with
smaller scores indicating better performance. The figure includes the
following models:

- A conventional FFA process used to estimate the river flow quantile
  prediction when flow data is available. This is the benchmark for
  comparison.
- The optimal ML model river flow quantile prediction in the absence of
  river flow data (RFFA), generated from the Linear Quantile Regression
  model.

> Note: the figure below does not show an RFFA baseline comparison
> because creating this baseline requires manual intervention by a
> domain expert. This is difficult to complete at the same scale (all
> test data) that the ML model and FFA are applied at, so it is not
> shown.

![Figure 7: Quantile Score Comparison of FFA Benchmark to DQM Quantile
Prediction](img/DQM-vs-FFA.png)

<span style="font-size: smaller;">Caption: Quantile loss score
comparison of FFA benchmark to DQM Linear Quantile regression
predictions at different quantile levels. Linear quantile predictions
approach FFA benchmark at higher quantiles. No RFFA baseline is
available for comparison. Data reference:
`reports/Approach2/quantile-overall.ipynb`</span>

The figure shows that the ML model losses are close to the FFA
benchmark. Due to the absence of an RFFA baseline, it is difficult to
quantify the improvement of the ML model against the conventional RFFA
approach. However, the quantile losses of the Linear Quantile Regression
model are similar to the quantile losses from the IFM predictions. As a
result, we could possibly arrive at a logical conclusion that the
ML-based DQM provides a similar improvement to the baseline than the
ML-based IFM does. However, without an accessible, concrete RFFA
baseline this conclusion cannot be made definitively.

# Limitations

The following table gives a high-level overview of various limitations
to the current analysis. An expanded version of this list can be viewed
in the [Appendix: Expanded Limitations](#appendix-limitations) section.

**Table 3: Project Limitations**

| Type                           | Description                                                                                                                                                                 |
|--------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Data Structure                 | Variable amounts of data at gauge level is not directly accounted for in the ML algorithms.                                                                                 |
| Data Structure                 | Grouped data (multiple years per gauge, multiple gauges) affects training/testing data split.                                                                               |
| Data Structure                 | River flow magnitude is skewed, which cannot be fully dealt with.                                                                                                           |
| Data Structure                 | Nested gauge structure means measurements across gauges are sometimes not statistically independent from each other.                                                        |
| Data Structure                 | Testing dataset size can be quite variable depending on which gauges are assigned to training/testing data.                                                                 |
| Algorithmic and ML Limitations | Quantile scores are not scaled to river size                                                                                                                                |
| Algorithmic and ML Limitations | There is no accessible ground truth for Direct Quantile Method                                                                                                              |
| Algorithmic and ML Limitations | The variation in instantaneous maximum daily flow within the prairies geo-region was not well-explained by the given watershed features.                                    |
| Statistical Assumptions        | Higher homogeneity could be achieved in unsupervised clustering.                                                                                                            |
| Statistical Assumptions        | The quantile predictions show some evidence of over-prediction. With the Index Flood model performing well, this is likely due to the residual ratio from the growth curve. |
| Statistical Assumptions        | The most suitable statistical distribution was chosen via quantile scores. It could be chosen with other selection methods in future iterations.                            |
| Data Product Limitations       | There is limited functionality for manual intervention (i.e. manual removal of river gauges).                                                                               |
| Data Product Limitations       | Retraining the ML models must be done via Jupyter Notebook.                                                                                                                 |
| Data Product Limitations       | Direct Quantile Method has limited hyperparameter tuning due to the large number of models to train.                                                                        |

# Future Improvements

The following list provides areas that the existing analysis could be
improved in the future by BGC or future MDS capstone projects. The
improvements listed are brief summaries, and the full description can be
viewed in the [Appendix: Expanded Future
Improvements](#appendix-improvements) section.

**Table 4: Project Limitations**

| Improvement                                   | Description                                                                                  |
|-----------------------------------------------|----------------------------------------------------------------------------------------------|
| Explore a mixed effects model                 | Explore the effect of year on quantile prediction                                            |
| Conduct data augmentation                     | Estimate instantaneous daily maximum via average daily maximum to increase data availability |
| Visualize outlier detection                   | Use clustering to create effective visualizations to find unusual watersheds                 |
| User more clusters                            | More clusters may improve the validity of homogeneity assumptions                            |
| Explore a new quantile prediction methodology | Create a model that serves as an intermediate between the IFM and DQM methodologies          |
| Implement continuous DQM predictions          | Implement a methodology to create continuous quantile predictions via DQM                    |

# Conclusions

Floods pose significant risk to both infrastructure and human life.
Traditional RFFA methods could benefit from integrating modern ML and
statistical techniques. This report justifies the transition to ML-based
RFFA for both the Index Flood Method and Direct Quantile Method for
RFFA. While the approaches detailed in this report have some statistical
limitations and room for further improvement, the benefits of ML-based
RFFA are apparent. The river flow quantile prediction script and
supporting analysis developed by the MDS team should be used to
supplement BGC’s Cambio software to improve the accuracy of BGC’s RFFAs.
This report and the documentation of the accompanying repository provide
a thorough proof-of-concept outline detailing both the improvements
made, as well as avenues for further research. The report is a helpful
reference to justify BGC’s transition toward ML-based RFFA.

# Appendices

## Appendix: Supporting Analysis for Index Flood Method

### IFM Data Preprocessing

Multiple preprocessing steps are required to get data suitable for the
IFM. The following steps are required to ensure data is suitable for
river flow quantile prediction.

**Table A1: IFM Preprocessing Steps**

| Step | What                                                                                      | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|------|-------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Remove erroneous/missing data and control for edge cases                                  | Removing data with N/A values ensures calculations can be completed successfully. Removing gauges with 1 or fewer years of data ensures within-gauge variance is \>0, which is important during model training. The model training algorithm explained in [IFM Evaluation Metric Selection](#appendix-ifm-eval-metric) outlines the requirement for a weight parameter that is derived from variance. If variance is zero, the weight parameter becomes infinity, which is a problematic edge case that must be avoided. |
| 2    | Convert instantaneous river flow values to log base-10 scale                              | Addresses skewness in the data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 3    | Scale all numeric watershed characteristics to a common scale (mean = 0 and variance = 1) | Scaling data ensures that all numeric features are on a similar scale, preventing one feature from dominating the model training process due to its larger magnitude.                                                                                                                                                                                                                                                                                                                                                    |

### IFM Evaluation Metric Selection

Loss functions are important in machine learning models because they
define a quantity to be minimized to improve the overall accuracy of the
model. The least squares loss function is ubiquitous, and is based on
minimizing the squared difference between actual and predicted values.
Common loss functions are coded directly into off-the-shelf machine
learning modeling libraries, such as `scikit-learn`.

There are inherent limitations with using scikit-learn’s built-in
squared error loss function for training the ML models on skewed data.
Taking river flow as an example, squared error in large rivers will
dominate overall squared error across all rivers (e.g. prediction errors
of 100 $m^3/s$ in large rivers vs prediction errors of 1 $m^3/s$ in
small rivers). As a result, squared error will prioritize reducing
residuals in large rivers, as this is the most efficient way to reduce
overall residuals. This issue is less prevalent on the log scale, but
due to this modeling process being applied most often to small rivers,
simply logging the target flows is insufficient to obtain the desired
model fit.

To address this issue with squared error, the MDS team implemented a
modified approach to training the scikit-learn models that incorporates
the variable scale of river flow data.

Scikit-learn offers a `sample_weight` parameter within multiple models’
`.fit()` method. This parameter allows the user to indicate the relative
importance or significance of each sample in the dataset. I.e. large
river flows can be assigned a smaller weight, and small river flows can
be assigned a larger weight. This prevents large rivers from dominating
the squared error algorithm, and ensures all rivers are prioritized
equally when the squared error algorithm is minimizing overall
residuals. The `sample_weight` parameter accepts a list of numerical
values that represent the weight to be assigned to each row in the
training data.

Using the `sample_weight` parameter required the creation of a method to
represent the weight of each gauge. The weight of each gauge was
calculated as the inverse of the variance of logged river flow (log
base-10 of the instantaneous maximum annual river flow). The magnitude
of prediction error should be proportional to the standard deviation
observed at that gauge, and therefore the squared prediction error
should be proportional to the variance (standard deviation squared).
Gauges on large rivers are assigned a small weight, as large rivers are
expected to have large flow variation. As a result, the relative
importance of all rivers in the squared error algorithm should be
roughly equal. This approach is similar to the approach of weighted
least squares, but the weights are fixed, rather than learned
parameters.

A similar issue arose during hyperparameter tuning, where the best model
hyperparameters minimize some objective scoring metric. Typical choices
for this metric are root mean squared error (RMSE), R-squared, or even
mean absolute percentage error (MAPE). However, all of these metrics
have issues:

- RMSE shares the same limitation as least squares due to the
  order-of-magnitude variation in flow data.
- R-squared explains the degree of variance captured by a model, but
  does not directly optimize prediction accuracy. It’s possible to have
  a high R-squared value but poor predictive performance.
- MAPE is a promising metric to optimize the models as it is a relative
  scoring metric, which is helpful when the target has a variable scale.
  However, there are multiple limitations with using MAPE as an
  optimization criterion:
  - Sensitivity to zero or small actual values: MAPE is sensitive to
    cases where the actual values are close to zero or very small. When
    the denominator (actual value) approaches zero, the MAPE can become
    extremely large or undefined, leading to misleading or unreliable
    results. This sensitivity can make it challenging to compare models
    or make informed decisions during hyperparameter tuning.
  - Asymmetric treatment of errors: MAPE treats overestimations and
    underestimations differently. It penalizes overestimations more
    severely than underestimations since the error is divided by the
    actual value. Depending on the specific application, this asymmetric
    treatment may not align with the desired behavior of the model. In
    this situation of predicting the index flood, balanced errors are
    more desirable.

In response, the MDS team used the same weighted squared error approach,
which incorporates the weights developed for model fitting, and applied
them in a similar fashion to calculate root mean squared error, which
tuning attempts to minimize.

Overall, there is no single scoring metric that can be used to select
the optimal ML model in isolation. Throughout this analysis, multiple
numeric evaluation metrics (weighted RMSE, R-squared, MAPE) and visual
evaluation strategies (e.g. Prediction vs. Actual Plots) are used to
choose the best overall model.

### Example Model Predictions

**Example 1 (07FB001): Example of an inaccurate IFM prediction**

*Qualification:*

- This gauge qualifies as inaccurate based on having a large difference
  between prediction and actual index flood value.

*Overview of predictions:*

**Table A2: Overview of 0.5% AEP Predictions for Gauge 07FB001**

| Component                      | From                 | Value           |
|--------------------------------|----------------------|-----------------|
| Observed Value                 | Observed Index Flood | $1569 m^3/s$    |
| IFM: Index Flood Prediction    | Baseline Model       | $1232 m^3/s$    |
| IFM: Index Flood Prediction    | ML Model             | $730 m^3/s$     |
| FFA: 0.5% AEP Flood Benchmark  | FFA                  | \$ 6477 m^3/s\$ |
| IFM: 0.5% AEP Flood Prediction | Baseline Model       | $4540 m^3/s$    |
| IFM: 0.5% AEP Flood Prediction | ML Model             | $8363 m^3/s$    |
| DQM: 0.5% AEP Flood Prediction | ML Model             | $11820 m^3/s$   |

*Feature Importance via SHAP*

![Figure A1: SHAP Force Plot of Features that Influence Prediction for
Gauge 07FB001](../ML-exploration/ma_plots/shap_force_plot_bad_case.png)

<span style="font-size: smaller;">Caption: To interpret the chart, note
two things: the “base value” and the “f(x)” value. The “base value”
represents a typical output, and the “f(x)” represents where the current
output is. If the “f(x)” is right of the base value, this prediction is
higher than expected, and vice versa. Columns on the red side of the
spectrum work to push the prediction higher, while blue values work to
push the prediction lower. In the chart above, `soils_total` is the
largest contributor to a high prediction, while `cn_arcii` is the
largest contributor to lowering the prediction. </span>

- The top 3 features pushing the prediction higher are HSG Soil Type B,
  Soils Total, and Runoff Curve Number (AMCII).
- Snow/Ice is the most significant feature pushing the prediction lower.

*Feature Summary:*

**Table A3: Feature Summary of Watershed Upstream of Gauge 07FB001**

| Feature                   | Value                   |
|---------------------------|-------------------------|
| StationNum                | 07FB001                 |
| NameNom                   | PINE RIVER AT EAST PINE |
| Status                    | active                  |
| Area_km2                  | 12136.3                 |
| Date                      | 2021-12-17              |
| Normal_1971_2000_PAS_mean | 415.137                 |
| Normal_1971_2000_MAT_mean | 0.963                   |
| Normal_1971_2000_MAP_mean | 830.29                  |
| PPT01_mean                | 75.878                  |
| PPT02_mean                | 61.525                  |
| PPT03_mean                | 56.537                  |
| PPT04_mean                | 46.149                  |
| PPT05_mean                | 50.813                  |
| PPT06_mean                | 87.087                  |
| PPT07_mean                | 91.739                  |
| PPT08_mean                | 66.525                  |
| PPT09_mean                | 66.452                  |
| PPT10_mean                | 70.344                  |
| PPT11_mean                | 83.404                  |
| PPT12_mean                | 73.827                  |
| PPT_wt                    | 211.23                  |
| PPT_sp                    | 153.499                 |
| PPT_sm                    | 245.35                  |
| PPT_fl                    | 220.2                   |
| Forest                    | 0.866                   |
| Shrub                     | 0.097                   |
| Wetland                   | 0.001                   |
| Cropland                  | 0.017                   |
| Urban-Barren              | 0.012                   |
| Water                     | 0.003                   |
| Snow-Ice                  | 0.003                   |
| landcover_total           | 193247.0                |
| HSG-A                     | 0.0                     |
| HSG-B                     | 0.0                     |
| HSG-C                     | 0.841                   |
| HSG-D                     | 0.003                   |
| HSG-AD                    | 0.0                     |
| HSG-BD                    | 0.0                     |
| HSG-CD                    | 0.156                   |
| HSG-DD                    | 0.0                     |
| soils_total               | 391997.0                |
| gmted_elev_min            | 518.0                   |
| gmted_elev_mean           | 1150.842                |
| gmted_elev_max            | 2297.0                  |
| gmted_elev_range          | 1779.0                  |
| centre_elev               | 1158.0                  |
| cn_arci                   | 56.033                  |
| cn_arcii                  | 74.352                  |
| cn_arciii                 | 87.561                  |
| centroid_lon              | -121.547                |
| centroid_lat              | 55.242                  |
| Area_sqkm                 | 12125.769               |
| Length_km                 | 688.739                 |
| Catchment_Length_m        | 17605.755               |
| Catchment_Slope           | 0.101                   |
| cluster_label             | 3                       |
| region_label              | Boreal Plain            |

Data Reference: watershed characteristics data, cluster and region label
generated by `flood_predictor.py`

**Example 2 (05PE028): Example of an accurate IFM prediction**

*Qualification:*

This gauge qualifies as accurate based on having a small difference
between prediction and actual index flood value.

*Overview of predictions:*

**Table A4: Overview of 0.5% AEP Predictions for Gauge 05PE028**

| Component                      | From                 | Value           |
|--------------------------------|----------------------|-----------------|
| Observed Value                 | Observed Index Flood | \$ 804 m^3/s\$  |
| IFM: Index Flood Prediction    | Baseline Model       | \$ 417 m^3/s\$  |
| IFM: Index Flood Prediction    | ML Model             | \$ 744 m^3/s\$  |
| FFA: 0.5% AEP Flood Benchmark  | FFA                  | \$ 1572 m^3/s\$ |
| IFM: 0.5% AEP Flood Prediction | Baseline Model       | \$ 1538 m^3/s\$ |
| IFM: 0.5% AEP Flood Prediction | ML Model             | \$ 2270 m^3/s\$ |
| DQM: 0.5% AEP Flood Prediction | ML Model             | \$ 1992 m^3/s\$ |

*Feature Importance via SHAP*

![Figure A2: SHAP Force Plot of Features that Influence Prediction for
Gauge 05PE028](../ML-exploration/ma_plots/shap_force_plot_good_case.png)

<span style="font-size: smaller;">Caption: To interpret the chart, note
two things: the “base value” and the “f(x)” value. The “base value”
represents a typical output, and the “f(x)” represents where the current
output is. If the “f(x)” is right of the base value, this prediction is
higher than expected, and vice versa. Columns on the red side of the
spectrum work to push the prediction higher, while blue values work to
push the prediction lower. In the chart above, `centre_elev` is the
largest contributor to a high prediction, while `cn_arcii` is the
largest contributor to lowering the prediction. </span>

- The top 3 features pushing the prediction lower are `HSG-B`,
  `soils_total`, and `cn_arcii`. - `HSG-C` is the most significant
  feature pushing the prediction higher.

*Feature summary:*

**Table A5: Feature Summary of Watershed Upstream of Gauge 05PE028**

| Feature                   | Value                                             |
|---------------------------|---------------------------------------------------|
| StationNum                | 05PE028                                           |
| NameNom                   | WINNIPEG RIVER WESTERN CHANNEL NEAR TUNNEL ISLAND |
| Status                    | active                                            |
| Area_km2                  | 69526.300                                         |
| Date                      | 2022-06-17                                        |
| Normal_1971_2000_PAS_mean | 147.408                                           |
| Normal_1971_2000_MAT_mean | 2.680                                             |
| Normal_1971_2000_MAP_mean | 663.397                                           |
| PPT01_mean                | 25.269                                            |
| PPT02_mean                | 20.841                                            |
| PPT03_mean                | 26.453                                            |
| PPT04_mean                | 37.140                                            |
| PPT05_mean                | 64.188                                            |
| PPT06_mean                | 102.489                                           |
| PPT07_mean                | 93.576                                            |
| PPT08_mean                | 86.378                                            |
| PPT09_mean                | 85.282                                            |
| PPT10_mean                | 57.692                                            |
| PPT11_mean                | 39.888                                            |
| PPT12_mean                | 24.200                                            |
| PPT_wt                    | 70.309                                            |
| PPT_sp                    | 127.781                                           |
| PPT_sm                    | 282.442                                           |
| PPT_fl                    | 182.862                                           |
| Forest                    | 0.680                                             |
| Shrub                     | 0.014                                             |
| Wetland                   | 0.112                                             |
| Cropland                  | 0.018                                             |
| Urban-Barren              | 0.001                                             |
| Water                     | 0.175                                             |
| Snow-Ice                  | 0.000                                             |
| landcover_total           | 1108977.000                                       |
| HSG-A                     | 0.000                                             |
| HSG-B                     | 0.288                                             |
| HSG-C                     | 0.266                                             |
| HSG-D                     | 0.018                                             |
| HSG-AD                    | 0.001                                             |
| HSG-BD                    | 0.174                                             |
| HSG-CD                    | 0.000                                             |
| HSG-DD                    | 0.000                                             |
| soils_total               | 1614456.000                                       |
| gmted_elev_min            | 287.000                                           |
| gmted_elev_mean           | 399.945                                           |
| gmted_elev_max            | 629.000                                           |
| gmted_elev_range          | 342.000                                           |
| centre_elev               | 334.000                                           |
| cn_arci                   | 55.188                                            |
| cn_arcii                  | 73.260                                            |
| cn_arciii                 | 86.729                                            |
| centroid_lon              | -92.917                                           |
| centroid_lat              | 48.630                                            |
| Area_sqkm                 | 69484.767                                         |
| Length_km                 | 2061.676                                          |
| Catchment_Length_m        | 33703.044                                         |
| Catchment_Slope           | 0.010                                             |
| cluster_label             | 4                                                 |
| region_label              | Boreal Shield                                     |

Data Reference: watershed characteristics data, cluster and region label
generated by `flood_predictor.py`

### Additional Figures

#### IFM: Gauge-level Quantile Loss

![Figure A3: Gauge-level Quantile Loss from Baseline RFFA
Model](../ML-exploration/ma_plots/quantile_loss_by_station_from_Baseline_RFFA.png)

Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`

![Figure A4: Gauge-level Quantile Loss from ML
Model](../ML-exploration/ma_plots/quantile_loss_by_station_from_ML_Model.png)

Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`

#### IFM: Quantile Loss by Region

![Figure A5: Gauge-level Quantile Loss from Baseline RFFA
Model](../ML-exploration/ma_plots/mean_quantile_scores_by_region_from_Baseline_RFFA.png)

Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`

![Figure A6: Gauge-level Quantile Loss from ML
Model](../ML-exploration/ma_plots/mean_quantile_scores_by_region_from_ML_Model.png)

Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`

## Appendix: Supporting Analysis for Direct Quantile Prediction

There are 4 notebooks under the folder ‘report/Approach2’. The codes are
basically the same, except the input/training data. Each notebook is for
a specific region (Prairies, Boreal Shield, Boreal Plain, Overall) and
contains everything, e.g models, coefficient importances, decreasing
portions and “Max-out” behavior exploration, which will be explained
below.

### DQM Data Preprocessing

Data preprocessing for DQM is pretty similar to IFM, except that loss
functions of models used in DQM are quantile loss, a.k.a fidelity loss,
in which case there is no need to implement weighted RMSE like what we
did for IFM.

For the sake of non-duplicates, results/findings will be mainly focused
on the ‘Overall’ region. For other regions, please refer to the
corresponding notebooks.

**Table A6: DQM Preprocessing Steps**

| Step | What                                                                                      | Why                                                                                                                                                                   |
|------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | Remove erroneous/missing data and control for edge cases                                  | Removing data with N/A values ensures calculations can be completed successfully.                                                                                     |
| 2    | Convert instantaneous river flow values to log base-10 scale                              | Addresses skewness in the data                                                                                                                                        |
| 3    | Scale all numeric watershed characteristics to a common scale (mean = 0 and variance = 1) | Scaling data ensures that all numeric features are on a similar scale, preventing one feature from dominating the model training process due to its larger magnitude. |

### Coefficient importance

Three models are used for DQM, Linear quantile regression, LightGBM
quantile regression and Quantile random forest regression. Feature
importances shown below come with models’ own implementation. Taking
‘overall’ region as an example:

Linear:

![Figure A7: Coefficient Importance: Linear Quantile
Regressor](img/DQM-appendix/linear.png)

Data reference: `reports/Approach2/quantile-overall.ipynb`

LightGBM:

![Figure A8: Coefficient Importance: LightGBM Quantile
Regressor](img/DQM-appendix/lightGBM.png)

Data reference: `reports/Approach2/quantile-overall.ipynb`

Random forest:

![Figure A9: Coefficient Importance: Random Forest Quantile
Regressor](img/DQM-appendix/random-forest.png)

Data reference: `reports/Approach2/quantile-overall.ipynb`

Generally speaking, feature importances present what we expect: flows
are most correlated with precipitation and soil types.

However, for tree-based models’ feature importance, one thing to note is
that models tend to favor numerical features more. Instead of taking
what has been already implemented from packages, we should also consider
doing a permutation feature importance, which has not been done in
notebooks yet.

### “Max-out” Behavior

This section is mainly for exploration.

Theoretically speaking, quantile functions should “max-out” at some
value, i.e. value beyond that quantile does not change. In order to
explore this kind of behavior, models have been trained on 400
quantiles.

After training and testing with each test gauge, we do not see “max-out”
behaviors with Linear and lightGBM, but we do see this with random
forest sometime. It’s not pretty clear if this is because of
package-specific implementation and needs to be further checked. Below
is just one example:

![Figure A10: Example of Max-Out Behaviour at One
Gauge](img/DQM-appendix/max-out.png')

Data reference: `reports/Approach2/quantile-overall.ipynb`

### Decreasing portion

This section is mainly for exploration.

Ideally, given the nature of quantiles, higher quantiles should always
be larger or equal in magnitude than smaller quantiles. Decreasing
portion refers to the proportion of quantiles from a model where higher
quantiles may not always be larger in magnitude than smaller quantiles.

107 rivers in test set. We have 107\*8 estimated quantiles. Numerator is
the number of lower quantiles that is \>= higher quantiles.

Linear: 105/856 Random Forest: 50/856 LighGB: 146/856

### Prediction Concerns

While the quantile loss scores of the models appear promising, it is
important to acknowledge that these scores are averaged, obscuring the
variability of the loss. Merely relying on the mean score fails to
capture crucial information about the spread or shape of the quantile
loss.

A concern of note with the direct quantile method is that the
consistency and accuracy of predictions can vary considerably for
different gauges. The figures below are examples of a bad and good
(respectively) set of quantile predictions from the modeling test set.
It is not currently understood what aspect of a gauge’s features cause
the models to generate poor predictions. As with the IFM, it is
necessary to exercise caution when employing direct quantile method
predictions.

![Figure A11: Example of Bad Gauge-level DQM River Flow Quantile
Predictions](img/DQM-predictions/bad.png)

Data Reference: `reports/Approach2/good_bad.R`

![Figure A12: Example of Good Gauge-level DQM River Flow Quantile
Predictions](img/DQM-predictions/good.png)

Data Reference: `reports/Approach2/good_bad.R`

Unfortunately, unlike the IFM, there is no way to verify that the DQM
presents empirical benefits over the existing method, as doing so would
require a hydrologist to identify similar rivers for each of the
hundreds of test set gauges. However, the ability of this method to
produce reasonable predictions without any need for manual intervention
suggests that this is a viable alternative to the direct quantile method
currently used. In the future, verification could be achieved by
comparing the predictions of this model to the model used in practice,
ideally for rivers with well known flow values to establish a ground
truth.

## Appendix: Choosing IFM or DQM

Both IFM and DQM are used in modern hydrology depending on context, and
as such there is validity to presenting improvements to both methods.
Consulting with BGC’s hydrologists, IFM is often the default selection,
as it has more physically interpretable components (an index flood
prediction and normalized growth curve) that provide transparency to the
approach. However, IFM requires a region to have relatively homogenous
watersheds. Homogeneity of gauges is required to fit a statistical
distribution to a defined group. When this is unrealistic, the direct
quantile method is used instead. Often more mountainous regions
(i.e. British Columbia) rely on the direct quantile method, while more
level regions (i.e. Boreal Plains) can use IFM.

## Appendix: Regional Homogeneity, Clustering and H-Index

The IFM relies on the assumption that all rivers pooled together to
generate growth curves share the same underlying distribution of
residual ratios (observed flow divided by the observed geomean of flow).

To validate this assumption, it is necessary to create subgroups of
rivers which share highly similar growth curves, such that their
normalized flow data can be pooled to create a single growth curve that
describes the homogenous group.

Hierarchical clustering on the features of the watersheds is done to
achieve this without manual intervention. Hierarchical clustering is
preferred due to its flexibility to manually define the desired number
of clusters, as well as the ability to define different linkage
criteria. The “ward” criterion in the `scipy` implementation of
hierarchical clustering was found to be ideal, as its criterion of
equalized variance within clusters tends to create more evenly sized
clusters and reduce the tendency for individual rivers to be labeled as
outliers.

The goal of the hierarchical clustering undertaken on this project was
to generate homogenous clusters, but also to provide physical
interpretability. It was important for the differences between the
clusters to be explainable, rather than being arbitrary labels.

The following criteria were used for clustering exploration, explained
in detail in the subsections below.

1.  Standard Deviation of Residual Ratios

2.  Distribution of Variance Ratios

3.  Physical/Principal Component Analysis (PCA) Interpretability

4.  Heterogeneity Index (M. and Wallis 1997)

Overall, 4 clusters were selected based on the above criteria, formed by
applying the clustering to all data pooled together. It was also
separately found that subdividing the data by geo-region also increased
homogeneity. The result was the formation of 12 clusters (4 clusters for
each of 3 geo-regions).

### Cluster Interpretation

Overall, the clustering process was dominated by watershed areas.
Because watershed area is known to account for the majority of variance
in flow (National Research Council Canada, 1989), this is generally
analogous to grouping rivers by their flow magnitude, though flow is not
directly included in the process to allow for clustering of unmonitored
rivers for prediction.

Cluster labels are generated arbitrarily, and it was found that Cluster
1 corresponds to very large watersheds (mean 955,000 km2), Cluster 2 to
large watersheds (mean 307,000 km2), Cluster 4 to medium watersheds
(mean 75,600 km2), and Cluster 3 to the smallest watersheds (mean 2,240
km2). It is worth noting that luster 3 represents approximately 90% of
the training dataset, and is generally the cluster of greatest interest,
as smallest rivers are least likely to be gauged. Clustering is not
directly equivalent to grouping by size, as variance in all features is
considered. However, many features are highly collinear with area. For
example, mean centroid elevation also varies significantly between
clusters, but this can be explained by considering that the largest
watersheds tend to have the lowest elevation, as smaller, more alpine
watersheds tend to be nested within large watersheds in lower valleys.

![Figure A13: Geographic Distribution of Gauges by Cluster
Label](../Clustering-Exploration/img/clust-map.png)

Caption: Geographic distribution of gauges, colored by cluster label.
Arbitrary cluster labels (1-4) have been replaced by relative watershed
area, and scaled for visual clarity.

Data Reference: `reports/Clustering-Exploration/clustering-map.ipynb`

This figure shows the geographical distribution of gauges by watershed
size. The large rivers can be easily oriented to major Canadian rivers,
such as the upper left ‘very large’ gauges being the Slave River, the
central parallel orange ‘large’ rivers being the Churchill and
Saskatchewan rivers, etc. This map also illustrates the nesting aspect
of some gauges described in the [Appendix: Nested Gauge
Structure](#appendix-limitations-nested), as many large rivers are
gauged in multiple places, and the observed flow of the downstream
gauges are clearly not independent of the upstream gauges.

It was found later in the project that the resolution of this clustering
is probably not sufficient to meet the requirements of the IFM (see
[Appendix: Unable to Achieve High Homogeneity in
Clustering](#appendix-limitations-hindex)). This was unable to be
adjusted due to time constraints, and the necessity of refitting
significant portions of the existing pipeline. However, this coarse
clustering provided a basis from which to explore new methods, along
with ensuring physical interpretability of clusters to guide downstream
decision making. Refined clustering could potentially further increase
the reliability of the IFM approach detailed here and is a route for
further exploration.

### Criterion: Standard Deviation of Residual Ratios

The initial criterion used to evaluate clusters was the homogeneity of
the standard deviation of residual ratios at each gauge. Gauges with
similar spread of residual ratios can be theorized to have similar
frequency magnitude curves. Gauges with large spread take on larger
values at the highest quantiles, with the reverse being true of gauges
with small spread, and so gauges that are identified as having similar
features and sharing a growth curve by the clustering algorithm should
also have similar standard deviations.

The figure below shows the initial results of this criterion. Standard
deviation is plotted as error-bars around a fixed value of 0. The y-axis
is in terms of residual ratio and is shared between all plots to allow
for comparison. When generating clusters, these plots were used as a
visual check, with the intention being to observe similar magnitudes of
standard deviation within a cluster, and different magnitudes of
standard deviation between the clusters. Initial exploration using elbow
plots suggested three or four clusters, and it was found that increasing
the number of clusters beyond four did not meaningfully improve the
visuals of this plot.

It can be observed on this plot that the largest rivers (cluster 1) have
the smallest standard deviation, and the smallest rivers (cluster 3)
have the largest. This has a reasonable physical interpretation in terms
of seasonal precipitation. The small cluster 3 rivers have relatively
small watersheds, and so can be significantly affected by a particularly
heavy period of local rain, or larger than normal snowmelt.
Additionally, a difference of even a few cubic meters per second of flow
would be represented as a large residual ratio for the smallest rivers
due to their small mean. Conversely, the very large watersheds of the
cluster 1 rivers would require higher than normal precipitation across
exceedingly large areas, which is much more unlikely. Similarly, an
increase of tens or even hundreds of cubic meters per second is required
to produce comparably large residual ratios as are observed in cluster 3
rivers.

![Figure A14: Standard Deviation of Residual Ratios by
Cluster](../Clustering-Exploration/img/std-dev.png)

Caption: Standard Deviation of Residual Ratios by Cluster (y-axis).
X-axis sorts by magnitude of flow. Similar y-axis magnitudes within each
cluster and different magnitudes between each cluster is desirable.

Data reference: `reports/Clustering-Exploration/clustering-expl.ipynb`

### Criterion: Distribution of Variance Ratios

Another criterion developed for this clustering approach was related to
whether the variability of within-gauge variances in each group was as
expected under homogeneity, or whether the spread was such that the
observed differences are more likely due to heterogeneity.

If the group is truly homogenous, then the residual ratio values of each
gauge can effectively be considered a random sample of the overall
group. These random samples have the same expected variance as the
overall group, with some variability due to sampling error. This
variability should have specific distributional properties, and if the
observed variability of gauges deviates from the expected distribution,
we can conclude the observed difference is due to heterogeneity.

The criterion shown below was derived from Cochran’s Theorem, being
specifically applicable to normally distributed random variables.
Observations of the distribution of log(flow) for gauges show that they
are roughly symmetrical and bell-shaped, and for gauges with sufficient
number of observations the Central Limit Theorem allows for a reasonable
assumption of normality.

$\frac{(n-1)\sigma^2_{gauge}}{\sigma_{group}^2} \sim \chi^2_{n-1}$

This provides us a variance ratio statistic and corresponding
distribution that can be computed for each gauge. The closer the
observed distribution of the statistics are to this theoretical
distribution, the closer the variances of each gauge are to the overall
variance, and the more homogenous the group.

In order to validate this, the non-exceedance probability of each
statistic value for its corresponding $\chi^2_{n-1}$ distribution is
computed. If the group is homogenous, these computed probabilities
should be Uniform(0,1) distributed. To apply a visual check, the
observed probabilities were plotted against the quantiles of the
uniform(0,1) distribution. The resulting P-P (Probability/Probability)
plot should lie as close as possible to the diagonal line. An example is
shown below. This criterion was specifically used to determine that
dividing rivers by geo-region produced more homogenous clusters.

![Figure A15: P-P Plots for overall data vs prairie region
only](../Clustering-Exploration/img/pp-overall.png)
![](../Clustering-Exploration/img/pp-prairie.png)

Caption: P-P Plots for overall data vs prairie region only, plotting
observed $\chi^2$ non-exceedance probabilities against the uniform (0,1)
distribution. A fit along the diagonal red line indicates homogeneity.

Data reference: `reports/Clustering-Exploration/clustering-expl.ipynb`

Distribution of probabilities for all gauges pooled together is observed
to be significantly farther from the diagonal than probabilities for all
prairie gauges, indicating the prairies are closer to homogeneity. This
supports subdividing gauges by region.

### Criterion: Physical/Principal Component Analysis (PCA) Interpretability

The physical interpretability of clusters was considered a significantly
important criterion to gain confidence in the new approach. It was
considered preferable to have significant physical evidence for the
separation of clusters, rather than a fine resolution of clustering,
with largely arbitrary labels and no clear way to distinguish between
them. This was done partially through evaluating the mean feature values
within each cluster, but also through Principal Component Analysis (PCA)
visualization.

PCA can provide a visual ‘summary’ of the similarities and differences
of highly multidimensional data points. In this case, 51 numeric
watershed characteristics are summarized by two latent feature vectors,
which cumulatively account for 55% of the total variation observed in
the data. The PCA feature values of individual gauges can be plotted in
two dimensions, with small distances between gauges indicating
similarity, and vice versa.

![Figure A16: PCA Visualization](../Clustering-Exploration/img/pca.png)

Caption: Plot showing latent similarities between gauges in each cluster
label. Feat2 (y-axis) corresponds mostly to watershed area, while Feat 1
(x-axis) corresponds mostly to precipitation. Cohesive groups with
minimal mixing is desirable.

Data reference: `reports/Clustering-Exploration/clustering-expl.ipynb`

The figure above shows the results of PCA on all gauges, along with
corresponding colored markings for their hierarchical cluster labels (n
= 4). Analysis of the feature weights showed that Feat2 (y-axis) is
largely correlated with the watershed area, and Feat1 (x-axis) is
largely correlated with precipitation. The relative dominance of
watershed area in the clustering process is visible here by the
horizontal color banding.

This plot represents a positive clustering outcome under this criterion,
with each hierarchical cluster label being also relatively grouped
together by the PCA algorithm, with minimal mixing between the clusters.
Experimentation with other cluster numbers, or clustering on
regionalized data, resulted in PCA plots with unclear boundaries and/or
significant mixing of labels, indicating less interpretable clusters.

t-SNE (t-distributed stochastic neighbor embedding) clustering was also
explored as an option for this criterion but was not found to be clearer
than PCA plotting. PCA plotting was selected due to the added benefit of
feature interpretability for the latent vectors.

### Criterion: Heterogeneity Index (Hosking and Wallis, 1997)

The heterogeneity test outlined in Regional Frequency Analysis (M. and
Wallis 1997) chapter 4.3 is the current standard criterion for group
homogeneity used in hydrological practice. This was unfortunately not
known to the MDS team until late in the project and would have
significantly changed the clustering approach taken.

The heterogeneity index, or H-Index, is a test for how heterogenous a
group is using the method of l-moments. Even if a group of gauges is
truly homogenous, with their residual ratio of flows being drawn from
the same underlying distribution, their observed l-moments will not
necessarily be the same due to sampling error. The H-index compares the
spread of observed L-CVs (L-coefficient of variation, observed variance
divided by observed mean) in a group to the L-CVs of a simulated
homogenous population, generated from a kappa distribution, with the
parameters being the mean l-moments of the observed group.

It is computed as follows, with $V$ being the standard deviation of
observed L-CVs weighted by observations per gauge, $\mu_V$ being the
mean $V$ generated by the simulation, and $\sigma_V$ being the standard
deviation of $V$ values generated by the simulation.

$H = \frac{V- \mu_V}{\sigma_V}$

H \< 2 is desirable to establish ‘possibly homogenous’ clusters.

The H-index was used as evidence confirm that dividing by region does
yield more homogenous clusters, and that generally the boreal shield is
the most heterogenous of the clusters in this investigation, with the
plains and prairies having more homogenous watersheds, which matches the
hydrological understanding.

However, H-index also shows that the resolution of clustering undertaken
in this analysis was not sufficient to validate the IFM assumptions,
particularly for cluster 3.

![Table A7: H-Index Per Region and
Cluster](../Clustering-Exploration/img/h-index.png)

Caption: Table showing H-index values for each region/ cluster
combination. Some combinations are missing due to insufficient data.

Data reference: `reports/Clustering-Exploration/h-index/h-index.ipynb`

The H-indexes of the region/cluster combinations are shown above. It is
worth noting that the standard of H \< 2 may be too harsh in practice,
as the only place this is validated is in Shield Cluster 2, which is
composed of multiple gauges along a single river.

Additionally, in normal practice, a discordancy test is administered,
and gauges that negatively affect the H-index of their corresponding
cluster can be removed from the training set. However, this should be
viewed as a fine-tuning measure and would not sufficiently improve these
scores.

### Classifying Clusters

When receiving new gauges, it is necessary to assign them a cluster
label to pass them to the correct growth curve. Due to the lack of
‘prediction’ capacity in the hierarchical clustering algorithm, it is
necessary to re-cluster each new gauge – effectively adding the new
gauge to the training data and reapplying the clustering algorithm.

It was found through experimentation that the difference of a single
data point does not appear sufficient to affect the structure of the
clusters – which would be a concern because the labels 1-4 are
arbitrary. To ensure consistency, when predicting multiple gauges with
the script, each is re-clustered separately to prevent the addition of
multiple gauges distorting clustering.

One issue of note is that providing very large numbers (i.e. all values
100,000, when many features are proportions ranging from 0-1) is
sufficient to restructure the clusters and generate an erroneous
arbitrary label. This type of error is potentially realistic for unit
scaling errors, data mis-entries, or other issues.

### Clustering Recommendations and Limitations

Unfortunately, due to lack of time, as well as significant downstream
dependencies on early clustering, the MDS team was not able to refit
clusters in accordance with the H-index criterion. Despite this, the
proposed IFM still shows significant promise, and could likely be
further improved by increasing the number of clusters.

It is important to note that there is a significant trade-off between
establishing homogeneity within clusters and maintaining the benefits of
pooling. The literature criterion of H \< 2 is likely too strict, as the
size of clusters required to meet this standard would be extremely small
and would likely result in many clusters containing only single gauges.
This effectively compromises the benefits of the IFM, as the intention
is to pool together data from multiple gauges to create a more reliable
estimation of the growth curve. Additionally, while predicting cluster
labels of new gauges is reliable for small numbers of clusters, it is
likely far more challenging for large numbers of clusters.
Interpretability of clusters would also be significantly decreased.

Overall, the H-index suggests that finer clustering is necessary to
increase homogeneity and improve the reliability of the residual ratio
distributions. This is a definite step that could be taken to further
improve the predictions generated by the MDS Team’s approach.
Discordancy checks could also be run to fine tune clusters by removing
problematic gauges. However, a balance must be struck to ensure the
value of pooling is maintained. Even if gauges in a cluster are not
fully homogeneous, they may at least be similar enough to justify their
grouping into a single cluster. Realistically, no two watersheds are
truly homogenous, and some relaxation of this IFM assumption appears
necessary for it to function. An alternative method that allows for
partial relaxation of the IFM assumption is also detailed in the
[Expanded Future Improvements](#appendix-improvements). However, it is
evident that this assumption was over-relaxed in the MDS Team’s current
approach, and there is significant potential benefit to finer
clustering.

## Appendix: Expanded Limitations and Future Improvements

### Expanded Limitations

#### Data Structure

##### Variable Amounts of Data at Gauge Level

The current ML algorithm for training the index flood prediction model
within the Index Flood Method does not directly account for the variable
number of years of data across different gauges. The issue with this is
that there may be a representation bias present in the data, where
well-gauged rivers are very well represented (longer data record). A
variability in the amount of data available per gauge may also lead to
unintended issues with flow variance at gauge level, as smaller sample
sizes may have larger variance due to observing a high return period
flood by chance. Our algorithm depends on the flow variance at gauge
level to train and tune models correctly (see [IFM Evaluation Metric
Selection](#appendix-ifm-eval-metric) section).

##### Grouped Data and Train-Test Splits

The data is structured in a “grouped” or “hierarchical” fashion, where
multiple years of data are available for each gauge, and there are
multiple gauges. To avoid leakage of training data into the test data,
data within a gauge cannot be split between the train and test set. A
consequence of this is that a 70% train / 30% test data split refers to
a 70-30 split at the gauge level only. For example, it could be possible
due to random chance that a 70% train split is equivalent to only 40% of
the actual rows of data if those 70% of gauges all happen to be gauges
with very few years of data. A symptom of this is that weighted-RMSE
scores in the test set are much higher than cross-validation scores, but
are vastly improved when the size of the test set is increased. This
shows that scores are sensitive to the amount of data being allocated to
the train versus test split. In the future, different sizes of train and
test split could be experimented with more thoroughly.

##### River Flow Skewness

River flow varies across multiple orders of magnitude, with river flows
ranging between $10_{-2}$ and $10^3$ m3/s. As a result, flows must be
analyzed on a log base-10 scale. Using a log scale helps to address the
wide-ranging scale of the data, but it does not fully solve all
problems.

##### Nested Gauge Structure

It is possible for a large watershed to contain multiple smaller
watersheds. As an extension of this, it is possible for the large
watershed to have one gauge, and the smaller watersheds within it to
also all have one gauge. This results in a nested data structure, where
river flow measurements across different gauges can no longer be
considered independent. This could be addressed via a data cleaning
exercise to eliminate known nested gauges where the area of the
different watersheds is very similar. There will always be some degree
of nesting in river flow data, as rivers flow toward the ocean and
combine to form larger rivers. It would be up to the domain experts at
BGC to determine what level of nesting is acceptable. A similar issue
arises when considering multiple gauging, as the largest rivers in the
dataset are often gauged in multiple locations. In the case above where
many small rivers contribute to a large river, the large river is only
partly dependent on each of the small rivers. However, with sequential
gauges, the downstream gauge observations may be fully dependent on the
upstream observations, with the difference of watershed area between the
gauges accounting for only a small fraction of the observed flow.

##### Testing Dataset Composition

The test set for model evaluation was constructed randomly. However,
there may have been a benefit to experimental ‘blocking’ by manually
constructing test sets to ensure that relevant subgroups (i.e. clusters,
regions) were equally represented in the test set. However, due to the
extensive manual effort required to create multiple large test sets,
this was not pursued.

#### Algorithmic and ML Limitations

##### Quantile Scores not Scaled to River Size

Quantile loss is used for model training in the quantile prediction
method. Quantile loss is a linear function of error magnitude. Larger
rivers with larger flow magnitudes will tend to have larger errors, and
therefore larger quantile scores. When training, minimization of large
river error will be favored. This could be addressed by introducing a
weighting parameter, similar to the strategy implemented in the Index
Flood Method, but is not as drastic as in IFM because it is linear, not
quadratic.

##### No Accessible “Ground Truth” for DQM

There is no easily accessible baseline for comparison in the DQM
approach. The conventional direct quantile prediction method requires an
experienced hydrologist to manually cluster and collect similar rivers.
This is difficult to recreate at scale for all rivers at the same level
of efficiency as the ML models.

##### Index Flood Method: Challenging Prairie Data

The prairie region is the worst-performing region for the index flood
prediction, with the worst numerical scores and the largest spread in
the predicted vs actual plots. This might be attributed to the absence
of suitable factors that properly explain the prairie hydrology.
Discussions with BGC hydrologists suggest that region specific features
such as the Prairie Potholes have significant influence over prairie
hydrology, and their influence may not be fully captured by our
available watershed features. Without these factors, the ML model is not
able to properly explain the variance in the annual maximum flows.

#### Statistical Assumptions

##### Unable to Achieve High Homogeneity in Clustering

Index flood model still relies on homogeneity. While we use
regionalization and clustering to improve, it isn’t fully achieved by
our clustering under the H-index criterion. This criterion was
unfortunately not known to the MDS team until late in the project, and
there were already many downstream dependencies to the early clustering
regime. Further improvements to the quantile predictions of the IFM
could be achieved by applying hierarchical clustering with a greater
number of clusters. It should be noted that this may introduce problems
with the statistical benefits of pooling, physical interpretability of
clusters, and prediction of cluster labels for new rivers. See Regional
Homogeneity, Clustering and H-Index appendix for a thorough discussion.

##### Overprediction / Choice of Statistical Distribution

The figures below visualize IFM gauge-level predictions of the RFFA
models compared to the FFA benchmark at the design flood (0.5% AEP). Low
dispersion around the diagonal line is ideal, indicating that the RFFA
prediction is close to the FFA benchmark. There is slight evidence of
lower dispersion in the ML model figure. This indicates the ML model
predictions are generally closer to the FFA benchmark than predictions
made by the conventional RFFA modeling process. The figures also show
evidence of overprediction by the ML model compared to the FFA, this is
discussed as a limitation in the [Appendix: Overprediction / Choice of
Statistical Distribution](#appendix-overprediction) section.

![Figure A17: Comparison of FFA Predictions to RFFA Predictions for
River Flow
Quantiles](../ML-exploration/ma_plots/ffa_vs_rffa_plot_grid.png)

Caption: Comparison of RFFA vs FFA predictions at gauge level for the
design flood (0.5% AEP). Chart show evidence of slightly lower
dispersion around diagonal in ML models vs baseline. Charts show
evidence of overprediction in ML models vs baseline model.

Data reference: `reports/ML-exploration/ML-approach1-ma.ipynb`

FFA vs RFFA plot shows that our model over-predicts at high quantiles,
while the baseline model exhibits underprediction at high quantiles.
This could be due to the use of Log Pearson Type III for residual
ratios. Log Pearson Type III exhibited the best fit via mean quantile
score for quantiles of interest, but it may be an overpredictor for
higher quantiles. In the short term this could be addressed by
quantifying the degree of bias and applying a scale factor, but further
investigation is needed.

##### Distribution Selection Methodology

The most appropriate statistical distribution was fit to the residual
ratio data based on the highest mean quantile score. There are other
ways to determine the best distribution, including AIC and BIC criteria,
QQ and PP plots, and tests such as the Kolmogorov-Smirnov and
Anderson-Darling tests.

#### Data Product Limitations

##### Limited Manual Intervention Ability

There is currently no functionality in the data product for domain
experts to manually intervene in the prediction pipeline in any way. For
example, if end users wish to remove a particular gauge from the
available data, they would need to remove it from the raw data, re-train
the ML models, override saved models, then re-run the prediction
scripts. To avoid filtering input data, users could instead modify the
ML model training code to include functionality to customize training
weights in the dataset (increase importance or omit gauges). This could
take the form of a new function argument that takes an array of length
X_train, defaulting to all 1s. Any gauges to exclude could be given a
value of 0; gauges that should be 50% weighted could be given a value of
0.5, etc. Multiply variance weight scores by this array to get final
sample weights. Regardless of the solution implemented, manual
intervention requires re-training models and possibly redoing
hyperparameter tuning.

##### Model Training not Scripted

The model training process was not scripted, but instead models were
iteratively developed within notebooks. If end users wish to re-train
the machine learning models explored in this analysis, they will need to
rerun the `.ipynb` notebooks used to create the models.

##### Limited DQM Hyperparameter Tuning

Hyperparameter tuning takes significant processing time. Per ML model,
it can take between 10 to 90 minutes to complete hyperparameter tuning.
The Direct Quantile Method requires one ML model per quantile of
interest. This can quickly inflate the number of ML models that require
hyperparameter tuning. As a result, hyperparameter tuning was not
explored in this project for DQM. Hyperparameter tuning has the capacity
to greatly improve the prediction accuracy of the quantile prediction
models, and should be explored in the future.

### Expanded Future Improvements

#### Mixed Effects Model

Explore a mixed effects model with year as its own effect. Some years
have more data due to more gauges in operation. Peaks in these years may
be very similar (i.e. a very rainy year) and so may be overrepresented
in the dataset. This was not explored in this analysis due to time
constraints.

#### Data Augmentation

Implement data augmentation by regression on daily maxima vs
instantaneous maxima. This analysis is based solely on instantaneous
daily maxima, which is more effective for FFA than daily average maxima.
However, there are techniques to estimate the instantaneous daily maxima
from the average daily maxima when instantaneous is not available. This
was not explored in this analysis due to time constraints and the
presence of a suitable amount of instantaneous daily maxima data.

#### Visual Outlier Detection

Could visualize when a watershed is “unusual” with respect to
clustering. This could be visualized using principal component analysis
plots (see [Appendix: Regional Homogeneity, Clustering and
H-Index](#appendix-clustering)), with the new watershed’s location in
the plot being highlighted to determine if its cluster label appears
appropriate, and that the watershed’s features are well represented in
the training data. Alternatively, a different clustering method able to
identify outliers (i.e. DBSCAN) could be applied to flag input rivers as
potential outliers. DBSCAN is probably not a suitable replacement for
hierarchical clustering in the creation of pooled distributions due to
the flexibility of hierarchical clustering, and the necessity of equal
density clusters for DBSCAN, which we do not have. However, if a new
data point is flagged as an outlier by an appropriately tuned DBSCAN
algorithm, this could be a sign that it is an outlier.

#### New Clustering Techniques

Use hierarchical clustering approach with finer clusters (greater ‘n’)
to increase the validity of the assumption of homogeneity. See
[Appendix: Regional Homogeneity, Clustering and
H-Index](#appendix-clustering).

#### New Quantile Modelling Paradigm

Given the difficulties with establishing truly homogenous clusters in
the IFM (see [Appendix: Regional Homogeneity, Clustering and
H-Index](#appendix-clustering)), this suggests that there may be room
for an intermediate model between the current IFM and DQM. Under the
current IFM, it is assumed that the only difference between grouped
gauges is the index flood value, whereas under the DQM, there are no
assumed similarities between gauges. An intermediate method could assume
that the geometric mean, variance, and possibly skewness (first three
l-moments) are different for grouped gauges. Like the index flood, these
values could be included as ML predictions. With these quantities
allowed to vary within groups, it would be far easier to satisfy the
IFM’s assumption of homogeneity. This is a novel approach that is not
represented in the literature, and so would take extensive
experimentation and theoretical validation to convince stakeholders of
its validity, and so is merely suggested as a future avenue of further
research.

#### Continuous DQM Predictions

Could implement a continuous prediction for DQM, similar to that of lFM.
Due to the nature of quantile prediction models, it is necessary to
either train a new model for each quantile, or to pre-specify a list of
quantiles for DQM. This means that the data product can only output
predictions for set, pretrained quantiles for DQM. It is possible to
create a continuous prediction by training more quantile models and
fitting a probability distribution to the resulting values, but this was
not implemented due to time constraints. More quantile models can be
trained using training notebooks.

# References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-EGBC_2018" class="csl-entry">

EGBC. 2018. “Legislated Flood Assessments in a Changing Climate in BC.”
*EGBC*.
<https://www.egbc.ca/getmedia/f5c2d7e9-26ad-4cb3-b528-940b3aaa9069/Legislated-Flood-Assessments-in-BC.pdf>.

</div>

<div id="ref-NRC_1989" class="csl-entry">

*Hydrology of Floods in Canada*. 1989. National Research Council,
Canada, Associate Committee on Hydrology.

</div>

<div id="ref-M_Wallis_1997" class="csl-entry">

M., Hosking J R, and James R. Wallis. 1997. *Regional Frequency Analysis
an Approach Based on l-Moments*. Cambridge University Press.

</div>

</div>
