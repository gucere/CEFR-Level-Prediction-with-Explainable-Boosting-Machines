# CEFR Level Prediction with Explainable Boosting Machines

The paper *CEFR Level Prediction with Explainable Boosting Machines* (submitted to NLP4CALL 2026), with its code and aggregated results.

The paper trains Explainable Boosting Machines (EBMs, InterpretML 0.7.8) on linguistic complexity and accuracy features of 406,062 texts from the cleaned EFCAMDAT subcorpus (main prompts) to predict the course level (1–15) and CEFR band (A1–C1). It reports:

- performance for held-out learners (97.7% CEFR accuracy, 94.4% exact-level accuracy on the test set) and in ten-fold cross-validation;
- the effect of excluding direct length measures, other raw counts, length-dependent measures and near-duplicate features;
- the contributions of five linguistic dimensions (lexical sophistication, lexical diversity and density, syntactic complexity, grammatical constructions, accuracy) and of the six tools, overall and per course level;
- generalisation to writing tasks withheld from training, where exact-level macro F1 falls to 0.19 and CEFR accuracy to 0.71.

## What is (and is not) in this repository

Included: the paper itself (LaTeX source, bibliography, figures and the submitted PDF in [3_NLP4CALL_Submission](2_Explainable_Boosting_Machine/7_Paper/3_NLP4CALL_Submission)), all our code (feature-extraction wrappers, dataframe construction, the EBM pipeline, cross-validation, validity checks and the script that builds the paper tables), the feature definitions and exclusion lists, and all aggregated results behind the paper.

Not included, because of the data licence and third-party rights:

- the EFCAMDAT texts, and anything with one row per text or learner (per-text feature values, predictions, the learner split and the cross-validation folds);
- the trained models;
- the code of the analysis tools (TAALES, LCA, L2SCA, TAASSC, POLKE) and the EGP catalogue text. Each tool folder has a note on where to get the tool and where to place it.

The EFCAMDAT data can be requested from the EF Research Lab (<https://ef-lab.mml.cam.ac.uk/EFCAMDAT.html>). We use the cleaned subcorpus of Shatz (2020), file *Final database (main prompts)*.

## Where the paper's results come from

| Paper content | File(s) |
| --- | --- |
| Data and splits | [table_1_data.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_1_data.csv), [level_and_cefr_counts.xml](0_Data/level_and_cefr_counts.xml) |
| Feature sets and their sizes | [table_2_feature_sets.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_2_feature_sets.csv), the three `blacklist_*.csv` lists in [1_Complexity_Analysis_Full_Module](1_Complexity_Analysis_Full_Module) |
| Validation and test results of all 16 configurations | [table_3_validation_results.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_3_validation_results.csv), [table_5_test_results.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_5_test_results.csv) |
| Per-band and per-level test results | `per_cefr_metrics_level_all_experiments.csv` and `per_level_metrics_level_all_experiments.csv` in [3_Level_Prediction/2_Prediction_Performance/0_All_Experiments](2_Explainable_Boosting_Machine/3_Level_Prediction/2_Prediction_Performance/0_All_Experiments) |
| Cross-validation (means, SDs, corrected t-tests, per-class results) | [5_Cross_Validation/3_Results](2_Explainable_Boosting_Machine/5_Cross_Validation/3_Results), [table_4_cross_validation.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_4_cross_validation.csv) |
| Dimension and tool shares | `cv_dimension_shares.csv`, `cv_tool_shares.csv` in [5_Cross_Validation/3_Results](2_Explainable_Boosting_Machine/5_Cross_Validation/3_Results); per fold in [5_Cross_Validation/2_Jobs](2_Explainable_Boosting_Machine/5_Cross_Validation/2_Jobs) |
| Stable individual features | `cv_feature_stability.csv`, [table_7_top_features.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_7_top_features.csv) |
| Performance by text length | [table_8_length_check.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_8_length_check.csv), [performance_by_text_length.csv](2_Explainable_Boosting_Machine/6_Validity_Checks/2_Length_Check/performance_by_text_length.csv) |
| Unseen writing tasks | [table_9_topic_check.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_9_topic_check.csv), [table_10_leave_one_task_out.csv](2_Explainable_Boosting_Machine/7_Paper/1_Tables/table_10_leave_one_task_out.csv), [6_Validity_Checks](2_Explainable_Boosting_Machine/6_Validity_Checks) |
| Figures | [7_Paper/2_Figures](2_Explainable_Boosting_Machine/7_Paper/2_Figures) |
| Feature contributions of every model and output | `feature_contributions_*_all_experiments.csv` in the two `3_Feature_Contributions/0_All_Experiments` folders |

All paper tables are also collected in [paper_tables.md](2_Explainable_Boosting_Machine/7_Paper/1_Tables/paper_tables.md). The manuscript compiles with the official NLP4CALL template files included in its folder (`nlp4call.sty`, `acl_natbib.bst`) and the `xurl` package, for example on Overleaf. The full pipeline additionally writes one report folder per model; this repository keeps the `0_All_Experiments` files, which combine them.

## Reproducing the results

Run the commands from the repository root. Two pinned environments are used (exact versions in [requirements](requirements)):

| Environment | Used for |
| --- | --- |
| Python 3.10 ([feature_extraction_python310.txt](requirements/feature_extraction_python310.txt)) | feature extraction (spaCy 3.8.14, `en_core_web_sm` 3.8.0, ERRANT 3.0.2) and the descriptive analyses |
| Python 3.13 ([ebm_and_llm_python313.txt](requirements/ebm_and_llm_python313.txt)) | the EBM pipeline, cross-validation, validity checks and paper tables |

1. **Data.** Place `Final database (main prompts).xlsx` in `0_Data` and run [Data_To_Txt_Files.py](0_Data/Data_To_Txt_Files.py) to export the original and corrected texts.
2. **Features.** Obtain the tools as described in the notes in each folder of [1_Complexity_Analysis_Methods](1_Complexity_Analysis_Full_Module/1_Complexity_Analysis_Methods) and run the `run_*.py` scripts there (TAALES through [TAALES-Reliable-Automation](1_Complexity_Analysis_Full_Module/1_Complexity_Analysis_Methods/TAALES/TAALES-Reliable-Automation), POLKE through `POLKE/polke-main/run_POLKE.py` inside the POLKE code). Then build the feature table:
   ```
   python "1_Complexity_Analysis_Full_Module/2_Basic_Analysis/0_Feature_Dataframe/create_dataframe.py"
   ```
3. **Models.** Train all feature configurations for both tasks and write the reports:
   ```
   python "2_Explainable_Boosting_Machine/run_ebm_pipeline.py"
   ```
   Learners are assigned to training, validation and test sets by `sha256("42:<learner_id>")`: the first 16 hex digits divided by 2^64 give a number below 0.70 for training, below 0.85 for validation and otherwise test.
4. **Cross-validation** (ten learner-grouped, level-stratified folds over training and validation texts, seed 42):
   ```
   python "2_Explainable_Boosting_Machine/5_Cross_Validation/cross_validation.py" prepare --tasks level --experiments combined combined_without_length_correlated_features majority median
   python "2_Explainable_Boosting_Machine/5_Cross_Validation/cross_validation.py" run --parallel 4
   python "2_Explainable_Boosting_Machine/5_Cross_Validation/cross_validation.py" summarize
   ```
5. **Validity checks and paper tables:**
   ```
   python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" length
   python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" topic --workers 1 --memory 1GB
   python "2_Explainable_Boosting_Machine/6_Validity_Checks/validity_checks.py" tasks --workers 1 --memory 1GB
   python "2_Explainable_Boosting_Machine/7_Paper/paper_tables.py"
   ```

A level classifier on the full training set takes about two hours on a laptop (i9-11900H, 16 GB), and an InterpretML fit briefly needs about 10 GB of memory near its end, so run at most two fits in parallel on such a machine. `cross_validation.py test` and `2_CEFR_Classification/cefr_classification.py test` check the procedures on small synthetic data.
