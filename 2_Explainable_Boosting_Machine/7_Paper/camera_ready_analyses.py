import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EBM = HERE.parent
sys.path.insert(0, str(EBM / '0_Shared_Data_And_Splits'))
import ebm_workflow as wf 

LEVEL_RUNS = EBM / '3_Level_Prediction/1_Intermediate_Calculations'
TABLES = HERE / '1_Tables'
VALIDITY_FOLDERS = sorted(EBM.glob('*_Validity_Checks'))
CI_FAMILIES = {'all features': 'combined', 'main': 'combined_without_length_correlated_features'}
LENGTH_FAMILIES = ('combined', 'combined_without_length_correlated_features',
                   'combined_without_length_correlated_and_duplicate_features', 'length_only')
WORDS = 'complexity__LCA__wordtokens'
LEVELS = list(range(1, 16))


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def write_csv(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f'{path}.tmp')
    table.to_csv(temporary, index=False)
    wf.os.replace(temporary, path)
    log(f'Saved {path}')


def read_predictions(run):
    return pd.read_csv(run / 'test_predictions.csv', dtype={'cefr_level': str, 'predicted_cefr': str}) \
        .sort_values('text_id', kind='stable').reset_index(drop=True)


def confidence_intervals(manifest):
    """Per-class and macro test F1 of the level classifiers with 95% learner-clustered bootstrap intervals."""
    rows = []
    for features, family in CI_FAMILIES.items():
        run = LEVEL_RUNS / f'classifier__{family}'
        predictions = read_predictions(run)
        frame = predictions[['text_id', 'learner_id']].assign(split='test')
        resamples = wf.run_resamples(run, 'test', frame, manifest, 'level')
        original = wf.resample_metrics(np.ones((1, len(predictions))), wf.prepare_resampling(predictions, manifest, 'level'))
        for target, truth, labels in (('level', 'level', LEVELS), ('cefr', 'cefr_level', manifest['labels'])):
            per_class = resamples[target]['labels']['f1-score']
            for column, label in enumerate(labels):
                texts = predictions[truth] == label
                rows.append({'features': features, 'target': target, 'class': label, 'texts': int(texts.sum()),
                             'learners': int(predictions.loc[texts, 'learner_id'].nunique()),
                             'f1': original[target]['labels']['f1-score'][0, column],
                             'bootstrap_sd': wf.spread(per_class[:, column]),
                             'ci_low': np.percentile(per_class[:, column], 2.5),
                             'ci_high': np.percentile(per_class[:, column], 97.5)})
            macro = resamples[target]['macro_f1']
            rows.append({'features': features, 'target': target, 'class': 'macro', 'texts': len(predictions),
                         'learners': int(predictions.learner_id.nunique()), 'f1': original[target]['macro_f1'][0],
                         'bootstrap_sd': wf.spread(macro), 'ci_low': np.percentile(macro, 2.5), 'ci_high': np.percentile(macro, 97.5)})
        log(f'{features}: intervals from {len(macro):,} learner resamples')
    return pd.DataFrame(rows)


def ranked_correlation(group):
    return group.prediction.rank().corr(group.words.rank())


def length_check(words):
    correlations, fifths = [], []
    for family in LENGTH_FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            run = LEVEL_RUNS / f'{algorithm}__{family}'
            if not (run / 'test_predictions.csv').is_file():
                log(f'skipped {algorithm}__{family}: no saved test predictions')
                continue
            table = read_predictions(run).merge(words, on='text_id', validate='one_to_one')
            table['prediction'] = (sum(table[f'probability_level_{k}'] * k for k in LEVELS) if algorithm == 'classifier'
                                   else table.predicted_level_continuous)
            per_level = table.groupby('level').apply(ranked_correlation, include_groups=False)
            sizes = table.level.value_counts().loc[per_level.index]
            key = {'family': family, 'algorithm': algorithm}
            correlations.append({**key, 'level': 'all (weighted)', 'texts': len(table),
                                 'spearman_prediction_words': float(np.average(per_level, weights=sizes))})
            correlations += [{**key, 'level': int(level), 'texts': int(sizes[level]), 'spearman_prediction_words': value}
                             for level, value in per_level.items()]
            table['fifth'] = table.groupby('level').words.transform(lambda s: pd.qcut(s.rank(method='first'), 5, labels=False) + 1)
            table['error'] = table.prediction - table.level
            for fifth, group in table.groupby('fifth'):
                fifths.append({**key, 'word_count_fifth': int(fifth), 'texts': len(group), 'median_words': group.words.median(),
                               'mean_signed_error': group.error.mean(), 'share_too_low': (group.error < -0.5).mean(),
                               'share_too_high': (group.error > 0.5).mean(),
                               'accuracy': (group.predicted_level == group.level).mean()})
    return pd.DataFrame(correlations), pd.DataFrame(fifths)


def main():
    if len(VALIDITY_FOLDERS) != 1:
        raise ValueError(f'Expected one validity-check folder next to {HERE.name}, found {len(VALIDITY_FOLDERS)}.')
    manifest = wf.read_json(wf.SHARED / 'dataset_manifest.json')
    data = wf.resolve_manifest_path(manifest['data_directory'])
    with duckdb.connect() as connection:
        words = connection.execute(f'SELECT text_id, "{WORDS}" AS words FROM read_parquet({wf.literal(data / "feature_dataframe.parquet")})').df()
    write_csv(TABLES / 'test_confidence_intervals.csv', confidence_intervals(manifest))
    correlations, fifths = length_check(words)
    write_csv(VALIDITY_FOLDERS[0] / '2_Length_Check' / 'within_level_length_check.csv', correlations)
    write_csv(VALIDITY_FOLDERS[0] / '2_Length_Check' / 'within_level_length_fifths.csv', fifths)


if __name__ == '__main__':
    sys.exit(main() or 0)
