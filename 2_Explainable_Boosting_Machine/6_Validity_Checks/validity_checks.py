"""Validity checks of the main level models.

  python validity_checks.py topic      retrain on half of each level's writing topics; compare seen and unseen topics
  python validity_checks.py length     performance by text length, from the saved validation predictions
  python validity_checks.py tasks      leave one writing task per level out (8 folds) for the main model
  python validity_checks.py test       checks of the subset metrics

Topic check: in EFCAMDAT every writing topic belongs to one course level, so a model could learn to recognise
the writing task instead of judging the English. Each model is retrained on the training texts of 4 of the 8
topics per level (the split used in section 3, then the reverse) and evaluated on all validation texts, separately
for topics it saw and topics it did not see. The main models, trained on all topics, show how hard the two groups
of topics are anyway; the topic effect is the extra gap of the topic-restricted models.

Leave-one-task-out check: fold k retrains the main model on the training texts of 7 of the 8 tasks per level (all but
the k-th by topic number), so every validation text is predicted once by a model that never saw its task. Pooled over
the folds, this estimates performance on new writing tasks, comparable with the main model's validation results.
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '0_Shared_Data_And_Splits'))
import ebm_workflow as wf  # noqa: E402

POST_TRAINING = wf.ROOT / '3_Language_Level_Prediction_with_LLMs/Code/Post-Training'
TOPICS, HOLDOUT = POST_TRAINING / 'Data/text_topics.csv', POST_TRAINING / 'topic_holdout.json'
FAMILIES = ('combined_without_length_correlated_features', 'combined_without_length_correlated_and_duplicate_features', 'combined')
LENGTH_FAMILIES = ('combined', 'combined_without_length_correlated_features',
                   'combined_without_length_correlated_and_duplicate_features', 'length_only')
LENGTH_BINS = (0, 50, 100, 150, 200, np.inf)
TASK_FOLDS, TASK_FAMILIES = 8, ('combined_without_length_correlated_features',)
TOPIC_FOLDER, LENGTH_FOLDER, TASK_FOLDER = HERE / '1_Topic_Check', HERE / '2_Length_Check', HERE / '3_Leave_One_Task_Out'


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def write_csv(path, table):
    temporary = Path(f'{path}.tmp')
    table.to_csv(temporary, index=False)
    wf.os.replace(temporary, path)


def subset_metrics(predictions, manifest, algorithm):
    """The pipeline's metrics for a subset of the saved predictions of a level model."""
    labels = manifest['labels']
    metrics = {'texts': len(predictions)}
    for target, truth, predicted, names in (('level', 'level', 'predicted_level', list(range(1, 16))),
                                            ('cefr', 'cefr_level', 'predicted_cefr', labels)):
        values = wf.class_metrics(predictions[truth], predictions[predicted], names)
        metrics.update({f'{target}_{name}': values[name] for name in wf.CLASS_METRICS})
    if algorithm == 'regressor':
        errors = predictions.predicted_level_continuous - predictions.level
        metrics.update({'regression_mae': float(errors.abs().mean()), 'regression_rmse': float(np.sqrt((errors ** 2).mean()))})
    return metrics


def topic_groups():
    """Topic of every text, and the two topic groups: section 3's held-out topics and the others."""
    holdout = wf.read_json(HOLDOUT)
    if wf.digest(TOPICS) != holdout['topic_map_sha256']:
        raise ValueError(f'{TOPICS.name} differs from the topic map used for {HOLDOUT.name}.')
    topics = pd.read_csv(TOPICS, dtype={'topic_id': str}, encoding='utf-8-sig')[['text_id', 'topic_id']]
    held_out = set(holdout['held_out_topics'])
    return topics, {'section_3_held_out': held_out, 'section_3_seen': set(topics.topic_id) - held_out}


def topic_check(args):
    manifest, frame = wf.verify_shared(args.shared)
    topics, groups = topic_groups()
    frame = frame.merge(topics, on='text_id', how='left', validate='one_to_one')
    if frame.topic_id.isna().any():
        raise ValueError(f'{int(frame.topic_id.isna().sum())} texts have no topic.')
    definitions = wf.experiment_definitions(argparse.Namespace(shared=args.shared, task='level', use_ranked=False, ranking=None), manifest)
    validation = frame[frame.split.eq('validation')].sort_values('text_id')
    runs = TOPIC_FOLDER / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    rows = []
    for family in args.families or FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            for trained_on, topic_set in groups.items():
                name = f'{algorithm}__{family}__trained_on_{trained_on}'
                training = frame[frame.split.eq('train') & frame.topic_id.isin(topic_set)].sort_values('text_id')
                seen = validation.topic_id.isin(topic_set)
                log(f'{name}: {len(training):,} training texts; validation {int(seen.sum()):,} seen / {int((~seen).sum()):,} unseen topics')
                if args.dry_run:
                    continue
                config = {'dataset_id': wf.token(manifest), 'family': family, 'algorithm': algorithm, 'trained_on': trained_on,
                          'topics': sorted(topic_set, key=lambda topic: int(topic)), 'columns_sha256': wf.token(definitions[family]['columns']),
                          'rounds': args.rounds, 'min_observations': args.min_observations, 'seed': manifest['seed']}
                folder = runs / name
                marker = folder / 'completed.json'
                if not (marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)):
                    with wf.training_lock(folder), tempfile.TemporaryDirectory(prefix='.fit_', dir=runs) as temporary:
                        if not (marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)):
                            predictions = wf.fit_run(manifest, training, validation, definitions[family]['columns'], 'level', algorithm, family,
                                                     args.rounds, args.min_observations, args.workers, args.memory, Path(temporary), name)[4]
                            predictions = predictions.merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one')
                            folder.mkdir(exist_ok=True)
                            predictions.to_csv(folder / 'validation_predictions.csv', index=False)
                            wf.write_json(folder / 'config.json', config)
                            wf.write_json(marker, {'configuration_id': wf.token(config), 'training_texts': len(training)})
                predictions = pd.read_csv(folder / 'validation_predictions.csv', dtype={'topic_id': str, 'cefr_level': str,
                                                                                         'predicted_cefr': str})
                main = pd.read_csv(wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' /
                                   'validation_predictions.csv', dtype={'cefr_level': str, 'predicted_cefr': str}) \
                    .merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one')
                for model, table in (('topic_restricted', predictions), ('main_model_all_topics', main)):
                    for subset, mask in (('seen_topics', table.topic_id.isin(topic_set)), ('unseen_topics', ~table.topic_id.isin(topic_set))):
                        rows.append({'family': family, 'algorithm': algorithm, 'trained_on': trained_on, 'model': model,
                                     'topics': subset, **subset_metrics(table[mask], manifest, algorithm)})
    if args.dry_run or args.families or not rows:
        return  # the tables are written by a run over all models
    results = pd.DataFrame(rows)
    write_csv(TOPIC_FOLDER / 'topic_check_results.csv', results)
    metrics = [name for name in ('level_macro_f1', 'level_accuracy', 'level_quadratic_weighted_kappa', 'cefr_macro_f1',
                                 'regression_mae') if name in results]
    wide = results.pivot_table(index=['family', 'algorithm', 'trained_on', 'model'], columns='topics', values=metrics)
    gaps = pd.DataFrame({metric: wide[(metric, 'seen_topics')] - wide[(metric, 'unseen_topics')] for metric in metrics}).reset_index()
    # Averaged over both directions, so topics that are simply easier or harder cancel out.
    summary = gaps.groupby(['family', 'algorithm', 'model'], as_index=False)[metrics].mean()
    restricted = summary[summary.model.eq('topic_restricted')].set_index(['family', 'algorithm'])[metrics]
    reference = summary[summary.model.eq('main_model_all_topics')].set_index(['family', 'algorithm'])[metrics]
    effect = (restricted - reference).reset_index().assign(model='topic_effect: restricted gap minus main-model gap')
    write_csv(TOPIC_FOLDER / 'topic_check_gaps.csv', pd.concat([summary, effect], ignore_index=True).rename(
        columns={metric: f'{metric}_seen_minus_unseen' for metric in metrics}))
    log(f'Saved {TOPIC_FOLDER}')


def task_folds_of(frame):
    """Fold k (1 to 8) holds out the k-th writing task, by topic number, of every level."""
    levels = frame.groupby('topic_id')['level'].unique()
    if levels.map(len).ne(1).any():
        raise ValueError('A writing task belongs to more than one level.')
    tasks = {}
    for topic, level in levels.map(lambda values: int(values[0])).items():
        tasks.setdefault(level, []).append(int(topic))
    if any(len(topics) != TASK_FOLDS for topics in tasks.values()):
        raise ValueError(f'Every level needs {TASK_FOLDS} writing tasks.')
    return {fold: {str(sorted(topics)[fold - 1]) for topics in tasks.values()} for fold in range(1, TASK_FOLDS + 1)}


def run_task_fold(manifest, frame, validation, definitions, family, algorithm, fold, held_out, args):
    """Train on the training texts of the other 7 tasks per level; save the predictions for all validation texts."""
    name = f'{algorithm}__{family}__without_task_fold_{fold}'
    training = frame[frame.split.eq('train') & ~frame.topic_id.isin(held_out)].sort_values('text_id')
    unseen = validation.topic_id.isin(held_out)
    log(f'{name}: {len(training):,} training texts; validation {int((~unseen).sum()):,} seen / {int(unseen.sum()):,} unseen tasks')
    if args.dry_run:
        return None
    config = {'dataset_id': wf.token(manifest), 'family': family, 'algorithm': algorithm, 'fold': fold,
              'held_out_topics': sorted(held_out, key=int), 'columns_sha256': wf.token(definitions[family]['columns']),
              'rounds': args.rounds, 'min_observations': args.min_observations, 'seed': manifest['seed']}
    folder = TASK_FOLDER / 'runs' / name
    marker = folder / 'completed.json'
    done = lambda: marker.is_file() and wf.read_json(marker)['configuration_id'] == wf.token(config)
    if args.tables_only and not done():
        log(f'{name}: not finished yet; left out of the tables')
        return None
    if not done():
        folder.parent.mkdir(parents=True, exist_ok=True)
        with wf.training_lock(folder), tempfile.TemporaryDirectory(prefix='.fit_', dir=folder.parent) as temporary:
            if not done():
                predictions = wf.fit_run(manifest, training, validation, definitions[family]['columns'], 'level', algorithm, family,
                                         args.rounds, args.min_observations, args.workers, args.memory, Path(temporary), name)[4]
                predictions = predictions.merge(validation[['text_id', 'topic_id']], on='text_id', validate='one_to_one') \
                    .assign(unseen_task=lambda table: table.topic_id.isin(held_out))
                folder.mkdir(exist_ok=True)
                predictions.to_csv(folder / 'validation_predictions.csv', index=False)
                wf.write_json(folder / 'config.json', config)
                wf.write_json(marker, {'configuration_id': wf.token(config), 'training_texts': len(training)})
    return pd.read_csv(folder / 'validation_predictions.csv', dtype={'topic_id': str, 'cefr_level': str, 'predicted_cefr': str})


def task_check(args):
    """Leave one writing task per level out (8 folds): every validation text is predicted once by a model that never
    saw its task, which estimates performance on new writing tasks; compared with the main model trained on all tasks."""
    manifest, frame = wf.verify_shared(args.shared)
    topics, _ = topic_groups()
    frame = frame.merge(topics, on='text_id', how='left', validate='one_to_one')
    if frame.topic_id.isna().any():
        raise ValueError(f'{int(frame.topic_id.isna().sum())} texts have no topic.')
    folds = task_folds_of(frame)
    definitions = wf.experiment_definitions(argparse.Namespace(shared=args.shared, task='level', use_ranked=False, ranking=None), manifest)
    validation = frame[frame.split.eq('validation')].sort_values('text_id')
    rows, pooled = [], []
    for family in args.families or TASK_FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            unseen_parts = []
            for fold in args.folds or range(1, TASK_FOLDS + 1):
                table = run_task_fold(manifest, frame, validation, definitions, family, algorithm, fold, folds[fold], args)
                if table is None:
                    continue
                for subset, mask in (('seen_tasks', ~table.unseen_task), ('unseen_tasks', table.unseen_task)):
                    rows.append({'family': family, 'algorithm': algorithm, 'fold': fold, 'tasks': subset,
                                 **subset_metrics(table[mask], manifest, algorithm)})
                unseen_parts.append(table[table.unseen_task])
            if len(unseen_parts) == TASK_FOLDS:
                main = pd.read_csv(wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' /
                                   'validation_predictions.csv', dtype={'cefr_level': str, 'predicted_cefr': str})
                for model, table in (('unseen_tasks_pooled_over_folds', pd.concat(unseen_parts)), ('main_model_all_tasks', main)):
                    pooled.append({'family': family, 'algorithm': algorithm, 'model': model, **subset_metrics(table, manifest, algorithm)})
    if args.dry_run or args.families or args.folds or not rows:
        return  # the tables are written by a run over all folds
    TASK_FOLDER.mkdir(parents=True, exist_ok=True)
    write_csv(TASK_FOLDER / 'task_check_folds.csv', pd.DataFrame(rows))
    if pooled:
        write_csv(TASK_FOLDER / 'task_check_summary.csv', pd.DataFrame(pooled))
    log(f'Saved {TASK_FOLDER}')


def length_check(args):
    manifest, frame = wf.verify_shared(args.shared)
    data = wf.resolve_manifest_path(manifest['data_directory'])
    column = 'complexity__LCA__wordtokens'
    with duckdb.connect() as connection:
        words = connection.execute(f'SELECT text_id, "{column}" AS words FROM read_parquet({wf.literal(data / "feature_dataframe.parquet")})').df()
    rows = []
    for family in LENGTH_FAMILIES:
        for algorithm in ('classifier', 'regressor'):
            path = wf.BASE / '3_Level_Prediction/1_Intermediate_Calculations' / f'{algorithm}__{family}' / 'validation_predictions.csv'
            if not path.is_file():
                log(f'skipped {algorithm}__{family}: no saved predictions')
                continue
            table = pd.read_csv(path, dtype={'cefr_level': str, 'predicted_cefr': str}).merge(words, on='text_id', validate='one_to_one')
            table['words_bin'] = pd.cut(table.words, LENGTH_BINS, right=False)
            for interval, group in table.groupby('words_bin', observed=True):
                label = f'{int(interval.left)}+' if np.isinf(interval.right) else f'{int(interval.left)}-{int(interval.right) - 1}'
                rows.append({'family': family, 'algorithm': algorithm, 'words': label,
                             'share_of_texts': len(group) / len(table), **subset_metrics(group, manifest, algorithm)})
    LENGTH_FOLDER.mkdir(parents=True, exist_ok=True)
    write_csv(LENGTH_FOLDER / 'performance_by_text_length.csv', pd.DataFrame(rows))
    log(f'Saved {LENGTH_FOLDER}')


def run_tests():
    import unittest

    class SubsetTests(unittest.TestCase):
        def test_subset_metrics_match_the_pipeline(self):
            generator = np.random.default_rng(1)
            level = generator.integers(1, 16, 300)
            predicted = np.clip(level + generator.integers(-1, 2, 300), 1, 15)
            manifest = {'labels': wf.LABELS[:5], 'level_to_cefr': {str(k): wf.LABELS[(k - 1) // 3] for k in range(1, 16)}}
            mapping = {int(k): v for k, v in manifest['level_to_cefr'].items()}
            table = pd.DataFrame({'level': level, 'predicted_level': predicted, 'cefr_level': [mapping[x] for x in level],
                                  'predicted_cefr': [mapping[x] for x in predicted], 'predicted_level_continuous': predicted + .2})
            metrics = subset_metrics(table, manifest, 'regressor')
            expected = wf.class_metrics(table.level, table.predicted_level, list(range(1, 16)))
            self.assertEqual(metrics['level_macro_f1'], expected['macro_f1'])
            self.assertAlmostEqual(metrics['regression_mae'], float(np.abs(predicted + .2 - level).mean()))
            self.assertEqual(metrics['texts'], 300)

        def test_each_fold_holds_out_one_task_per_level(self):
            frame = pd.DataFrame({'topic_id': [str(topic) for topic in range(1, 17)] * 2,
                                  'level': [1] * 8 + [2] * 8 + [1] * 8 + [2] * 8})
            folds = task_folds_of(frame)
            self.assertEqual(sorted(folds), list(range(1, 9)))
            self.assertEqual(folds[1], {'1', '9'})
            self.assertEqual(set().union(*folds.values()), {str(topic) for topic in range(1, 17)})
            with self.assertRaises(ValueError):
                task_folds_of(frame.assign(level=[1] * 16 + [2] * 16))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SubsetTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['topic', 'length', 'tasks', 'test'])
    parser.add_argument('--shared', type=Path, default=wf.SHARED)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--min-observations', type=int, default=20)
    parser.add_argument('--memory', default='1GB', help='DuckDB memory for loading features.')
    parser.add_argument('--dry-run', action='store_true', help='Show the topic groups and text counts without fitting.')
    parser.add_argument('--tables-only', action='store_true',
                        help='Leave-one-task-out check: write the tables from the finished folds without training the others.')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, TASK_FOLDS + 1),
                        help='Leave-one-task-out check: only these folds, e.g. one per parallel process.')
    parser.add_argument('--families', nargs='+', choices=FAMILIES,
                        help='Topic check: only these models, e.g. one per parallel process; a run without it adds the finished runs of all models to the tables.')
    args = parser.parse_args(argv)
    args.shared = args.shared.resolve()
    if args.command == 'test':
        return run_tests()
    return {'topic': topic_check, 'length': length_check, 'tasks': task_check}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
