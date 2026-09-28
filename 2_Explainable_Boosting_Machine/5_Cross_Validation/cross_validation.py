"""10-fold cross-validation of every EBM experiment.

Folds group texts by learner and balance course levels over the training and validation texts; the reserved
test texts are never read. Every (task, model, fold) is one job, fitted exactly like the main pipeline's runs,
so the jobs can run on one machine or be spread over the tasks of a cluster array:

  python cross_validation.py prepare               folds and the job list
  python cross_validation.py run --parallel 30     work through all jobs with 30 processes
  python cross_validation.py run --job-index 7     one job, e.g. the task number of a cluster array
  python cross_validation.py status                completed, running and failed jobs
  python cross_validation.py summarize             mean and standard deviation over folds, corrected tests
  python cross_validation.py test                  checks on small synthetic data
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '0_Shared_Data_And_Splits'))
import ebm_workflow as wf  # noqa: E402

FOLDS_FOLDER, JOBS_FOLDER, RESULTS_FOLDER = '1_Folds', '2_Jobs', '3_Results'
RANKED_EXPERIMENTS = {'training_mutual_information_features': 'mi', 'training_information_gain_features': 'ig'}
# Relative cost of one fit per feature, measured on the main runs; longest jobs start first.
COST = {('level', 'classifier'): 1.0, ('cefr', 'classifier'): 0.4, ('level', 'regressor'): 0.07}
RANK_COST = 600
MAIN_MODEL = 'combined_without_length_correlated_features'
TOP_FEATURES = 20
# Linguistic dimensions of the tools: feature-group importance beside the per-tool view.
DIMENSIONS = {'TAALES': 'lexical_sophistication', 'TAALES_COVERAGE': 'lexical_sophistication',
              'LCA': 'lexical_diversity_and_density', 'L2SCA': 'syntactic_complexity', 'TAASSC': 'syntactic_complexity',
              'POLKE': 'grammatical_constructions', 'ERRANT': 'accuracy'}


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def write_csv(path, table):
    temporary = Path(f'{path}.tmp')
    table.to_csv(temporary, index=False)
    try:
        os.replace(temporary, path)
    except PermissionError as error:
        raise PermissionError(f'Close {Path(path).name} in Excel or any other program, then rerun the same command.') from error


def columns_token(columns):
    return wf.token(list(columns))


def verify_shared(shared):
    try:
        return wf.verify_shared(shared)
    except ValueError as error:
        raise ValueError(f'{error} If this copy was cloned on Linux or macOS, clone it again with '
                         '"git clone -c core.autocrlf=true": the checksums were made with Windows line endings.') from error


# Folds and jobs --------------------------------------------------------------------------------------------

def build_folds(frame, folds, seed):
    """Learner-grouped folds, balanced by course level, over the training and validation texts only."""
    pool = frame[frame.split.isin(['train', 'validation'])].sort_values('text_id').reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold = np.full(len(pool), -1)
    for number, (_, held_out) in enumerate(splitter.split(pool, pool.level, groups=pool.learner_id), 1):
        fold[held_out] = number
    return pool[['text_id', 'learner_id', 'level', 'cefr_level']].assign(fold=fold)


def experiment_columns(shared, task, manifest, experiments=None):
    """The main pipeline's feature sets for one task; MI and IG sets come from each fold's own ranking."""
    definitions = wf.experiment_definitions(SimpleNamespace(shared=shared, task=task, use_ranked=False, ranking=None), manifest)
    families = {name: definition['columns'] for name, definition in definitions.items()}
    families.update({name: None for name in RANKED_EXPERIMENTS})
    return {name: columns for name, columns in families.items() if experiments is None or name in experiments}


def job_list(shared, manifest, folds, experiments=None, top_k=200, tasks=None):
    rows = [{'job_id': f'rank__fold{fold:02d}', 'kind': 'rank', 'task': '', 'algorithm': '', 'family': '', 'fold': fold,
             'depends_on': '', 'cost': RANK_COST} for fold in range(1, folds + 1)]
    for task, algorithms in (('cefr', ['classifier']), ('level', ['classifier', 'regressor'])):
        if tasks and task not in tasks:
            continue
        for family, columns in experiment_columns(shared, task, manifest, experiments).items():
            for algorithm in algorithms:
                if (family == 'majority' and algorithm != 'classifier') or (family == 'median' and algorithm != 'regressor'):
                    continue
                features = top_k if columns is None else len(columns)
                for fold in range(1, folds + 1):
                    rows.append({'job_id': f'{task}__{algorithm}__{family}__fold{fold:02d}', 'kind': 'fit', 'task': task,
                                 'algorithm': algorithm, 'family': family, 'fold': fold,
                                 'depends_on': f'rank__fold{fold:02d}' if family in RANKED_EXPERIMENTS else '',
                                 'cost': round(features * COST[(task, algorithm)], 1)})
    jobs = pd.DataFrame(rows)
    if not jobs.loc[jobs.kind.eq('fit'), 'depends_on'].str.len().gt(0).any():
        jobs = jobs[jobs.kind.eq('fit')]
    # Rankings first, as MI and IG fits wait for them; then the experiments in the order given, longest fits first.
    position = {name: index for index, name in enumerate(experiments or [])}
    jobs = jobs.assign(order=jobs.kind.ne('rank'), priority=jobs.family.map(position).fillna(0))
    return jobs.sort_values(['order', 'priority', 'cost', 'job_id'], ascending=[True, True, False, True]) \
               .drop(columns=['order', 'priority']).reset_index(drop=True)


def prepare(args):
    manifest, frame = verify_shared(args.shared)
    folder = args.output / FOLDS_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    folds = build_folds(frame, args.folds, manifest['seed'])
    counts = folds.groupby('fold').size()
    fold_manifest = {'dataset_id': wf.token(manifest), 'folds': args.folds, 'seed': manifest['seed'],
                     'method': 'StratifiedGroupKFold over training and validation texts: grouped by learner, stratified by '
                               'course level; the test texts are not used',
                     'texts': len(folds), 'learners': int(folds.learner_id.nunique()),
                     'held_out_texts': {str(k): int(v) for k, v in counts.items()},
                     'mean_training_texts': float(len(folds) - counts.mean()), 'mean_held_out_texts': float(counts.mean())}
    path = folder / 'folds.csv'
    if path.is_file():
        saved = wf.read_json(folder / 'fold_manifest.json')
        if saved['dataset_id'] != fold_manifest['dataset_id'] or saved['folds'] != args.folds or \
                not pd.read_csv(path, dtype={'learner_id': str, 'cefr_level': str}).equals(folds):
            raise ValueError(f'{path} belongs to another setup; use a new --output folder.')
        fold_manifest = saved
    else:
        write_csv(path, folds)
        fold_manifest['folds_sha256'] = wf.digest(path)
        wf.write_json(folder / 'fold_manifest.json', fold_manifest)
    jobs = job_list(args.shared, manifest, args.folds, args.experiments, args.top_k, args.tasks)
    (args.output / JOBS_FOLDER).mkdir(parents=True, exist_ok=True)
    write_csv(args.output / JOBS_FOLDER / 'jobs.csv', jobs)
    log(f'{len(folds):,} texts of {fold_manifest["learners"]:,} learners in {args.folds} folds; '
        f'{len(jobs):,} jobs ({int(jobs.kind.eq("rank").sum())} rankings, {int(jobs.kind.eq("fit").sum())} fits)')


class Context:
    """Everything a job needs, read once per process."""

    def __init__(self, args):
        self.args = args
        self.manifest, frame = verify_shared(args.shared)
        folder = args.output / FOLDS_FOLDER
        self.fold_manifest = wf.read_json(folder / 'fold_manifest.json')
        if self.fold_manifest['dataset_id'] != wf.token(self.manifest) or wf.digest(folder / 'folds.csv') != self.fold_manifest['folds_sha256']:
            raise ValueError('The folds belong to another setup or were changed; run prepare with a new --output folder.')
        folds = pd.read_csv(folder / 'folds.csv', dtype={'learner_id': str, 'cefr_level': str})
        pool = frame.merge(folds[['text_id', 'fold']], on='text_id', how='inner', validate='one_to_one')
        if len(pool) != len(folds) or pool.split.eq('test').any():
            raise ValueError('The folds must cover exactly the training and validation texts.')
        self.pool = pool.sort_values('text_id')
        self.jobs = pd.read_csv(args.output / JOBS_FOLDER / 'jobs.csv', keep_default_na=False).set_index('job_id', drop=False)
        self.definitions = {task: experiment_columns(args.shared, task, self.manifest) for task in ('cefr', 'level')}
        self.original = sorted({spec['column'] for spec in wf.read_json(args.shared / 'feature_definitions.json')['original']} | set(wf.DERIVED))

    def split(self, fold):
        return self.pool[self.pool.fold.ne(fold)], self.pool[self.pool.fold.eq(fold)]

    def folder(self, job_id):
        return self.args.output / JOBS_FOLDER / job_id


def job_configuration(context, job):
    base = {'dataset_id': wf.token(context.manifest), 'folds_sha256': context.fold_manifest['folds_sha256'],
            'fold': int(job.fold), 'min_observations': context.args.min_observations, 'seed': context.manifest['seed']}
    if job.kind == 'rank':
        return {**base, 'kind': 'rank', 'top_k': context.args.top_k, 'methods': wf.RANKING_METHODS,
                'columns_sha256': columns_token(context.original)}
    columns = job_columns(context, job)
    config = {**base, 'kind': 'fit', 'task': job.task, 'algorithm': job.algorithm, 'feature_set': job.family,
              'columns_sha256': columns_token(columns), 'rounds': context.args.rounds, 'interactions': 0, 'max_bins': 128,
              'learning_rate': .04, 'outer_bags': 1,
              'class_weight': 'balanced' if job.algorithm == 'classifier' and job.family != 'majority' else 'none',
              'packages': {name: wf.importlib.metadata.version(name) for name in ('interpret', 'scikit-learn', 'numpy', 'duckdb', 'pandas')}}
    if job.depends_on:
        config['ranking_completed_sha256'] = wf.digest(context.folder(job.depends_on) / 'completed.json')
    return config


def job_columns(context, job):
    if job.family in RANKED_EXPERIMENTS:
        table = pd.read_csv(context.folder(job.depends_on) / f'ranking_{RANKED_EXPERIMENTS[job.family]}.csv')
        table = table[table.target.eq(job.task) & table.selected].sort_values('rank')
        if table.empty:
            raise ValueError(f'{job.depends_on}: no selected features for {job.task}.')
        return table.feature.tolist()
    return context.definitions[job.task][job.family]


def complete(context, job):
    folder = context.folder(job.job_id)
    marker = folder / 'completed.json'
    if not marker.is_file():
        return False
    if job.depends_on and not complete(context, context.jobs.loc[job.depends_on]):
        return False
    saved = wf.read_json(marker)
    if saved['configuration_id'] != wf.token(job_configuration(context, job)):
        raise ValueError(f'{job.job_id}: settings or inputs changed since it was run; use a new --output folder.')
    for name, expected in saved['files'].items():
        if wf.digest(folder / name) != expected:
            raise ValueError(f'Changed job result: {folder / name}')
    return True


# Running jobs ----------------------------------------------------------------------------------------------

def run_job(context, job):
    """Run one job into a staging folder and publish it only when every file is written."""
    args, folder = context.args, context.folder(job.job_id)
    config = job_configuration(context, job)
    training, held_out = context.split(int(job.fold))
    started = time.time()
    # The feature matrices (several GB) can go on a fast local disk; the small results stay beside the job folders.
    scratch = args.scratch or folder.parent
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.job_', dir=folder.parent) as temporary, \
            tempfile.TemporaryDirectory(prefix='.matrices_', dir=scratch) as matrices:
        work = Path(matrices)
        stage = Path(temporary) / 'result'
        stage.mkdir()
        if job.kind == 'rank':
            tables = wf.rank_features(context.manifest, training, context.original, args.min_observations, args.top_k, work, args.memory)
            for method, table in tables.items():
                table.to_csv(stage / f'ranking_{method}.csv', index=False)
        else:
            columns = job_columns(context, job)

            def score(model, matrix, features):
                # Groups are read from the second part of a column name, so relabel features by dimension for the second pass.
                by_dimension = [f"group__{DIMENSIONS.get(name.split('__')[1], 'other')}__{name}" for name in features]
                return (wf.feature_and_tool_contributions(model, matrix, features, job.algorithm),
                        wf.feature_and_tool_contributions(model, matrix, by_dimension, job.algorithm))

            scored = wf.fit_run(context.manifest, training, held_out, columns, job.task, job.algorithm, job.family, args.rounds,
                                args.min_observations, args.workers, args.memory, work, job.job_id, score)
            _, selected, excluded, metrics, _, contributions = scored
            wf.write_json(stage / 'metrics.json', {**metrics, 'training_texts': len(training), 'held_out_texts': len(held_out)})
            wf.write_json(stage / 'excluded_features.json', excluded)
            if contributions is not None:
                (outputs, tools, per_feature, per_tool), (_, dimensions, _, per_dimension) = contributions
                pd.DataFrame({'feature': selected, 'mean_absolute_contribution': per_feature.mean(axis=1)}) \
                    .to_csv(stage / 'feature_importance.csv', index=False)
                for name, groups, totals in (('tool', tools, per_tool), ('dimension', dimensions, per_dimension)):
                    pd.DataFrame([{name: group, 'output': output, 'features': len(groups[group]), 'contribution': float(totals[group][i])}
                                  for group in groups for i, output in enumerate(outputs)]).to_csv(stage / f'{name}_contributions.csv', index=False)
        wf.write_json(stage / 'job_config.json', config)
        wf.write_json(stage / 'completed.json', {
            'configuration_id': wf.token(config), 'minutes': round((time.time() - started) / 60, 1),
            'code_sha256': {'cross_validation': wf.digest(Path(__file__)), 'ebm_workflow': wf.digest(Path(wf.__file__))},
            'files': {path.name: wf.digest(path) for path in stage.iterdir() if path.is_file()}})
        if folder.exists():
            shutil.rmtree(folder)  # an unfinished earlier attempt; completed jobs never reach this point
        stage.rename(folder)
    log(f'{job.job_id}: done in {(time.time() - started) / 60:.1f} min')


def try_lock(folder):
    """Claim a job without waiting; returns the lock path, or None when a live process holds it."""
    lock = folder.parent / f'.{folder.name}.lock'
    for _ in range(2):
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                content = lock.read_text(encoding='ascii', errors='replace')
                age = time.time() - lock.stat().st_mtime
            except OSError:
                continue
            owner = int(content) if content.strip().isdigit() else 0
            if (owner == 0 and age > 60) or (owner and not psutil.pid_exists(owner)):
                lock.unlink(missing_ok=True)
                continue
            return None
        with os.fdopen(descriptor, 'w', encoding='ascii') as handle:
            handle.write(str(os.getpid()))
        return lock
    return None


def execute(context, job):
    """Run a job and, first, its unfinished ranking; waits while another process runs either of them."""
    if job.depends_on:
        execute(context, context.jobs.loc[job.depends_on])
    folder = context.folder(job.job_id)
    with wf.training_lock(folder, poll_seconds=30):
        if not complete(context, job):
            run_job(context, job)


def worker(args):
    """Take the next ready job (longest first), run it, and repeat until every job is done or has failed."""
    context = Context(args)
    failed = set()
    while True:
        pending = [job for _, job in context.jobs.iterrows() if job.job_id not in failed and not complete(context, job)]
        if not pending:
            break
        started = False
        for job in pending:
            if job.depends_on and not complete(context, context.jobs.loc[job.depends_on]):
                continue
            lock = try_lock(context.folder(job.job_id))
            if lock is None:
                continue
            try:
                if not complete(context, job):
                    run_job(context, job)
            except Exception as error:  # noqa: BLE001 - the job is recorded and the others continue
                failed.add(job.job_id)
                (context.args.output / JOBS_FOLDER / 'failed').mkdir(exist_ok=True)
                (context.args.output / JOBS_FOLDER / 'failed' / f'{job.job_id}.txt').write_text(repr(error), encoding='utf-8')
                log(f'{job.job_id}: FAILED: {error!r}')
            finally:
                lock.unlink(missing_ok=True)
            started = True
            break
        if not started:
            time.sleep(30)  # the remaining jobs run in other processes or wait for a ranking
    return 1 if failed else 0


def run(args):
    if args.job or args.job_index is not None:
        context = Context(args)
        job = context.jobs.loc[args.job] if args.job else context.jobs.iloc[args.job_index]
        execute(context, job)
        return 0
    if args.parallel <= 1:
        return worker(args)
    logs = args.output / JOBS_FOLDER / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    memory = args.memory
    if memory == 'auto':
        memory = f'{max(512, int(psutil.virtual_memory().available / 2**20 * 0.6 / args.parallel))}MiB'
    command = [sys.executable, '-u', str(Path(__file__).resolve()), 'worker', '--shared', str(args.shared), '--output', str(args.output),
               '--rounds', str(args.rounds), '--workers', str(args.workers), '--memory', memory,
               '--min-observations', str(args.min_observations), '--top-k', str(args.top_k)] \
        + (['--scratch', str(args.scratch)] if args.scratch else [])
    processes = []
    for number in range(1, args.parallel + 1):
        handle = open(logs / f'worker_{number:02d}.log', 'a', encoding='utf-8')
        processes.append((subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), handle))
        log(f'started worker {number}/{args.parallel}')
        if number < args.parallel:
            time.sleep(args.stagger)  # spread the memory-heavy loading phases
    codes = [process.wait() for process, _ in processes]
    for _, handle in processes:
        handle.close()
    status(args)
    return max(codes)


def status(args):
    context = Context(args)
    jobs = context.jobs
    done = [job.job_id for _, job in jobs.iterrows() if complete(context, job)]
    running = [path.name[1:-5] for path in (args.output / JOBS_FOLDER).glob('.*.lock')]
    failed = sorted(path.stem for path in (args.output / JOBS_FOLDER / 'failed').glob('*.txt')) \
        if (args.output / JOBS_FOLDER / 'failed').is_dir() else []
    minutes = [wf.read_json(context.folder(job) / 'completed.json').get('minutes', 0) for job in done]
    log(f'{len(done):,}/{len(jobs):,} jobs complete ({sum(minutes) / 60:.1f} process-hours so far); '
        f'{len(running)} running; {len(set(failed) - set(done))} failed')
    for job in running:
        print(f'  running: {job}')
    for job in sorted(set(failed) - set(done)):
        print(f'  failed:  {job}')


# Summaries ------------------------------------------------------------------------------------------------

def fold_metric_rows(context):
    rows, tools, features, dimensions, classes, confusions = [], [], [], [], [], []
    for _, job in context.jobs[context.jobs.kind.eq('fit')].iterrows():
        if not complete(context, job):
            continue
        folder = context.folder(job.job_id)
        metrics = wf.read_json(folder / 'metrics.json')
        experiment = f'{job.algorithm}__{job.family}'
        identity = {'task': job.task, 'experiment': experiment, 'algorithm': job.algorithm, 'family': job.family, 'fold': int(job.fold),
                    'training_texts': metrics['training_texts'], 'held_out_texts': metrics['held_out_texts']}
        routes = [('cefr', 'Direct CEFR classification')] if job.task == 'cefr' else \
            [('cefr', 'Level probabilities summed into CEFR bands' if job.algorithm == 'classifier' else
              'Numerical level rounded and mapped to CEFR'), ('level', 'Course level')]
        for target, route in routes:
            values = {name: metrics[target][name] for name in wf.CLASS_METRICS}
            if target == 'level' and 'regression' in metrics:
                values.update({f'regression_{name}': metrics['regression'][name] for name in wf.REGRESSION_METRICS})
            rows.append({**identity, 'evaluated_target': target, 'prediction_route': route, **values})
            view = {**identity, 'evaluated_target': target, 'prediction_route': route}
            labels = [str(label) for label in metrics[target]['labels']]
            report = metrics[target]['per_class']
            classes += [{**view, 'label': label, **{name: report[label][name] for name in ('precision', 'recall', 'f1-score', 'support')}}
                        for label in labels if label in report]
            confusions += [{**view, 'true_label': true, 'predicted_label': predicted, 'count': int(count)}
                           for true, counts in zip(labels, metrics[target]['confusion_matrix'])
                           for predicted, count in zip(labels, counts)]
        if (folder / 'tool_contributions.csv').is_file():
            tools.append(pd.read_csv(folder / 'tool_contributions.csv', dtype={'output': str}).assign(**identity))
            dimensions.append(pd.read_csv(folder / 'dimension_contributions.csv', dtype={'output': str}).assign(**identity))
            features.append(pd.read_csv(folder / 'feature_importance.csv').assign(**identity))
    extras = {'dimensions': dimensions, 'classes': pd.DataFrame(classes), 'confusions': pd.DataFrame(confusions)}
    return pd.DataFrame(rows), tools, features, extras


def group_shares(tables, column):
    """Each group's share of the contributions per output (and 'overall', the mean over outputs), with its
    standard deviation over folds. MI and IG select features per fold, so a group absent from a fold counts as zero."""
    table = pd.concat(tables, ignore_index=True)
    overall = table.groupby(['task', 'experiment', 'fold', column], as_index=False).agg(
        features=('features', 'first'), contribution=('contribution', 'mean')).assign(output='overall')
    table = pd.concat([overall, table], ignore_index=True)
    table['share'] = table.contribution / table.groupby(['task', 'experiment', 'fold', 'output']).contribution.transform('sum')
    summaries = []
    for (task, experiment), group in table.groupby(['task', 'experiment']):
        wide = {name: group.pivot_table(index=['output', 'fold'], columns=column, values=name, fill_value=0.)
                for name in ('share', 'contribution', 'features')}
        long = pd.concat({name: frame.stack() for name, frame in wide.items()}, axis=1).reset_index()
        summaries.append(long.groupby(['output', column], as_index=False).agg(
            features=('features', 'mean'), share_mean=('share', 'mean'), share_standard_deviation=('share', 'std'),
            contribution_mean=('contribution', 'mean')).assign(task=task, experiment=experiment))
    shares = pd.concat(summaries)[['task', 'experiment', 'output', column, 'features', 'share_mean',
                                   'share_standard_deviation', 'contribution_mean']]
    return shares.sort_values(['task', 'experiment', 'output', 'share_mean'], ascending=[True, True, True, False])


def corrected_comparison(differences, training, held_out):
    """Nadeau and Bengio's corrected resampled t-test for k-fold differences (two-sided)."""
    k = len(differences)
    mean, variance = float(np.mean(differences)), float(np.var(differences, ddof=1))
    if variance == 0:
        return mean, 0.0, None, (0.0 if mean else 1.0)
    statistic = mean / np.sqrt((1 / k + held_out / training) * variance)
    return mean, float(np.sqrt(variance)), float(statistic), float(2 * stats.t.sf(abs(statistic), df=k - 1))


def summarize(args):
    context = Context(args)
    folds, tools, features, extras = fold_metric_rows(context)
    if folds.empty:
        raise ValueError('No completed cross-validation fits yet.')
    keys = ['task', 'experiment', 'evaluated_target', 'prediction_route']
    counts = folds.groupby(keys).fold.nunique()
    complete_models = counts[counts.eq(context.fold_manifest['folds'])].index
    if complete_models.empty:
        log(f'No model has all {context.fold_manifest["folds"]} folds yet; {len(folds):,} model-fold results so far.')
        return
    folds = folds.set_index(keys).loc[complete_models].reset_index()
    finished = set(zip(folds.task, folds.experiment))
    tools, features, dimensions = ([table for table in tables if (table.task.iloc[0], table.experiment.iloc[0]) in finished]
                                   for tables in (tools, features, extras['dimensions']))
    metrics = [name for name in [*wf.CLASS_METRICS, *(f'regression_{n}' for n in wf.REGRESSION_METRICS)] if name in folds]
    summary = folds.groupby(keys)[metrics].agg(['mean', 'std'])
    summary.columns = [f'{name}_mean' if statistic == 'mean' else f'{name}_standard_deviation' for name, statistic in summary.columns]
    summary = summary.reset_index()
    summary.insert(4, 'folds', context.fold_manifest['folds'])
    summary = summary.sort_values(['evaluated_target', 'macro_f1_mean'], ascending=[True, False])
    summary.insert(0, 'rank', summary.groupby('evaluated_target').cumcount() + 1)
    training, held_out = context.fold_manifest['mean_training_texts'], context.fold_manifest['mean_held_out_texts']
    comparisons = []
    boards = [('cefr', 'macro_f1', None), ('level', 'macro_f1', None), ('level', 'regression_mae', 'regressor')]
    for target, metric, algorithm in boards:
        board = summary[summary.evaluated_target.eq(target)]
        if algorithm:
            board = board[board.experiment.str.startswith(algorithm)]
        if board.empty or f'{metric}_mean' not in board or board[f'{metric}_mean'].isna().all():
            continue
        lower_is_better = metric.endswith('mae')
        best = board.loc[board[f'{metric}_mean'].idxmin() if lower_is_better else board[f'{metric}_mean'].idxmax()]
        main = board[board.experiment.str.endswith(f'__{MAIN_MODEL}') & (
            board.prediction_route.str.startswith('Level probabilities') | ~board.evaluated_target.eq('cefr'))]
        references = [('rank_1', best)] + ([('main_model', main.iloc[0])] if not main.empty else [])
        for name, reference in references:
            ref = folds[(folds[keys] == reference[keys]).all(axis=1)].set_index('fold')[metric]
            for _, row in board.iterrows():
                values = folds[(folds[keys] == row[keys]).all(axis=1)].set_index('fold')[metric]
                mean, deviation, statistic, p_value = corrected_comparison((values - ref).loc[ref.index].to_numpy(), training, held_out)
                same = row.experiment == reference.experiment and row.prediction_route == reference.prediction_route
                comparisons.append({'reference': name, 'reference_experiment': reference.experiment,
                                    'reference_route': reference.prediction_route, 'metric': metric, **row[keys].to_dict(),
                                    f'{metric}_mean': row[f'{metric}_mean'], 'difference_mean': mean,
                                    'difference_standard_deviation': deviation, 'corrected_t': statistic, 'p_value': p_value,
                                    'significant_at_0_05': bool(not same and p_value is not None and p_value < .05)})
    destination = args.output / RESULTS_FOLDER
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / 'cv_fold_metrics.csv', folds.sort_values(keys + ['fold']))
    write_csv(destination / 'cv_summary.csv', summary)
    write_csv(destination / 'cv_comparisons.csv', pd.DataFrame(comparisons))
    views = extras['classes'].set_index(keys).index.isin(complete_models)
    per_class = extras['classes'][views].groupby(keys + ['label'], sort=False, as_index=False).agg(
        precision_mean=('precision', 'mean'), precision_standard_deviation=('precision', 'std'),
        recall_mean=('recall', 'mean'), recall_standard_deviation=('recall', 'std'),
        f1_mean=('f1-score', 'mean'), f1_standard_deviation=('f1-score', 'std'), support_mean=('support', 'mean'))
    write_csv(destination / 'cv_per_class_metrics.csv', per_class)
    confusion = extras['confusions'][extras['confusions'].set_index(keys).index.isin(complete_models)]
    confusion = confusion.groupby(keys + ['true_label', 'predicted_label'], sort=False, as_index=False)['count'].sum()
    confusion['percentage_of_true_label'] = 100 * confusion['count'] / confusion.groupby(keys + ['true_label'])['count'].transform('sum')
    write_csv(destination / 'cv_confusion_matrices.csv', confusion)
    if tools:
        write_csv(destination / 'cv_tool_shares.csv', group_shares(tools, 'tool'))
        write_csv(destination / 'cv_dimension_shares.csv', group_shares(dimensions, 'dimension'))
        importance = pd.concat(features, ignore_index=True)
        importance['rank_in_fold'] = importance.groupby(['task', 'experiment', 'fold']).mean_absolute_contribution \
            .rank(ascending=False, method='first')
        stability = importance.groupby(['task', 'experiment', 'feature'], as_index=False).agg(
            total=('mean_absolute_contribution', 'sum'), squares=('mean_absolute_contribution', lambda values: float((values ** 2).sum())),
            folds_present=('fold', 'nunique'), mean_rank_when_present=('rank_in_fold', 'mean'),
            folds_in_top_20=('rank_in_fold', lambda ranks: int((ranks <= TOP_FEATURES).sum())))
        # A feature missing from a fold (MI and IG) contributes zero there.
        folds_per_model = importance.groupby(['task', 'experiment']).fold.nunique()
        count = pd.Series(list(zip(stability.task, stability.experiment))).map(folds_per_model).to_numpy()
        stability['importance_mean'] = stability.total / count
        stability['importance_standard_deviation'] = np.sqrt(np.maximum(
            0, (stability.squares - count * stability.importance_mean ** 2) / np.maximum(count - 1, 1)))
        stability = stability.drop(columns=['total', 'squares'])[
            ['task', 'experiment', 'feature', 'importance_mean', 'importance_standard_deviation', 'folds_present',
             'mean_rank_when_present', 'folds_in_top_20']]
        stability = stability.sort_values(['task', 'experiment', 'importance_mean'], ascending=[True, True, False])
        stability.insert(3, 'overall_rank', stability.groupby(['task', 'experiment']).cumcount() + 1)
        write_csv(destination / 'cv_feature_stability.csv', stability)
    missing = len(counts) - len(complete_models)
    log(f'Summarized {len(complete_models):,} model views over {context.fold_manifest["folds"]} folds'
        + (f'; {missing} still incomplete' if missing else '') + f': {destination}')


# Checks -----------------------------------------------------------------------------------------------------

def run_tests():
    import unittest

    class CrossValidationTests(unittest.TestCase):
        def test_folds_jobs_runs_and_summary(self):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root / 'data'
                data.mkdir()
                ids = np.arange(1, 1801)
                level = (ids - 1) % 15 + 1
                cefr = np.array(wf.LABELS)[(level - 1) // 3]
                generator = np.random.default_rng(0)
                table = pd.DataFrame({'text_id': ids, 'cefr_level': cefr,
                                      'complexity__LCA__ld': level + generator.normal(0, .5, len(ids)),
                                      'complexity__POLKE__polke_1_per_100_words': 2. * level + generator.normal(0, 1, len(ids)),
                                      'accuracy__ERRANT__total_errors': 10 + ids % 4, 'accuracy__ERRANT__missing_errors': ids % 3,
                                      'accuracy__ERRANT__unnecessary_errors': ids % 2, 'accuracy__ERRANT__replacement_errors': 5 + ids % 2,
                                      'accuracy__ERRANT__other_errors': ids % 2,
                                      'accuracy__ERRANT__errors_per_100_words': (10 + ids % 4) / level})
                metadata = pd.DataFrame({'text_id': ids, 'cefr_level': cefr, 'level': level})
                import duckdb
                with duckdb.connect() as connection:
                    connection.register('features', table)
                    connection.register('metadata', metadata)
                    connection.execute(f'COPY features TO {wf.literal(data / "feature_dataframe.parquet")} (FORMAT PARQUET)')
                    connection.execute(f'COPY metadata TO {wf.literal(data / "text_metadata.parquet")} (FORMAT PARQUET)')
                pd.DataFrame([{'column': column, 'group': column.split('__')[0], 'source': column.split('__')[1],
                               'original_feature': column.split('__')[2],
                               'measure_type': 'error_count' if column.endswith('_errors') and column.startswith('accuracy') else 'test'}
                              for column in table.columns[2:]]).to_csv(data / 'feature_dictionary.csv', index=False)
                learners = metadata[['text_id', 'cefr_level']].assign(learner_id=((ids - 1) // 2).astype(str))
                learners.to_csv(root / 'learners.csv', index=False)
                wf.main('cefr', ['prepare', '--shared', str(root / 'shared'), '--data', str(data), '--metadata', str(root / 'learners.csv'),
                                 '--cefr-output', str(root / 'cefr'), '--level-output', str(root / 'level')])
                common = ['--shared', str(root / 'shared'), '--output', str(root / 'cv'), '--rounds', '2', '--workers', '1',
                          '--min-observations', '2', '--top-k', '1']
                main(['prepare', *common, '--folds', '3', '--experiments', 'majority', 'median', 'combined',
                      'training_mutual_information_features'])
                folds = pd.read_csv(root / 'cv' / FOLDS_FOLDER / 'folds.csv', dtype={'learner_id': str})
                _, frame = wf.verify_shared(root / 'shared')
                pool = frame[frame.split.isin(['train', 'validation'])]
                self.assertEqual(set(folds.text_id), set(pool.text_id))
                self.assertEqual(folds.groupby('learner_id').fold.nunique().max(), 1)
                self.assertEqual(sorted(folds.fold.unique()), [1, 2, 3])
                jobs = pd.read_csv(root / 'cv' / JOBS_FOLDER / 'jobs.csv', keep_default_na=False)
                # 3 rankings; CEFR: majority, combined, MI; level: 2 x (combined, MI) + majority + median; each for 3 folds.
                self.assertEqual(len(jobs), 3 + 3 * (3 + 6))
                self.assertEqual(list(jobs.kind[:3]), ['rank'] * 3)
                main(['run', *common, '--scratch', str(root / 'scratch'),
                      '--job', 'level__classifier__training_mutual_information_features__fold02'])
                self.assertTrue((root / 'cv' / JOBS_FOLDER / 'rank__fold02' / 'completed.json').is_file())
                self.assertEqual(list((root / 'scratch').iterdir()), [])  # temporary matrices are removed
                self.assertEqual(main(['run', *common]), 0)
                main(['prepare', *common, '--folds', '3', '--experiments', 'majority', 'median', 'combined',
                      'training_mutual_information_features'])
                before = {path: wf.digest(path) for path in (root / 'cv' / JOBS_FOLDER).rglob('completed.json')}
                self.assertEqual(len(before), len(jobs))
                self.assertEqual(main(['run', *common]), 0)
                self.assertEqual(before, {path: wf.digest(path) for path in (root / 'cv' / JOBS_FOLDER).rglob('completed.json')})
                main(['summarize', *common])
                results = root / 'cv' / RESULTS_FOLDER
                summary = pd.read_csv(results / 'cv_summary.csv')
                # CEFR board: 3 direct + 3 summed level classifiers + 3 mapped regressors; level board: 3 classifiers + 3 regressors.
                self.assertEqual(sorted(summary.groupby('evaluated_target').size().items()), [('cefr', 9), ('level', 6)])
                self.assertTrue((summary.folds == 3).all() and summary.macro_f1_mean.between(0, 1).all())
                comparisons = pd.read_csv(results / 'cv_comparisons.csv')
                self.assertEqual(set(comparisons.reference), {'rank_1'})
                self.assertTrue(comparisons[comparisons.experiment.eq(comparisons.reference_experiment)
                                            & comparisons.prediction_route.eq(comparisons.reference_route)].difference_mean.eq(0).all())
                for name in ('tool', 'dimension'):
                    shares = pd.read_csv(results / f'cv_{name}_shares.csv', dtype={'output': str})
                    self.assertTrue(np.allclose(shares.groupby(['task', 'experiment', 'output']).share_mean.sum(), 1))
                dimensions = pd.read_csv(results / 'cv_dimension_shares.csv', dtype={'output': str})
                self.assertEqual(set(dimensions.dimension), {'lexical_diversity_and_density', 'grammatical_constructions', 'accuracy'})
                per_class = pd.read_csv(results / 'cv_per_class_metrics.csv', dtype={'label': str})
                level_view = per_class[per_class.evaluated_target.eq('level') & per_class.experiment.eq('classifier__combined')]
                self.assertEqual(sorted(level_view.label, key=int), [str(level) for level in range(1, 16)])
                self.assertTrue(level_view.f1_mean.between(0, 1).all())
                confusion = pd.read_csv(results / 'cv_confusion_matrices.csv', dtype={'true_label': str, 'predicted_label': str})
                pooled = confusion[confusion.evaluated_target.eq('cefr') & confusion.experiment.eq('classifier__combined')
                                   & confusion.task.eq('cefr')]
                self.assertEqual(pooled['count'].sum(), len(pool))  # every text is held out exactly once
                self.assertTrue(np.allclose(pooled.groupby('true_label').percentage_of_true_label.sum(), 100))
                level_only = job_list(root / 'shared', wf.verify_shared(root / 'shared')[0], 3, ['combined'], 1, ['level'])
                self.assertEqual(set(level_only.task), {'level'})
                self.assertEqual(len(level_only), 3 * 2)
                ordered = job_list(root / 'shared', wf.verify_shared(root / 'shared')[0], 3, ['majority', 'median', 'combined'], 1, ['level'])
                self.assertEqual(list(ordered.family), ['majority'] * 3 + ['median'] * 3 + ['combined'] * 6)  # priority order
                stability = pd.read_csv(results / 'cv_feature_stability.csv')
                self.assertTrue(stability.folds_in_top_20.between(0, 3).all() and stability.folds_present.between(1, 3).all())
                combined = stability[stability.experiment.eq('classifier__combined') & stability.task.eq('level')]
                self.assertTrue(combined.folds_present.eq(3).all() and combined.overall_rank.tolist() == list(range(1, len(combined) + 1)))
                # The CV fit of a fold equals the pipeline's fitting of the same rows.
                context = Context(SimpleNamespace(shared=root / 'shared', output=root / 'cv', rounds=2, workers=1, memory='auto',
                                                  min_observations=2, top_k=1))
                training, held_out = context.split(1)
                with tempfile.TemporaryDirectory() as work:
                    direct = wf.fit_run(context.manifest, training, held_out, context.definitions['level']['combined'], 'level',
                                        'classifier', 'combined', 2, 2, 1, 'auto', Path(work), 'check')
                saved = wf.read_json(context.folder('level__classifier__combined__fold01') / 'metrics.json')
                self.assertEqual(direct[3]['level']['macro_f1'], saved['level']['macro_f1'])

        def test_corrected_comparison(self):
            mean, deviation, statistic, p_value = corrected_comparison(np.array([.01, .02, .015, .012]), 900., 100.)
            self.assertAlmostEqual(mean, .01425)
            expected = mean / np.sqrt((1 / 4 + 100 / 900) * np.var([.01, .02, .015, .012], ddof=1))
            self.assertAlmostEqual(statistic, expected)
            self.assertAlmostEqual(p_value, 2 * stats.t.sf(expected, 3))
            self.assertEqual(corrected_comparison(np.zeros(4), 900., 100.)[3], 1.0)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CrossValidationTests)
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'run', 'worker', 'status', 'summarize', 'test'])
    parser.add_argument('--shared', type=Path, default=wf.SHARED)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--experiments', nargs='+', help='Limit prepare to these feature sets (default: all).')
    parser.add_argument('--tasks', nargs='+', choices=['cefr', 'level'], help='Limit prepare to these tasks (default: both).')
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2, help='n_jobs of each fit, as in the main pipeline.')
    parser.add_argument('--memory', default='auto', help='DuckDB memory for loading features, per process.')
    parser.add_argument('--min-observations', type=int, default=20)
    parser.add_argument('--top-k', type=int, default=200)
    parser.add_argument('--parallel', type=int, default=1, help='Number of job processes on this machine.')
    parser.add_argument('--stagger', type=int, default=60, help='Seconds between starting job processes.')
    parser.add_argument('--job', help='Run one job by its job_id.')
    parser.add_argument('--job-index', type=int, help='Run one job by its row in jobs.csv (0-based), e.g. a cluster array index.')
    parser.add_argument('--scratch', type=Path, help='Folder for the temporary feature matrices, ideally a fast local disk '
                                                     '(default: beside the job results).')
    args = parser.parse_args(argv)
    args.shared, args.output = args.shared.resolve(), args.output.resolve()
    args.scratch = args.scratch.resolve() if args.scratch else None
    if min(args.folds, args.rounds, args.workers, args.min_observations, args.top_k, args.parallel) < 1 or args.folds < 2:
        parser.error('Invalid nonpositive setting.')
    if args.command == 'test':
        return run_tests()
    if args.command == 'prepare':
        return prepare(args)
    if args.command == 'status':
        return status(args)
    if args.command == 'summarize':
        return summarize(args)
    if args.command == 'worker':
        return worker(args)
    return run(args)


if __name__ == '__main__':
    sys.exit(main() or 0)
