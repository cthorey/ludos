from __future__ import print_function

import datetime
import hashlib
import json
import os
import sqlite3
import time
import uuid
from functools import wraps

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from box import Box
from clearml import Task
from ludos.models import common
from ludos.utils import dictionary, orm, s3

ROOT_DIR = os.environ['ROOT_DIR']


def spin_on_error(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        print('*' * 50)
        print('TrainingLand')
        print('*' * 50)
        try:
            function(*args, **kwargs)
        except Exception as e:
            print(e)
            print('sleeping now')
            time.sleep(10000000)


def retrieve_summary(
        model_task='.*',
        model_name='.*',
        dataset_name='.*',
        expname='.*',
        score_name='.*',
        split='.*',
        model_id=None):
    """
    Retrieve previously computed summary
    """
    with orm.session_scope() as sess:
        query = sess.query(orm.Modelzoo).\
                filter(orm.Modelzoo.model_name.op('~')(model_name)).\
                filter(orm.Modelzoo.dataset_name.op('~')(dataset_name)).\
                filter(orm.Modelzoo.model_task.op('~')(model_task)).\
                filter(orm.Modelzoo.score_name.op('~')(score_name)).\
                filter(orm.Modelzoo.split.op('~')(split)).\
                filter(orm.Modelzoo.expname.op('~')(expname))
        if model_id is not None:
            query = query.filter(orm.Modelzoo.model_id == model_id)
        records = query.all()
        entries = []
        for record in records:
            record = Box(record.__dict__)
            record.pop('_sa_instance_state')
            entries.append(record)
    return pd.DataFrame(entries)


class Experiment(object):
    """Experiment manager - given a task and a model, the next step
    is usually to experiment with the parameters to otpimize some
    kind of metrics.

    The class help you managing your experiment process by create
    new experiment ID for each new trial (a new set of parameters).

    We will use indifferentl task/trial in the following :).

    Args:
        model_task (str): Task of the model
        model_name (str): Name of the model
        config_name (str): Name of the config

    """
    def __init__(self, model_task: str, model_name: str, config_name: str):
        self.model_task = model_task
        self.model_name = model_name
        self.config_name = config_name
        self.bucket = s3.S3Bucket(bucket_name='s3ludos')
        self.model_folder = os.path.join(
            ROOT_DIR, "models", self.model_task, self.model_name)

    def get_model_id(self, expname):
        """Return a uniaue ID for the model
        """
        entry = dict(
            model_task=self.model_task,
            model_name=self.model_name,
            expname=expname)
        return hashlib.sha1(json.dumps(entry).encode()).hexdigest()

    def next_trial_name(self):
        """Returns the next trial name
        """
        with orm.session_scope() as sess:
            results = sess.query(orm.Modelzoo).filter(
                orm.Modelzoo.model_task == self.model_task).filter(
                    orm.Modelzoo.model_name == self.model_name).filter(
                        orm.Modelzoo.expname.op('~')(
                            "{}.*".format(self.config_name))).all()
            expnames = [r.expname for r in results]
            if not expnames:
                idx = 0
            else:
                idx = max([int(expname.split('t')[-1]) for expname in expnames])
        expname = "{}t{}".format(self.config_name, idx + 1)
        return expname

    def log_task(self, expname: str):
        """Log the existence of this task to the
        modelzoo.
        """
        with orm.session_scope() as sess:
            entry = dict(
                model_task=self.model_task,
                model_name=self.model_name,
                expname=expname)
            model_id = self.get_model_id(expname)
            entry = orm.Modelzoo(
                model_id=model_id,
                status="started",
                created_on=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                **entry)
            sess.add(entry)

    def update_task(self, expname, **kwargs):
        """Update the modelzoo record for this task.
        """
        with orm.session_scope() as sess:
            entry = dict(
                model_task=self.model_task,
                model_name=self.model_name,
                expname=expname)
            model_id = hashlib.sha1(json.dumps(entry).encode()).hexdigest()
            entry = sess.query(
                orm.Modelzoo).filter(orm.Modelzoo.model_id == model_id).all()[0]
            for key, value in kwargs.items():
                setattr(entry, key, value)

    def start(self, expname: str):
        self.log_task(expname)

    def end(self, expname: str, status: str):
        self.update_task(expname, status=status)


class LightningExperiment(Experiment):
    """Experiment taking advantage of trains (from allegro.ai).

    The trains client Task will log everything for you : ).

    Examples:

        .. code-block:: python

           # Get your config
           cfg = get_config()
           task = LightningExperimentightningExperiment(model_task="instance_segmentation",
                               model_name="fancy_detectron2",
                               config_name=config_name)
           # Get nex trial name for this exp -- config_nametX
           expname = task.next_trial_name()
           # Get a Trains task
           logger = task.start(expname)
           # Connect the config - should be a dict
           logger.connect(cfg)
           # Train your model and get your results as a dict
           ...
           # report back
           task.upload_checkpoints(get_checkpoint_file(),
                            expname,
                            cfg=trainer.cfg_to_dict(tr.cfg),
                            meta_data=meta_data,
                            train_meta=train_meta,
                            **results)
    """
    def start(self, expname: str):
        self.log_task(expname)
        task = Task.init(
            project_name=self.model_task,
            task_name="{}/{}".format(self.model_name, expname),
            reuse_last_task_id=False)
        return task

    def upload_checkpoints(self, checkpoint_path, expname, **kwargs):
        if checkpoint_path == "":
            raise ValueError('Could not retrieve checkpoints')
        # add some stuff in there
        data = torch.load(checkpoint_path)
        data['extra'] = kwargs
        torch.save(data, checkpoint_path)
        # upload to s3
        key = 'models/{}/{}/{}_weights.pth'.format(
            self.model_task, self.model_name, expname)
        print('Uploading {} to s3'.format(key))
        self.bucket.upload_from_file(checkpoint_path, key, overwrite=True)

    def end(
            self,
            expname: str,
            dataset_name: str,
            score_name: str,
            score,
            split: str = "validation",
            status: str = "success",
            maintainer: str = "clement"):
        self.update_task(
            expname,
            dataset_name=dataset_name,
            score_name=score_name,
            score=score,
            status=status,
            split=split,
            maintainer=maintainer)


class Optuna(object):
    """Optuna based hyperparameters explorer.

    The task of finding a good set of hyperparameters can be
    tiring... or you can use Optuna :).

    Args:
        model_task (str): Task of the model
        model_name (str): Name of the model
        exploration_name (str): Name of the exploration
        training_method (str): Training method
        direction (str): Pass to optuna to decide what is the good
        direction to look for

    Note:
        The exploration should exist in
        training_config/models/model_task/model_name/explorations/exploration_name.yaml

        It should have three sections:

        1. common: defined the arguments ultimately passed to the train method
        2. base_cfg: defined the base config (which overwrite the config.py) common to all
        3. parameter_space: defined the parameter space itself


    Example:

        .. code-block::

           parameter_space:
             g0:
               method: 'suggest_loguniform'
               name: 'solver.default_lr'
               values: [0.00001, 0.01]
             g1:
               method: 'suggest_categorical'
               name: 'loss.loss_prob.params.weight'
               values: [[[1.0, 1.0], [0.49,0.51]]]
             g2:
               method: 'suggest_categorical'
               name: 'model.arch_details.head.lin_ftrs'
               values: [[[512,1024,2048], [4096, 4096], [512,256], [256,256], [1024, 1024]]]
    """
    def __init__(
            self,
            model_task,
            model_name,
            exploration_name,
            training_method,
            direction='minimize',
            reset=False):
        """
        Optuna based explorer.
        """
        self.config_folder = os.path.join(
            common.CONFIG_FOLDER, 'models', model_task, model_name)
        self.exploration_name = exploration_name
        path = os.path.join(
            self.config_folder, 'explorations',
            '{}.yaml'.format(exploration_name))
        self.exploration = Box.from_yaml(open(path, 'r'))

        optuna_folder = os.path.join(
            self.config_folder, 'explorations', 'optuna')
        if not os.path.isdir(optuna_folder):
            os.makedirs(optuna_folder)
        db_path = '{}/{}.db'.format(optuna_folder, self.exploration_name)
        if reset and os.path.isfile(db_path):
            os.remove(db_path)

        self.study = optuna.create_study(
            study_name=self.exploration_name,
            storage='sqlite:///{}'.format(db_path),
            load_if_exists=True,
            direction=direction)
        self.training_method = training_method

    def suggest_params(self, trial):
        params = dict()
        parameter_space = self.exploration.parameter_space.to_dict()
        for key, param in parameter_space.items():
            v = param['values']
            n = param['name']
            if param['method'] == 'suggest_categorical':
                v = [param['values']]
            params[param['name']] = getattr(trial, param['method'])(n, *v)
        return params

    def optimize(self, trial):
        base_cfg = self.exploration.base_cfg.to_dict()
        base = dictionary.flatten(base_cfg, sep='.')
        params = self.suggest_params(trial)
        base.update(params)
        new_cfg = dictionary.unflatten(base, sep='.')
        cfg_name = '{}_{}'.format(self.exploration_name, uuid.uuid4().hex)
        with open(os.path.join(self.config_folder, '{}.yaml'.format(cfg_name)),
                  'w+') as f:
            yaml.dump(new_cfg, f)
        return self.training_method(cfg_name, **self.exploration.common)

    def run(self, n_trials=1):
        self.study.optimize(self.optimize, n_trials=n_trials)


class Bender(object):
    """Random exploration for hyperparameters

    Args:
        model_task (str): Task for the model
        model_name (str): Name of the model
        exploration_name (str): Name of the exploration

    Note:
        The exploration should exist in
        training_config/models/model_task/model_name/explorations/exploration_name.yaml

        It should have three sections:

        1. common: defined the arguments ultimately passed to the train method
        2. base_cfg: defined the base config (which overwrite the config.py) common to all
        3. parameter_space: defined the parameter space itself


    Example:
        Each param can be off three types [uniform, params, constant]

        .. code-block::

           parameter_space:
             g0:
               type: 'uniform'
               name: 'input.radius'
               values:
                 low: 0.08
                 high: 0.15
             g1:
               name: 'input.voxelize'
               type: 'params'
               values: [0.001, 0.005, 0.01]
             g2:
               type: 'uniform'
               name: 'solver.default_lr'
               values:
                 low: 0.00006
                 high: 0.0005
    """
    def __init__(self, model_task, model_name, exploration_name):
        self.config_folder = os.path.join(
            common.CONFIG_FOLDER, 'models', model_task, model_name)
        self.exploration_name = exploration_name
        path = os.path.join(
            self.config_folder, 'explorations',
            '{}.yaml'.format(exploration_name))
        self.exploration = Box.from_yaml(open(path, 'r'))

    def _suggest_params(self):
        params = dict()
        parameter_space = self.exploration.parameter_space.to_dict()
        for key, param in parameter_space.items():
            if param['type'] == 'params':
                probs = param.get('probs', None)
                values = param['values']
                p = dict(zip(range(len(values)), values))
                value = p[np.random.choice(list(p.keys()), p=probs).item()]
                params[param['name']] = value
            elif param['type'] == 'uniform':
                values = param['values']
                params[param['name']] = np.random.uniform(**values)
            elif param['type'] == 'constant':
                params[param['name']] = param['value']
            else:
                raise RuntimeError('Wrong parameter type')
        return params

    def suggest(self):
        """Suggest a new combination of parameters

        Returns:
            cfg_name (name): Name of the autogenerated config
            with the new parameters

        """
        base_cfg = self.exploration.base_cfg.to_dict()
        base = dictionary.flatten(base_cfg, sep='.')
        params = self._suggest_params()
        base.update(params)
        new_cfg = dictionary.unflatten(base, sep='.')
        cfg_name = '{}_{}'.format(self.exploration_name, uuid.uuid4().hex)
        with open(os.path.join(self.config_folder, '{}.yaml'.format(cfg_name)),
                  'w+') as f:
            yaml.dump(new_cfg, f)
        return cfg_name
