# -*- coding: utf-8 -*-
from __future__ import print_function
import shutil
import time
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import trange
from tqdm import tqdm, tqdm_notebook

import keras
import keras.backend as K
from keras.callbacks import (ModelCheckpoint,
                             EarlyStopping,
                             TensorBoard,
                             ReduceLROnPlateau)
from keras.metrics import top_k_categorical_accuracy

import sklearn
from sklearn.model_selection import StratifiedKFold, KFold

from scipy.io import wavfile
import librosa

import constants
from constants import *
import utils
import archs


def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def top_2_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def raw_labels_to_indices(raw_labels):
    labels = LABELS or list(raw_labels.unique())
    label_idx = {label: i for i, label in enumerate(LABELS)}
    labels_enc = raw_labels.apply(lambda x: label_idx[x])
    return labels_enc


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5


class PrepareData:
    important_keys = ['audio_length', 'dim', 'sampling_rate', 'n_mfcc', 'use_mfcc']
    
    def __init__(self, preprocessing_fn=None):
        self.preprocessing_fn = (
            preprocessing_fn or
            audio_norm)
        self.X = None
        self.previous_params = {
            key: None for key in PrepareData.important_keys
        }
        self.previous_hash = None
        
    def _prepare_data(self, df, config, data_dir):
        X = np.empty(shape=(df.shape[0], *config.dim))
        input_length = config.audio_length
        for i, fname in enumerate(df.index):
            file_path = os.path.join(data_dir, fname)
            data, _ = librosa.core.load(
                file_path, sr=config.sampling_rate,
                res_type='kaiser_fast')

            if len(data) > input_length:
                max_offset = len(data) - input_length
                offset = np.random.randint(max_offset)
                data = data[offset:(input_length+offset)]
            else:
                if input_length > len(data):
                    max_offset = input_length - len(data)
                    offset = np.random.randint(max_offset)
                else:
                    offset = 0
                data = np.pad(data, (offset, input_length - len(data) - offset), "constant")
            
            if config.use_mfcc:
                data = librosa.feature.mfcc(
                    data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
            else:
                data = self.preprocessing_fn(data)[:, np.newaxis]
            X[i,] = data
        return X
        
    def __call__(self, df, config, data_dir):
        new_params = {
            key: getattr(config, key) for key in PrepareData.important_keys
        }
        self.previous_params = {
            key: self.previous_params[key] for key in PrepareData.important_keys
        }
        new_hash = df.nframes.sum()  # tricky way!
        if new_params == self.previous_params and new_hash == self.previous_hash:
            return self.X
        else:
            self.previous_params = new_params
            self.previous_hash = new_hash
            self.X = self._prepare_data(df, config, data_dir)
            return self.X


def get_class_weights(y, smooth_factor=0):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}


class StoreStatistics(keras.callbacks.Callback):
    metrics_to_watch = ['loss', 'map_3']  # just for reference
    
    def __init__(self, path, slide_tick_for_first_epoch, train_data, validation_data):
        self.path = path
        self.slide_tick_for_first_epoch = slide_tick_for_first_epoch
        self.train_data = train_data
        self.validation_data = validation_data
        
    def log_messages(self, logs):
        def _map_k(y_true, preds, k=3):
            label_idx_true = np.argmax(y_true, axis=1)
            sum_ = 0
            for index in range(preds.shape[0]):
                top3_now = np.argpartition(preds[index], -k)[-k:]
                top3_now = top3_now[np.argsort(preds[index][top3_now])]
                sum_ += np.sum(np.equal(top3_now, label_idx_true[index]).astype(float) * np.array([0.33, 0.5, 1.0]))
            return sum_ / y_true.shape[0]
                              
        X_train = self.train_data[0]
        y_train = self.train_data[1]
        X_val = self.validation_data[0]
        y_val = self.validation_data[1]
        
        for prefix, x, y in [('val', X_val, y_val), ('train', X_train, y_train)]:
            preds = self.model.predict(x)
            self.log_frame.loc[self.tick, '{}_loss'.format(prefix)] = (
                logs.get('{}loss'.format('' if prefix == 'train' else 'val_')))
            self.log_frame.loc[self.tick, '{}_{}'.format(prefix, 'map_3')] = (
                _map_k(y, preds, k=3))
            self.log_frame.loc[self.tick, '{}_{}'
                .format(prefix, 'most_frequent_proportion')] = (
                    np.max(np.sum(preds, axis=0)) / preds.shape[0])
            self.log_frame.loc[self.tick, 'time'] = time.time() - self.time_start
            self.log_frame.loc[self.tick, 'epoch'] = self.epoch
            a = logs.get('{}top_1_accuracy'.format('val_' if prefix == 'val' else ''))
            b = logs.get('{}top_2_accuracy'.format('val_' if prefix == 'val' else ''))
            c = logs.get('{}top_3_accuracy'.format('val_' if prefix == 'val' else ''))
            self.log_frame.loc[self.tick, '{}_acc_3'.format(prefix)] = c
            self.log_frame.loc[self.tick, '{}_acc_2'.format(prefix)] = b
            self.log_frame.loc[self.tick, '{}_acc_1'.format(prefix)] = a
            self.log_frame.loc[self.tick, '{}_map_3_with_keras'.format(prefix)] = (
                a + (b - a) * 0.5 + (c - b) * 0.33)

        
    def on_train_begin(self, logs={}):
        self.tick = 0  # count of updates (forward passes)
        self.epoch = 0
        self.log_frame = pd.DataFrame(
            columns=['train_{}'.format(metric) for metric in StoreStatistics.metrics_to_watch] +
                    ['val_{}'.format(metric) for metric in StoreStatistics.metrics_to_watch] +
                    ['time'])
        self.time_start = time.time()
        
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.log_messages(logs)
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.tick += 1
        if self.slide_tick_for_first_epoch:
            if self.epoch == 0 and self.tick % self.slide_tick_for_first_epoch == 0:
                self.log_messages(logs)
        return
    
    
def train_and_store_results(
        name_experiment,
        func_to_create_model,
        config,
        X_train,
        y_train,
        y_train_label_idx,
        X_test,
        slide_tick_for_first_epoch,
        verbose_keras,
        make_predictions,
        do_kfold):
    y_train_label_idx_must_be = np.argmax(y_train, axis=1)
    if len(y_train_label_idx.shape) != len(y_train_label_idx_must_be.shape):
        message = 'Wow! {} and {} shapes of Ys are not consistent'
        print(message, file=sys.stderr)
        raise ValueError(message)
    if not np.all(y_train_label_idx_must_be == y_train_label_idx):
        message = 'Wow! shapes of Ys are not consistent'
        print(message, file=sys.stderr)
        raise ValueError(message)

    prediction_folder = os.path.join(constants.PREDICTIONS, name_experiment)
    tensorboard_folder = os.path.join(constants.TENSORBOARDS, name_experiment)
    log_folder = os.path.join(constants.LOGS, name_experiment)
    model_folder = os.path.join(constants.MODELS, name_experiment)
    if os.path.exists(prediction_folder):
        shutil.rmtree(prediction_folder)
    os.mkdir(prediction_folder)
    if os.path.exists(model_folder):
        shutil.rmtree(model_folder)
    os.mkdir(model_folder)
    if os.path.exists(log_folder):
        shutil.rmtree(log_folder)
    os.mkdir(log_folder)
    if os.path.exists(tensorboard_folder):
        shutil.rmtree(tensorboard_folder)
    # this folder is for tensorflow that is why other behavior
        
    if do_kfold:
        if config.complete_run:
            skf = StratifiedKFold(n_splits=config.n_folds)
            splits = skf.split(X=np.zeros(X_train.shape[0]), y=y_train_label_idx)
            # sklearn' skf is weird: requires X as the first arg
        else:
            kf = KFold(n_splits=config.n_folds)
            splits = kf.split(X=np.zeros(X_train.shape[0]), y=None)
    else:
        if config.complete_run:
            splits = [
                sklearn.model_selection.train_test_split(
                    range(X_train.shape[0]),
                    test_size=0.11,
                    stratify=y_train_label_idx)]
        else:
            splits = [
                sklearn.model_selection.train_test_split(
                    range(X_train.shape[0]),
                    test_size=0.11)]
        
    for i, pair_of_indices in enumerate(splits):
        K.clear_session()
        train_split = pair_of_indices[0]
        val_split = pair_of_indices[1]
        X, y, X_val, y_val = (X_train[train_split], y_train[train_split],
                              X_train[val_split], y_train[val_split])
        model_fold_path = os.path.join(model_folder, 'best_%d.h5' % i)
        log_fold_path = os.path.join(log_folder, 'history_%i' % i)
        tensorboard_fold_folder = os.path.join(tensorboard_folder, 'fold_%i' % i)
        store_statistics_callback = StoreStatistics(
            path=log_fold_path,
            slide_tick_for_first_epoch=slide_tick_for_first_epoch,
            train_data=(X, y),
            validation_data=(X_val, y_val))
        checkpoint = ModelCheckpoint(
            model_fold_path,
            monitor='val_loss',
            verbose=verbose_keras, save_best_only=True)
        early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
        tb = TensorBoard(log_dir=tensorboard_fold_folder, write_graph=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, min_lr=0.00001)
        callbacks_list = [checkpoint, early, tb, store_statistics_callback, reduce_lr]
        model = func_to_create_model(config)
        history = model.fit(
            X, y, validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            batch_size=config.batch_size,
            epochs=config.max_epochs,
            verbose=verbose_keras,
            class_weight=config.class_weights)
        
        # Save log_frame to csv
        store_statistics_callback.log_frame.to_csv(log_fold_path, index=True, index_label='tick')
        
        if make_predictions:
            model.load_weights(model_fold_path)
            
            # Save train predictions
            predictions = model.predict(X_train, batch_size=64, verbose=verbose_keras)
            np.save(prediction_folder + "/train_predictions_%d.npy" % i, predictions)

            # Save test predictions
            predictions = model.predict(X_test, batch_size=64, verbose=verbose_keras)
            np.save(prediction_folder + "/test_predictions_%d.npy" % i, predictions)

            # Make a submission file
            top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
            predicted_labels = [' '.join(list(x)) for x in top_3]
            test = pd.read_csv(os.path.join(constants.DATA, 'sample_submission.csv'))
            if not config.complete_run:
                test = test.iloc[:config.size_when_not_complete_run, :]
            test['label'] = predicted_labels
            test[['label']].to_csv(prediction_folder + "/predictions_%d.csv" % i)
        
    return store_statistics_callback  # only one of all folds! just for debugging
