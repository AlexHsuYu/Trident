import os, sys, joblib, click
import pandas as pd, numpy as np
from scipy import fftpack
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from . import PROJECT_ROOT, BASE_PATH, CLASS_MAP
from .utils import (
    generate_file_index, read_data, preprocess_input, dataset
)
from .config import Config

@click.group()
def main():
    pass

@main.command('train')
def rand_forest_train():
    ''' preparing dataset '''
    # load file index if exists, create one if not exist
    try:
        file_index = pd.read_csv(os.path.join(Config.TRAIN_PATH, 'index.csv'))
    except FileNotFoundError:
        generate_file_index(Config.TRAIN_PATH)
        file_index = pd.read_csv(os.path.join(Config.TRAIN_PATH, 'index.csv'))

    train_df, valid_df = train_test_split(file_index, test_size=0.25, shuffle=True)
    train_df = train_df.reset_index(drop=True)

    # load dataset from dataframe of indices
    X, y = dataset(train_df, Config.TRAIN_PATH, mode='train')

    valid_df.to_csv(os.path.join(PROJECT_ROOT, 'assets/valid_data.csv'), index=False)

    # random forest setup
    classifier = RandomForestClassifier(
        n_estimators=30,
        max_depth=13,
        max_leaf_nodes=128,
        min_samples_leaf=20,
        max_features=7
    )

    classifier.fit(X, y) # training

    ''' evaluation '''
    X, y = dataset(valid_df, Config.TRAIN_PATH, mode='train')
    pred = classifier.predict(X)
    acc = accuracy_score(valid_df['category'], pred)
    print('model has accuracy: {}'.format(acc))

    joblib.dump(classifier, os.path.join(PROJECT_ROOT, 'weights/coral.joblib'))

@main.command('test')
@click.option('--save-path', default=os.path.join(PROJECT_ROOT, 'assets'))
def rand_forest_test(save_path):
    # load file index if exists, create one if not exist
    try:
        test_index = pd.read_csv(os.path.join(Config.TEST_PATH, 'index.csv'))
    except FileNotFoundError:
        generate_file_index(Config.TEST_PATH)
        test_index = pd.read_csv(os.path.join(Config.TEST_PATH, 'index.csv'))

    X, _ = dataset(test_index, Config.TEST_PATH, mode='test')

    # load model
    classifier = joblib.load(os.path.join(PROJECT_ROOT, 'weights/coral_success.joblib'))

    pred = classifier.predict(X)
    test_index['category'] = pd.Series(pred).map(CLASS_MAP)

    # sort by filename (the right way)
    test_index['sort'] = test_index['path'].str.extract('(\d+)', expand=False).astype(int)
    test_index = test_index.sort_values('sort').reset_index(drop=True)
    test_index = test_index.drop('sort', axis=1)

    # save model to file
    print(test_index)
    test_index.to_csv(os.path.join(save_path, 'stage1_test_result.csv'), index=False)

# def main():
#     ''' reading data '''
#     file_index = pd.read_csv(Config.FILE_PATH)
#     train_data, valid_data = train_test_split(
#         file_index,
#         test_size=0.2,
#         shuffle=True
#     )
#     valid_data, test_data = train_test_split(
#         valid_data,
#         test_size=0.5,
#         shuffle=True
#     )

#     # save test data for later use
#     test_data.to_csv(os.path.join(PROJECT_ROOT, 'assets/cnn_test.csv'), index=False)
#     # print(train_data)
#     # print(train_data.index.values)
#     ''' generators '''
#     # custom generators doesn't work
#     # try the ones provided by keras
#     train_gen = Tsunami(
#         Config.DATA_PATH,
#         train_data,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=True
#     )
#     valid_gen = Tsunami(
#         Config.DATA_PATH,
#         valid_data,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False
#     )

#     ''' load model & train '''
#     tnn = TridentNN()
#     tnn.summary()
#     # sys.exit(0)

#     try:
#         os.makedirs(os.path.join(PROJECT_ROOT, 'weights'))
#     except OSError:
#         pass

#     early = KC.EarlyStopping(
#         monitor='val_loss',
#         mode='min',
#         patience=10,
#         min_delta=0.005
#     )
#     ckpt = KC.ModelCheckpoint(
#         os.path.join(PROJECT_ROOT, 'weights/conv_w{}s{}.h5'.format(Config.WINDOW_SIZE, Config.STRIDE)),
#         monitor='val_loss',
#         save_best_only=True,
#         save_weights_only=True
#     )

#     tnn.fit_generator(
#         train_gen,
#         validation_data=valid_gen,
#         callbacks=[early, ckpt],
#         # use_multiprocessing=True,
#         # workers=3,
#         epochs=Config.EPOCHS
#     )

# def cnn_test():
#     ''' load data '''
#     test_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'assets/cnn_test.csv'))
#     # print(test_data)
#     test_gen = Tsunami(
#         Config.DATA_PATH,
#         test_data,
#         batch_size=Config.BATCH_SIZE,
#         shuffle=False
#     )

#     ''' load model '''
#     tnn = TridentNN()
#     # load trained weights
#     tnn.load_weights(
#         os.path.join(PROJECT_ROOT, 'weights/conv_w{}s{}.h5'.format(Config.WINDOW_SIZE, Config.STRIDE))
#     )

#     prob = tnn.predict_generator(test_gen)
#     # print(results)
#     results = np.argmax(prob, axis=1)
#     print(results)

#     truth = np.zeros(results.shape, dtype=np.int64)
#     for i, t in enumerate(test_gen):
#         # print(np.argmax(t[1], axis=1))
#         truth[i*Config.BATCH_SIZE : (i+1)*Config.BATCH_SIZE] = np.argmax(t[1], axis=1)
#         if i + 1 == len(test_gen):
#             break

#     print(truth)

#     acc = accuracy_score(truth, results)
#     print(acc)
