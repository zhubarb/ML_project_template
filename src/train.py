from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import os
import joblib
from src import dispatcher
from src import constants
from fastai.tabular import *

FOLD = 0
MODEL = 'fastai'

FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
    }

if __name__ == '__main__':

    os.chdir("..") # get one level up to project dir
    df = pd.read_csv(constants.TRAINING_DATA)
    df_test = pd.read_csv(constants.TEST_DATA)

    for FOLD in range(5):
        print(FOLD)
        train_df = df.loc[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df.loc[df.kfold==FOLD].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
        valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

        valid_df = valid_df[train_df.columns]

        if MODEL != 'fastai':
            label_encoders = {}
            for c in train_df.columns:
                lbl = preprocessing.LabelEncoder()
                # so the encoder is aware of the values in test as well and knows to encode everything
                lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() +
                        df_test[c].values.tolist())
                train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
                valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
                label_encoders[c] = lbl

            # data is ready to train
            clf = dispatcher.MODELS[MODEL]
            clf.fit(train_df, ytrain)

            preds = clf.predict_proba(valid_df)[:, 1]

            joblib.dump(label_encoders, os.path.join(constants.MODELS, f"{MODEL}_{FOLD}_label_encoder.pkl"))
            joblib.dump(clf, os.path.join(constants.MODELS, f"{MODEL}_{FOLD}.pkl"))
            joblib.dump(train_df.columns, os.path.join(constants.MODELS, f"{MODEL}_{FOLD}_columns.pkl"))

        elif MODEL == 'fastai':
            valid_idx = df.loc[df.kfold==FOLD].index.tolist() # after this, drop kfold col
            df_for_fastai = df.drop(['id', 'kfold'], axis=1) # drop fileds not necessary to predict
            procs = [Categorify]
            cat_vars = train_df.columns.tolist()
            dep_var = 'target'
            data = (TabularList.from_df(df_for_fastai, cat_names=cat_vars,  procs=procs)
                    .split_by_idx(valid_idx)
                    .label_from_df(cols=dep_var)
                    .databunch()
                    )
            # data is ready to train
            clf= tabular_learner(data, layers=[100, 50], ps=[0.001, 0.01], emb_drop=0.04,
                                      metrics=accuracy)
            clf.fit_one_cycle(cyc_len=5, max_lr=5e-3, wd=0.2)
            preds=clf.get_preds(DatasetType.Valid)[0][:,1]

            clf.export(os.path.join(constants.MODELS, f"{MODEL}_{FOLD}.pkl"))
            joblib.dump(train_df.columns, os.path.join(constants.MODELS, f"{MODEL}_{FOLD}_columns.pkl"))

        else:
            raise Exception('Model: $s not defined'%MODEL)


        print(metrics.roc_auc_score(yvalid,preds))

