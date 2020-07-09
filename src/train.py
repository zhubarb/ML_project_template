from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import os
import joblib
from src import dispatcher
from src import constants

FOLD = 0#os.environ.get('FOLD')
MODEL = 'randomforest'

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
        train_df = df.loc[df.kfold.isin(FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
        valid_df = df.loc[df.kfold==FOLD].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(['id', 'target', 'kfold'], axis=1)
        valid_df = valid_df.drop(['id', 'target', 'kfold'], axis=1)

        valid_df = valid_df[train_df.columns]

        label_encoders = {}
        for c in train_df.columns:
            lbl = preprocessing.LabelEncoder()
            # so the encoder is aware of the values in test as well and knows to encode
            # everything
            lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() +
                    df_test[c].values.tolist())
            train_df.loc[:,c] = lbl.transform(train_df[c].values.tolist())
            valid_df.loc[:,c] = lbl.transform(valid_df[c].values.tolist())
            label_encoders[c] = lbl

        # data is ready to train
        clf = dispatcher.MODELS[MODEL]
        clf.fit(train_df, ytrain)

        preds=clf.predict_proba(valid_df)[:,1]
        print(metrics.roc_auc_score(yvalid,preds))

        joblib.dump(label_encoders, os.path.join(constants.MODELS,f"{MODEL}_{FOLD}_label_encoder.pkl"))
        joblib.dump(clf, os.path.join(constants.MODELS,f"{MODEL}_{FOLD}.pkl"))
        joblib.dump(train_df.columns, os.path.join(constants.MODELS,f"{MODEL}_{FOLD}_columns.pkl"))