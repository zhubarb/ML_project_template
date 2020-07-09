import os
import pandas as pd
import joblib
import numpy as np
from src import constants
from fastai.tabular import *

MODEL = "fastai"


def predict():
    '''
    Taking the average of the 5-fold trained 5 random forests
    :return:
    '''
    os.chdir("..")  # get one level up to project dir
    df = pd.read_csv(constants.TEST_DATA)
    test_idx = df["id"].values
    predictions = None

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(constants.TEST_DATA)
        if MODEL!= 'fastai':
            encoders = joblib.load(os.path.join(constants.MODELS,f"{MODEL}_{FOLD}_label_encoder.pkl"))
            cols = joblib.load(os.path.join(constants.MODELS,f"{MODEL}_{FOLD}_columns.pkl"))
            for c in encoders:
                print(c)
                lbl = encoders[c]
                df.loc[:, c] = lbl.transform(df[c].values.tolist())

            # data is ready to train
            clf = joblib.load(os.path.join(constants.MODELS,f"{MODEL}_{FOLD}.pkl"))

            df = df[cols]
            preds = clf.predict_proba(df)[:, 1]
        elif MODEL =='fastai':
            cols = joblib.load(os.path.join(constants.MODELS, f"{MODEL}_{FOLD}_columns.pkl"))
            clf = load_learner(path=constants.MODELS, file=f"{MODEL}_{FOLD}.pkl",
                         test=TabularList.from_df(df[cols]) )
            preds=clf.get_preds(ds_type=DatasetType.Test)[0][:,1]
        else:
            raise Exception('Model: $s not defined'%MODEL)

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub


if __name__ == "__main__":
    submission = predict()
    submission.id = submission.id.astype(int) # otherwise id is stored as str, and submission fails
    submission.to_csv(f"models/{MODEL}.csv", index=False)
