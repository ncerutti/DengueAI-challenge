from turtle import setworldcoordinates
from evaluation import evaluate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

def fit_predict_evaluate(
    pl,
    train_clean,
    test_clean,
    test_features,
    expname,
    crossval=False,
    create_submission=False,
):

    y = train_clean["total_cases"]
    X = train_clean.drop("total_cases", axis=1)

    if crossval:
        cycles=5
        scores = cross_val_score(pl, X, y, cv=cycles, scoring='neg_mean_absolute_error')
        print("Over %i cycles: MAE: %0.2f with std: %0.2f" % (cycles,-1*scores.mean(), -1*scores.std()))
    else:
        # sub-train-test split of the train dataset
        #random_states = [42,43,44,45,46]
        random_states = [42]

        # cycle through different random states
        scoresL = []
        for i, random_state in enumerate(random_states):
            print(f"Fit-predict cycle:{i}")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=random_state
            )

            # Fit - Predict
            print("Fitting the model...")
            pl.fit(X_train, y_train)

            # Evaluate
            print("Predicting and evaluating...")
            y_test_predict = pl.predict(X_test)
            scoresi = evaluate(
                y_test, y_test_predict, expname + "_" + str(i), visualize=True
            )
            scoresL.append(scoresi)

        # calculate average scores
        keys = scoresL[0].keys()
        scores = {}
        for i, scoresi in enumerate(scoresL):
            for key in keys:
                if i == 0:
                    scores[key] = scoresi[key]
                else:
                    scores[key] = scores[key] + scoresi[key]
        for key in keys:
            scores[key] = scores[key] / len(scoresL)

        if len(random_states>1):
            print(
                f"Averaged scores over {len(random_states)} train-test splits:\n R2 = {scores['R2']} \n RMSE:{scores['RMSE']} \n MAE:{scores['MAE']}"
            )

    # Create a submission
    if create_submission:

        # retrain the model using the entire dataset:
        pl.fit(X, y)

        fname = expname + "_submission" + ".csv"
        submission = export_submission(test_features, test_clean, fname, pl)
        print(f"saved submission file: {fname}")
    return scores


def export_submission(test_clean, test_processed, out_path, pl):

    test_clean["total_cases"] = pl.predict(test_processed)
    test_clean["total_cases"] = test_clean["total_cases"].astype(int)
    submission = test_clean[["city", "year", "weekofyear", "total_cases"]]
    #submission["city"] = submission["city"].map({1: "sj", 0: "iq"})
    submission.to_csv(out_path, index=False)
    # print(submission.dtypes)
    
    return submission
