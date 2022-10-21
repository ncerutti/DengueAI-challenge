from evaluation import evaluate

def fit_predict_evaluate(pl,sj_train,figname):
    #Split Train-test
    sj_train_subtrain = sj_train.head(800)
    sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)
    
    sj_train_subtrain_X = sj_train_subtrain.drop(['total_cases'], axis=1)
    sj_train_subtrain_y = sj_train_subtrain['total_cases']

    sj_train_subtest_X = sj_train_subtest.drop(['total_cases'], axis=1)
    sj_train_subtest_y = sj_train_subtest['total_cases']

    #Fit - Predict
    print('Fitting the model...')
    pl.fit(sj_train_subtrain_X, sj_train_subtrain_y)
    sj_train_subtest_yhat = pl.predict(sj_train_subtest_X)

    #Evaluate
    print('Predicting and evaluating...')
    scores = evaluate(sj_train_subtest_y, sj_train_subtest_yhat,figname,visualize=True)

    return scores