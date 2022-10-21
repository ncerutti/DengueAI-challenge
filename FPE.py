from evaluation import evaluate
from sklearn.model_selection import train_test_split

def fit_predict_evaluate(pl,train_clean,test_clean,figname):

    Y=train_clean['total_cases']
    X=train_clean.drop('total_cases', axis=1)

    X_train, X_test, y_train, y_test=train_test_split(X, Y, random_state=42)

    # #Split Train-test
    # sj_train_subtrain = sj_train.head(800)
    # sj_train_subtest = sj_train.tail(sj_train.shape[0] - 800)
    
    # sj_train_subtrain_X = sj_train_subtrain.drop(['total_cases'], axis=1)
    # sj_train_subtrain_y = sj_train_subtrain['total_cases']

    # sj_train_subtest_X = sj_train_subtest.drop(['total_cases'], axis=1)
    # sj_train_subtest_y = sj_train_subtest['total_cases']

    #Fit - Predict
    print('Fitting the model...')

    # pl.fit(sj_train_subtrain_X, sj_train_subtrain_y)
    # sj_train_subtest_yhat = pl.predict(sj_train_subtest_X)

    pl.fit(X_train, y_train)
    #y_test_predict = pl.predict(X_test)

    #this is baseline.py:
    test_clean['total_cases'] = pl.predict(test_clean)
    test_clean['total_cases'] = test_clean['total_cases'].astype(int)

    submission=test_clean[['city', 'year', 'weekofyear', 'total_cases']]
    submission['city']=submission['city'].map({1:'sj', 0:'iq'})
    submission.to_csv('submission.csv', index=False)
    print(submission.dtypes)

    #pl.fit(sj_train_subtrain_X, sj_train_subtrain_y)
    #sj_train_subtest_yhat = pl.predict(sj_train_subtest_X)

    #Evaluate
    #print('Predicting and evaluating...')
    #scores = evaluate(sj_train_subtest_y, sj_train_subtest_yhat,figname,visualize=True)

    #return scores