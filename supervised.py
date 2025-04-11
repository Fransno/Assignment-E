import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

def supervisedrf(x_train, x_test, y_train, y_test):

    results = []
    
    for n_trees in [50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=n_trees)
        rf.fit(x_train, y_train)
        train_acc = accuracy_score(y_train, rf.predict(x_train))
        test_acc = accuracy_score(y_test, rf.predict(x_test))
        results.append({
            'Model': f'RF (n={n_trees})',
            'Test Accuracy': test_acc,
            'Train-Test Gap': train_acc - test_acc
        })

    results_df = pd.DataFrame(results)
    print("\n---------------- Random Forest ----------------")
    print(results_df[['Model', 'Test Accuracy', 'Train-Test Gap']].sort_values('Test Accuracy', ascending=False))

    best_rf = results_df.iloc[results_df['Test Accuracy'].idxmax()]
    print("\nBest Model:")
    print(classification_report(y_test, 
          RandomForestClassifier(n_estimators=int(best_rf['Model'].split('=')[1][:-1]), random_state=1)
          .fit(x_train, y_train).predict(x_test)))



def supervisedxgb(x_train, x_test, y_train, y_test):

    results = []

    for max_depth in [3, 5, 7]:
        xgb = XGBClassifier(max_depth=max_depth, objective='multi:softmax', num_class=3, random_state=1)
        xgb.fit(x_train, y_train)
        test_acc = accuracy_score(y_test, xgb.predict(x_test))
        train_acc = accuracy_score(y_train, xgb.predict(x_train))
        results.append({
            'Model': f'XGB (depth={max_depth})',
            'Test Accuracy': test_acc,
            'Train-Test Gap': train_acc - test_acc,
            'Report': classification_report(y_test, xgb.predict(x_test), output_dict=True)
        })
    
    results_df = pd.DataFrame(results)
    print("\n---------------- XGBoost ----------------")
    print(results_df[['Model', 'Test Accuracy', 'Train-Test Gap']].sort_values('Test Accuracy', ascending=False))

    best_xgb = results_df.iloc[results_df['Test Accuracy'].idxmax()]
    best_depth = int(best_xgb['Model'].split('=')[1].replace(')', ''))  # Fix for "5)"
    
    print("\nBest Model:")
    print(classification_report(y_test, 
          XGBClassifier(max_depth=best_depth,
                       objective='multi:softmax', 
                       num_class=3, 
                       random_state=1)
          .fit(x_train, y_train).predict(x_test)))