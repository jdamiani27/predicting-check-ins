from zipfile import ZipFile
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df_train = pd.read_csv(ZipFile('data/train.csv.zip').open('train.csv'), index_col='row_id')
    kNN5 = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree', n_jobs=-1)
    kNN5.fit(df_train[['x', 'y']].values, df_train['place_id'].values)

    train_preds = []
    rows = []

    for row in df_train.itertuples():
        rows.append([row.x, row.y])

        if len(rows) % 1000 == 0:
            train_preds.append(kNN5.predict(rows)[:])
            rows = []
            print "Done predicting this many:", len(train_preds)

    print "Done predicting"
    print "Accuracy:", accuracy_score(df_train['place_id'].values,
                                        train_preds)
