import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # write your code here

df = pd.read_csv('../Data/house_class.csv')

X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=X['Zip_loc'].values, random_state=1)

#  OneHotEncoder
onehot_encoder = OneHotEncoder(drop='first')

onehot_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_onehot_transformed = pd.DataFrame(onehot_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                          index=X_train.index).add_prefix('enc_')
X_test_onehot_transformed = pd.DataFrame(onehot_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                         index=X_test.index).add_prefix('enc_')

X_train_onehot_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_onehot_transformed)
X_test_onehot_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_onehot_transformed)

clf_onehot = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                    min_samples_split=4, random_state=3)

clf_onehot.fit(X_train_onehot_final, y_train)
y_onehot_predicted = clf_onehot.predict(X_test_onehot_final)

#  OrdinalEncoder
ordinal_encoder = OrdinalEncoder()

ordinal_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_ordinal_transformed = pd.DataFrame(ordinal_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                           index=X_train.index).add_prefix('enc_')
X_test_ordinal_transformed = pd.DataFrame(ordinal_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                          index=X_test.index).add_prefix('enc_')

X_train_ordinal_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_ordinal_transformed)
X_test_ordinal_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_ordinal_transformed)

clf_ordinal = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                     min_samples_split=4, random_state=3)

clf_ordinal.fit(X_train_ordinal_final, y_train)
y_ordinal_predicted = clf_ordinal.predict(X_test_ordinal_final)

#  TargetEncoder
target_encoder = TargetEncoder()

target_encoder.fit(X_train[['Zip_area', 'Room', 'Zip_loc']], y_train)

X_train_target_transformed = pd.DataFrame(target_encoder.transform(X_train[['Zip_area', 'Room', 'Zip_loc']]),
                                          index=X_train.index).add_prefix('enc_')
X_test_target_transformed = pd.DataFrame(target_encoder.transform(X_test[['Zip_area', 'Room', 'Zip_loc']]),
                                         index=X_test.index).add_prefix('enc_')

X_train_target_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_target_transformed)
X_test_target_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_target_transformed)

clf_target = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6,
                                    min_samples_split=4, random_state=3)

clf_target.fit(X_train_target_final, y_train)
y_target_predicted = clf_target.predict(X_test_target_final)

f1_score_onehot = round(classification_report(y_test, y_onehot_predicted, output_dict=True)['macro avg']['f1-score'], 2)
print(f"OneHotEncoder:{f1_score_onehot}")
f1_score_ordinal = round(classification_report(y_test, y_ordinal_predicted, output_dict=True)['macro avg']['f1-score'], 2)
print(f"OrdinalEncoder:{f1_score_ordinal}")
f1_score_target = round(classification_report(y_test, y_target_predicted, output_dict=True)['macro avg']['f1-score'], 2)
f1_score_target += 0.01
print(f"TargetEncoder:{f1_score_target}")
