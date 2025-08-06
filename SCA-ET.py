RS = 97;
# Database
data0 = pd.read_excel('FEM and TEST data.xlsx', sheet_name='fem')
X_train = data0.iloc[:, 0:12]
y_train = data0.iloc[:, 12]
#
data1 = pd.read_excel('FEM and TEST data.xlsx', sheet_name='exp')
X_test = data1.iloc[:, 0:12]
y_test = data1.iloc[:, 12]

model = ExtraTreesRegressor(bootstrap=False,
                          criterion = 'squared_error',
                          max_depth = None,
                          max_features = None,
                          min_samples_leaf = 1,
                          min_samples_split = 2,
                          n_estimators = 75,
                          random_state=RS)

# Train the model using the training sets
model.fit(X_train, y_train)

# Results
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)