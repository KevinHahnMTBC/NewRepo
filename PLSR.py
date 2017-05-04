 Your PLSR models here.
# Number of datapoints and features 
n = 1000
p = 7

#hey

# Create random normally distributed data for parameters.
    # Makes an array of tuples based on the number of parameters/features
X = np.random.normal(size=n * p).reshape((n, p))

# Create normally distributed outcome related to parameters but with noise.
    # Messing with the values of tuples in X to prep the correlation map 
y = X[:, 0]
+ 0.72 * X[:, 1] 
+ 300000 * X[:, 2]
+ np.random.normal(size=n) * 4000
+ 1234

# Check out correlations. First column is the outcome.
f, ax = plt.subplots(figsize=(12, 9))
corrmat = pd.DataFrame(np.insert(X, 0, y, axis=1)).corr()
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

# Fit a linear model with all 10 features.
regr = linear_model.LinearRegression()
regr.fit(X, y)

# Save predicted values.
Y_pred = regr.predict(X)
print('R-squared regression:', regr.score(X, y))

# Fit a linear model using Partial Least Squares Regression.
# Reduce feature space to 3 dimensions.
pls1 = PLSRegression(n_components=3)

# Reduce X to R(X) and regress on y.
pls1.fit(X, y)

# Save predicted values.
Y_PLS_pred = pls1.predict(X)
print('R-squared PLSR:', pls1.score(X, y))

# Compare the predictions of the two models
plt.scatter(Y_pred,Y_PLS_pred) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 3 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()



# Replicate PLSR with same data to reduce feature space to 2 dimensions based on corrmap
pls2 = PLSRegression(n_components=2)

pls2.fit(X, y)

Y_PLS_pred2 = pls2.predict(X)
print('R-squared PLSR:', pls2.score(X, y))

plt.scatter(Y_pred,Y_PLS_pred2) 
plt.xlabel('Predicted by original 10 features')
plt.ylabel('Predicted by 3 features')
plt.title('Comparing LR and PLSR predictions')
plt.show()