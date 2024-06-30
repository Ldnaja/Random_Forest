# Project - Decision Trees

## Summary

This repository contains the code and analysis for a project which involves building and evaluating Random Forest models to predict the maximum temperature in Seattle. The dataset used is `temps_extended.xlsx`, and various hyperparameter configurations were tested, including the use of bootstrap, number of estimators, maximum tree depth, minimum samples per leaf, and minimum samples required to split a node.

## Methodology

1. **Dataset**: `temps_extended.xlsx`
   - Variables used:
     - `year`: Year of the record.
     - `ws_1`: Average wind speed one day before.
     - `temp_2`: Maximum temperature two days before.
     - `temp_1`: Maximum temperature one day before.
     - `average`: Historical average of maximum temperatures.

2. **Data Preprocessing**:
   - The predictor variables were normalized using `StandardScaler` to ensure all variables have the same scale.

3. **Model**:
   - Random Forest, an ensemble method based on multiple decision trees.

4. **Hyperparameter Configurations**:
   - Ten different configurations were tested, varying:
     - `bootstrap`: Whether bootstrap samples are used.
     - `n_estimators`: Number of trees in the forest.
     - `max_depth`: Maximum depth of each tree.
     - `min_samples_leaf`: Minimum number of samples per leaf.
     - `min_samples_split`: Minimum number of samples required to split a node.

5. **Evaluation**:
   - Models were evaluated using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
   - 5-fold cross-validation was employed to assess the generalization ability of the models.

## Results and Discussion

### Configurations Tested

| Configuration | Bootstrap | n_estimators | max_depth | min_samples_leaf | min_samples_split | MAE   | MSE   |
|---------------|-----------|--------------|-----------|------------------|-------------------|-------|-------|
| Config 1      | True      | 200          | 4         | 4                | 5                 | 3.8645| 24.0683|
| Config 2      | True      | 400          | 10        | 2                | 5                 | 3.8525| 24.2265|
| Config 3      | True      | 600          | 30        | 1                | 2                 | 3.9042| 24.8388|
| Config 4      | True      | 300          | 20        | 3                | 10                | 3.8365| 24.0217|
| Config 5      | True      | 500          | 50        | 2                | 2                 | 3.8864| 24.6002|
| Config 6      | False     | 400          | 15        | 1                | 5                 | 5.0772| 43.4178|
| Config 7      | False     | 600          | 60        | 1                | 2                 | 5.1918| 45.0251|
| Config 8      | False     | 300          | 25        | 4                | 2                 | 4.6434| 35.8506|
| Config 9      | False     | 500          | 35        | 2                | 10                | 4.7427| 37.0381|
| Config 10     | True      | 700          | 60        | 1                | 5                 | 3.8839| 24.6022|

### General Evaluation of Metrics

- **Bootstrap Impact**: Configurations with bootstrap showed better performance in terms of MAE and MSE, highlighting Config 4 as the most efficient.
- **MAE and MSE**: Configurations with bootstrap consistently provided lower MAE and MSE, indicating more accurate and stable predictions.
- **Variable Importance**: `temp_1` (maximum temperature one day before) was the most influential variable, followed by `average` (historical average of temperatures).

## Figures and Graphs

#### Figure 1. Comparison of Errors by Configuration
![Figure 1](figura_1.png)

This bar chart compares the mean absolute error (MAE) and mean squared error (MSE) for each tested Random Forest configuration. Configurations 1, 2, 4, and 5 had the lowest MAE and MSE values, indicating better predictive performance.

#### Figure 2. Error Convergence Curves
![Figure 2](figura_2.png)

This graph shows the evolution of errors (MAE and MSE) over epochs for each Random Forest configuration. Configurations like Config 1, Config 2, Config 4, and Config 5 show rapid stabilization of errors, indicating good convergence.

#### Figure 3. Boxplot of Mean Absolute Errors (MAE)
![Figure 3](figura_3.png)

The boxplot shows the distribution of mean absolute errors (MAE) for each configuration. Configurations 1, 2, 4, and 5 have lower variability and lower median MAE values, indicating consistent and accurate predictions.

#### Figure 4. Boxplot of Mean Squared Errors (MSE)
![Figure 4](figura_4.png)

The boxplot shows the distribution of mean squared errors (MSE) for each configuration. Configurations 1, 2, 4, and 5 have lower variability and lower median MSE values, indicating more precise and stable predictions.

#### Figure 5. Variable Importance
![Figure 5](figura_5.png)

The variable `temp_1` (maximum temperature one day before) has the highest relative importance in the model, confirming that recent temperature is a crucial predictor for the next day's maximum temperature. The `average` (historical average of temperatures) is also significant, reflecting long-term climatic patterns.

## Conclusion

The results demonstrate the effectiveness of the Random Forest model in predicting the maximum temperature in Seattle using the `temps_extended.xlsx` dataset. The configurations tested varied in the number of estimators, tree depth, minimum samples per leaf, minimum samples required to split a node, and the use of bootstrap. Key findings include the significant impact of bootstrap on model performance and the critical importance of recent temperature data (`temp_1`) in making accurate predictions.

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## References

- Breiman, Leo. "Random Forests." Machine Learning 45.1 (2001): 5-32.
- Liaw, Andy, and Matthew Wiener. "Classification and Regression by randomForest." R News 2.3 (2002): 18-22.
- Pedregosa, Fabian, et al. "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research 12 (2011): 2825-2830.

## Appendix: Program Listing

The following link directs to the Colab notebook where the Random Forest was developed along with its results and graphs:

[Código: Trabalho4_RandomForest.ipynb](https://colab.research.google.com/drive/1G_oPN1kYTvCkOSIdl4GMu3bh-PYhKx1I?usp=sharing)
