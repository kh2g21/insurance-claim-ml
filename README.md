# Travel Insurance Claim Prediction

This is a model that predicts whether a customer will make a travel insurance claim using machine learning. It includes an extensive ML pipeline; from preprocessing, handling data imbalances, model training, hyperparameter tuning, and evaluation. Finally, the model is deployed to a Streamlit app according to the machine learning algorithm that yields the best ROC-AUC results after evaluation. 

## Dataset and Preprocessing

The dataset used for this project is the [Travel Insurance](https://www.kaggle.com/datasets/mhdzahier/travel-insurance) dataset which can be found on Kaggle. In this repository, it is represented by `travel insurance.csv`. The target variable is `Claim`, which indicates whether a customer has made a travel insurance claim. In the preprocessing step, the target values were mapped to numerical form: "Yes" was converted to 1, and "No" was converted to 0. The dataset contains both numerical features, such as age and travel duration, and categorical features, such as gender and destination. As this dataset also included some categorical features, these had to be converted into dummy variables using one-hot encoding, and the first category was dropped to avoid multicollinearity. 

To address the issue of class imbalance, several approaches were tested. Initially, downsizing the majority class and random undersampling were tried, but these methods either removed too much data or did not improve model performance. Ultimately, the Synthetic Minority Oversampling Technique (SMOTE) was applied to the training data, as it generated synthetic samples for the smaller, minority class (“Yes” claims) and provided the best results. Logistic Regression models are sensitive to feature scaling, so the numerical features were standardized using the `StandardScaler` library.. Tree-based models, such as Random Forest, Gradient Boosting, and XGBoost, do not require feature scaling, so scaling was not applied to those models.



## Model Training and Evaluation

Four different machine learning models were trained in this project: Logistic Regression, Random Forest, Gradient Boosting, and XGBoost. Each model was combined with SMOTE (and scaling in the case of Logistic Regression) in a pipeline to ensure proper preprocessing within cross-validation. Hyperparameter tuning was performed for each model using `GridSearchCV` with five-fold stratified cross-validation, optimizing for the area under the ROC curve (ROC-AUC).

For Logistic Regression, the solver "saga" was used with both L1 and L2 penalties, and the regularization strength C was tested with values 0.01, 0.1, and 1. For Random Forest, the number of estimators was tested with 100 and 150 trees, maximum tree depths of 5 and 8 were evaluated, and the minimum number of samples required to split an internal node was set to either 2 or 5. For Gradient Boosting, the number of estimators was tested with 100 and 150, maximum tree depths of 3 and 5 were evaluated, the learning rate was tested with 0.05 and 0.1, and the subsample ratio was tested with 0.8 and 1.0. For XGBoost, the number of estimators was set to 100 and 150, maximum tree depths of 3 and 5 were evaluated, the learning rate was tested with 0.05 and 0.1, and both the subsample ratio and the column sample by tree ratio were set to 0.8.

After training and cross-validation, the best model was selected based on the ROC-AUC score on the test set. The evaluation metrics included the classification report (precision, recall, and F1-score), the ROC-AUC score, the Precision-Recall AUC (PR-AUC), and the confusion matrix. These metrics allow us to evaluate both overall model performance and performance on the minority class, which is critical given the imbalanced nature of the dataset.

The pipeline also includes confusion matrix visualizations for each model, plotted as heatmaps with Seaborn. The confusion matrix provides a detailed breakdown of model predictions, showing the number of true positives (correctly predicted claims), true negatives (correctly predicted non-claims), false positives (predicted a claim when there wasn’t one), and false negatives (missed actual claims). By comparing confusion matrices across models, it is possible to see which algorithms strike a better balance between detecting claims (recall) and avoiding false alarms (precision).

The final trained model pipeline, including SMOTE, scaling (for Logistic Regression), and the classifier, was saved to a file named `best_insurance_claim_model.pkl` using joblib. This file can be loaded and used to make predictions on new data, provided the new data has the same features and preprocessing steps as the training data.

## Testing Results

The following table shows the cross-validation ROC-AUC and the test set performance of each model. Each model was trained and validated using stratified cross-validation and then evaluated on a held-out test set.

| Model               | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
|----------------------|------------|--------------|-------------|
| Logistic Regression  | 0.72       | 0.71         | 0.35        |
| Random Forest        | 0.84       | 0.83         | 0.52        |
| Gradient Boosting    | 0.86       | 0.85         | 0.55        |
| XGBoost              | 0.87       | 0.86         | 0.57        |

These results show that XGBoost achieved the best overall performance, closely followed by Gradient Boosting. The higher precision-recall AUC scores for the tree-based methods indicate that they are better suited to handling the imbalanced dataset and capturing non-linear feature interactions. 

The Precision-Recall AUC (PR-AUC) values are lower than the ROC-AUC values across all models. This is expected given the highly imbalanced nature of the dataset, where only a small proportion of customers make an insurance claim. This means the model is trained on many more negatives than positives. In the context of this problem - what precision is trying to determine is, "_of all the customers I predicted would make a claim, how many actually did?”_ - but because actual claims are so rare, in such cases, precision drops quickly because even a small number of false positives significantly affects the score. For example, if 10 people are predicted to claim, but only 3 actually do, then the precision would only be 30%. On the other hand, recall is asking "_of all the people who actually claimed, how many did I catch?_" - but since there are so few claim cases, this makes it easier for the model to miss them, lowering recall as a result.

Unlike ROC-AUC, which can remain relatively high under imbalance, PR-AUC provides a more realistic picture of how well the model identifies the minority class. Therefore, while the PR-AUC values appear modest, they still represent meaningful improvements over a naive classifier and demonstrate the models’ ability to detect rare claim events.

In context, these results suggest that non-linear, tree-based models capture the complex relationships in the travel insurance data better than linear methods. The strong performance of XGBoost in particular highlights its suitability for this type of imbalanced classification problem, where subtle interactions between features like age, product type, and duration affect claim likelihood.

The best-performing model was saved for deployment, allowing predictions on new customer data.

## Possible Improvements

One of the main areas where the project could be improved is additional feature engineering. Although the current dataset uses the original features and applies one-hot encoding for categorical variables, additional features could help the models capture more complex patterns. For example, categorical variables like `Destination` and `Travel Type` could be grouped to form a new feature, and destinations could be grouped into broader regions such as “Europe,” or “Asia-Pacific,” and combined with the travel type (e.g., business or leisure). This would reduce the number of categories while potentially highlighting meaningful patterns, such as higher claim rates for leisure trips to certain regions.

Model improvements are another important area to consider. While the current project trains Logistic Regression, Random Forest, Gradient Boosting, and XGBoost, additional experimentation could further enhance performance. For instance, ensemble methods could combine the strengths of multiple classifiers to produce more robust predictions. In Logistic Regression, elastic-net regularization, which combines L1 and L2 penalties, could be explored to improve generalization and automatically perform feature selection. For tree-based models, experimenting with additional hyperparameters, such as the minimum samples per leaf or the learning rate could also yield improvements. 

Cross-validation is a crucial step in evaluating model performance, and there are several ways it could be enhanced in this project. Currently, a five-fold stratified cross-validation is used, which ensures that each fold has a similar distribution of the target classes. This approach could be extended by using repeated stratified K-fold cross-validation, which repeats the splitting process multiple times to provide more stable and reliable estimates of model performance. This is particularly important in imbalanced datasets like this one, where variance between folds can affect the evaluation metrics. Moreover, exploring multiple evaluation metrics during cross-validation, such as Precision-Recall AUC or F1-score for the minority class, would provide a better understanding of the model’s performance beyond ROC-AUC.



