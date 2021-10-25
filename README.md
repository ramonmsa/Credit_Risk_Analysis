# Credit_Risk_Analysis
## Purpose
The purpose of this Analysis is to build and evaluate models using the credit card dataset. Therefore,  different techniques will be applied to train and evaluate models with unbalanced classes. 

The techniques used to adjust the class distribution of the data set in this analysis are: 
- RandomOverSampler  
- SMOTE 
- ClusterCentroids
- SMOTEENN
- BalancedRandomForestClassifier
- EasyEnsembleClassifier


## Source and Resources
- LendingClub's credit card dataset
- Libraries: `imbalanced-learn` and `scikit-learn` for Python. 

## Results
In this section the results for each technique will be displayed within its bulleted list level.

- Oversampling
  - Naive Random Oversampling <br>
   The __*accuracy score*__ shown for the Naive Random Oversampling model is __*67%*__ while the __*precision*__ and __*recall scores*__ show average of __*99%*__ and __*64%*__ respectively. Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*64%*__ while the metrics for the minority class indicate a  __*precision*__ of __*1%*__ and a __*recall*__ of __*70%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138537587-4d9cbb78-ce2d-41c1-8ce9-24bcce814843.png" alt="Naive Random Oversampling Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138537637-27fad9af-cdf0-4a4d-9c9d-1c292171052e.png" alt="Naive Random Oversampling Precision and Recall Scores"  style="width:40%">
  - SMOTE Oversampling <br>
   The __*accuracy score*__ shown for the SMOTE Oversampling model is __*66%*__ while the __*precision*__ and __*recall scores*__ show average of __*99%*__ and __*69%*__ respectively.  Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*69%*__ while the metrics for the minority class indicate a  __*precision*__ of __*1%*__ and a __*recall*__ of __*63%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138537791-1ec2c81e-23cb-4a04-bb37-7ffb49b66df6.png" alt="SMOTE Oversampling Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138537823-bb362da9-fb6f-4234-ba50-d12449dd0165.png" alt="SMOTE Oversampling Precision and Recall Scores"  style="width:40%">

- Undersampling <br>
   The __*accuracy score*__ shown for the Undersampling model is __*59%*__ while the __*precision*__ and __*recall scores*__ show average of __*99%*__ and __*55%*__ respectively. Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*55%*__ while the metrics for the minority class indicate a  __*precision*__ of __*1%*__ and a __*recall*__ of __*62%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138537930-50ad90fe-6928-48bc-83f8-a2b6b0bcf022.png" alt="Undersampling Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138537974-a2c0a3be-e1e5-4a48-ae5d-bf26fa0544c3.png" alt="Undersampling Precision and Recall Scores"  style="width:40%">

- Combination Sampling <br>
   The __*accuracy score*__ shown for the Combination Sampling model is __*66%*__ while the __*precision*__ and __*recall scores*__ show average of __*99%*__ and __*57%*__ respectively. Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*57%*__ while the metrics for the minority class indicate a  __*precision*__ of __*1%*__ and a __*recall*__ of __*75%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138538012-4a0cbb14-9998-4457-8445-2449e1922766.png" alt="Combination Sampling Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138538056-41cdc3f7-f0c0-4f42-97cd-6940c7168955.png" alt="Combination Sampling Precision and Recall Scores"  style="width:40%">

- Ensemble Learners <br>
  - Balanced Random Forest Classifier <br>
   The __*accuracy score*__ shown for the Balanced Random Forest Classifier model is __*79%*__ while the __*precision*__ and __*recall scores*__ show average of __*99%*__ and __*88%*__ respectively. Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*88%*__ while the metrics for the minority class indicate a  __*precision*__ of __*3%*__ and a __*recall*__ of __*70%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138538093-e140ebe2-189d-40ac-8b38-937b3b6f18b5.png" alt="Ensemble Learners Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138538110-1e599992-5419-4198-8c91-a6e7fb6131bd.png" alt="Ensemble Learners Precision and Recall Scores"  style="width:40%">

  - Easy Ensemble AdaBoost Classifier <br>
   The __*accuracy score*__ shown for the Easy Ensemble AdaBoost Classifier model is __*93%*__ while the __*precision*__ and __*recall scores*__  show average of __*99%*__ and __*94%*__ respectively. Therefore, the metrics for the majority class demonstrate a __*precision*__ of __*100%*__ and a __*recall*__ of __*94%*__ while the metrics for the minority class indicate a  __*precision*__ of __*9%*__ and a __*recall*__ of __*92%*__ .<br>
<img src="https://user-images.githubusercontent.com/69650068/138538581-6082cd70-5ec7-4df3-9f3b-f10ea1432339.png" alt="Easy Ensemble AdaBoost Classifier Accuracy Scores"  style="width:40%"> | <img src="https://user-images.githubusercontent.com/69650068/138538192-abbca728-c986-4fd2-bcae-41cb4ca4d3de.png" alt="Easy Ensemble AdaBoost Classifier Precision and Recall Scores"  style="width:40%">

## Summary
Based on the results it is possible to conclude that the two Ensemble Learners techniques used in this analysis demonstrated better metrics than the metrics reveled by the other four techniques, Furthermore, the Easy Ensemble AdaBoost Classifier technique showed higher accuracy score (93%) having the same behavior observed in for the recall which presents scores over 90% for both majority and minority classes.

Moreover, the Easy Ensemble AdaBoost Classifier model precision slightly improved for the minority class over the rest of the models for that same class. However, its inexpressive 9% score for the precision's minority class against the 100% score for the majority class precision exposes that this model is still imbalanced. 

As a result of the imbalance in the all six techniques, since the extremely low score for the precision's minority class and the maximum score determined for the majority class precision is also true for the other five techniques exploited in this analysis, it is recommended not to use any of the above techniques.

