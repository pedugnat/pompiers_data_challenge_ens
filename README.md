# pompiers_data_challenge_ens
_Predicting response times of the Paris Fire Brigade vehicles_

This repository contains the code I used to take part in the Predicting response times of the Paris Fire Brigade vehicles challenge
See challenge page : https://challengedata.ens.fr/participants/challenges/21/

For a more detailed description of the dataset, see : https://paris-fire-brigade.github.io/data-challenge/challenge.html

## Challenge context
The response time is one of the most important factors for emergency services because their ability to save lives and rescue people depends on it. A non-optimal choice of an emergency vehicle for a rescue request may lengthen the arrival time of the rescuers and impact the future of the victim. This choice is therefore highly critical for emergency services and directly rely on their ability to predict precisely the arrival time of the different units available.

## Challenge goals
The task is to predict the delay between the selection of a rescue vehicle (the time when a rescue team is warned) and the time when it arrives at the scene of the rescue request (manual reporting via portable radio).
Overall, there are two continuous variables to predict. The metrics used is the **R squared** between the prediction and the true values

## Approach
### Main features
* distance between departure and intervention
* OSRM responses (ie estimated distance by OpenStreetMap)
* GPS tracks
* ratios between numeric variables
* target encoding + regularization for categorical variables like : 
1. Alert reason
2. Type of vehicle
3. Day of the week
4. Hour of the day

### Modeling technique
Using XGBoost Regressor with : 
* 100 estimators
* early_stopping
* 6 max_depth to prevent overfitting
* 4-Folds cross validation
