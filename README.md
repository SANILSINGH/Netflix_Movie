# Netflix_Movie

## 1. Business Problem
### 1.1 Problem Description

Netflix is all about connecting people to the movies they love. To help customers find those movies, they developed world-class movie recommendation system: CinematchSM. Its job is to predict whether someone will enjoy a movie based on how much they liked or disliked other movies. Netflix use those predictions to make personal movie recommendations based on each customer’s unique tastes. And while Cinematch is doing pretty well, it can always be made better.

Now there are a lot of interesting alternative approaches to how Cinematch works that netflix haven’t tried. Some are described in the literature, some aren’t. We’re curious whether any of these can beat Cinematch by making better predictions. Because, frankly, if there is a much better approach it could make a big difference to our customers and our business.

Credits: https://www.netflixprize.com/rules.html
### 1.2 Problem Statement

Netflix provided a lot of anonymous rating data, and a prediction accuracy bar that is 10% better than what Cinematch can do on the same training data set. (Accuracy is a measurement of how closely predicted ratings of movies match subsequent actual ratings.)
### 1.3 Sources

    https://www.netflixprize.com/rules.html
    https://www.kaggle.com/netflix-inc/netflix-prize-data
    Netflix blog: https://medium.com/netflix-techblog/netflix-recommendations-beyond-the-5-stars-part-1-55838468f429 (very nice blog)
    surprise library: http://surpriselib.com/ (we use many models from this library)
    surprise library doc: http://surprise.readthedocs.io/en/stable/getting_started.html (we use many models from this library)
    installing surprise: https://github.com/NicolasHug/Surprise#installation
    Research paper: http://courses.ischool.berkeley.edu/i290-dm/s11/SECURE/a1-koren.pdf (most of our work was inspired by this paper)
    SVD Decomposition : https://www.youtube.com/watch?v=P5mlg91as1c

### 1.4 Real world/Business Objectives and constraints

Objectives:

    Predict the rating that a user would give to a movie that he ahs not yet rated.
    Minimize the difference between predicted and actual rating (RMSE and MAPE)

Constraints:

    Some form of interpretability.

## 2. Machine Learning Problem
### 2.1 Data
### 2.1.1 Data Overview

Get the data from : https://www.kaggle.com/netflix-inc/netflix-prize-data/data

Data files :

    combined_data_1.txt

    combined_data_2.txt

    combined_data_3.txt

    combined_data_4.txt

    movie_titles.csv

  
The first line of each file [combined_data_1.txt, combined_data_2.txt, combined_data_3.txt, combined_data_4.txt] contains the movie id followed by a colon. Each subsequent line in the file corresponds to a rating from a customer and its date in the following format:

CustomerID,Rating,Date

MovieIDs range from 1 to 17770 sequentially.
CustomerIDs range from 1 to 2649429, with gaps. There are 480189 users.
Ratings are on a five star (integral) scale from 1 to 5.
Dates have the format YYYY-MM-DD.

### 2.1.2 Example Data point

1:
1488844,3,2005-09-06
822109,5,2005-05-13
885013,4,2005-10-19
30878,4,2005-12-26
823519,3,2004-05-03
893988,3,2005-11-17
124105,4,2004-08-05
1248029,3,2004-04-22
1842128,4,2004-05-09
2238063,3,2005-05-11
1503895,4,2005-05-19
2207774,5,2005-06-06
2590061,3,2004-08-12
2442,3,2004-04-14
543865,4,2004-05-28
1209119,4,2004-03-23
804919,4,2004-06-10
1086807,3,2004-12-28
1711859,4,2005-05-08
372233,5,2005-11-23
1080361,3,2005-03-28
1245640,3,2005-12-19
558634,4,2004-12-14
2165002,4,2004-04-06
1181550,3,2004-02-01
1227322,4,2004-02-06
427928,4,2004-02-26
814701,5,2005-09-29
808731,4,2005-10-31
662870,5,2005-08-24
337541,5,2005-03-23
786312,3,2004-11-16
1133214,4,2004-03-07
1537427,4,2004-03-29
1209954,5,2005-05-09
2381599,3,2005-09-12
525356,2,2004-07-11
1910569,4,2004-04-12
2263586,4,2004-08-20
2421815,2,2004-02-26
1009622,1,2005-01-19
1481961,2,2005-05-24
401047,4,2005-06-03
2179073,3,2004-08-29
1434636,3,2004-05-01
93986,5,2005-10-06
1308744,5,2005-10-29
2647871,4,2005-12-30
1905581,5,2005-08-16
2508819,3,2004-05-18
1578279,1,2005-05-19
1159695,4,2005-02-15
2588432,3,2005-03-31
2423091,3,2005-09-12
470232,4,2004-04-08
2148699,2,2004-06-05
1342007,3,2004-07-16
466135,4,2004-07-13
2472440,3,2005-08-13
1283744,3,2004-04-17
1927580,4,2004-11-08
716874,5,2005-05-06
4326,4,2005-10-29

### 2.2 Mapping the real world problem to a Machine Learning Problem
### 2.2.1 Type of Machine Learning Problem

For a given movie and user we need to predict the rating would be given by him/her to the movie. 
The given problem is a Recommendation problem 
It can also seen as a Regression problem 

### 2.2.2 Performance metric

    Mean Absolute Percentage Error: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    Root Mean Square Error: https://en.wikipedia.org/wiki/Root-mean-square_deviation

### 2.2.3 Machine Learning Objective and Constraints

    Minimize RMSE.
    Try to provide some interpretability.

## Exploratory Data Analysis

### Preprocessing

### Spliting data into Train and Test(80:20)

### Exploratory Data Analysis on Train data

### Distribution of ratings

### Number of Ratings per a month

### Analysis on the Ratings given by user

### Analysis of ratings of a movie given by a user

### number of ratings on each day of the week

### Creating sparse matrix from data frame

### Finding Global average of all movie ratings, Average rating per user, and Average rating per movie

### Cold Start problem

### Computing Similarity matrices

### Computing User-User Similarity matrix

### Note: Calculating User User Similarity_Matrix is not very easy(unless we have huge Computing Power and lots of time) because of number of. usersbeing lare. we can try if we want to. our system could crash or the program stops with Memory Error.

### Trying with all dimensions (17k dimensions per user)
### Trying with reduced dimensions (Using TruncatedSVD for dimensionality reduction of user vector)
### Computing Movie-Movie Similarity matrix

### Note: Even though we have similarity measure of each movie, with all other movies, We generally don't care much about least similar movies.Most of the times, only top_xxx similar items matters. It may be 10 or 100.We take only those top similar movie ratings and store them in a saperate dictionary.

### Finding most similar movies using similarity matrix

Does Similarity really works as the way we expected...? Let's pick some random movie and check for its similar movies.

# Machine Learning Models

### Sampling Data
### Build sample train data from the train data
### Build sample test data from the test data4.2 Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)
### Finding Global Average of all movie ratings, Average rating per User, and Average rating per Movie (from sampled train)
### Featurizing data
### Featurizing data for regression problem
### Featurizing train data

GAvg : Average rating of all the ratings

Similar users rating of this movie:
    sur1, sur2, sur3, sur4, sur5 ( top 5 similar users who rated that movie.. )

Similar movies rated by this user:
    smr1, smr2, smr3, smr4, smr5 ( top 5 similar movies rated by this movie.. )

UAvg : User's Average rating

MAvg : Average rating of this movie

rating : Rating of this movie by this user.

### Featurizing test data

GAvg : Average rating of all the ratings

Similar users rating of this movie:
    sur1, sur2, sur3, sur4, sur5 ( top 5 simiular users who rated that movie.. )

Similar movies rated by this user:
    smr1, smr2, smr3, smr4, smr5 ( top 5 simiular movies rated by this movie.. )

UAvg : User AVerage rating

MAvg : Average rating of this movie

rating : Rating of this movie by this user.

### Transforming data for Surprise models

### Transforming train data

We can't give raw data (movie, user, rating) to train the model in Surprise library.They have a saperate format for TRAIN and TEST data, which will be useful for training the models like SVD, KNNBaseLineOnly....etc..,in Surprise.We can form the trainset from a file, or from a Pandas DataFrame. http://surprise.readthedocs.io/en/stable/getting_started.html#load-dom-dataframe-py

### Transforming test data

Testset is just a list of (user, movie, rating) tuples. (Order in the tuple is impotant)

### Applying Machine Learning models

Global dictionary that stores rmse and mape for all the models....

It stores the metrics in a dictionary of dictionaries

keys : model names(string)

value: dict(key : metric, value : value )

### XGBoost with initial 13 features

Test Data
------------------------------
RMSE : 1.0761851474385373 
MAPE : 34.504887593204884

### Suprise BaselineModel

Test Data
------------------------------
RMSE : 1.0730330260516174
MAPE : 35.04995544572911

### XGBoost with initial 13 features + Surprise Baseline predictor

TEST DATA
------------------------------
RMSE :  1.0763419061709816
MAPE :  34.491235560745295

### Surprise KNNBaseline with user user similarities

Test Data
---------------
RMSE : 1.0726493739667242
MAPE : 35.02094499698424

### Surprise KNNBaseline with movie movie similarities

Test Data
---------------
RMSE : 1.072758832653683
MAPE : 35.02269653015042

### XGBoost with initial 13 features + Surprise Baseline predictor + KNNBaseline predictor

First we will run XGBoost with predictions from both KNN's ( that uses User_User and Item_Item similarities along with our previous features.
Then we will run XGBoost with just predictions form both knn models and preditions from our baseline model.

TEST DATA
------------------------------
RMSE :  1.0763602465199797
MAPE :  34.48862808016984

### Matrix Factorization Techniques
### SVD Matrix Factorization User Movie intractions

Test Data
---------------
RMSE : 1.0726046873826458
MAPE : 35.01953535988152

### SVD Matrix Factorization with implicit feedback from user ( user rated movies )

Test Data
---------------
RMSE : 1.0728491944183447
MAPE : 35.03817913919887

### XgBoost with 13 features + Surprise Baseline + Surprise KNNbaseline + MF Techniques

TEST DATA
------------------------------
RMSE :  1.0763580984894978
MAPE :  34.487391651053336

### XgBoost with Surprise Baseline + Surprise KNNbaseline + MF Techniques

TEST DATA
------------------------------
RMSE :  1.0763580984894978
MAPE :  34.487391651053336

# Comparision between all models

MODEL 	           RMSE

svd               1.0726046873826458

knn_bsl_u         1.0726493739667242

knn_bsl_m         1.072758832653683

svdpp             1.0728491944183447

bsl_algo          1.0730330260516174

xgb_knn_bsl_mu    1.0753229281412784

xgb_all_models    1.075480663561971

first_algo        1.0761851474385373

xgb_bsl           1.0763419061709816

xgb_final         1.0763580984894978

xgb_knn_bsl       1.0763602465199797
