import logging
import numpy as np
import pandas as pd
import pyspark
from pyspark.sql.types import *
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

class MovieRecommender():
    """Template class for a Movie Recommender system."""

    def __init__(self):
        """Constructs a MovieRecommender"""
        self.logger = logging.getLogger('reco-cs')
        self.logger.info("Starting up the spark Engine: ")
        spark = pyspark.sql.SparkSession.builder.getOrCreate()
        sc = spark.sparkContext
        self.sc = sc
        self.spark = spark

        #Initialize model parameter
        self.rank = 10
        self.reg_param = 0.01
        self.max_iter = 20

        self.als_model = ALS(userCol='user',
                itemCol='movie',
                ratingCol='rating',
                nonnegative=True,
                regParam=self.reg_param,
                maxIter=self.max_iter,
                rank=self.rank ,
                coldStartStrategy = "nan"
               )

    def crossvalidate(self, train, test):
        #find out which paramater is giving best result and coss validate
        model = self.recommender 
        #remove rating, timestamp from test
        regularizations = [0.1, 1., 10., 100.]
        regularizations.sort()
        iter_array = [5, 10, 15, 20]
        test = test.drop(['rating', 'timestamp'], axis=1)
        model.transform(test)

        pass

    def fit(self, ratings):
        """
        Trains the recommender on a given set of ratings.

        Parameters
        ----------
        ratings : pandas dataframe, shape = (n_ratings, 4)
                  with columns 'user', 'movie', 'rating', 'timestamp'

        Returns
        -------
        self : object
            Returns self.
        """
        self.logger.debug("starting fit")
        ratings = ratings.sort_values(by='timestamp', ascending=False)
        #take 20% as validation/test set and 80% as training set
        n_rows = ratings.shape[0]
        train = ratings.head(int(n_rows*.8))
        test  = ratings.tail(int(n_rows*.2))

        sp_train = self.spark.createDataFrame(train.drop('timestamp', axis=1))
        sp_test  = self.spark.createDataFrame(test.drop('timestamp', axis=1))

        self.recommender = self.als_model.fit(sp_train)
        
        #params = self.crossvalidate(sp_train, sp_test)
        
        self.logger.debug("finishing fit")
        return(self)


    def transform(self, requests):
        """
        Predicts the ratings for a given set of requests.

        Parameters
        ----------
        requests : pandas dataframe, shape = (n_ratings, 2)
                  with columns 'user', 'movie'

        Returns
        -------
        dataframe : a pandas dataframe with columns 'user', 'movie', 'rating'
                    column 'rating' containing the predicted rating
        """
        self.logger.debug("starting predict")
        self.logger.debug("request count: {}".format(requests.shape[0]))

        #requests['rating'] = np.random.choice(range(1, 5), requests.shape[0])
        sp_req = self.spark.createDataFrame(requests)
        prediction_sp = self.recommender.transform(sp_req)
        predictions_pd = prediction_sp.toPandas()

        # Change 4.5 to mean or some reasonable paramter, discover ways to fill na's
        predictions_pd = predictions_pd.fillna(4.5)
        self.predict = predictions_pd
        self.logger.debug("finishing predict")
        return(predictions_pd)

    def print_pretty_ratings(self):
        #for future improvements add movie title info, 
        # run this initially to see top rated movies and after running model
        movies_df = pd.read_table("data/movies.dat", sep="::", header=None)
        movies_df = movies_df.iloc[:,0:2 ]
        movies_df.columns = ['movie', 'title']
        merged_df = pd.merge(self.predict, movies_df, on='movie')
        merged_df['user'] = 1 # to count no of users
        g = merged_df.groupby(['movie', 'title'])
        top_10_rated = g.agg({"user": np.sum, "rating":np.mean}).sort_values(by=['user'], ascending=False).head(10)
        print("**Top 10 rated movies by user***")
        print(top_10_rated)

if __name__ == "__main__":
    logger = logging.getLogger('reco-cs')
    logger.critical('you should use run.py instead')
