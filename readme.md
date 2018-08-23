# Preface
This case study was conducted as a group with contributing members: Nabor Reyna, Sonia Malik, and Hamilton Carpenter. It was completed under a 6 hour time cap. 

# Movie Recommender Case Study

Today you are going to have a little friendly competition with your classmates.

You are going to building a recommendation system based off data from the
[MovieLens dataset](http://grouplens.org/datasets/movielens/). It includes movie
information, user information, and the users' ratings. Your goal is to build a
recommendation system and to suggest movies to users!

The **movies data** and **user data** are in `data/movies.dat` and `data/users.dat`.

The **ratings data** can be found in `data/training.csv`. The users' ratings have been broken into a training and test set for you (to obtain the testing set, we have split the 20% of **the most recent** ratings).


## Your mission [read carefully]

You are provided a **request** file in `data/requests.csv`. It contains a list of `user,movie` pairs.

**Your mission** is to provide a rating for each of those `user,movie` pairs. You will submit a csv file with three columns `user,movie,rating` as created by the script `src/run.py` (see below).

Your **score** will be measured based on how well you predict the ratings for the users' ratings compared to our test set. 


## How to implement your recommender

The file `src/recommender.py` is your main template for creating your recommender. You can work from this file and implement whatever strategy you think is best. You need to implement both the `.fit()` and the `.transform()` methods.

**Tips**: You might want to consider working in a notebook first, in order to establish a proper training strategy (proof of concept). In practice, it is not necessary to implement the file `src/recommender.py` to provide a submission file (a notebook can perfectly do that without running through `src/run.py`). If you don't do that during the case study, eventually we recommend you to try to integrate your implementation into the `src/recommender.py` file.


## How to run your recommender

`src/run.py` has been prepared for your convenience (doesn't need modification). By executing it you create an instance of a `MovieRecommender` class (see file `src/recommender.py`), feeds it with the training data and outputs the results in a file.

It outputs a _properly formatted_ file of recommendations for you!

  Here's how to use this script:
  ```bash
  usage: run.py [-h] [--train TRAIN] [--requests REQUESTS] [--silent] outputfile

  positional arguments:
    outputfile           output file (where predictions are stored)

  optional arguments:
    -h, --help           show this help message and exit
    --train TRAIN        path to training ratings file (to fit)
    --requests REQUESTS  path to the input requests (to predict)
    --silent             deactivate debug output
  ```

When running this script, **you need to** specify your prediction output file as an argument (the one you will submit).

**Try now** to create a random prediction file by typing:

```bash
python src/run.py data/sample_submission.csv
```

## How to submit your prediction for scoring

Next, make sure that you are able to post your score to Slack. To begin, you'll
need to pip install the `performotron` library: 

```bash 
pip install git+https://github.com/gschool/dsi-performotron.git --upgrade
```

After that, you should be able to use the `src/slack_poster.py` file to 
post your score to slack. The `src/slack_poster.py` file takes a _properly 
formatted_ file of recommendations (see `data/sample_submission.csv` for an 
example) and reports your score to Slack. When prompted, enter a slack channel name to 
post results to, **prefacing the channel with a `#` (i.e. #dsi_...)**. 
Test it out with the following command:
    
```bash
python src/slack_poster.py data/sample_submission.csv
```

See if you can get the best score! 

## Evaluation: how the score is computed

For each user, our scoring metric will select the 5% of movies you thought would be most highly rated by that user. It then looks at the actual ratings (in the test data) that the user gave those movies.  Your score is the average of those ratings.

Thus, for an algorithm to score well, it only needs to identify which movies a user is likely to rate most highly (so the absolute accuracy of your ratings is less important than the rank ordering).

As mentioned above, your submission should be in the same format as the sample
submission file, and the only thing that will be changed is the ratings column.


## Note on running your script with Spark

If your `recommender.py` script relies on spark, you may want to use the script `run_on_spark.sh` to execute your code.

In a terminal, use: `bash run_on_spark.sh src/run.py` with arguments to run your recommender.

The `src/slack_poster.py` doesn't need to run on spark, as it simply reads the result file produced by `run.py`.
