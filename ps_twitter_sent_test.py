'''
TRAINING THE MODEL FOR TESTING THE TRAINED SENTIMENT ANALYSIS MODEL
This code does following things:
	1) Read Raw Tweets from file in HDFS or Local file system
	2) Remove Punctuation,stopwords,stemmer,any http,twoOrMore letters
	3) Tokenize words for Bag-of-Words	
	4) Convert tokens into TF-IDF with default smart hashing with 200 buckets
	5) Convert Tweets and Bag-of-Words into Spark Dataframe to take advantage of catalyst
	6) Saving the prediction into csv file
'''
##################################
# Importing all necessary Modules
##################################

# import spark context
import string
import json
import re

# NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Spark ML
from pyspark.ml.feature import HashingTF, IDF, Word2Vec

# Spark MLlib
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import NaiveBayes,NaiveBayesModel
from pyspark.mllib.evaluation import MulticlassMetrics

# Spark Basic
from pyspark import SparkContext
from pyspark.sql import SQLContext,Row
from pyspark.sql.types import *
from pyspark.sql.functions import udf

##################
## Functions    ##
##################

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

# Tokenize string
####################
def tokenize(text):    
    tokens = word_tokenize(text)
    lowercased = [t.lower() for t in tokens]
    no_punctuation = []
    for word in lowercased:
        punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
        no_punctuation.append(punct_removed)
    no_stopwords = [w.strip() for w in no_punctuation if not w in STOPWORDS]
    stemmed = [STEMMER.stem(w) for w in no_stopwords]
    return [w for w in stemmed]
    #return no_stopwords

# Cleaning the text field
##########################
def cleaningText(text):
    # The following regex just strips of an URL (not just http), any punctuations, 
    # User Names or Any non alphanumeric characters.    
    preprocesstext1 = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\d+)"," ",text).split()).strip().lower()
    
    #start replaceTwoOrMore
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    preprocesstext2 = pattern.sub(r"\1\1", preprocesstext1)    
    return preprocesstext2

# Creating a TFIDF function
###########################
def add_tfidf_to_dataframe(d,inputcolName):
    htf = HashingTF(numFeatures=200, inputCol=inputcolName, outputCol="xHashTF")
    idfinstance = IDF(inputCol="xHashTF", outputCol="xTFIDF")
    tfidf = idfinstance.fit(htf.transform(d))
    outdf = tfidf.transform(htf.transform(d))
    return outdf   

# Called function containing all logic
######################################
def main(sc,sqlContext):

	# Creating UDF
	udfct = udf(cleaningText,StringType())
	udftokenize = udf(tokenize,ArrayType(StringType()))

	# Import full dataset of newsgroup posts as text file
	twitter_raw = sc.textFile('tweetdata.json')

	# Parse JSON entries in dataset
	data = twitter_raw.map(lambda line: json.loads(line))

	# Take only english tweets
	data_english = data.filter(lambda line: line['lang']=='en')
	data_english.count()

	# Extract relevant fields in dataset -- category label and text content
	data_text = data_english.map(lambda line: (line['id'],line['text']))

	# ## Creating Spark Dataframe
	tweet_df = sqlContext.createDataFrame(data_text,('id','text'))

	# Print Dataframe Schema
	tweet_df.printSchema()

	# Cleaning text data
	cleantweet = tweet_df.select('id','text',udfct(tweet_df.text).alias('ctext'))
	cleantweet.head(1)
	
	# Adding token field in the dataframe
	cleantweet2 = cleantweet.withColumn("tokens",udftokenize(cleantweet.ctext))

	# Printing Schema
	cleantweet2.printSchema()
	cleantweet2.show(3)

	# ## Creating Training LabeledPoint Dataset
	# * hashingTF
	# * TF_IDF

	rescaledData = add_tfidf_to_dataframe(cleantweet2,"tokens")
	rescaledData.printSchema()

	# Converting Dataframe to RDD
	testData = rescaledData.select('id','text','xTFIDF')

	# Printing Schema
	testData.printSchema()

	# Load the model
	trainedmodelNB = NaiveBayesModel.load(sc,'NB_SentimentModel')

	# Predicting based on the trained model
	pred = testData.map(lambda p: (p.id,trainedmodelNB.predict(p.xTFIDF))).toDF(['id','pred_sentiment'])

	# Saving the predicted dataframe into csv file
	# Saving the output to csv file

	pred.write.format("com.databricks.spark.csv").option("header", "true").save("tweetpred_out.csv")

## Main functionality
if __name__ == "__main__":

    # Setting Spark Context
    sc = SparkContext(master="local[*]", appName="Twitter Test Application")
    sqlContext = SQLContext(sc)

    # Execute Main functionality
    main(sc,sqlContext)
