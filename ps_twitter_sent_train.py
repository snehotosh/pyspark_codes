'''
TRAINING THE MODEL FOR SENTIMENT ANALYSIS
This code does following things:
	1) Read Raw Tweets from file in HDFS or Local file system
	2) Remove Punctuation,stopwords,stemmer,any http,twoOrMore letters
	3) Tokenize words for Bag-of-Words
	4) Using VADER Sentiment to do sentence polarization for getting label in Supervised Learning
	5) Convert tokens into TF-IDF with default smart hashing with 200 buckets
	6) Convert Tweets and Bag-of-Words into Spark Dataframe to take advantage of catalyst
	7) Dividing the dataset into training and validation set with 70:30 ratio before TFIDF for generalization
	8) Training the Multinomial NaiveBayes Model
	9) Applying the trained model to predict on Validation set for metrics evaluation
       10) Getting all relevant metrics
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
from nltk.sentiment.vader import SentimentIntensityAnalyzer

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

# Getting Polarity Score
##########################
def sentPolarityscore(text):
    ss = sid.polarity_scores(text)
    sent = ss['compound']      
    return float(sent)

# Converting polarity into 1:Positive,0:Neutral and -1:Negative
###############################################################
def sentPolarity(text):
    ss = sid.polarity_scores(text)
    sent = ss['compound']  
    if sent >= 0.0 and sent < 0.20:
        return 0
    elif sent >= 0.20 and sent <= 1.0:
        return 1
    else:
        return -1
    return sent

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
def main(sc,sqlContext,sid):

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

	# ## Using VADER - Valency (Sentiment) for Creating Labels ( Sentiment )
	# * For automatic polarization of sentence for training label

	# UDF for sentiment score
	sentiscoreUDF = udf(sentPolarityscore,FloatType())

	# UDF for sentiment Polarity
	sentiUDF = udf(sentPolarity,IntegerType()) 

	# Transforming on original tweets
	cleantweet1 = cleantweet.withColumn("sentiscore",sentiscoreUDF(cleantweet.text))
	cleantweet2 = cleantweet1.withColumn("sentiment",sentiUDF(cleantweet.text))
	cleantweet3 = cleantweet2.withColumn("tokens",udftokenize(cleantweet.ctext))

	# Printing Schema
	cleantweet3.printSchema()
	cleantweet3.show(3)

	tweetFinal = cleantweet3.select('id','text','tokens','sentiscore','sentiment')
	tweetFinal.printSchema()

	# ## Divide into Train and Validation Set before TF-IDF conversion
	trainingset, validationset = tweetFinal.randomSplit([0.7, 0.3],seed = 12345)

	print 'TrainingSet count:',trainingset.count()
	print 'ValidationSet count:',validationset.count()
	print 'Total count:',tweetFinal.count() 


	# ## Creating Training LabeledPoint Dataset
	# * hashingTF
	# * TF_IDF

	train = add_tfidf_to_dataframe(trainingset,"tokens")
	train.printSchema()

	# Converting Dataframe to RDD
	tweetRDDtrain = train.select('sentiment','xTFIDF').rdd

	# Create an RDD of LabeledPoints using sentiment as Labels and TFIDF as Feature Vectors
	train_labelpoint = tweetRDDtrain.map(lambda (label, text): LabeledPoint(label, text))
	train_labelpoint.take(2)

	# Ask Spark to persist the RDD so it won't have to be re-created later
	train_labelpoint.persist()

	# ## Creating Validation LabeledPoint Dataset
	validation = add_tfidf_to_dataframe(validationset,"tokens")
	validation.printSchema()

	# Converting Dataframe to RDD
	tweetRDDvalid = validation.select('sentiment','xTFIDF').rdd

	# Create an RDD of LabeledPoints using sentiment as Labels and TFIDF as Feature Vectors
	valid_labelpoint = tweetRDDvalid.map(lambda (label, text): LabeledPoint(label, text))
	valid_labelpoint.take(2)

	# Ask Spark to persist the RDD so it won't have to be re-created later
	valid_labelpoint.persist()

	# ## Model Training using Multinomial NaiveBayes
	# Train a Multinomial Naive Bayes model on the training data
	modelNB = NaiveBayes.train(train_labelpoint)

	# Compare predicted labels to actual labels
	predictionAndLabel = valid_labelpoint.map(lambda p: (float(modelNB.predict(p.features)), p.label))

	# Filter to only correct predictions
	correct = predictionAndLabel.filter(lambda (predicted, actual): predicted == actual)

	# Calculate and print accuracy rate
	accuracy = correct.count() / float(valid_labelpoint.count())

	print "Classifier correctly predicted category " + str(accuracy * 100) + " percent of the time"

	# ####### Evaluation Metrics ###########

	# Instantiate metrics object
	metrics = MulticlassMetrics(predictionAndLabel)

	# Confusion Metrics
	confusion_matrix = metrics.confusionMatrix()

	# They are ordered by class label ascending (i.e. [-1, 0, 1])
	print confusion_matrix

	# Overall statistics
	precision = metrics.precision()
	recall = metrics.recall()
	f1Score = metrics.fMeasure()

	print("Summary Stats")
	print("--------------")

	print("Precision = %s" % precision)
	print("Recall = %s" % recall)
	print("F1 Score = %s" % f1Score)

	# Statistics by class
	labels = valid_labelpoint.map(lambda lp: lp.label).distinct().collect()

	for label in sorted(labels):
    		print("Class %s precision = %s" % (label, metrics.precision(label)))
    		print("Class %s recall = %s" % (label, metrics.recall(label)))
    		print("Class %s F1 Measure = %s" % (label, metrics.fMeasure(label, beta=1.0)))

	# Weighted stats
	print("Weighted recall = %s" % metrics.weightedRecall)
	print("Weighted precision = %s" % metrics.weightedPrecision)
	print("Weighted F(1) Score = %s" % metrics.weightedFMeasure())
	print("Weighted F(0.5) Score = %s" % metrics.weightedFMeasure(beta=0.5))
	print("Weighted false positive rate = %s" % metrics.weightedFalsePositiveRate)

	# Saving the model
	modelNB.save(sc,'NB_SentimentModel')

## Main functionality
if __name__ == "__main__":

    # Setting Spark Context
    sc = SparkContext(master="local[*]", appName="Twitter Application")
    sqlContext = SQLContext(sc)

    # Instantiating SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Execute Main functionality
    main(sc,sqlContext,sid)
