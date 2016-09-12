'''
THIS CODE READ Historical TWEET from Twitter through Twitter Application
	1) This code can read multiple pages with 100 tweets per page
	2) Historical Tweets
'''

import tweepy
import sys
import jsonpickle
import os


# Replace the API_KEY and API_SECRET with your application's key and secret.
# OAuth Keys in Twitter API
consumerKey = '.............'
consumerSecret = '............'

#auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
auth = tweepy.OAuthHandler(consumer_key=consumerKey, consumer_secret=consumerSecret)
api = tweepy.API(auth,proxy="<HOST>:<PORT>")
 
if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# this is what we're searching for
searchQuery = 'samsung'  

# Some arbitrary large number
maxTweets = 10000 

# this is the max the API permits
tweetsPerQry = 100  

# We'll store the tweets in a text file.
fName = 'tweetsSamsung.json' 

# If results from a specific ID onwards are reqd, set since_id to that ID.
# else default to no lower limit, go as far back as API allows
sinceId = None

# If results only below a specific ID are, set max_id to that ID.
# else default to no upper limit, start from the most recent tweet matching the search query.
max_id = -1L
tweetCount = 0

print("Downloading max {0} tweets".format(maxTweets))

# Opening file in write mode and writing to a file all Tweets
with open(fName, 'w') as f:
    while tweetCount < maxTweets:
        try:
            if (max_id <= 0):
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry)
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            since_id=sinceId)
            else:
                if (not sinceId):
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1))
                else:
                    new_tweets = api.search(q=searchQuery, count=tweetsPerQry,
                                            max_id=str(max_id - 1),
                                            since_id=sinceId)
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                f.write(jsonpickle.encode(tweet._json, unpicklable=False) +
                        '\n')
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            max_id = new_tweets[-1].id
        except tweepy.TweepError as e:
            # Just exit if any error
            print("some error : " + str(e))
            break

# Final printing the number of tweets
print ("Downloaded {0} tweets, Saved to {1}".format(tweetCount, fName))