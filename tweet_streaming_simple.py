'''
THIS CODE will get the Twitter Stream.
	1) Its a simple non-production code
	2) All output stored to output.json (one tweet  per line)
	3) Text of tweets also printed as received.
'''

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json
from auth import TwitterAuth

class StdOutListener(StreamListener):
	
	#This function gets called every time a new tweet is received on the stream
	def on_data(self, data):
		#Just write data to one line in the file
		fhOut.write(data)
		
		#Convert the data to a json object (shouldn't do this in production; might slow down and miss tweets)
		j=json.loads(data)

		#See Twitter reference for what fields are included -- https://dev.twitter.com/docs/platform-objects/tweets
		text=j["text"] 	# The text of the tweet
		print(text) 	# Print it out

	def on_error(self, status):
		print("ERROR")
		print(status)

if __name__ == '__main__':
	try:
		#Create a file to store output. "a" means append (add on to previous file)
		fhOut = open("tweetdata.json","a")

		#Create the listener
		l = StdOutListener()
		auth = OAuthHandler(TwitterAuth.consumer_key, TwitterAuth.consumer_secret)
		auth.set_access_token(TwitterAuth.access_token, TwitterAuth.access_token_secret)

		#Connect to the Twitter stream
		stream = Stream(auth, l)	

		#Terms to track
		stream.filter(track=["samsung"])
		
		#Alternatively, location box  for geotagged tweets
		#stream.filter(locations=[-0.530, 51.322, 0.231, 51.707])

	except KeyboardInterrupt:
		#User pressed ctrl+c -- get ready to exit the program
		pass

	#Close the 
	fhOut.close()