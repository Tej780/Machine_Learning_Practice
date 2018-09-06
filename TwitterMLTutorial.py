import tweepy
from textblob import TextBlob

consumer_key = "UJCcUww1VA1OheIS3oND4hRUm"
consumer_secret = "u4JxGTOo5IX8N4Mdd942pGNiUsFvCjsikL8FSXBs7SSAaC0DY5"

access_token = "570293893-D2ny7pX7pyV0a0RXppsxhl6iZdA6aEVE9f2XMIon"
access_secret = "mbq9YoyunjJJ1sqwyM0QOjNB1yiyb2VI5YjWTPYX4LSpf"


auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

api = tweepy.API(auth)

public_tweet = api.search("Machine Learning")

for tweet in public_tweet:
    text = tweet.text
    print(text)
    analysis = TextBlob(text)
    print(analysis.sentiment,"\n")

