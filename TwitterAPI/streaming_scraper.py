import settings
import tweepy
import dataset
from textblob import TextBlob
from sqlalchemy.exc import ProgrammingError
import json

from Classifiers.TweetClassificationModel import TweetClassificationModel
from DataModels.DataModels import Tweet

db = dataset.connect(settings.CONNECTION_STRING)
# Load classification models
model_dir = "../Resources/Models/t_test_model.pk1"
vectorizer_dir = "../Resources/Vectorizers/t_test_model.vec"
ch2_dir = "../Resources/Vectorizers/t_test_model.ch2"

model = TweetClassificationModel()
model.load_model(model_dir)
model.load_vectorizer(vectorizer_dir)
model.load_ch2(ch2_dir)

def main():
    # load streaming API auth info
    auth = tweepy.OAuthHandler(settings.TWITTER_APP_KEY, settings.TWITTER_APP_SECRET)
    auth.set_access_token(settings.TWITTER_KEY, settings.TWITTER_SECRET)
    api = tweepy.API(auth)

    # start listening
    stream_listener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
    stream.filter(track=settings.TRACK_TERMS)



class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        if status.retweeted:
            return

        description = status.user.description
        loc = status.user.location
        text = status.text
        coords = status.coordinates
        geo = status.geo
        name = status.user.screen_name
        user_created = status.user.created_at
        followers = status.user.followers_count
        id_str = status.id_str
        created = status.created_at
        retweets = status.retweet_count
        bg_color = status.user.profile_background_color
        blob = TextBlob(text)
        sent = blob.sentiment
        polarity=sent.polarity

        # Cast tweet to obj for classs prediction
        tweet = Tweet(name,text,created,description, sent, polarity, loc, followers, user_created)
        prediction = model.predict([tweet])
        print(str(prediction) + "========" + tweet.text)
        # print(text)

        if geo is not None:
            geo = json.dumps(geo)

        if coords is not None:
            coords = json.dumps(coords)

        table = db[settings.TABLE_NAME]
        try:
            table.insert(dict(
                user_description=description,
                user_location=loc,
                coordinates=coords,
                text=text,
                geo=geo,
                user_name=name,
                user_created=user_created,
                user_followers=followers,
                id_str=id_str,
                created=created,
                retweet_count=retweets,
                user_bg_color=bg_color,
                polarity=sent.polarity,
                subjectivity=sent.subjectivity,
                classification=prediction[0]
            ))
        except ProgrammingError as err:
            print(err)

    def on_error(self, status_code):
        if status_code == 420:
            #returning False in on_data disconnects the stream
            print "Disconnected with a 420"
            return False

if __name__ =="__main__":
    main()
