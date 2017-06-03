from Classifiers.TweetClassificationModel import TweetClassificationModel
from TweetLoader.TweetLoader import TweetLoader


def main():
    #-------Train----------
    tweet_dir = "Resources/test_twitter_coin_training_data.csv"
    model_out_dir="Resources/Models/t_test_model.pk1"

    tl = TweetLoader(tweet_dir)
    tweets = tl.load_tweets()
    classes = tl.get_classes()

    model = TweetClassificationModel(classes)
    model.train(tweets)
    model.dump_model(model_out_dir)

    # -------Predict----------
    predictions = model.predict(tweets)

if __name__ == '__main__':
    main()