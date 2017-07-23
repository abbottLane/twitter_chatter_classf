from Classifiers.TweetClassificationModel import TweetClassificationModel
from TweetLoader.TweetLoader import TweetLoader


def main():
    #-------Train----------
    tweet_dir = "/home/wlane/PycharmProjects/coin_twitter_chatter/Resources/bitcoin_tweets_training.csv"
    model_out_dir="Resources/Models/t_test_model.pk1"
    vectorizer_out_dir = "Resources/Vectorizers/t_test_model.vec"
    ch2_out_dir = "Resources/Vectorizers/t_test_model.ch2"

    tl = TweetLoader(tweet_dir)
    tweets = tl.load_tweets()

    model = TweetClassificationModel()
    model.train(tweets)
    model.dump_model(model_out_dir)
    model.dump_vectorizer(vectorizer_out_dir)
    model.dump_ch2(ch2_out_dir)

    # -------Predict----------
    test_tweet_dir = "/home/wlane/PycharmProjects/coin_twitter_chatter/Resources/bitcoin_tweets_testing.csv"
    test_loader = TweetLoader(test_tweet_dir)
    test_tweets = test_loader.load_tweets(test=True)
    model.load_model(model_out_dir)
    model.load_vectorizer(vectorizer_out_dir)
    predictions = model.predict(test_tweets)

    for i, p in enumerate(predictions):
        print (p + "\t"+ test_tweets[i].text)

if __name__ == '__main__':
    main()