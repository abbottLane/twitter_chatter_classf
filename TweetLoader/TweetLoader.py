import csv

from DataModels.DataModels import Tweet


class TweetLoader():
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.classes =None

    def load_tweets(self, test=False):
        '''
        Reads data from input dir, builds Tweet objects, returns list of all Tweet objs
        :return: list of Tweet objects
        '''
        tweets=list()
        with open(self.input_dir,  "rb") as csv_file:
            data = csv.reader(csv_file)
            col2lab_map = dict()
            for i, row in enumerate(data):
                if i==0:
                    # create column number-to-label mapping
                    label_cols = row[16:]
                    for i, col in enumerate(label_cols):
                        col2lab_map[i]=col
                    self.classes = col2lab_map.values()

                elif i !=0: # skip the header
                    polarity=round(float(row[1]), 1)
                    time=row[4]
                    text=row[5].lower().decode("utf-8")
                    desc=row[6].lower().decode("utf-8")
                    followers=int(row[7])
                    loc=row[8]
                    subjectivity=round(float(row[11]),1)
                    user_created=row[12]
                    handle=row[15].lower().decode("utf-8")
                    label = self._get_label_from_multiple_columns(row[16:], col2lab_map)
                    if not label and not test: # if set isn't annotated past here, return what we have
                        return tweets
                    tweets.append(Tweet(handle,text,time,desc, polarity, subjectivity, loc, followers, user_created,label))
        return tweets

    def _get_label_from_multiple_columns(self, list_of_tag_cols, col2label):
        '''
        List of label columns should have a single marked column. Assumes all tweets have 1 label.
        :param list_of_tag_cols: a list of 0's or empty strings, and a single character of some other kind indicating class label column
        :return:
        '''
        for i, item in enumerate(list_of_tag_cols):
            if item != "" and item != "0":
                return col2label[i]
        return None

    def get_classes(self):
        return self.classes