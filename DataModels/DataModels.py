class Tweet(object):
    def __init__(self, handle,text,time,desc, polarity, subjectivity, loc, followers, user_created,label=None):
        self.text=text
        self.time=time
        self.desc = desc
        self.polarity = polarity
        self.subjectivity=subjectivity
        self.loc = loc
        self.followers=followers
        self.user_created=user_created
        self.handle = handle
        self.id = id
        self.gold_label = label