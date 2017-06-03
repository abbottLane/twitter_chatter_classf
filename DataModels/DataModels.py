class Tweet(object):
    def __init__(self, handle, text,timestamp, name, id, gold_label):
        self.text=text
        self.handle = handle
        self.timestamp = timestamp
        self.name = name
        self.id = id
        self.gold_label = gold_label