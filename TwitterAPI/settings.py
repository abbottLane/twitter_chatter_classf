TRACK_TERMS = ["bitcoin"]
CONNECTION_STRING="sqlite:///tweets.db"
CSV_NAME = "bitcoin_tweets.csv"
TABLE_NAME = "bitcoin"

try:
    from private import *
except Exception:
    pass