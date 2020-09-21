import redis
import os
import time
import pandas as pd

def redis_load_data():
    DATA_SOURCE = "data.csv"
    df = pd.read_csv(DATA_SOURCE)
    records = df.to_dict('records')
    r = redis.Redis()
    import time
    start_time = time.time()
    for i, record in enumerate(records):
        r.hmset('record_'+str(i), mapping=record)
    print("--- %s seconds ---" % (time.time() - start_time))

# To run this pipeline from the python CLI:
#   $python taxi_pipeline_redis.py
if __name__ == '__main__':
    redis_load_data()