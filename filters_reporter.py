import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import pandas as pd
import os

class Reporter(Filter):
    def __init__(self, event_csv="cafe_events.csv"):
        super().__init__()
        self.event_csv = event_csv
        if not os.path.exists(self.event_csv):
            pd.DataFrame(columns=["type","id","timestamp","duration_sec"]).to_csv(self.event_csv, index=False)

    def run(self, data):
        if "events" in data:
            df = pd.DataFrame(data["events"])
            df.to_csv(self.event_csv, mode="a", header=False, index=False)
        return data
