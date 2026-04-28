# Fifth and final filter in the pipeline — persists coffee events to a CSV file using Pandas.
# Most frames produce no events so this filter does nothing. Only writes when ZoneLogic
# detected a coffee this frame and added an entry to data["events"].

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openfilter_stub import Filter
import pandas as pd  # handles CSV creation and append writes cleanly
import os            # os.path.exists check prevents overwriting an existing log on restart


class Reporter(Filter):
    """
    Append-only event logger. Creates the CSV with headers on first run,
    then appends one row per event on subsequent frames. Survives restarts —
    events from previous runs are preserved because we never overwrite the file.
    """

    # Creates the CSV file with column headers if it doesn't already exist.
    # On restart the file is left untouched so previous session data is preserved.
    def __init__(self, event_csv="cafe_events.csv"):
        super().__init__()
        # Store the output path — defaults to cafe_events.csv next to the running script
        self.event_csv = event_csv
        # Only create the file on a fresh run — skip if it already exists from a previous session
        if not os.path.exists(self.event_csv):
            pd.DataFrame(columns=["type", "id", "timestamp", "duration_sec"]).to_csv(self.event_csv, index=False)

    # Appends any events from this frame to the CSV — skips write entirely if no events occurred.
    # mode="a" appends without overwriting, header=False prevents duplicate column headers.
    def run(self, data):
        # Most frames have no events — guard prevents unnecessary disk I/O every frame
        if "events" in data:
            # pd.DataFrame converts the list of event dicts into a table with one row per event
            df = pd.DataFrame(data["events"])
            # Append to existing file — mode="a" adds rows at the end, never overwrites
            df.to_csv(self.event_csv, mode="a", header=False, index=False)
        return data