class Filter:
    def __init__(self): pass
    def stop(self): pass
    def run(self, data): return data
    def close(self): pass

class Graph:
    def __init__(self, filters): 
        self.filters = filters
        self._out = {}
    def step(self):
        data = {}
        for f in self.filters:
            data = f.run(data)
            if data is None:
                return False
        self._out = data
        return True
    def last_output(self): 
        return self._out
    def close(self):
        for f in self.filters:
            if hasattr(f, "close"):
                f.close()
