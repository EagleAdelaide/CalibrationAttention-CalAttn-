import time


class Logger:
    def __init__(self, path):
        self.path = path

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(self.path, "a") as f:
            f.write(line + "\n")
