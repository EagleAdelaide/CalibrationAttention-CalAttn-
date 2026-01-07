class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.cnt = 0
        self.avg = 0.0

    def update(self, v, n=1):
        self.sum += float(v) * n
        self.cnt += int(n)
        self.avg = self.sum / max(1, self.cnt)
