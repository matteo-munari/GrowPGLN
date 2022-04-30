import torch


class MeanStdEstimator:
    """Online mean and standard deviation estimator using Welford's algorithm."""
    def __init__(self):
        self.count = 0
        self.mean = None
        self.m2 = None

    def __call__(self, sample, test=False):
        if not test:  # do not update statistics using test data
            if self.count == 0:
                self.mean = torch.zeros(size=sample.shape)
                self.m2 = torch.zeros(size=sample.shape)

            self.count += 1
            delta = sample - self.mean
            self.mean += delta / self.count
            delta_2 = sample - self.mean
            self.m2 += delta * delta_2

        stddev = torch.sqrt(torch.div(self.m2, self.count))
        return self.mean, stddev
