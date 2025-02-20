class Normalizer:
    """
    Normalizer class to normalize and denormalize data using mean and std.

    mean = sum(x) / N
    std = sqrt(sum((x - mean)^2) / N)
    """

    def __init__(self, data):
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0, unbiased=False)

    def normalize(self, data):
        return (data - self.mean) / (self.std + 1e-8)

    def denormalize(self, data):
        return data * (self.std + 1e-8) + self.mean
