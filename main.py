import numpy
from sklearn import linear_model

from sample_set import SampleSet

datasets = []

TRAINING_SETS = [
    "ERE_200",
    "CRP",
    "hnf1",
    "reb1",
    "nfkb",
    "pdr3"
]

TESTING_SETS = [
    "E2F200",
    "mcb",
    "creb"
]


if __name__ == '__main__':
    model = linear_model.LinearRegression()

    samples = SampleSet(TRAINING_SETS, TESTING_SETS)

    model.fit(samples.training_x_samples, samples.training_y_samples)

    print("Coefficients: ", model.coef_)

    print(f"Training set variance score: {model.score(samples.training_x_samples, samples.training_y_samples)}")
    print(f"Testing set variance score: {model.score(samples.testing_x_samples, samples.testing_y_samples)}")

    coef_file = open("coef.txt", "w")
    for x in model.coef_:
        coef_file.write("{:.10f}\n".format(x))

    mean_file = open("mean.txt", "w")
    for x in samples.mean:
        mean_file.write("{:.10f}\n".format(x))

    for i in range(10):
        print(samples.training_y_samples[i], numpy.dot(samples.training_x_samples[i], model.coef_))

    for i in range(10):
        print(samples.testing_y_samples[i], numpy.dot(samples.testing_x_samples[i], model.coef_))

