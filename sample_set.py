import copy
import random
from typing import List, Dict

import numpy

from dataset import Dataset, MOTIF_LENGTH, Instance, LEFT_SIDE, RIGHT_SIDE

EXPO = 4
SIDE = 6
N_RANDOM_INSTANCES = 2000
LAMBDA = 1


class SampleSet:
    training_x_samples: List[numpy.ndarray]
    training_y_samples: List[float]

    mean: numpy.ndarray

    testing_x_samples: List[numpy.ndarray]
    testing_y_samples: List[float]

    def __init__(self, training_sets: List[str], testing_sets: List[str]):
        self.training_x_samples = []
        self.training_y_samples = []

        self.mean = numpy.array([0] * (12 * SIDE * EXPO + 1))

        training_datasets = []

        for dataset_name in training_sets:
            training_datasets.append(
                Dataset(
                    MOTIF_LENGTH[dataset_name],
                    f"data/{dataset_name}/dataset.fa",
                    f"data/{dataset_name}/positions.conf"
                )
            )

            self.training_x_samples.append(
                SampleSet.frequencies_to_vector(training_datasets[-1], training_datasets[-1].motif_frequencies)
            )
            self.mean = numpy.add(self.mean, self.training_x_samples[-1])

        self.mean = numpy.divide(self.mean, len(training_sets))

        for i in range(len(training_datasets)):
            self.training_y_samples.append(
                SampleSet.score_candidate_instances(training_datasets[i], training_datasets[i].instances) +
                LAMBDA * numpy.linalg.norm(numpy.subtract(self.training_x_samples[i], self.mean))
            )

        for dataset in training_datasets:
            SampleSet.generate_samples(
                dataset,
                self.mean,
                self.training_x_samples,
                self.training_y_samples
            )
            print(f"Current number of training samples: {len(self.training_x_samples)}")

        self.testing_x_samples = []
        self.testing_y_samples = []

        for dataset_name in testing_sets:
            print(f"Prepare testing samples from dataset {dataset_name}")
            dataset = Dataset(
                MOTIF_LENGTH[dataset_name],
                f"data/{dataset_name}/dataset.fa",
                f"data/{dataset_name}/positions.conf"
            )

            self.testing_x_samples.append(SampleSet.frequencies_to_vector(dataset, dataset.motif_frequencies))
            self.testing_y_samples.append(
                SampleSet.score_candidate_instances(dataset, dataset.instances) +
                LAMBDA * numpy.linalg.norm(numpy.subtract(self.testing_x_samples[-1], self.mean))
            )

            SampleSet.generate_samples(
                dataset,
                self.mean,
                self.testing_x_samples,
                self.testing_y_samples
            )
            print(f"Current number of testing samples: {len(self.testing_x_samples)}")

    @staticmethod
    def score_candidate_instances(dataset: Dataset, candidate_instances: List[List[Instance]]) -> float:
        correct = 0

        for i in range(dataset.n):
            current_correct = 0
            for candidate_instance in candidate_instances[i]:
                for instance in dataset.instances[i]:
                    for j in range(dataset.w):
                        if instance.core[j] == candidate_instance.core[j]:
                            current_correct += 1
            correct += current_correct / len(dataset.instances[i]) / len(candidate_instances[i])

        return correct / dataset.n / dataset.w

    @staticmethod
    def append_expo_to_vector(vector: List[float], x: float):
        for i in range(1, EXPO + 1):
            vector.append(x ** i)

    @staticmethod
    def frequencies_to_vector(dataset: Dataset, frequencies: Dict[str, List[float]]) -> numpy.ndarray:
        vector = [1]
        for j in range(SIDE):
            front_side = []
            back_side = []
            middle_side = []
            for ch in dataset.alphabet:
                front_side.append(frequencies[ch][j] / dataset.background_frequencies[ch])
                back_side.append(frequencies[ch][dataset.w - j - 1] / dataset.background_frequencies[ch])

                frequency_sum = 0
                for k in range(j, j + dataset.w - SIDE + 1):
                    frequency_sum += frequencies[ch][k] / dataset.background_frequencies[ch]
                frequency_sum /= dataset.w - SIDE + 1
                middle_side.append(frequency_sum)

            front_side.sort(reverse=True)
            back_side.sort(reverse=True)
            middle_side.sort(reverse=True)

            for value in front_side:
                SampleSet.append_expo_to_vector(vector, value)

            for value in back_side:
                SampleSet.append_expo_to_vector(vector, value)

            for value in middle_side:
                SampleSet.append_expo_to_vector(vector, value)

        return numpy.array(vector)

    @staticmethod
    def generate_samples(dataset: Dataset, mean: numpy.ndarray, x_samples: List[numpy.ndarray], y_samples: List[float]):
        # # replace 1 instances
        instances = copy.deepcopy(dataset.instances)

        for i in range(dataset.n):
            for j in range(len(dataset.instances[i])):
                note = instances[i][j]

                for k in range(len(dataset.sequences[i].content) - dataset.w + 1):
                    instances[i][j] = Instance(
                        dataset.sequences[i].content[k - LEFT_SIDE:k],
                        dataset.sequences[i].content[k:k + dataset.w],
                        dataset.sequences[i].content[k + dataset.w:k + dataset.w + RIGHT_SIDE]
                    )

                    if random.randrange(10) in range(5):
                        x_samples.append(
                            SampleSet.frequencies_to_vector(
                                dataset,
                                dataset.get_frequencies_of_instances(instances)
                            )
                        )

                        y_samples.append(
                            SampleSet.score_candidate_instances(
                                dataset,
                                instances
                            ) + LAMBDA*numpy.linalg.norm(numpy.subtract(x_samples[-1], mean))
                        )

                instances[i][j] = note

        # random generate instances
        for _ in range(N_RANDOM_INSTANCES):
            instances = []
            for i in range(dataset.n):
                j = random.randrange(len(dataset.sequences[i].content) - dataset.w + 1)
                instances.append([
                    Instance(
                        dataset.sequences[i].content[j - LEFT_SIDE:j],
                        dataset.sequences[i].content[j:j + dataset.w],
                        dataset.sequences[i].content[j + dataset.w:j + dataset.w + RIGHT_SIDE]
                    )
                ])

            x_samples.append(
                SampleSet.frequencies_to_vector(
                    dataset,
                    dataset.get_frequencies_of_instances(instances)
                )
            )

            y_samples.append(
                SampleSet.score_candidate_instances(dataset, instances) +
                LAMBDA*numpy.linalg.norm(numpy.subtract(x_samples[-1], mean))
            )
