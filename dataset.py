import operator
from dataclasses import dataclass
from typing import List, Dict, Tuple

MOTIF_LENGTH = {
    "ERE_200": 13,
    "E2F200": 11,
    "creb": 12,
    "CRP": 22,
    "hnf1": 14,
    "mcb": 6,
    # "mef2": 10,
    # "myod": 6,
    "nfkb": 12,
    "pdr3": 8,
    "reb1": 10,
    # "srf": 12,
    # "tbp": 7
}

LEFT_SIDE = 0
RIGHT_SIDE = 0


@dataclass
class Sequence:
    name: str
    content: str


@dataclass
class Instance:
    left: str
    core: str
    right: str


class Dataset:
    alphabet: str
    a: int
    n: int
    w: int
    sequences: List[Sequence]

    n_motif = 0
    motif: str
    offsets: List[List[int]]
    instances: List[List[Instance]]

    motif_frequencies: Dict[str, List[float]]
    background_frequencies: Dict[str, float]

    def __init__(self, w, object_file_name, answer_file_name, motif=None, alphabet=None):
        if not alphabet:
            alphabet = "ACGT"
        self.alphabet = alphabet
        self.a = len(alphabet)
        self.w = w
        self.read_from_file(object_file_name, answer_file_name)
        self.n_motif, self.instances = self.get_instances_from_offsets(self.offsets)
        self.motif_frequencies = self.get_frequencies_of_instances(self.instances)
        self.motif = motif or self.pick_motif()

    def get_instances_from_offsets(self, offsets: List[List[int]]) -> Tuple[int, List[List[Instance]]]:
        n_motif = 0

        instances = [[] for _ in range(self.n)]

        for i in range(self.n):
            for j in offsets[i]:
                if j < LEFT_SIDE or j + self.w > len(self.sequences[i].content) - RIGHT_SIDE:
                    continue
                instances[i].append(
                    Instance(
                        self.sequences[i].content[j - LEFT_SIDE:j],
                        self.sequences[i].content[j:j + self.w],
                        self.sequences[i].content[j + self.w:j + self.w + RIGHT_SIDE]
                    )
                )

            n_motif += len(instances[i])

        return n_motif, instances

    def read_from_file(self, object_file_name: str, answer_file_name: str):
        object_file = open(object_file_name, "r")

        self.sequences = []

        for line in object_file:
            line = line.strip()
            if not line:
                continue
            if line[0] == '>':
                name = line[1:-1].split()[0]
                self.sequences.append(Sequence(name=name, content=""))
            else:
                self.sequences[-1].content += line.upper()

        self.n = len(self.sequences)

        answer_file_name = open(answer_file_name, "r")
        self.offsets = [[] for _ in range(self.n)]

        for line in answer_file_name:
            line.strip()
            if not line:
                continue
            p = line.find("=")
            i = int(line[:p]) - 1
            if line[p+1:-1]:
                self.offsets[i] = list(map(int, line[p+1:-1].split(",")))

        total = 0
        for sequence in self.sequences:
            total += len(sequence.content)

        occurrences = {ch: 0 for ch in self.alphabet}
        for sequence in self.sequences:
            for ch in sequence.content:
                occurrences[ch] += 1

        self.background_frequencies = {
            ch: occurrences[ch] / total
            for ch in self.alphabet
        }

    def pick_motif(self) -> str:
        counter = [dict() for _ in range(self.w)]
        for i in range(self.n):
            for instance in self.instances[i]:
                for k in range(self.w):
                    if counter[k].get(instance.core[k]):
                        counter[k][instance.core[k]] += 1
                    else:
                        counter[k][instance.core[k]] = 1
        motif = "".join([
            max(counter[k].items(), key=operator.itemgetter(1))[0]
            for k in range(self.w)
        ])
        return motif

    def get_frequencies_of_instances(self, instances: List[List[Instance]]) -> Dict[str, List[float]]:
        occurrences = {ch: [0]*self.w for ch in self.alphabet}
        counter = 0

        for i in range(self.n):
            counter += len(instances[i])
            for instance in instances[i]:
                for j in range(self.w):
                    occurrences[instance.core[j]][j] += 1

        return {
            ch: [
                occurrences[ch][j] / counter
                for j in range(self.w)
            ]
            for ch in self.alphabet
        }
