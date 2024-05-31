from enum import IntEnum

# Agent kinds and their order used in the SEIRD model


class Compartments(IntEnum):
    susceptible = 0
    exposed = 1
    infected = 2
    recovered = 3
    hospitalized = 4
    critical = 5
    deceased = 6