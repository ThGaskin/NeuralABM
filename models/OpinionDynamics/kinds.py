from enum import IntEnum

# Agent kinds and their order used in the SEIRD model


class Compartments(IntEnum):
    susceptible = 0
    exposed = 1
    infected = 2
    symptomatic = 3
    recovered = 4
    hospitalized = 5
    critical = 6
    deceased = 7