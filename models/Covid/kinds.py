from enum import IntEnum

# Agent kinds and their order used in the SEIRD+ model


class Compartments(IntEnum):
    susceptible = 0
    exposed = 1
    infected = 2
    recovered = 3
    symptomatic = 4
    hospitalized = 5
    critical = 6
    deceased = 7
    quarantine_S = 8
    quarantine_E = 9
    quarantine_I = 10
    contact_traced = 11
