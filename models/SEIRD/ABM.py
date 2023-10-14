import copy
import sys
from enum import IntEnum
from os.path import dirname as up
from typing import Sequence, Union

import numpy as np
import torch
from dantro._import_tools import import_module_from_path

sys.path.append(up(up(up(__file__))))
base = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include")

Vector = base.Vector
distance = base.distance

""" The SIR agent-based model of infectious diseases """


class kinds(IntEnum):
    SUSCEPTIBLE = 0
    EXPOSED = 1
    INFECTED = 2
    RECOVERED = 3
    SYMPTOMATIC = 4
    HOSPITALIZED = 5
    CRITICAL = 6
    DECEASED = 7
    QUARANTINE_S = 8
    QUARANTINE_E = 9
    QUARANTINE_I = 10
    CONTACT_TRACED = 11


# --- The agent class --------------------------------------------------------------------------------------------------
class Agent:
    def __init__(self, *, id: int, kind: kinds, position: Vector):
        """

        :param id: the agent id (fixed)
        :param kind: the agent kind
        :param pos: the agent position
        """
        self.id = id
        self.kind = kind
        self.position = position
        self.init_position = position
        self.init_kind = kind

    # Move an agent along a direction vector
    def move(self, direction: Vector):
        self.position += direction

    # Moves an agent within a square domain, repelling it from the boundary walls
    def repel_from_wall(self, direction: Vector, space: Union[Vector, Sequence]):
        if isinstance(space, Vector):
            x_0, x_1 = 0, space.x
            y_0, y_1 = 0, space.y
        else:
            x_0, x_1 = space[0]
            y_0, y_1 = space[1]

        if not (self.position + direction).within_space(space):
            new_pos = self.position + direction
            if new_pos.x < x_0:
                direction.x = -(direction.x + 2 * (self.position.x - x_0))
            elif new_pos.x > x_1:
                direction.x = -(direction.x - 2 * (x_1 - self.position.x))
            if new_pos.y < y_0:
                direction.y = -(direction.y + 2 * (self.position.y - y_0))
            elif new_pos.y > y_1:
                direction.y = -(direction.y - 2 * (y_1 - self.position.y))
        self.move(direction)

    def move_in_periodic_space(self, direction: Vector, space: Union[Vector, Sequence]):

        if isinstance(space, Vector):
            x_0, x_1 = 0, space.x
            y_0, y_1 = 0, space.y
        else:
            (
                x_0,
                x_1,
            ) = space[0]
            y_0, y_1 = space[1]

        new_position = self.position + direction

        in_space = new_position.within_space(space)
        while not in_space:
            if new_position.x < x_0:
                new_position.x = x_1 - abs(x_0 - new_position.x)
            elif new_position.x > x_1:
                new_position.x = x_0 + abs(new_position.x - x_1)
            if new_position.y < y_0:
                new_position.y = y_1 - abs(y_0 - new_position.y)
            elif new_position.y > y_1:
                new_position.y = y_0 + abs(new_position.y - y_1)
            in_space = new_position.within_space(space)

        self.position = new_position

    def move_randomly_in_space(
        self,
        *,
        space: Union[Vector, Sequence],
        diffusion_radius: float,
        periodic: bool = False,
    ):
        """Move an agent randomly within a space with a given diffusivity. If the boundaries are periodic,
        the agent moves through the boundaries

        :param space: the space within which to move
        :param diffusion: the diffusivity
        :param periodic: whether the boundaries are periodic
        """

        # Get a random direction in the sphere with radius diffusion_radius
        direction = Vector((2 * np.random.rand() - 1), (2 * np.random.rand() - 1))
        direction.normalise(norm=diffusion_radius)

        # Non-periodic case: move within space, repelling from walls
        if not periodic:
            self.repel_from_wall(direction, space)

        # Periodic case: move through boundaries
        else:
            self.move_in_periodic_space(direction, space)

    def reset(self):
        self.position = self.init_position
        self.kind = self.init_kind

    def __repr__(self):
        return f"Agent {self.id}; " f"kind: {self.kind}; " f"position: {self.position}"


# --- The SIR ABM ------------------------------------------------------------------------------------------------------
class SEIRD_ABM:
    def __init__(
        self,
        *,
        N: int,
        space: tuple,
        sigma_s: float,
        sigma_i: float,
        sigma_r: float,
        r_infectious: float,
        k_E: float,
        k_I: float,
        k_SY: float,
        k_H: float,
        k_C: float,
        k_R: float,
        k_D: float,
        k_Q: float,
        k_S: float,
        is_periodic: bool,
        **__,
    ):
        """

        :param r_infectious: the radius of contact within which infection occurs
        :param k_E: the probability of infecting an agent within the infection radius
        :param k_I: the rate of becoming infectious
        :param k_SY: the rate of showing symptoms
        :param k_H: the rate of being hospitalized
        :param k_C: the rate of becoming a critical patient
        :param k_R: the rate of recovering
        :param k_D: the rate of mortality
        :param k_Q: the rate of going into self-isolation/quarantine
        :param k_S: the rate of going out of quarantine
        """

        # Parameters for the dynamics
        self.space = Vector(space[0], space[1])
        self.is_periodic = is_periodic
        self.sigma_s = sigma_s
        self.sigma_i = sigma_i
        self.sigma_r = sigma_r
        self.r_infectious = r_infectious
        self.k_E = torch.tensor(k_E)
        self.k_I = torch.tensor(k_I)
        self.k_SY = torch.tensor(k_SY)
        self.k_H = torch.tensor(k_H)
        self.k_C = torch.tensor(k_C)
        self.k_R = torch.tensor(k_R)
        self.k_D = torch.tensor(k_D)
        self.k_Q = torch.tensor(k_Q)
        self.k_S = torch.tensor(k_S)

        # Set up the cells and initialise their location at a random position in space.
        # All cells are initialised as susceptible
        self.N = N
        self.init_kinds = [kinds.INFECTED] + [kinds.SUSCEPTIBLE] * (self.N - 1)

        # Initialise the agent positions and kinds
        self.agents = {
            i: Agent(
                id=i,
                kind=self.init_kinds[i],
                position=Vector(
                    np.random.rand() * self.space.x, np.random.rand() * self.space.y
                ),
            )
            for i in range(self.N)
        }

        # Track the ids of the susceptible, infected, and recovered cells
        self.kinds = None

        # Track the current kinds, positions, and total kind counts of all the agents
        self.current_kinds = None
        self.current_positions = None
        self.current_counts = None

        # Count the number of susceptible, infected, and recovered agents.
        self.susceptible = None
        self.exposed = None
        self.infected = None
        self.recovered = None
        self.symptomatic = None
        self.hospitalized = None
        self.critical = None
        self.deceased = None
        self.quarantined_s = None
        self.quarantined_e = None
        self.quarantined_i = None
        self.contact_traced = None

        # Track the times since infection occurred for each agent. Index = time since infection
        self.times_since_infection = None

        # Initialise all the datasets
        self.initialise()

    # Initialises the ABM data containers
    def initialise(self):

        # Initialise the ABM with one infected agent.
        self.kinds = {
            kinds.SUSCEPTIBLE: {i: None for i in range(1, self.N)},
            kinds.EXPOSED: {},
            kinds.INFECTED: {0: None},
            kinds.RECOVERED: {},
            kinds.SYMPTOMATIC: {},
            kinds.HOSPITALIZED: {},
            kinds.CRITICAL: {},
            kinds.DECEASED: {},
            kinds.QUARANTINE_S: {},
            kinds.QUARANTINE_E: {},
            kinds.QUARANTINE_I: {},
        }

        self.current_kinds = [int(self.agents[i].kind) for i in range(self.N)]
        # S E I R SY H C D Qs Qe Qi C
        self.current_counts = torch.tensor(
            [
                [self.N - 1],
                [0.0],
                [1.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
            ],
            dtype=torch.float,
        )
        self.susceptible = torch.tensor(self.N - 1, dtype=torch.float)
        self.exposed = torch.tensor(0, dtype=torch.float)
        self.infected = torch.tensor(1, dtype=torch.float)
        self.recovered = torch.tensor(0, dtype=torch.float)
        self.symptomatic = torch.tensor(0, dtype=torch.float)
        self.hospitalized = torch.tensor(0, dtype=torch.float)
        self.critical = torch.tensor(0, dtype=torch.float)
        self.deceased = torch.tensor(0, dtype=torch.float)
        self.quarantined_s = torch.tensor(0, dtype=torch.float)
        self.quarantined_e = torch.tensor(0, dtype=torch.float)
        self.quarantined_i = torch.tensor(0, dtype=torch.float)
        self.contact_traced = torch.tensor(0, dtype=torch.float)

        self.current_positions = [
            (self.agents[i].position.x, self.agents[i].position.y)
            for i in range(self.N)
        ]

    # --- Update functions ---------------------------------------------------------------------------------------------

    # Updates the agent kinds
    def update_kinds(self, id=None, kind=None):

        if id is None:
            self.current_kinds = [int(self.agents[i].kind) for i in range(self.N)]
        else:
            self.current_kinds[id] = int(kind)

    # Updates the kind counts
    def update_counts(self):

        self.current_counts = torch.tensor(
            [
                [len(self.kinds[kinds.SUSCEPTIBLE])],
                [len(self.kinds[kinds.EXPOSED])],
                [len(self.kinds[kinds.INFECTED])],
                [len(self.kinds[kinds.RECOVERED])],
                [len(self.kinds[kinds.SYMPTOMATIC])],
                [len(self.kinds[kinds.HOSPITALIZED])],
                [len(self.kinds[kinds.CRITICAL])],
                [len(self.kinds[kinds.DECEASED])],
                [len(self.kinds[kinds.QUARANTINE_S])],
                [len(self.kinds[kinds.QUARANTINE_E])],
                [len(self.kinds[kinds.QUARANTINE_I])],
                [len(self.kinds[kinds.CONTACT_TRACED])],
            ]
        ).float()

    # Moves the agents randomly in space
    def move_agents_randomly(self):

        for kind in [kinds.SUSCEPTIBLE, kinds.EXPOSED]:
            for agent_id in self.kinds[kind].keys():
                self.agents[agent_id].move_randomly_in_space(
                    space=self.space,
                    diffusion_radius=self.sigma_s,
                    periodic=self.is_periodic,
                )

        for kind in [kinds.INFECTED, kinds.SYMPTOMATIC]:
            for agent_id in self.kinds[kind].keys():
                self.agents[agent_id].move_randomly_in_space(
                    space=self.space,
                    diffusion_radius=self.sigma_i,
                    periodic=self.is_periodic,
                )

        for agent_id in self.kinds[kinds.RECOVERED].keys():
            self.agents[agent_id].move_randomly_in_space(
                space=self.space,
                diffusion_radius=self.sigma_r,
                periodic=self.is_periodic,
            )

    # Updates the agent positions
    def update_positions(self):

        self.current_positions = [
            (self.agents[i].position.x, self.agents[i].position.y)
            for i in range(self.N)
        ]

    # Resets the ABM to the initial state
    def reset(self):
        for i in range(self.N):
            self.agents[i].reset()
        self.initialise()

    # Performs agent transitions
    def transition(self, transition_list, time: int):
        for t in transition_list:
            agent_ids = list(self.kinds[t[0]].keys())
            for agent_id in agent_ids:
                if t[2].dim() > 0:
                    t[2] = t[2][time]
                if np.random.rand() < t[2]:
                    self.agents[agent_id].kind = t[1]
                    self.kinds[t[0]].pop(agent_id)
                    self.kinds[t[1]].update({agent_id: None})
                    self.update_kinds(agent_id, t[1])
                    self.current_counts[t[0]] -= 1
                    self.current_counts[t[1]] += 1

    # --- Run function -------------------------------------------------------------------------------------------------

    # Runs the ABM for a single iteration
    def run_single(self, *, time: int):

        # Perform the transitions before moving the agents around
        self.transition(
            [
                (kinds.CRITICAL, kinds.DECEASED, self.k_D),
                (kinds.CRITICAL, kinds.RECOVERED, self.k_R),
                (kinds.HOSPITALIZED, kinds.CRITICAL, self.k_C),
                (kinds.HOSPITALIZED, kinds.RECOVERED, self.k_R),
                (kinds.SYMPTOMATIC, kinds.HOSPITALIZED, self.k_H),
                (kinds.SYMPTOMATIC, kinds.RECOVERED, self.k_R),
                (kinds.INFECTED, kinds.SYMPTOMATIC, self.k_SY),
                (kinds.INFECTED, kinds.QUARANTINE_I, self.k_Q),
                (kinds.INFECTED, kinds.RECOVERED, self.k_R),
                (kinds.EXPOSED, kinds.INFECTED, self.k_I),
                (kinds.EXPOSED, kinds.QUARANTINE_E, self.k_Q),
                (kinds.QUARANTINE_E, kinds.QUARANTINE_I, self.k_I),
                (kinds.SUSCEPTIBLE, kinds.QUARANTINE_S, self.k_Q),
                (kinds.QUARANTINE_S, kinds.SUSCEPTIBLE, self.k_S),
                (kinds.QUARANTINE_I, kinds.SYMPTOMATIC, self.k_SY),
                (kinds.QUARANTINE_I, kinds.RECOVERED, self.k_R),
            ],
            time,
        )

        # Collect the ids of the infected agents
        exposed_agent_ids = []

        if self.kinds[kinds.SUSCEPTIBLE] and (
            self.kinds[kinds.INFECTED] or self.kinds[kinds.SYMPTOMATIC]
        ):

            # For each susceptible agent, calculate the number of contacts to an infected agent.
            # A contact occurs when the susceptible agent is within the infection radius of an infected agent.
            num_contacts = torch.sum(
                torch.vstack(
                    [
                        torch.hstack(
                            [
                                torch.ceil(
                                    torch.relu(
                                        1
                                        - distance(
                                            self.agents[s].position,
                                            self.agents[i].position,
                                            space=self.space,
                                            periodic=self.is_periodic,
                                        )
                                        / self.r_infectious
                                    )
                                )
                                for i in list(self.kinds[kinds.INFECTED].keys())
                                + list(self.kinds[kinds.SYMPTOMATIC].keys())
                            ]
                        )
                        for s in self.kinds[kinds.SUSCEPTIBLE].keys()
                    ]
                ),
                dim=1,
            ).long()

            # Get the ids of susceptible agents that had a non-zero number of contacts with infected agents
            risk_contacts = torch.nonzero(num_contacts).long()

            if len(risk_contacts) != 0:

                k_E = self.k_E if self.k_E.dim() == 0 else self.k_E[time]

                # Move all susceptible agents that were in contact with an infected agent with probability
                # 1 - (1- p_infect)^n, where n is the number of contacts.
                exposures = torch.flatten(
                    torch.ceil(
                        torch.relu(
                            (1 - torch.pow((1 - k_E), num_contacts[risk_contacts]))
                            - torch.rand((len(risk_contacts), 1))
                        )
                    )
                )

                # Get the ids of the exposed agents
                exposed_agent_ids = [
                    list(self.kinds[kinds.SUSCEPTIBLE].keys())[_]
                    for _ in torch.flatten(
                        risk_contacts[torch.nonzero(exposures != 0.0, as_tuple=True)]
                    )
                ]

            if exposed_agent_ids:

                # Update the counts of susceptible and infected agents accordingly
                self.current_counts[0] -= len(exposed_agent_ids)
                self.current_counts[1] += len(exposed_agent_ids)

                # Update the agent kind to 'exposed'
                for agent_id in exposed_agent_ids:
                    self.agents[agent_id].kind = kinds.EXPOSED
                    self.kinds[kinds.SUSCEPTIBLE].pop(agent_id)
                    self.kinds[kinds.EXPOSED].update({agent_id: None})
                    self.update_kinds(agent_id, kinds.EXPOSED)

        # Move the susceptible, infected, and recovered agents with their respective diffusivities
        self.move_agents_randomly()
        self.update_positions()
