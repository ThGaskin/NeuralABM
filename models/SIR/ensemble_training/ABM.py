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
    INFECTED = 1
    RECOVERED = 2


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
class SIR_ABM:
    def __init__(
        self,
        *,
        N: int,
        space: tuple,
        sigma_s: float,
        sigma_i: float,
        sigma_r: float,
        r_infectious: float,
        p_infect: float,
        t_infectious: float,
        is_periodic: bool,
        **__,
    ):
        """

        :param r_infectious: the radius of contact within which infection occurs
        :param p_infect: the probability of infecting an agent within the infection radius
        :param t_infectious: the time for which an agent is infectious
        """

        # Parameters for the dynamics
        self.space = Vector(space[0], space[1])
        self.is_periodic = is_periodic
        self.sigma_s = sigma_s
        self.sigma_i = sigma_i
        self.sigma_r = sigma_r
        self.r_infectious = torch.tensor(r_infectious)
        self.p_infect = torch.tensor(p_infect)
        self.t_infectious = torch.tensor(t_infectious)

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
        self.infected = None
        self.recovered = None

        # Track the times since infection occurred for each agent. Index = time since infection
        self.times_since_infection = None

        # Initialise all the datasets
        self.initialise()

    # Initialises the ABM data containers
    def initialise(self):
        # Initialise the ABM with one infected agent.
        self.kinds = {
            kinds.SUSCEPTIBLE: {i: None for i in range(1, self.N)},
            kinds.INFECTED: {0: None},
            kinds.RECOVERED: {},
        }
        self.current_kinds = [int(self.agents[i].kind) for i in range(self.N)]
        self.current_counts = torch.tensor(
            [[self.N - 1], [1.0], [0.0]], dtype=torch.float
        )
        self.susceptible = torch.tensor(self.N - 1, dtype=torch.float)
        self.infected = torch.tensor(1, dtype=torch.float)
        self.recovered = torch.tensor(0, dtype=torch.float)

        self.current_positions = [
            (self.agents[i].position.x, self.agents[i].position.y)
            for i in range(self.N)
        ]

        self.times_since_infection = [[0]]

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
                [len(self.kinds[kinds.INFECTED])],
                [len(self.kinds[kinds.RECOVERED])],
            ]
        ).float()

    # Moves the agents randomly in space
    def move_agents_randomly(self):
        for agent_id in self.kinds[kinds.SUSCEPTIBLE].keys():
            self.agents[agent_id].move_randomly_in_space(
                space=self.space,
                diffusion_radius=self.sigma_s,
                periodic=self.is_periodic,
            )

        for agent_id in self.kinds[kinds.INFECTED].keys():
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

    # --- Run function -------------------------------------------------------------------------------------------------

    # Runs the ABM for a single iteration
    def run_single(self, *, parameters: torch.tensor = None):
        p_infect = self.p_infect if parameters is None else parameters[0]
        t_infectious = self.t_infectious if parameters is None else parameters[1]

        # Collect the ids of the infected agents
        infected_agent_ids = []

        if self.kinds[kinds.SUSCEPTIBLE] and self.kinds[kinds.INFECTED]:
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
                                for i in self.kinds[kinds.INFECTED].keys()
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
                # Infect all susceptible agents that were in contact with an infected agent with probability
                # 1 - (1- p_infect)^n, where n is the number of contacts.
                infections = torch.flatten(
                    torch.ceil(
                        torch.relu(
                            (1 - torch.pow((1 - p_infect), num_contacts[risk_contacts]))
                            - torch.rand((len(risk_contacts), 1))
                        )
                    )
                )

                # Get the ids of the newly infected agents
                infected_agent_ids = [
                    list(self.kinds[kinds.SUSCEPTIBLE].keys())[_]
                    for _ in torch.flatten(
                        risk_contacts[torch.nonzero(infections != 0.0, as_tuple=True)]
                    )
                ]

            if infected_agent_ids:
                # Update the counts of susceptible and infected agents accordingly
                self.current_counts[0] -= len(infected_agent_ids)
                self.current_counts[1] += len(infected_agent_ids)

                # Update the agent kind to 'infected'
                for agent_id in infected_agent_ids:
                    self.agents[agent_id].kind = kinds.INFECTED
                    self.kinds[kinds.SUSCEPTIBLE].pop(agent_id)
                    self.kinds[kinds.INFECTED].update({agent_id: None})
                    self.update_kinds(agent_id, kinds.INFECTED)

        # Track the time since infection of the newly infected agents
        self.times_since_infection.insert(0, infected_agent_ids)

        # Change any 'infected' agents that have surpassed the maximum infected time to 'recovered'.
        if len(self.times_since_infection) > t_infectious:
            # The agents that have been infectious for the maximum amount of time have recovered
            recovered_agents = self.times_since_infection.pop()

            # Update the counts accordingly
            self.current_counts[1] -= len(recovered_agents)
            self.current_counts[2] += len(recovered_agents)

            # Update the agent kinds
            for agent_id in recovered_agents:
                self.kinds[kinds.INFECTED].pop(agent_id)
                self.kinds[kinds.RECOVERED].update({agent_id: None})
                self.update_kinds(agent_id, kinds.RECOVERED)

        # Move the susceptible, infected, and recovered agents with their respective diffusivities
        self.move_agents_randomly()
        self.update_positions()
