import sys
from os.path import dirname as up

import pytest
import torch
from dantro._import_tools import import_module_from_path
from dantro._yaml import load_yml
from pkg_resources import resource_filename

sys.path.insert(0, up(up(up(__file__))))

SIR = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="models.SIR")
vec = import_module_from_path(mod_path=up(up(up(__file__))), mod_str="include.vector")

Agent = SIR.ABM.Agent
Vector = vec.Vector

# Load the test config
CFG_FILENAME = resource_filename("tests", "cfgs/SIR_Dynamics.yml")
test_cfg = load_yml(CFG_FILENAME)


def test_agent():
    """Test agent creation and movement"""

    agent = Agent(id=0, kind="some_kind", position=Vector(0, 0))

    agent.move(Vector(1, 1))
    assert agent.position == Vector(1, 1)

    agent.move(Vector(-1, -1))
    assert agent.position == Vector(0, 0)

    space = [[-1, 1], [-1, 1]]

    # Test repelling from wall
    agent.position = Vector(-0.5, 0)
    agent.repel_from_wall(space=space, direction=Vector(-1, 0))
    assert agent.position == Vector(-0.5, 0)

    agent.repel_from_wall(space=space, direction=Vector(-1, 1))
    assert agent.position == Vector(-0.5, 1)

    # Test movement in periodic space
    agent.move_in_periodic_space(direction=Vector(1, 1), space=space)
    assert agent.position == Vector(0.5, 0)

    agent.move_in_periodic_space(direction=Vector(1, 2), space=space)
    assert agent.position == Vector(-0.5, 0)

    agent.position = Vector(0, 0)
    agent.move_in_periodic_space(direction=Vector(2, 2), space=space)
    assert agent.position == Vector(0, 0)

    space = Vector(4, 5)

    # Test diffusion
    agent.move_randomly_in_space(space=space, diffusion_radius=0.5)
    assert agent.position.within_space(space)
    assert agent.position != Vector(0, 0)
    assert abs(agent.position) == pytest.approx(0.5, 1e-10)

    agent.move_randomly_in_space(space=space, diffusion_radius=4)
    assert agent.position.within_space(space)
    agent.position = Vector(0, 0)
    agent.move_randomly_in_space(space=space, diffusion_radius=10, periodic=True)

    agent.position = Vector(0, 0)
    space = [[-1, 1], [-1, 1]]
    agent.move_randomly_in_space(space=space, diffusion_radius=1)
    assert agent.position != Vector(0, 0)

    # Test reset
    agent.kind = "some_other_kind"
    agent.reset()
    assert agent.position == Vector(0, 0)
    assert agent.kind == "some_kind"

    # Test representation
    assert str(agent) == "Agent 0; kind: some_kind; position: (0, 0)"


def test_ABM():
    """Test ABM initialisation"""

    for entry in test_cfg:
        ABM_cfg = test_cfg[entry]
        ABM = SIR.SIR_ABM(**ABM_cfg)

        assert ABM
        assert ABM.N == ABM_cfg["N"]
        assert ABM.p_infect == ABM_cfg["p_infect"]
        assert ABM.t_infectious == ABM_cfg["t_infectious"]
        assert ABM.space == Vector(ABM_cfg["space"][0], ABM_cfg["space"][1])

        assert len(ABM.current_kinds) == ABM.N
        assert (
            ABM.current_counts
            == torch.tensor([[ABM.N - 1], [1], [0]], dtype=torch.float)
        ).all()

        # Test the ABM runs and obeys basic properties
        for n in range(ABM_cfg["num_steps"]):
            ABM.run_single()

            # Test the agent count stays constant
            assert torch.sum(ABM.current_counts) == ABM.N
            assert torch.sum(torch.abs(ABM.current_counts)) == ABM.N

            # Test the agents remain within the space
            for _, agent in ABM.agents.items():
                assert agent.position.within_space(ABM.space)


def test_dynamics():
    """Test basic dynamics work"""
    cfg = test_cfg["dynamics"]
    ABM = SIR.SIR_ABM(**cfg)
    for n in range(cfg["num_steps"]):
        ABM.run_single()
        assert torch.sum(ABM.current_counts) == ABM.N

    assert len(ABM.current_kinds) == ABM.N
    assert (
        ABM.current_counts != torch.tensor([[ABM.N - 1], [1], [0]], dtype=torch.float)
    ).all()

    ABM.reset()
    assert (
        ABM.current_counts == torch.tensor([[ABM.N - 1], [1], [0]], dtype=torch.float)
    ).all()


def test_no_dynamics():
    """Test nothing happens when p_infect is 0 and t_infectious > num_steps"""

    cfg = test_cfg["no_dynamics"]
    ABM = SIR.SIR_ABM(**cfg)
    for n in range(cfg["num_steps"]):
        ABM.run_single()
        assert torch.sum(ABM.current_counts) == ABM.N

    assert len(ABM.current_kinds) == ABM.N
    assert (
        ABM.current_counts == torch.tensor([[ABM.N - 1], [1], [0]], dtype=torch.float)
    ).all()

    ABM.reset()


def test_full_recovery():
    """Test all agents make a full recovery"""

    cfg = test_cfg["full_recovery"]
    ABM = SIR.SIR_ABM(**cfg)
    for n in range(cfg["num_steps"]):
        ABM.run_single()

    assert (ABM.current_counts[1] == 0).all()
