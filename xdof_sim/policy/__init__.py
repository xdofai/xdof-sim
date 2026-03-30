"""Policy interfaces for running trained models in the sim.

The policy module requires additional dependencies (torch).
Install with: pip install xdof-sim[policy]
"""

from xdof_sim.policy.base import PolicyConfig, BasePolicy

__all__ = ["PolicyConfig", "BasePolicy"]

try:
    from xdof_sim.policy.lbm_policy import LBMPolicy, LBMPolicyConfig

    __all__ += ["LBMPolicy", "LBMPolicyConfig"]
except ImportError:
    pass
