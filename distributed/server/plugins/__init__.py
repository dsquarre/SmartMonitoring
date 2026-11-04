"""SmartMonitoring plugins: client selectors, server rules, server controllers and client behaviours.

Shared by the distributed system (server/selector.py, server/aggregator.py) and the simple simulator
(simple/smartmon/simulation.py), so a method behaves the same in both. See base.py for the interfaces.

    from plugins import build_controller, get_selector_by_name
    controller = build_controller("select_train", rule_args={"fedyogi": {"eta": 0.01}})
    selector = get_selector_by_name("oort", seed=0)
"""
from .attacks import ATTACKS, craft_malicious_weights
from .base import ClientUpdate, Controller, RoundContext, Rule
from .controllers import CONTROLLERS, build_controller, parse_spec
from .rules import RULES, build_rule
from .selectors import SELECTORS, ClientSelector, OortClientSelector, RandomClientSelector, get_selector_by_name

__all__ = ["ATTACKS", "CONTROLLERS", "RULES", "SELECTORS", "ClientSelector", "ClientUpdate", "Controller",
           "OortClientSelector", "RandomClientSelector", "RoundContext", "Rule", "build_controller", "build_rule",
           "craft_malicious_weights", "get_selector_by_name", "parse_spec"]
