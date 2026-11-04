"""SmartMonitoring, simple single-process version.

The methods (client selectors, server rules and controllers, attacks) are not implemented here: they are the plugins
of the distributed system, distributed/server/plugins, loaded as the package `plugins`. A method therefore runs the
same code in the simulator and in the deployed system.
"""
import importlib.util
import os
import sys

PLUGINS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "distributed", "server", "plugins"))

if "plugins" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("plugins", os.path.join(PLUGINS_DIR, "__init__.py"),
                                                   submodule_search_locations=[PLUGINS_DIR])
    _module = importlib.util.module_from_spec(_spec)
    sys.modules["plugins"] = _module
    _spec.loader.exec_module(_module)
