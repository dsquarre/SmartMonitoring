"""Client selection for the coordinator. The selectors are plugins (plugins/selectors.py: oort, random), shared with
the simple simulator; this module keeps the import path main.py uses."""
from plugins.selectors import (SELECTORS, ClientSelector, OortClientSelector, RandomClientSelector,  # noqa: F401
                               get_selector_by_name)
