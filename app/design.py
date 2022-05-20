from .utils import YamlMixin
from dataclasses import dataclass


@dataclass
class ReinforcedConcreteFrame(YamlMixin):
    name: str = None

    def force_design(self):
        return True
