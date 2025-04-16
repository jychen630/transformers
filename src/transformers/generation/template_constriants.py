from abc import ABC
from typing import List, Union, Optional

class TemplateConstraint(Constraint):
    r"""Constraint that enforces a template pattern with fixed and flexible slots."""
    
    def __init__(self, template: List[Union[int, List[int], str]]):
        super(Constraint, self).__init__()
        pass

    def advance(self) -> Optional[Union[int, List[int]]]:
        pass

    def does_advance(self, token_id: int) -> bool:
        pass

    def update(self, token_id: int):
        pass

    def reset(self):
        pass

    def remaining(self) -> int:
        pass

    def copy(self, stateful=False):
        pass