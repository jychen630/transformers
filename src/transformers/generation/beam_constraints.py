from abc import ABC, abstractmethod
from typing import List, Optional, Set
import logging
from datetime import datetime
import os
os.makedirs("logs", exist_ok=True)
# Set up logging
logger = logging.getLogger('template_constraint')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(f'logs/template_constraint-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
file_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(file_handler)

class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    def __init__(self):
        # test for the above condition
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            stepped, completed, reset = self.update(advance)
            counter += 1

            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token(s) that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids (Union[int, List[int], None]):
                - A single token ID (int) that advances the constraint, or
                - A list of token IDs that could advance the constraint
                - None if the constraint is completed or cannot be advanced
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class PhrasalConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """

    def __init__(self, token_ids: List[int]):
        super(Constraint, self).__init__()

        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids

        self.seqlen = len(self.token_ids)
        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.completed = False

    def advance(self):
        if self.completed:
            return None
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False

        return token_id == self.token_ids[self.fulfilled_idx + 1]

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.fulfilled_idx += 1
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True
            self.completed = completed
        else:
            # failed to make progress.
            reset = True
            self.reset()
        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.fulfilled_idx = 0

    def remaining(self):
        return self.seqlen - (self.fulfilled_idx + 1)

    def copy(self, stateful=False):
        new_constraint = PhrasalConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint


class DisjunctiveTrie:
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max([len(one) for one in nested_token_ids])

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for tidx, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count


class DisjunctiveConstraint(Constraint):
    r"""
    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`):
            A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
            the list of words.
    """

    def __init__(self, nested_token_ids: List[List[int]]):
        super(Constraint, self).__init__()

        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids

        self.seqlen = self.trie.max_height
        self.current_seq = []
        self.completed = False

    def advance(self):
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens

    def update(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed

        return stepped, completed, reset

    def reset(self):
        self.completed = False
        self.current_seq = []

    def remaining(self):
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        else:
            return self.seqlen - len(self.current_seq)

    def copy(self, stateful=False):
        new_constraint = DisjunctiveConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.current_seq = self.current_seq
            new_constraint.completed = self.completed

        return new_constraint


class ConstraintListState:
    r"""
    A class for beam scorers to track its progress through a list of constraints.

    Args:
        constraints (`List[Constraint]`):
            A list of [`Constraint`] objects that must be fulfilled by the beam scorer.
    """

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints

        # max # of steps required to fulfill a given constraint
        self.max_seqlen = max([c.seqlen for c in constraints])
        self.n_constraints = len(constraints)
        self.completed = False

        self.init_state()

    def init_state(self):
        self.complete_constraints = []
        self.inprogress_constraint = None
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]

    def get_bank(self):
        add = 0
        if self.inprogress_constraint:
            # extra points for having a constraint mid-fulfilled
            add += self.max_seqlen - self.inprogress_constraint.remaining()

        return (len(self.complete_constraints) * self.max_seqlen) + add

    def advance(self):
        """The list of tokens to generate such that we can make progress.
        By "list" we don't mean the list of token that will fully fulfill a constraint.

        Given constraints `c_i = {t_ij | j == # of tokens}`, If we're not in the middle of progressing through a
        specific constraint `c_i`, we return:

        `[t_k1 for k in indices of unfulfilled constraints]`

        If we are in the middle of a constraint, then we return:
            `[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

        Though we don't care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint,
        that's the only one we'll return.
        """
        token_list = []
        if self.inprogress_constraint is None:
            for constraint in self.pending_constraints:  # "pending" == "unfulfilled yet"
                advance = constraint.advance()
                if isinstance(advance, int):
                    token_list.append(advance)
                elif isinstance(advance, list):
                    token_list.extend(advance)
        else:
            advance = self.inprogress_constraint.advance()
            if isinstance(advance, int):
                token_list.append(advance)
            elif isinstance(advance, list):
                token_list.extend(advance)

        if len(token_list) == 0:
            return None
        else:
            return token_list

    def reset(self, token_ids: Optional[List[int]]):
        """
        token_ids: the tokens generated thus far to reset the state of the progress through constraints.
        """
        self.init_state()

        if token_ids is not None:
            for token in token_ids:
                # completes or steps **one** constraint
                complete, stepped = self.add(token)

                # the entire list of constraints are fulfilled
                if self.completed:
                    break

    def add(self, token_id: int):
        if not isinstance(token_id, int):
            raise TypeError(f"`token_id` should be an `int`, but is `{token_id}`.")

        complete, stepped = False, False

        if self.completed:
            complete = True
            stepped = False
            return complete, stepped

        if self.inprogress_constraint is not None:
            # In the middle of fulfilling a constraint. If the `token_id` *does* makes an incremental progress to current
            # job, simply update the state

            stepped, complete, reset = self.inprogress_constraint.update(token_id)
            if reset:
                # 1. If the next token breaks the progress, then we must restart.
                #     e.g. constraint = "I love pies" and sequence so far is "I love" but `token_id` == "books".

                #     But that doesn't mean we self.init_state(), since we only reset the state for this particular
                #     constraint, not the full list of constraints.

                self.pending_constraints.append(self.inprogress_constraint.copy(stateful=False))
                self.inprogress_constraint = None

            if complete:
                # 2. If the next token completes the constraint, move it to completed list, set
                #     inprogress to None. If there are no pending constraints either, then this full list of constraints
                #     is complete.

                self.complete_constraints.append(self.inprogress_constraint)
                self.inprogress_constraint = None

                if len(self.pending_constraints) == 0:
                    # we're done!
                    self.completed = True

        else:
            # Not in the middle of fulfilling a constraint. So does this `token_id` helps us step towards any of our list
            # of constraints?

            for cidx, pending_constraint in enumerate(self.pending_constraints):
                if pending_constraint.does_advance(token_id):
                    stepped, complete, reset = pending_constraint.update(token_id)

                    if not stepped:
                        raise Exception(
                            "`constraint.update(token_id)` is not yielding incremental progress, "
                            "even though `constraint.does_advance(token_id)` is true."
                        )

                    if complete:
                        self.complete_constraints.append(pending_constraint)
                        self.inprogress_constraint = None

                    if not complete and stepped:
                        self.inprogress_constraint = pending_constraint

                    if complete or stepped:
                        # If we made any progress at all, then it's at least not a "pending constraint".

                        self.pending_constraints = (
                            self.pending_constraints[:cidx] + self.pending_constraints[cidx + 1 :]
                        )

                        if len(self.pending_constraints) == 0 and self.inprogress_constraint is None:
                            # If there's no longer any pending after this and no inprogress either, then we must be
                            # complete.

                            self.completed = True

                        break  # prevent accidentally stepping through multiple constraints with just one token.

        return complete, stepped

    def copy(self, stateful=True):
        new_state = ConstraintListState(self.constraints)  # we actually never though self.constraints objects
        # throughout this process. So it's at initialization state.

        if stateful:
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]

        return new_state


class TemplateConstraintSlow(Constraint):
    r"""
    Enforces generation according to a template with fixed and variable parts.

    Args:
        template_parts (List[List[int]]): List where each element is either:
            - Non-empty list: Fixed token sequence (must be generated exactly)
            - Empty list: Variable part (exactly one token, not starting next fixed part)
        all_token_ids (List[int]): List of all possible token IDs in the vocab.
    """

    def __init__(self, template_parts: List[List[int]], all_token_ids: List[int]):
        # template_parts = [[262], [], [], [2059], [], [287]]
        super(Constraint, self).__init__()
        logger.info(f"[__init__] Initializing with template_parts={template_parts}")
        logger.info(f"[__init__] Number of all_token_ids: {len(all_token_ids)}") #50257

        self.template = template_parts  # 6
        self.all_token_ids = all_token_ids
        self.current_part = 0
        self.active_constraint = None
        self.next_forbidden = None
        self.completed = False
        self.seqlen = sum(len(part) if part else 1 for part in template_parts)
        logger.info(f"[__init__] Expected sequence length: {self.seqlen}") #6
        self._init_next_constraint()

    def _init_next_constraint(self):
        logger.info(f"[__init_next_constraint] Current part: {self.current_part}")
        if self.current_part >= len(self.template):
            logger.info("[__init_next_constraint] All parts completed, marking as done")
            self.completed = True
            return

        current_part = self.template[self.current_part]
        logger.info(f"[__init_next_constraint] Processing part: {current_part}")
        if current_part:
            logger.info("[__init_next_constraint] Initializing PhrasalConstraint for fixed part")
            self.active_constraint = PhrasalConstraint(current_part)
            self.next_forbidden = None
        else:
            next_fixed = self._get_next_fixed()
            logger.info(f"[__init_next_constraint] Variable part, next fixed tokens: {next_fixed}")
            self.next_forbidden = next_fixed[0] if next_fixed else None
            self.active_constraint = None

    def _get_next_fixed(self):
        logger.info(f"[__get_next_fixed] Searching for next fixed part after {self.current_part}")
        for i in range(self.current_part + 1, len(self.template)):
            part = self.template[i]
            if part:
                logger.info(f"[__get_next_fixed] Found next fixed part: {part}")
                return part
        logger.info("[__get_next_fixed] No more fixed parts found")
        return None

    def advance(self):
        logger.info(f"[__advance] Current state - part: {self.current_part}, completed: {self.completed}")
        if self.completed:
            logger.info("[__advance] Already completed, returning None")
            return None

        if self.active_constraint:
            logger.info("[__advance] Delegating to active constraint")
            return self.active_constraint.advance()
        else:
            if self.next_forbidden is not None:
                logger.info(f"[__advance] Variable part with forbidden token: {self.next_forbidden}")
                get_top_k(self.all_token_ids, k = 5)
                allowed = [tid for tid in self.all_token_ids if tid != self.next_forbidden]
                logger.info(f"[__advance] Number of allowed tokens: {len(allowed)}")
                return allowed
            else:
                logger.info("[__advance] Variable part with no restrictions")
                return self.all_token_ids.copy()

    def does_advance(self, token_id: int):
        logger.info(f"[__does_advance] Checking token_id: {token_id}")
        if self.completed:
            logger.info("[__does_advance] Already completed, returning False")
            return False

        if self.active_constraint:
            result = self.active_constraint.does_advance(token_id)
            logger.info(f"[__does_advance] Active constraint result: {result}")
            return result
        else:
            result = token_id != self.next_forbidden if self.next_forbidden is not None else True
            logger.info(f"[__does_advance] Variable part result: {result}")
            return result

    def update(self, token_id: int):
        logger.info(f"[__update] Processing token_id: {token_id}")
        stepped = False
        completed = False
        reset = False

        if self.completed:
            logger.info("[__update] Already completed")
            return stepped, completed, reset

        if self.active_constraint:
            logger.info("[__update] Updating active constraint")
            stepped, part_completed, reset = self.active_constraint.update(token_id)
            logger.info(f"[__update] Active constraint result - stepped: {stepped}, completed: {part_completed}, reset: {reset}")

            if part_completed:
                logger.info("[__update] Fixed part completed, moving to next part")
                self.current_part += 1
                self._init_next_constraint()
                if self.current_part >= len(self.template):
                    logger.info("[__update] All parts completed")
                    self.completed = True
            elif reset:
                logger.info("[__update] Resetting active constraint")
                self.active_constraint.reset()
                reset = True
        else:
            logger.info("[__update] Processing variable part")
            if self.does_advance(token_id):
                stepped = True
                self.current_part += 1
                logger.info(f"[__update] Variable part accepted, moving to part {self.current_part}")
                self._init_next_constraint()
                if self.current_part >= len(self.template):
                    logger.info("[__update] All parts completed")
                    self.completed = True
            else:
                logger.info("[__update] Invalid token for variable part, resetting")
                reset = True
                self.reset()

        if self.current_part >= len(self.template):
            logger.info("[__update] Template generation completed")
            self.completed = True

        logger.info(f"[__update] Final result - stepped: {stepped}, completed: {self.completed}, reset: {reset}")
        return stepped, self.completed, reset

    def reset(self):
        logger.info("[__reset] Resetting constraint state")
        self.current_part = 0
        self.completed = False
        self._init_next_constraint()
        if self.active_constraint:
            logger.info("[__reset] Resetting active constraint")
            self.active_constraint.reset()

    def remaining(self):
        logger.info(f"[__remaining] Checking remaining tokens from part {self.current_part}")
        if self.completed:
            logger.info("[__remaining] Already completed, returning 0")
            return 0
        remaining = 0
        for part in self.template[self.current_part:]:
            remaining += len(part) if part else 1
        logger.info(f"[__remaining] Tokens remaining: {remaining}")
        return remaining

    def copy(self, stateful=False):
        logger.info(f"[__copy] Creating copy with stateful={stateful}")
        new_constraint = TemplateConstraint(self.template, self.all_token_ids)
        if stateful:
            logger.info("[__copy] Copying current state")
            new_constraint.current_part = self.current_part
            new_constraint.completed = self.completed
            new_constraint.next_forbidden = self.next_forbidden
            new_constraint.seqlen = self.seqlen
            new_constraint.active_constraint = (
                self.active_constraint.copy(stateful=True) if self.active_constraint else None
            )
        return new_constraint


class TrieNode:
    def __init__(self):
        self.children = {}  # Map of token_id to TrieNode
        self.is_variable = False  # Marks if this node represents a variable token
        self.is_end = False  # Marks if this node completes a valid template
        self.forbidden_next = None  # Token that cannot follow this node (for variable parts)

class TemplateConstraint(Constraint):
    def __init__(self, template_parts: List[List[int]], all_token_ids: List[int]):
        super(Constraint, self).__init__()
        self.template = template_parts
        self.all_token_ids = all_token_ids
        self.root = self._build_trie(template_parts)
        self.current_node = self.root
        self.completed = False
        self.seqlen = sum(len(part) if part else 1 for part in template_parts)

    def _build_trie(self, template_parts: List[List[int]]) -> TrieNode:
        root = TrieNode()
        current = root
        
        for i, part in enumerate(template_parts):
            if not part:  # Variable part
                new_node = TrieNode()
                new_node.is_variable = True
                # Find next fixed part's first token to forbid
                for j in range(i + 1, len(template_parts)):
                    if template_parts[j]:
                        new_node.forbidden_next = template_parts[j][0]
                        break
                # Create a separate node for each allowed token
                for tid in self.all_token_ids:
                    if new_node.forbidden_next is None or tid != new_node.forbidden_next:
                        token_node = TrieNode()
                        token_node.is_end = (i == len(template_parts) - 1)
                        current.children[tid] = token_node
                # Move to next part without changing current
                continue
            else:  # Fixed part
                for token_id in part:
                    new_node = TrieNode()
                    current.children[token_id] = new_node
                    current = new_node
        
        current.is_end = True
        print(f"[TemplateConstraint] Trie built with {len(self.all_token_ids)} tokens")
        return root
    
    def _get_next_fixed_start(self, template_parts: List[List[int]], current_part: List[int]) -> Set[int]:
        print(f"[TemplateConstraint] Getting next fixed start for {current_part}")
        start_idx = template_parts.index(current_part)
        for part in template_parts[start_idx + 1:]:
            if part:  # Found next fixed part
                return {part[0]}  # Return first token of next fixed part
        return set()  # No more fixed parts
    
    def advance(self):
        if self.completed:
            return None
            
        valid_tokens = list(self.current_node.children.keys())
        if not valid_tokens:
            return self.all_token_ids.copy()
        print(f"[TemplateConstraint] Advancing with {len(valid_tokens)} options")  
        return valid_tokens

    def does_advance(self, token_id: int):
        if self.completed:
            return False
        return token_id in self.current_node.children

    def update(self, token_id: int):
        if self.completed:
            return False, True, False
            
        if token_id not in self.current_node.children:
            print(f"[TemplateConstraint] Invalid token {token_id}, resetting")
            self.reset()
            return False, False, True
            
        self.current_node = self.current_node.children[token_id]
        stepped = True
        completed = self.current_node.is_end
        self.completed = completed
        
        return stepped, completed, False

    def reset(self):
        print("[TemplateConstraint] Resetting to root node")
        self.current_node = self.root
        self.completed = False

    def remaining(self):
        if self.completed:
            return 0
        return self.seqlen

    def copy(self, stateful=False):
        new_constraint = TemplateConstraint(self.template, self.all_token_ids)
        if stateful:
            new_constraint.current_node = self.current_node
            new_constraint.completed = self.completed
            new_constraint.seqlen = self.seqlen
        return new_constraint