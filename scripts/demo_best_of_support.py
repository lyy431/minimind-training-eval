"""Web-only recovery and history handling around the frozen selection core."""
import copy
import gc

import torch

from best_of_n import RewardSelector


class SafeRewardSelector(RewardSelector):
    def score(self, messages, candidates, device='cuda'):
        # The frozen core's cleanup starts after its GPU transfer. Cover a
        # transfer that fails halfway, as well as interruption of scoring.
        try:
            return super().score(messages, candidates, device)
        finally:
            self.model.cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def trim_context(messages, history_rounds, system_messages=None):
    """Keep whole user/assistant/tool rounds, including the current user turn."""
    if not isinstance(history_rounds, int) or history_rounds < 0:
        raise ValueError('History rounds must be a nonnegative integer')
    rounds = []
    for message in messages:
        if message['role'] == 'system':
            continue
        if message['role'] == 'user':
            rounds.append([])
        # Drop orphan assistant/tool entries from any older, already truncated
        # history. A retained tool response must stay with its original user.
        if rounds:
            rounds[-1].append(message)
    kept = [message for turn in rounds[-(history_rounds + 1):] for message in turn]
    return copy.deepcopy(list(system_messages or []) + kept)
