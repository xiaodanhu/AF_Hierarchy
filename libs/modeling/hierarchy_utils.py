"""Hierarchy utilities: verbalizer loading and projection matrix construction."""

import json
import torch


def load_verbalizer(verbalizer_path):
    """Load verbalizer.json and return the parsed dict."""
    with open(verbalizer_path, 'r') as f:
        return json.load(f)


def build_projection_matrices(verbalizer, num_actions=99, num_phrases=14, num_activities=4):
    """
    Build fixed projection matrices from verbalizer mappings.

    Returns:
        V_ap: (num_actions, num_phrases) binary matrix, action-to-phrase
        V_pa: (num_phrases, num_activities) binary matrix, phrase-to-activity
        V_aa: (num_actions, num_activities) derived matrix, action-to-activity
    """
    V_ap = torch.zeros(num_actions, num_phrases)
    V_pa = torch.zeros(num_phrases, num_activities)

    phrase_to_idx = {pid: int(pid[1:]) for pid in verbalizer['phrases']}
    activity_to_idx = {aid: int(aid[1:]) for aid in verbalizer['activities']}

    for action_id, action_info in verbalizer['actions'].items():
        action_idx = int(action_id[1:])
        phrase_idx = phrase_to_idx[action_info['phrase']]
        V_ap[action_idx, phrase_idx] = 1.0

    for phrase_id, phrase_info in verbalizer['phrases'].items():
        phrase_idx = phrase_to_idx[phrase_id]
        activity_idx = activity_to_idx[phrase_info['activity']]
        V_pa[phrase_idx, activity_idx] = 1.0

    V_aa = V_ap @ V_pa
    return V_ap, V_pa, V_aa


def get_action_to_phrase_map(verbalizer):
    """Return dict mapping action index -> phrase index."""
    return {int(aid[1:]): int(info['phrase'][1:])
            for aid, info in verbalizer['actions'].items()}


def get_action_to_activity_map(verbalizer):
    """Return dict mapping action index -> activity index."""
    activity_name_to_idx = {v['short']: int(k[1:])
                            for k, v in verbalizer['activities'].items()}
    return {int(aid[1:]): activity_name_to_idx[info['activity']]
            for aid, info in verbalizer['actions'].items()}
