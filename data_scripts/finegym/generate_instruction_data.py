#!/usr/bin/env python3
"""
Generate instruction-following format for fine-tuning Qwen3VL model.
Converts gym99_train_label.txt and gym99_val_label.txt to instruction format.
"""

import json
import re
from typing import Dict, List, Tuple


def load_phrase_categories(filepath: str) -> Dict[int, str]:
    """Load phrase categories from set_categories.txt."""
    phrase_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "set: 0; vault"
            match = re.match(r'set:\s*(\d+);\s*(.+)', line)
            if match:
                phrase_id = int(match.group(1))
                phrase_name = match.group(2).strip()
                phrase_map[phrase_id] = phrase_name
    return phrase_map


def load_action_categories(filepath: str) -> Dict[int, Tuple]:
    """Load action categories from gym99_categories.txt with phrase mapping."""
    action_map = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "Clabel: 0; set: 1; ... (VT) round-off, flic-flac, ..."
            match = re.match(r'Clabel:\s*(\d+);\s*set:\s*(\d+);\s*.*?\(([A-Z]+)\)\s*(.+)', line)
            if match:
                action_id = int(match.group(1))
                set_id = int(match.group(2))
                # Get the action description (everything after the event type)
                action_name = match.group(4).strip()
                action_map[action_id] = (set_id, action_name)
    return action_map


def build_activity_descriptions() -> Dict[int, str]:
    """Build activity descriptions mapping."""
    return {
        0: "Vault (VT)",
        1: "Floor Exercise (FX)",
        2: "Balance Beam (BB)",
        3: "Uneven Bars (UB)"
    }


def map_set_to_phrase(action_to_phrase: Dict[int, int]) -> Dict[int, int]:
    """
    Map set IDs to phrase IDs based on action-to-phrase mappings.
    From the data, we can deduce:
    - set 1 (VT actions 0-5) -> phrase 0
    - set 21 (FX leap/jump actions 6-16) -> phrase 1
    - set 22 (FX turns actions 17-23) -> phrase 2
    - set 24 (FX front salto actions 24-30) -> phrase 3
    - set 25 (FX back salto actions 31-49) -> phrase 4
    - set 31 (BB leap/jump actions 41-52) -> phrase 5
    - set 32 (BB turns actions 53-56) -> phrase 6
    - set 33 (BB flight salto actions 57-64) -> phrase 7
    - set 34 (BB flight handspring actions 65-68) -> phrase 8
    - set 35 (BB dismounts actions 69-73) -> phrase 9
    - set 41 (UB circles actions 74-88) -> phrase 10
    - set 42 (UB flight same bar actions 89-93) -> phrase 11
    - set 43 (UB transition flight actions 94-95) -> phrase 12
    - set 44 (UB dismounts actions 96-98) -> phrase 13
    """
    return {
        1: 0, 21: 1, 22: 2, 24: 3, 25: 4,
        31: 5, 32: 6, 33: 7, 34: 8, 35: 9,
        41: 10, 42: 11, 43: 12, 44: 13
    }


def get_activity_from_phrase(phrase_id: int) -> int:
    """Get activity ID from phrase ID."""
    if phrase_id == 0:
        return 0  # Vault
    elif 1 <= phrase_id <= 4:
        return 1  # Floor Exercise
    elif 5 <= phrase_id <= 9:
        return 2  # Balance Beam
    elif 10 <= phrase_id <= 13:
        return 3  # Uneven Bars
    return -1


def build_hierarchical_prompt(
        activity_map: Dict[int, str],
        phrase_map: Dict[int, str],
        action_map: Dict[int, Tuple],
        set_to_phrase: Dict[int, int]
) -> str:
    """Build hierarchical human prompt organized by activity and phrase."""
    
    # Group actions by activity and phrase
    hierarchy = {}  # activity_id -> phrase_id -> [(action_id, action_name), ...]

    for action_id, (set_id, action_name) in action_map.items():
        phrase_id = set_to_phrase.get(set_id)
        if phrase_id is None:
            continue

        activity_id = get_activity_from_phrase(phrase_id)
        if activity_id == -1:
            continue

        if activity_id not in hierarchy:
            hierarchy[activity_id] = {}
        if phrase_id not in hierarchy[activity_id]:
            hierarchy[activity_id][phrase_id] = []

        hierarchy[activity_id][phrase_id].append((action_id, action_name))

    # Build prompt
    prompt_lines = [
        "<video>",
        "Task: Analyze the gymnastics routine. Locate the hierarchical activities and actions.",
        "",
        "### Definitions & Hierarchy",
    ]

    # Generate for each activity
    for activity_id in sorted(hierarchy.keys()):
        activity_name = activity_map[activity_id]
        prompt_lines.append(f"**Activity: <a{activity_id}> {activity_name}**")

        # Generate for each phrase within this activity
        for phrase_id in sorted(hierarchy[activity_id].keys()):
            phrase_name = phrase_map.get(phrase_id, "Unknown")
            prompt_lines.append(f"  [Phrase: <p{phrase_id:02d}> ({phrase_name})]")
            prompt_lines.append("  Actions:")

            # List all actions in this phrase
            for action_id, action_name in hierarchy[activity_id][phrase_id]:
                prompt_lines.append(f"    - <c{action_id:02d}>: {action_name}")

            prompt_lines.append("")  # Empty line after each phrase

    # Add instructions
    prompt_lines.extend([
        "### Instructions",
        "1. Output valid JSON.",
        "2. Identify the Activity ID (e.g., <a0>) and its total time span.",
        "3. List all atomic Actions chronologically.",
        "4. For each action, predict [Phrase_ID, Action_ID] (e.g., [<p00>, <c02>]) and the time span.",
        "5. Time Format: Use \"<s.s seconds>\" inside a list [start, end]."
    ])

    return "\n".join(prompt_lines)


def convert_annotation_to_instruction(annotation: Dict, human_prompt: str) -> Dict:
    """Convert a single annotation to instruction format."""
    video_name = annotation['video']
    # Add .mp4 extension
    video_file = f"{video_name}.mp4"

    # Get new_value (relative timestamps)
    new_values = annotation['new_value']

    # Build GPT response — handle multiple activities
    gpt_responses = []
    for activity in new_values:
        activity_id = activity['activity_id']
        span = activity['span']

        # Build actions list
        actions_list = []

        for action in activity['actions']:
            phrase_id = action['phrase_id']
            action_id = action['action_id']
            action_span = action['span']

            # Use token format: <p00>, <c02>
            actions_list.append({
                'ids': [f'{phrase_id}', f'{action_id}'],
                'span': action_span
            })

        # Use token format: <a0>
        activity_response = {
            'activity_id': f'{activity_id}',
            'span': span,
            'actions': actions_list
        }

        gpt_responses.append(activity_response)

    # If multiple activities, wrap in array; if single, use single object
    if len(gpt_responses) == 1:
        gpt_value = json.dumps(gpt_responses[0], ensure_ascii=False)
    else:
        gpt_value = json.dumps(gpt_responses, ensure_ascii=False)

    # Build instruction format
    instruction = {
        'video': video_file,
        'conversations': [
            {
                'from': 'human',
                'value': human_prompt
            },
            {
                'from': 'gpt',
                'value': gpt_value
            }
        ]
    }

    return instruction


def process_label_file(
        input_file: str,
        output_file: str,
        human_prompt: str
):
    """Process a label file and generate instruction file."""
    print(f"Processing {input_file}...")

    # Load annotations
    annotations = []

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    print(f"Loaded {len(annotations)} annotations")

    # Convert to instruction format
    instructions = []
    for annotation in annotations:
        instruction = convert_annotation_to_instruction(annotation, human_prompt)
        instructions.append(instruction)

    # Write to output file
    with open(output_file, 'w') as f:
        for instruction in instructions:
            f.write(json.dumps(instruction, ensure_ascii=False) + '\n')

    print(f"Generated {len(instructions)} instructions in {output_file}")


def main():
    print("Loading category mappings...")

    # Load categories
    phrase_map = load_phrase_categories('set_categories.txt')
    action_map = load_action_categories('gym99_categories.txt')

    print(f"Loaded {len(phrase_map)} phrase categories")
    print(f"Loaded {len(action_map)} action categories")

    # Build mappings
    activity_map = build_activity_descriptions()
    set_to_phrase = map_set_to_phrase({})  # Just need the mapping

    # Build hierarchical human prompt
    human_prompt = build_hierarchical_prompt(activity_map, phrase_map, action_map, set_to_phrase)

    print("\n=== Human Prompt Preview ===")
    print(human_prompt[:800] + "...")

    # Process train set
    print("\n=== Processing Train Set ===")
    process_label_file(
        'Dec16/gym99_train_label.txt',
        'Dec16/gym99_train_instruct.txt',
        human_prompt
    )

    # Process val set
    print("\n=== Processing Val Set ===")
    process_label_file(
        'Dec16/gym99_val_label.txt',
        'Dec16/gym99_val_instruct.txt',
        human_prompt
    )

    print("\n=== Done! ===")
    print("Generated files:")
    print(" - Dec16/gym99_train_instruct.txt")
    print(" - Dec16/gym99_val_instruct.txt")


if __name__ == '__main__':
    main()
