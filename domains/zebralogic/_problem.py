"""
Zebra puzzle generator utilities used by the current ZebraLogic training and evaluation pipelines.

This module intentionally keeps only the shared puzzle-generation and
curriculum symbols used by the current dataset pipeline.
"""

import random
import collections
import time
from typing import List
from dataclasses import dataclass


# ═════════════════════════════════════════════════════════════════════════════
# PART 1: Enigmata Puzzle Generator (original logic preserved, interface cleaned)
# ═════════════════════════════════════════════════════════════════════════════

# # --- Category Domains ---
# CATEGORY_DOMAINS = {
#     "Nationality": {
#         "american", "argentine", "australian", "brazilian", "british",
#         "canadian", "chinese", "colombian", "dutch", "egyptian",
#         "french", "german", "indian", "indonesian", "italian",
#         "japanese", "malaysian", "mexican", "nigerian", "pakistani",
#         "polish", "russian", "spanish", "thai", "turkish",
#     },
#     "Food": {
#         "apple", "apricot", "artichoke", "asparagus", "avocado",
#         "banana", "blueberry", "broccoli", "cabbage", "carrot",
#         "cauliflower", "cherry", "corn", "cranberry", "cucumber",
#         "eggplant", "garlic", "grapefruit", "grapes", "kale",
#         "kiwi", "lemon", "lettuce", "lime", "mango",
#         "nectarine", "onion", "orange", "papaya", "peach",
#         "pear", "peas", "pepper", "pineapple", "plum",
#         "pomegranate", "potato", "pumpkin", "radish", "raspberry",
#         "spinach", "strawberry", "tomato", "watermelon", "zucchini",
#     },
#     "Pet": {
#         "bird", "cat", "chinchilla", "dog", "ferret",
#         "fish", "frog", "goat", "goldfish", "guinea-pig",
#         "hamster", "hedgehog", "horse", "lizard", "mouse",
#         "pony", "rabbit", "rat", "snake", "turtle",
#     },
#     "Job": {
#         "accountant", "analyst", "architect", "bartender", "chef",
#         "coach", "dancer", "designer", "doctor", "dressmaker",
#         "electrician", "engineer", "entrepreneur", "firefighter", "fisherman",
#         "freelancer", "journalist", "lawyer", "librarian", "manager",
#         "mechanic", "musician", "nurse", "paramedic", "photographer",
#         "pilot", "police-officer", "project-manager", "scientist", "security-guard",
#         "social-worker", "software-developer", "teacher", "videographer", "writer",
#     },
#     "Beverage": {
#         "7up", "almond-milk", "coffee", "cola", "fanta",
#         "hot-chocolate", "iced-tea", "juice", "lemonade", "milk",
#         "mirinda", "soy-milk", "sprite", "tea", "water",
#     },
#     "Transport": {
#         "airplane", "bike", "boat", "bus", "car",
#         "helicopter", "jet-ski", "motorbike", "quad-bike", "roller",
#         "scooter", "ship", "skateboard", "snowmobile",
#         "subway", "taxi", "train", "tram", "trike", "van",
#     },
#     "Music-Genre": {
#         "ambient", "blues", "classical", "country", "d&b",
#         "disco", "dubstep", "electronic", "folk", "funk",
#         "gospel", "hip-hop", "house", "indie", "jazz",
#         "metal", "pop", "punk", "r&b", "reggae",
#         "rock", "salsa", "soul", "techno", "trance",
#     },
#     "Movie-Genre": {
#         "action", "adventure", "animation", "comedy", "crime",
#         "disaster", "documentary", "drama", "epic", "family",
#         "fantasy", "horror", "martial-arts", "musical", "mystery",
#         "romance", "satire", "scientific", "sports", "spy",
#         "superhero", "thriller", "time-travel", "western", "zombie",
#     },
#     "Sport": {
#         "badminton", "baseball", "basketball", "biathlon", "climbing",
#         "cricket", "cycling", "golf", "handball", "ice-hockey",
#         "lacrosse", "parkour", "rowing", "rugby", "sailing",
#         "skateboarding", "skiing", "snowboarding", "soccer", "surfing",
#         "swimming", "tennis", "volleyball", "water-polo", "weightlifting",
#     },
#     "Hobby": {
#         "baking", "board-games", "camping", "card-games", "chess",
#         "collecting", "cooking", "dancing", "drawing", "filmmaking",
#         "fishing", "gardening", "hiking", "magic-tricks", "photography",
#         "puzzles", "reading", "rock-climbing", "singing", "skydiving",
#         "sudoku", "traveling", "video-games", "woodworking", "writing",
#     },
#     "Color": {
#         "red", "blue", "green", "yellow", "orange",
#         "purple", "pink", "white", "black", "brown",
#         "gray", "turquoise", "magenta", "teal", "maroon",
#     },
# }

# ─────────────────────────────────────────────────────────────
# Single-token entities only (base LLaDA vocab, no special tokens)
# All values encode as exactly 1 token with " " prefix (Ġ-form).
# Collisions resolved: each value appears in exactly one category.
# ─────────────────────────────────────────────────────────────
CATEGORY_DOMAINS = {
    "Nationality": {
        "african", "american", "arab", "asian", "british",
        "canadian", "chinese", "european", "french", "german",
        "greek", "indian", "italian", "japanese", "latin",
        "polish", "roman", "russian", "spanish", "swiss",
    },  # 20 values
    "Food": {
        "almond", "apple", "asparagus", "avocado", "bacon",
        "banana", "basil", "bean", "beef", "beet",
        "berry", "blueberry", "bread", "broccoli", "cabbage",
        "cake", "candy", "carrot", "cauliflower", "celery",
        "cherry", "chili", "clam", "coconut", "corn",
        "crab", "cranberry", "cream", "cucumber", "curry",
        "date", "dill", "duck", "egg", "eggplant",
        "fig", "flour", "garlic", "grape", "grapefruit",
        "grapes", "ham", "honey", "jam", "kale",
        "lamb", "lemon", "lettuce", "lime", "mango",
        "meat", "mint", "nut", "oat", "olive",
        "onion", "orange", "pasta", "peach", "pear",
        "peas", "pepper", "pie", "pineapple", "pizza",
        "plum", "pork", "potato", "prune", "pumpkin",
        "raspberry", "rice", "rye", "sage", "salad",
        "salt", "soup", "spinach", "steak", "strawberry",
        "sugar", "sushi", "taco", "tofu", "tomato",
        "tuna", "walnut", "watermelon", "wheat", "zucchini",
    },  # 90 values
    "Pet": {
        "ant", "bat", "bear", "bee", "bird",
        "bull", "cat", "crow", "deer", "dog",
        "dove", "fish", "fox", "frog", "goat",
        "hawk", "hen", "horse", "mouse", "owl",
        "parrot", "pig", "pony", "rabbit", "ram",
        "rat", "seal", "snake", "turtle", "wolf",
    },  # 30 values
    "Job": {
        "accountant", "actor", "agent", "analyst", "architect",
        "artist", "baker", "barber", "bartender", "chef",
        "clerk", "coach", "dancer", "dean", "dentist",
        "designer", "doctor", "driver", "electrician", "engineer",
        "entrepreneur", "farmer", "firefighter", "fisherman", "freelancer",
        "guard", "journalist", "judge", "king", "lawyer",
        "librarian", "manager", "mason", "mayor", "mechanic",
        "miner", "model", "monk", "musician", "nurse",
        "photographer", "pilot", "poet", "priest", "queen",
        "sailor", "scientist", "tailor", "teacher", "tutor",
        "vet", "waiter", "writer",
    },  # 53 values
    "Beverage": {
        "beer", "broth", "cider", "cocoa", "coffee",
        "gin", "juice", "latte", "lemonade", "milk",
        "nectar", "punch", "rum", "sake", "soda",
        "sprite", "tea", "tonic", "water", "wine",
    },  # 20 values
    "Transport": {
        "airplane", "bike", "boat", "bus", "cab",
        "canoe", "car", "cart", "ferry", "helicopter",
        "jet", "kayak", "raft", "roller", "scooter",
        "ship", "skateboard", "sled", "subway", "tank",
        "taxi", "train", "tram", "truck", "van",
        "wagon", "yacht",
    },  # 27 values
    "Music-Genre": {
        "ambient", "blues", "classical", "country", "disco",
        "electronic", "folk", "funk", "gospel", "house",
        "hymn", "indie", "jazz", "metal", "opera",
        "pop", "punk", "rock", "salsa", "soul",
        "swing", "techno",
    },  # 22 values (latin → Nationality)
    "Movie-Genre": {
        "action", "adventure", "animation", "bio", "comedy",
        "crime", "cult", "disaster", "documentary", "drama",
        "epic", "family", "fantasy", "horror", "musical",
        "mystery", "parody", "romance", "satire", "scientific",
        "sports", "spy", "superhero", "thriller", "war",
        "western", "zombie",
    },  # 27 values
    "Sport": {
        "baseball", "basketball", "boxing", "climbing", "cricket",
        "curling", "cycling", "diving", "fencing", "golf",
        "lacrosse", "marathon", "polo", "rowing", "rugby",
        "sailing", "skiing", "soccer", "sprint", "squash",
        "surfing", "swimming", "tennis", "volleyball", "wrestling",
    },  # 25 values
    "Hobby": {
        "baking", "camping", "carving", "chess", "collecting",
        "cooking", "dancing", "drawing", "filmmaking", "fishing",
        "gaming", "gardening", "hiking", "hunting", "knitting",
        "painting", "photography", "pottery", "puzzles", "quilting",
        "reading", "running", "sewing", "singing", "traveling",
        "weaving", "woodworking", "writing", "yoga",
    },  # 29 values
    "Color": {
        "amber", "beige", "black", "blue", "bronze",
        "brown", "charcoal", "copper", "coral", "crimson",
        "gold", "gray", "green", "ivory", "maroon",
        "navy", "pink", "purple", "red", "ruby",
        "salmon", "silver", "tan", "teal", "turquoise",
        "violet", "white", "yellow",
    },  # 28 values
}

def _update_range(wns, rns, cmp):
    changed = False
    for rn in rns:
        classified_words = set()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) == 1:
                classified_words.add(next(iter(set_of_words)))
        word_to_cols = dict()
        for n_col, set_of_words in enumerate(rn):
            if len(set_of_words) != 1:
                prev_length = len(set_of_words)
                set_of_words.difference_update(classified_words)
                changed |= prev_length != len(set_of_words)
                for word in set_of_words:
                    word_to_cols.setdefault(word, set()).add(n_col)
        for word, cols in word_to_cols.items():
            if len(cols) == 1:
                x = rn[next(iter(cols))]
                if len(x) != 1:
                    x.clear()
                    x.add(word)
                    changed = True
    new_rns = [[{x for x in xs if x != wn} for xs in rn] for wn, rn in zip(wns, rns)]
    pairs = []
    for wn, rn in zip(wns, rns):
        new_pairs = []
        break_condition = True
        for cn, setn in enumerate(rn):
            if wn in setn:
                break_condition = False
                if not pairs:
                    pairs = [[]]
                for v in pairs:
                    new_pairs.append([*v, cn])
        pairs = new_pairs
        if break_condition:
            break
    for pair in pairs:
        if cmp(*pair):
            for nrn, cn, wn in zip(new_rns, pair, wns):
                nrn[cn].add(wn)
    changed |= any(rn != new_rn for rn, new_rn in zip(rns, new_rns))
    if changed:
        for rn, new_rn in zip(rns, new_rns):
            for old, new in zip(rn, new_rn):
                old.intersection_update(new)
    return changed


def _update_ranges(relations, ranges):
    changed = False
    for ins, wns, callable_object, *_ in relations:
        changed |= _update_range(wns, [ranges[i] for i in ins], callable_object)
    return changed


def generate_puzzle(
    table: List[List[str]], *,
    level: int = 2,
    minimal_conditions: bool = False,
    max_seconds_for_minimizing: float = None,
    tries: int = 10,
):
    """
    Enigmata puzzle generator. Original logic preserved.
    table: [[category, val1, val2, ...], ...]
    Returns: (premises_list, relations_list)
    """
    if level not in range(1, 21):
        raise ValueError('level must be >= 1 and <= 20')

    table_wo_left = [row[1:] for row in table]
    n_attributes = len(table_wo_left)
    m_objects = len(table_wo_left[0])
    center = m_objects // 2
    except_flag = True

    rules_for_relations = [
        (2, lambda j1, j2: j1 == j2, ['{0}: {1} == {2}: {3}', '{2}: {3} == {0}: {1}']),
        (2, lambda j1, j2: j1 == j2 - 1, ['{0}: {1} is on the left of {2}: {3}']),
        (2, lambda j1, j2: j1 == j2 + 1, ['{0}: {1} is on the right of {2}: {3}']),
        (1, lambda j1: j1 == 0, ['{0}: {1} is on the far left']),
        (1, lambda j1, last_index=m_objects - 1: j1 == last_index, ['{0}: {1} is on the far right']),
    ] + (m_objects % 2 != 0) * [
        (1, lambda j1, mid=center: j1 == mid, ['{0}: {1} is in the middle'])
    ]

    if level >= 2:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 + 1 == j1 == j3 - 1 or j3 + 1 == j1 == j2 - 1,
             ['{0}: {1} is between {2}: {3} and {4}: {5}',
              '{0}: {1} is between {4}: {5} and {2}: {3}']),
        ]
    if level >= 3:
        rules_for_relations += [
            (2, lambda j1, j2: j1 == j2 - 1 or j1 == j2 + 1,
             ['{0}: {1} is on the left or right of {2}: {3}']),
            (1, lambda j1, last_index=m_objects - 1: j1 == 0 or j1 == last_index,
             ['{0}: {1} is on the far left or far right']),
        ]
    if level >= 4:
        rules_for_relations += [
            (1, lambda j1: (j1 + 1) % 2 != 0, ['{0}: {1} is in an odd position']),
            (1, lambda j1: (j1 + 1) % 2 == 0, ['{0}: {1} is in an even position']),
        ]
    if level >= 5:
        rules_for_relations += [
            (2, lambda j1, j2: j1 < j2, ['{0}: {1} is somewhere to the left of {2}: {3}']),
            (2, lambda j1, j2: j1 > j2, ['{0}: {1} is somewhere to the right of {2}: {3}']),
        ]
    if level >= 6:
        rules_for_relations += [
            (2, lambda j1, j2: j1 != j2,
             ['{0}: {1} != {2}: {3}', '{2}: {3} != {0}: {1}'], except_flag),
        ]
    if level >= 7:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j2 < j1 < j3 or j3 < j1 < j2,
             ['{0}: {1} is somewhere between {2}: {3} and {4}: {5}',
              '{0}: {1} is somewhere between {4}: {5} and {2}: {3}']),
        ]
    if level >= 8:
        rules_for_relations += [
            (2, lambda j1, j2: j1 >= j2, ['{0}: {1} is not to the left of {2}: {3}']),
            (2, lambda j1, j2: j1 <= j2, ['{0}: {1} is not to the right of {2}: {3}']),
        ]
    if level >= 9:
        rules_for_relations += [
            (2, lambda j1, j2: j1 % 2 != j2 % 2,
             ['{0}: {1} and {2}: {3} have different parity positions',
              '{2}: {3} and {0}: {1} have different parity positions'], except_flag),
            (2, lambda j1, j2: j1 % 2 == j2 % 2,
             ['{0}: {1} and {2}: {3} have the same parity positions',
              '{2}: {3} and {0}: {1} have the same parity positions'], except_flag),
        ]
    if level >= 10:
        rules_for_relations += [
            (3, lambda j1, j2, j3: (j1 == j2 and j1 != j3) or (j1 != j2 and j1 == j3),
             ['{0}: {1} == {2}: {3} or {0}: {1} == {4}: {5}, but not both',
              '{0}: {1} == {4}: {5} or {0}: {1} == {2}: {3}, but not both'], except_flag),
            (3, lambda j1, j2, j3: (j1 == j2 and j2 != j3) or (j1 != j2 and j2 == j3),
             ['{0}: {1} == {2}: {3} or {2}: {3} == {4}: {5}, but not both',
              '{2}: {3} == {4}: {5} or {0}: {1} == {2}: {3}, but not both'], except_flag),
        ]
    if level >= 11:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 == j2 or j1 == j3,
             ['{0}: {1} == {2}: {3} or {0}: {1} == {4}: {5} or both',
              '{0}: {1} == {4}: {5} or {0}: {1} == {2}: {3} or both'], except_flag),
            (3, lambda j1, j2, j3: j1 == j2 or j2 == j3,
             ['{0}: {1} == {2}: {3} or {2}: {3} == {4}: {5} or both',
              '{2}: {3} == {4}: {5} or {0}: {1} == {2}: {3} or both'], except_flag),
        ]
    if level >= 12:
        rules_for_relations += [
            (3, lambda j1, j2, j3: j1 != j2 or j1 != j3,
             ['{0}: {1} != {2}: {3} or {0}: {1} != {4}: {5} or both',
              '{0}: {1} != {4}: {5} or {0}: {1} != {2}: {3} or both'], except_flag),
            (3, lambda j1, j2, j3: j1 != j2 or j2 != j3,
             ['{0}: {1} != {2}: {3} or {2}: {3} != {4}: {5} or both',
              '{2}: {3} != {4}: {5} or {0}: {1} != {2}: {3} or both'], except_flag),
        ]
    if level >= 13:
        rules_for_relations.pop(0)
    if level >= 14:
        rules_for_relations.pop(0)
        rules_for_relations.pop(0)
    if level >= 15:
        rules_for_relations.pop(0)
        rules_for_relations.pop(0)
        if m_objects % 2 != 0:
            rules_for_relations.pop(0)
    if level >= 16:
        rules_for_relations.pop(0)
    if level >= 17:
        rules_for_relations.pop(0)
        rules_for_relations.pop(0)
    if level >= 18:
        rules_for_relations.pop(0)
        rules_for_relations.pop(0)
    if level >= 19:
        rules_for_relations.pop(0)
        rules_for_relations.pop(0)
    if level >= 20:
        rules_for_relations.pop(0)

    is_minimized = False
    time_elapsed = False
    min_relations = None

    while True:
        ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))]
                  for i in range(len(table_wo_left))]
        relations = list()
        fail = False

        while not fail:
            needs_clarification = list()
            no_solutions = False
            solved = True
            for i, rng in enumerate(ranges):
                for j, rs in enumerate(rng):
                    if len(rs) == 0:
                        no_solutions = True
                        solved = False
                        break
                    elif len(rs) > 1:
                        solved = False
                        needs_clarification.append((i, j))
                if no_solutions:
                    break

            if solved or min_relations is not None and len(relations) >= len(min_relations):
                tries -= 1
                if min_relations is None or len(relations) < len(min_relations):
                    min_relations = relations
                if tries > 0:
                    fail = True
                    continue

            if tries <= 0:
                relations = min_relations
                if not minimal_conditions:
                    break
                number_of_relations_min = len(relations)
                number_of_relations_before = len(relations)
                start_time = time.monotonic()
                main_q = collections.deque([relations])
                while main_q:
                    current_relations = main_q.popleft()
                    for k in range(len(current_relations)):
                        new_ranges = [[set(table_wo_left[i]) for _ in range(len(table_wo_left[i]))]
                                      for i in range(len(table_wo_left))]
                        new_relations = current_relations.copy()
                        new_relations.pop(k)
                        changed = True
                        while changed:
                            changed = _update_ranges(new_relations, new_ranges)
                        q = collections.deque([new_ranges])
                        possible_solutions = []
                        while q:
                            current_ranges = q.popleft()
                            no_solutions = False
                            solved = True
                            for rng in current_ranges:
                                for rs in rng:
                                    if len(rs) == 0:
                                        no_solutions = True
                                        solved = False
                                        break
                                    elif len(rs) > 1:
                                        solved = False
                                if no_solutions or not solved:
                                    break
                            if no_solutions:
                                continue
                            if solved:
                                if current_ranges not in possible_solutions:
                                    possible_solutions.append(current_ranges)
                                    if len(possible_solutions) >= 2:
                                        break
                                continue
                            for n_group, rng in enumerate(current_ranges):
                                founded = False
                                for n_x, rs in enumerate(rng):
                                    if len(rs) > 1:
                                        founded = True
                                        for r in rs:
                                            new_ranges2 = [[x.copy() for x in row]
                                                           for row in current_ranges]
                                            new_ranges2[n_group][n_x] = {r}
                                            changed = True
                                            while changed:
                                                changed = _update_ranges(new_relations, new_ranges2)
                                            q.appendleft(new_ranges2)
                                        break
                                if founded:
                                    break
                        if len(possible_solutions) == 1:
                            number_of_relations_after = len(new_relations)
                            if number_of_relations_min > number_of_relations_after:
                                number_of_relations_min = number_of_relations_after
                                relations = new_relations
                                main_q.append(new_relations)
                        if (max_seconds_for_minimizing is not None and
                                time.monotonic() >= start_time + max_seconds_for_minimizing):
                            time_elapsed = True
                            break
                    if time_elapsed:
                        break
                is_minimized = number_of_relations_min < number_of_relations_before or not time_elapsed
                break

            if no_solutions or not needs_clarification:
                fail = True
                continue

            i, j = item = random.choice(needs_clarification)
            next2_i, next2_j = None, None
            if level >= 2 and len(needs_clarification) > 1:
                needs_clarification.remove(item)
                next2_i, next2_j = random.choice(needs_clarification)

            neighbours = []
            right_neighbours = []
            for dj in range(-1, 1 + 1):
                if not (0 <= j + dj < m_objects):
                    continue
                for new_i in range(0, n_attributes):
                    if new_i == i and dj == 0:
                        continue
                    new_item = (new_i, j + dj)
                    neighbours.append(new_item)
                    if level >= 2 and dj == 1:
                        right_neighbours.append(new_item)

            if not neighbours:
                continue

            next_i, next_j = random.choice(neighbours)
            if level >= 2 and next2_i is None and right_neighbours:
                next2_i, next2_j = random.choice(right_neighbours)

            permutations3 = [
                ((i, j), (next_i, next_j), (next2_i, next2_j)),
                ((i, j), (next2_i, next2_j), (next_i, next_j)),
                ((next_i, next_j), (i, j), (next2_i, next2_j)),
                ((next_i, next_j), (next2_i, next2_j), (i, j)),
                ((next2_i, next2_j), (i, j), (next_i, next_j)),
                ((next2_i, next2_j), (next_i, next_j), (i, j)),
            ] if next2_i is not None else []

            permutations2 = [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (next2_i, next2_j)),
                ((i, j), (next2_i, next2_j)),
                ((next_i, next_j), (i, j)), ((next2_i, next2_j), (next_i, next_j)),
                ((next2_i, next2_j), (i, j)),
            ] if next2_i is not None else [
                ((i, j), (next_i, next_j)), ((next_i, next_j), (i, j))
            ]

            possible_variants = []
            for (n_args, cmp_function, str_variants, *flags) in rules_for_relations:
                if n_args == 3:
                    for items in permutations3:
                        (ti, tj), (t_next_i, t_next_j), (t_next2_i, t_next2_j) = items
                        if flags and flags[0] and (ti == t_next_i or ti == t_next2_i or t_next_i == t_next2_i):
                            continue
                        if cmp_function(tj, t_next_j, t_next2_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 2:
                    for items in permutations2:
                        (ti, tj), (t_next_i, t_next_j) = items
                        if flags and flags[0] and ti == t_next_i:
                            continue
                        if cmp_function(tj, t_next_j):
                            possible_variants.append((n_args, items, cmp_function, random.choice(str_variants)))
                elif n_args == 1 and cmp_function(j):
                    possible_variants.append((n_args, [(i, j)], cmp_function, random.choice(str_variants)))

            if not possible_variants:
                continue

            n_args, list_of_ij, cmp_function, string_format = random.choice(possible_variants)
            list_for_format = []
            ins, wns = [], []
            for ii, jj in list_of_ij:
                list_for_format.extend([table[ii][0], table_wo_left[ii][jj]])
                ins.append(ii)
                wns.append(table_wo_left[ii][jj])
            relations.append((ins, wns, cmp_function, string_format.format(*list_for_format)))
            changed = True
            while changed:
                changed = _update_ranges(relations, ranges)

        if not fail:
            if minimal_conditions and not is_minimized and not time_elapsed:
                continue
            break

    premises = [t[-1] for t in relations]
    random.shuffle(premises)
    return premises, relations


def generate_zebra_problem(n_categories: int = 4, n_positions: int = 3, level: int = 2):
    """
    Generate a single ZebraLogic puzzle.
    Returns dict with: table_raw, relations, clue_texts, categories
    """
    all_categories = sorted(CATEGORY_DOMAINS.keys())
    chosen = sorted(random.sample(all_categories, k=n_categories))
    table = [
        [cat] + random.sample(sorted(CATEGORY_DOMAINS[cat]), k=n_positions)
        for cat in chosen
    ]

    premises, relations = generate_puzzle(
        table, level=level, minimal_conditions=True, max_seconds_for_minimizing=15
    )

    return {
        "table_raw": table,           # [[cat, v1, v2, ...], ...]
        "relations": relations,        # [(ins, wns, cmp_func, clue_text), ...]
        "clue_texts": premises,        # [clue_text, ...] (shuffled)
        "categories": chosen,
        "n_positions": n_positions,
        "level": level,
    }

# ═════════════════════════════════════════════════════════════════════════════
# PART 2: Difficulty Curriculum
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DifficultyConfig:
    """Configuration for a single difficulty tier."""
    name: str
    n_categories: int
    n_positions: int
    level: int
    weight: float = 1.0  # sampling weight


# Diverse difficulty distribution for heterogeneous constraint coverage
CURRICULUM = [
    # -- Tier 1: Simple (sudoku-like, few constraint types) --
    DifficultyConfig("easy-3x3",    n_categories=3, n_positions=3, level=1, weight=1.0),
    DifficultyConfig("easy-3x3-L2", n_categories=3, n_positions=3, level=2, weight=1.5),
    DifficultyConfig("easy-4x3",    n_categories=4, n_positions=3, level=2, weight=1.5),

    # -- Tier 2: Medium (mixed constraint types) --
    DifficultyConfig("med-4x3-L5",  n_categories=4, n_positions=3, level=5, weight=2.0),
    DifficultyConfig("med-4x4-L3",  n_categories=4, n_positions=4, level=3, weight=2.0),
    DifficultyConfig("med-4x4-L5",  n_categories=4, n_positions=4, level=5, weight=2.5),
    DifficultyConfig("med-5x3-L5",  n_categories=5, n_positions=3, level=5, weight=2.0),

    # -- Tier 3: Hard (many heterogeneous constraints) --
    DifficultyConfig("hard-4x4-L7", n_categories=4, n_positions=4, level=7, weight=2.0),
    DifficultyConfig("hard-4x5-L5", n_categories=4, n_positions=5, level=5, weight=2.0),
    DifficultyConfig("hard-5x4-L7", n_categories=5, n_positions=4, level=7, weight=1.5),
    DifficultyConfig("hard-4x4-L9", n_categories=4, n_positions=4, level=9, weight=1.5),

    # -- Tier 4: Expert (complex relation types) --
    DifficultyConfig("expert-5x5-L7",  n_categories=5, n_positions=5, level=7,  weight=1.0),
    DifficultyConfig("expert-4x4-L12", n_categories=4, n_positions=4, level=12, weight=0.8),
    DifficultyConfig("expert-5x4-L10", n_categories=5, n_positions=4, level=10, weight=0.8),
]


def sample_difficulty(curriculum: List[DifficultyConfig] = None) -> DifficultyConfig:
    """Weighted random sampling from curriculum."""
    if curriculum is None:
        curriculum = CURRICULUM
    weights = [d.weight for d in curriculum]
    return random.choices(curriculum, weights=weights, k=1)[0]

__all__ = [
    "CATEGORY_DOMAINS",
    "DifficultyConfig",
    "CURRICULUM",
    "sample_difficulty",
    "generate_zebra_problem",
]
