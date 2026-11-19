"""
curriculum/services/high_yield.py

Intelligent VTU Exam High-Yield Topic Analysis and Classification Engine.
Replaces naive hardcoded 'order <= 2' heuristics with contextual examination
weight analysis, distinguishing 8-10 mark core questions (algorithms, architectures,
working principles, mathematical derivations) from introductory or survey topics.
"""

import logging
from django.db import transaction

logger = logging.getLogger(__name__)

# Strong negative signals: Introductory, survey, motivation, and historical topics
# that almost never appear as standalone 10-mark VTU SEE exam questions.
LOW_YIELD_PREFIXES = (
    "introduction to", "intro to", "overview of", "why use", "need for",
    "history of", "evolution of", "applications of", "basic concepts of",
    "what is", "background on", "summary of", "case studies in", "case studies",
    "future directions", "trends in", "challenges and future", "conclusion",
)

# High-yield domain keywords scored by their typical VTU question weight
HIGH_YIELD_KEYWORDS = {
    # Algorithms & Core Math (10M / 8M questions)
    "algorithm": 10,
    "backpropagation": 12,
    "loss function": 9,
    "gradient descent": 10,
    "optimization": 8,
    "mathematical": 10,
    "derivation": 11,
    "working principle": 10,
    "markov decision": 11,
    "mdp": 11,
    "bellman": 12,
    "function approximation": 9,
    "numerical": 9,
    "dynamic programming": 11,

    # Deep Learning & Neural Architectures
    "convolution": 11,
    "pooling": 9,
    "receptive field": 9,
    "weight sharing": 9,
    "lstm": 12,
    "gru": 10,
    "recurrent": 9,
    "bptt": 11,
    "backpropagation through time": 12,
    "computational graph": 8,
    "dropout": 8,
    "activation function": 8,
    "vanishing gradient": 9,
    "adagrad": 9,
    "rmsprop": 9,
    "adam": 9,

    # System Architectures & Hardware
    "architecture": 10,
    "pipeline": 10,
    "hazard": 10,
    "cache": 9,
    "virtual memory": 10,
    "paging": 9,
    "segmentation": 9,
    "deadlock": 10,
    "banker": 11,
    "scheduling": 9,
    "synchronization": 9,
    "semaphore": 9,
    "process": 7,

    # Cryptography & Security
    "cryptosystem": 11,
    "rsa": 12,
    "des": 10,
    "aes": 11,
    "cipher": 10,
    "ecc": 11,
    "encryption": 9,
    "decryption": 9,
    "digital signature": 10,
    "hash function": 9,
    "diffie": 11,
    "key management": 10,
    "ssl/tls": 10,
    "watermarking": 9,
    "security": 7,
    "privacy": 7,

    # Mechanical / Manufacturing / Interdisciplinary
    "mechanism": 10,
    "metal removal": 10,
    "process parameters": 9,
    "edm": 11,
    "ecm": 11,
    "usm": 11,
    "ajm": 11,
    "laser": 9,
    "plasma": 9,

    # Core CS / Data Structures
    "tree": 8,
    "heap": 8,
    "graph": 8,
    "b-tree": 10,
    "shortest path": 10,
    "dijkstra": 11,
    "syntax directed": 10,
    "parsing": 9,
    "protocol": 8,
    "state machine": 9,
    "regression": 9,
    "classification": 8,
    "clustering": 8,
    "svm": 10,
    "support vector": 10,
}


def score_topic_exam_importance(title: str, order: int = 1) -> float:
    """
    Computes an academic exam weight score for a syllabus topic.
    Positive scores indicate high probability of being an 8-10 mark question.
    Negative scores indicate introductory/overview material.
    """
    clean_title = (title or "").strip().lower()
    if not clean_title:
        return 0.0

    score = 0.0

    # Strong penalty for introductory / overview topics
    if any(clean_title.startswith(p) for p in LOW_YIELD_PREFIXES):
        score -= 15.0

    # Keyword bonuses
    matched_keywords = 0
    for kw, weight in HIGH_YIELD_KEYWORDS.items():
        if kw in clean_title:
            score += weight
            matched_keywords += 1

    # If no keywords matched and not an intro, give neutral base score
    if matched_keywords == 0 and score >= 0:
        # Topics with colon or hyphen often denote detailed subtopics (e.g. "Architecture: Layers and Protocols")
        if ":" in clean_title or " - " in clean_title:
            score += 4.0
        # Give mild preference to middle core topics over topic 1
        if order in (2, 3):
            score += 2.0

    return score


def tag_module_high_yield_topics(module, max_picks: int = 2):
    """
    Analyzes all topics in a module, identifies the genuine top 2-3 high-yield
    exam concepts based on past VTU questions and semantic exam scoring,
    and updates `is_high_yield` and `vtu_frequency_tag` in the database.
    """
    topics = list(module.topics.all().order_by('order', 'id'))
    if not topics:
        return []

    # Map exam question associations if present
    exam_q_topics = set()
    for eq in module.exam_questions.all():
        if eq.topic_id:
            exam_q_topics.add(eq.topic_id)

    scored_topics = []
    for t in topics:
        sc = score_topic_exam_importance(t.title, t.order)
        # Topics directly linked to past exam questions receive top priority
        if t.id in exam_q_topics:
            sc += 25.0
        scored_topics.append((sc, t))

    # Sort descending by score, breaking ties by order
    scored_topics.sort(key=lambda item: (-item[0], item[1].order))

    # Determine picks: topics with score > 0, up to max_picks (minimum 1 if any topic exists)
    high_yield_picks = []
    for sc, t in scored_topics:
        if sc > 0 and len(high_yield_picks) < max_picks:
            high_yield_picks.append(t)

    # Edge case fallback: if every topic scored <= 0 (e.g. all generic titles), pick the middle topic (topic 2)
    if not high_yield_picks and topics:
        fallback_idx = 1 if len(topics) > 1 else 0
        high_yield_picks.append(topics[fallback_idx])

    high_yield_ids = {t.id for t in high_yield_picks}
    mod_q_label = f"Q{module.module_number * 2 - 1} or Q{module.module_number * 2}"

    with transaction.atomic():
        for t in topics:
            if t.id in high_yield_ids:
                t.is_high_yield = True
                if not t.vtu_frequency_tag:
                    t.vtu_frequency_tag = f"Core VTU Concept ({mod_q_label} Choice)"
                t.save(update_fields=['is_high_yield', 'vtu_frequency_tag'])
            else:
                t.is_high_yield = False
                t.vtu_frequency_tag = ""
                t.save(update_fields=['is_high_yield', 'vtu_frequency_tag'])

    logger.info(
        "Tagged Module %d (%s) high-yield topics: %s",
        module.module_number,
        module.title,
        [t.title for t in high_yield_picks]
    )
    return high_yield_picks


def tag_subject_high_yield_topics(subject, max_picks_per_module: int = 2):
    """
    Tags high-yield topics across all modules of a subject.
    """
    total_tagged = 0
    for mod in subject.modules.all().order_by('module_number'):
        picks = tag_module_high_yield_topics(mod, max_picks=max_picks_per_module)
        total_tagged += len(picks)
    return total_tagged


def tag_all_subjects_high_yield_topics(max_picks_per_module: int = 2):
    """
    Scans and tags high-yield topics for all subjects currently in the database.
    """
    from curriculum.models import Subject
    total = 0
    for s in Subject.objects.all():
        total += tag_subject_high_yield_topics(s, max_picks_per_module=max_picks_per_module)
    return total
