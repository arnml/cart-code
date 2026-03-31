"""Text similarity utilities."""


def jaccard(t1: str, t2: str) -> float:
    s1, s2 = set(t1.lower().split()), set(t2.lower().split())
    return len(s1 & s2) / len(s1 | s2) if s1 and s2 else 0.0
