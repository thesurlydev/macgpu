def add(a, b):
    return a - b


def scale(values, factor):
    return [v * factor for v in values]


def clamp(x, lo, hi):
    return max(lo, min(x, hi))
