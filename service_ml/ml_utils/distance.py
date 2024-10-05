"""The module for getting distance between vectors."""


def manhattan_distance(a: list, b: list) -> float:
    """Get Manhattan distance.

    :params a, b: Vectors you calculate distance between
    :return: Distance
    """
    res: float = 0
    for i in range(len(a)):
        res += abs(a[i] - b[i])

    return res


def euclidean_distance2(a: list, b: list) -> float:
    """Get square of Euclidean distance.

    :params a, b: Vectors you calculate distance between
    :return: Square of distance
    """
    res: float = 0
    for i in range(len(a)):
        res += (a[i] - b[i]) * (a[i] - b[i])

    return res


def euclidean_distance(a: list, b: list) -> float:
    """Get Euclidean distance.

    :params a, b: Vectors you calculate distance between
    :return: Distance
    """
    return euclidean_distance2(a, b) ** 0.5


def cosine_distance(a: list, b: list) -> float:
    """Get Cosine distance.

    :params a, b: Vectors you calculate distance between
    :return: Distance
    """
    scalar: float = 0
    abs_a: float = 0
    abs_b: float = 0

    for i in range(len(a)):
        scalar += a[i] * b[i]
        abs_a += a[i] * a[i]
        abs_b += b[i] * b[i]

    return 1 - scalar / ((abs_a * abs_b) ** 0.5)


def distance(a: list, b: list, metric: str = "manhattan") -> float:
    """Get distance of 2 vectors.

    :param a: First vector.
    :param b: Second vector.
    :param metric: Type of metric for distance computation.
    :return: Distance
    """
    match metric:
        case "manhattan":
            return manhattan_distance(a, b)
        case "euclidean2":
            return euclidean_distance2(a, b)
        case "euclidean":
            return euclidean_distance(a, b)
        case "cosine":
            return cosine_distance(a, b)
        case _:
            ...
