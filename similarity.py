
import math


def validate_vectors(a, b):
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Vectors cannot be empty")

    if len(a) != len(b):
        raise ValueError("Vectors must have equal dimensions")

    if not all(math.isfinite(x) for x in a + b):
        raise ValueError("Vector values must be finite numbers")


def dot_product(a, b):
    validate_vectors(a, b)
    return sum(x * y for x, y in zip(a, b))


def magnitude(vector):
    if len(vector) == 0:
        raise ValueError("Vector cannot be empty")

    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Vector values must be finite numbers")

    return math.sqrt(sum(x * x for x in vector))


def cosine_similarity(a, b):
    validate_vectors(a, b)

    mag_a = magnitude(a)
    mag_b = magnitude(b)

    if mag_a == 0 or mag_b == 0:
        raise ValueError("Zero vectors cannot be compared")

    return dot_product(a, b) / (mag_a * mag_b)


if __name__ == "__main__":
    a = [1, 0]
    b = [0.8, 0.6]

    print("Dot product:", dot_product(a, b))
    print("Magnitude of A:", magnitude(a))
    print("Cosine similarity:", cosine_similarity(a, b))
