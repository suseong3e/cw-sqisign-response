"""Recover the CRT norm from one branch's recovered inverse and coordinates."""
from __future__ import annotations

from math import gcd

from sympy import factorint, isprime, nextprime


BASE = 1 << 192
FACTOR_LIMIT = 100_000


def recover_norm(inverse: int, x: int, y: int, field_prime: int) -> tuple[int, int]:
    """Return the unique large prime norm and its small gcd cofactor.

    Key generation tests the short equivalent norm for primality before
    removing a possible square factor. A nontrivial final norm remains prime.
    The gcd cofactor divides ``BASE*inverse-1`` divided by the norm and is
    therefore smaller than BASE. An unfactored cofactor is an explicit block.
    """
    if inverse <= 0 or x < 0 or y < 0 or field_prime <= 0:
        raise ValueError("CRT inverse and field prime must be positive; coordinates nonnegative")
    if max(inverse, x, y) <= BASE:
        raise RuntimeError("candidate values do not establish norm greater than the 192-bit base")
    numerator = 4 * x * x + (2 * y + 1) ** 2 + field_prime
    if numerator % 4:
        raise ValueError("coordinate norm expression is not integral")
    common = gcd(BASE * inverse - 1, numerator // 4)
    if common <= BASE:
        raise RuntimeError("no CRT norm larger than the 192-bit base survives")
    factors = factorint(common, limit=FACTOR_LIMIT)
    if any(not isprime(int(factor)) for factor in factors):
        raise RuntimeError("CRT norm gcd has an unresolved factor")
    large = [int(factor) for factor in factors if factor > BASE]
    if len(large) != 1:
        raise RuntimeError("CRT norm gcd has no unique large prime factor")
    norm = large[0]
    if not max(inverse, x, y) < norm or pow(BASE, -1, norm) != inverse:
        raise RuntimeError("CRT norm candidate fails canonical coordinate or inverse checks")
    return norm, common // norm


def self_test() -> None:
    norm = int(nextprime(1 << 260))
    inverse = pow(BASE, -1, norm)
    x, y = 1, 1
    field_prime = 4 * norm - 4 * x * x - (2 * y + 1) ** 2
    assert recover_norm(inverse, x, y, field_prime) == (norm, 1)
    try:
        recover_norm(inverse, x, y, field_prime + 2)
    except ValueError:
        pass
    else:
        raise AssertionError("nonintegral norm expression was accepted")


if __name__ == "__main__":
    self_test()
    print("self-test passed")
