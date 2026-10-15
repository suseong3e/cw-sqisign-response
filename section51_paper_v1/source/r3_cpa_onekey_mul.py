#!/usr/bin/env python3
"""Source-order ibz_mul stores for an eight- or nine-word CRT inverse."""
from __future__ import annotations

import numpy as np

import r3_cpa_crt_analyze as crt


MASK = np.uint64(0xffffffff)


def words(value: int) -> np.ndarray:
    count = (int(value).bit_length() + 31) // 32
    if count not in (8, 9):
        raise ValueError(f"unsupported inverse word count: {count}")
    return np.array([(int(value) >> (32*j)) & 0xffffffff for j in range(count)],
                    dtype=np.uint64)


def update_row(state: np.ndarray, row: int, diff_word: np.ndarray,
               u_words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    result = state.copy()
    carry = np.zeros(state.shape[0], dtype=np.uint64)
    stores = np.empty((len(u_words), state.shape[0]), dtype=np.uint8)
    for j, word in enumerate(u_words):
        total = diff_word*word + result[:, row+j] + carry
        result[:, row+j] = total & MASK
        carry = total >> np.uint64(32)
        stores[j] = crt.low.matrix_attack.limb.low.hw32(result[:, row+j])
    result[:, row+len(u_words)] = carry
    return result, stores


def initial_state(secret: int, public: np.ndarray,
                  u_words: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((len(public), 34), dtype=np.uint64)
    borrow = np.zeros(len(public), dtype=np.uint64)
    for row in range(6):
        secret_word = np.uint64((secret >> (32*row)) & 0xffffffff)
        subtrahend = public[:, row] + borrow
        difference = (secret_word-subtrahend) & MASK
        borrow = (secret_word < subtrahend).astype(np.uint64)
        state, _ = update_row(state, row, difference, u_words)
    return state, borrow


def simulate_product(diff: np.ndarray, u: int) -> np.ndarray:
    u_words = words(u)
    state = np.zeros((len(diff), 34), dtype=np.uint64)
    stores = np.empty((17, len(u_words), len(diff)), dtype=np.uint8)
    for row in range(17):
        state, stores[row] = update_row(state, row, diff[:, row], u_words)
    return stores


def self_test() -> None:
    rng = np.random.default_rng(7)
    for count in (8, 9):
        u = (1 << (32*(count-1))) | int(rng.integers(1, 1 << 31))
        diff = rng.integers(0, 1 << 32, (3, 17), dtype=np.uint64)
        observed = simulate_product(diff, u)
        for row in range(3):
            state = [0]*34
            for i in range(17):
                carry = 0
                for j in range(count):
                    total = int(diff[row, i])*((u >> (32*j)) & 0xffffffff) + state[i+j] + carry
                    state[i+j] = total & 0xffffffff
                    carry = total >> 32
                    if observed[i, j, row] != state[i+j].bit_count():
                        raise AssertionError("ibz_mul store differs from scalar source order")
                state[i+count] = carry


if __name__ == "__main__":
    self_test()
    print("self-test passed")
