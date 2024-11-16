# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torch import Generator, Tensor


def supports_cuda() -> bool: ...


def create_random_device_generator(token: str = "") -> Generator: ...


def create_mt19937_generator(seed: int = 0): ...


def encrypt(input: Tensor, output: Tensor, key: Tensor, cipher, mode): ...


def decrypt(input: Tensor, output: Tensor, key: Tensor, cipher, mode): ...


def __version__() -> str: ...


def git_version() -> str: ...


class PRG:
    def __init__(self): ...

    def set_seeds(self, seeds: Tensor): ...

    def bit_random(self, bits: int): ...

    def random(self, length: int): ...