##############################################################################
#
# Copyright (c) 2024 Quantinuum LLC All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
##############################################################################

import json
from json import JSONDecodeError

import deal

from pytket.circuit import Circuit
from pytket.phir.api import pytket_to_phir


class TestPhirGen:
    @deal.has("stdout")
    @deal.raises(AssertionError, JSONDecodeError, KeyError, TypeError)
    def test_classicalexpbox(self) -> None:
        """From https://github.com/CQCL/pytket-phir/issues/86 ."""
        circ = Circuit(1)
        a = circ.add_c_register("a", 2)
        b = circ.add_c_register("b", 2)
        c = circ.add_c_register("c", 3)
        circ.add_classicalexpbox_register(a + b, c.to_list())

        phir = json.loads(pytket_to_phir(circ))
        assert phir["ops"][4] == {
            "cop": "=",
            "returns": ["c"],
            "args": [{"cop": "+", "args": ["a", "b"]}],
        }

    @deal.has("stdout")
    @deal.raises(AssertionError, JSONDecodeError, KeyError, TypeError)
    def test_nested_arith(self) -> None:
        """From https://github.com/CQCL/pytket-phir/issues/87 ."""
        circ = Circuit(1)
        a = circ.add_c_register("a", 2)
        b = circ.add_c_register("b", 2)
        c = circ.add_c_register("c", 3)
        circ.add_classicalexpbox_register(a + b // c, c.to_list())

        phir = json.loads(pytket_to_phir(circ))
        assert phir["ops"][4] == {
            "cop": "=",
            "returns": ["c"],
            "args": [{"cop": "+", "args": ["a", {"cop": "/", "args": ["b", "c"]}]}],
        }

    @deal.has("stdout")
    @deal.raises(AssertionError, JSONDecodeError, KeyError, TypeError)
    def test_arith_with_int(self) -> None:
        """From https://github.com/CQCL/pytket-phir/issues/88 ."""
        circ = Circuit(1)
        a = circ.add_c_register("a", 2)
        circ.add_classicalexpbox_register(a << 1, a.to_list())

        phir = json.loads(pytket_to_phir(circ))
        assert phir["ops"][2] == {
            "cop": "=",
            "returns": ["a"],
            "args": [{"cop": "<<", "args": ["a", 1]}],
        }

    @deal.has("stdout")
    @deal.raises(AssertionError, JSONDecodeError, KeyError, TypeError)
    def test_bitwise_ops(self) -> None:
        """From https://github.com/CQCL/pytket-phir/issues/91 ."""
        circ = Circuit(1)
        a = circ.add_c_register("a", 2)
        b = circ.add_c_register("b", 2)
        c = circ.add_c_register("c", 1)
        expr = a[0] ^ b[0]
        circ.add_classicalexpbox_bit(expr, [c[0]])

        phir = json.loads(pytket_to_phir(circ))
        assert phir["ops"][4] == {
            "cop": "=",
            "returns": [["c", 0]],
            "args": [{"cop": "^", "args": [["a", 0], ["b", 0]]}],
        }
