import unittest

import inventory


def noise(tag, lines=30):
    # Stands in for a chatty test suite: every test logs a screenful.
    for i in range(lines):
        print(
            f"DEBUG worker={i % 4} tag={tag} step={i:03d} cache=warm "
            f"latency_ms={(i * 37) % 250} msg=processing batch item {i} of {lines}"
        )


class TestInventory(unittest.TestCase):
    def test_apply_discount(self):
        noise("discount")
        self.assertAlmostEqual(inventory.apply_discount(200.0, 25), 150.0)

    def test_total_value(self):
        noise("total")
        stock = {"apple": 3, "pear": 0, "plum": 2}
        prices = {"apple": 0.5, "pear": 9.0, "plum": 1.25}
        self.assertAlmostEqual(inventory.total_value(stock, prices), 4.0)


def _restock_test(i):
    def test(self):
        noise(f"restock{i:02d}")
        self.assertEqual(inventory.restock({"a": i}, "a", 1)["a"], i + 1)

    return test


for _i in range(40):
    setattr(TestInventory, f"test_restock_{_i:02d}", _restock_test(_i))


if __name__ == "__main__":
    unittest.main()
