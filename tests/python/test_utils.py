import unittest
from bitstring import BitArray
from src.python.utils import byte_align, parse_entry


class TestUtils(unittest.TestCase):
    def test_byte_align(self):
        assert(byte_align(7) == 8)
        assert(byte_align(8) == 8)
        assert(byte_align(11) == 16)

    def test_parse_entry(self):
        k = 30
        entry = (BitArray(uint=234234, length=k) + BitArray(uint=98732, length=k)).tobytes()
        y, _, _, metadata = parse_entry(entry, k, 0, 0, k)
        assert(y.uint == 234234)
        assert(metadata.uint == 98732)

        pos_len = 31
        offset_len = 8
        entry = (BitArray(uint=234234, length=k) + BitArray(uint=7776, length=pos_len)
                 + BitArray(uint=77, length=offset_len) + BitArray(uint=98732, length=k)).tobytes()
        y_, pos, offset, metadata = parse_entry(entry, k, pos_len, offset_len, k)
        assert(y.uint == 234234)
        assert(pos.uint == 7776)
        assert(offset.uint == 77)
        assert(metadata.uint == 98732)


if __name__ == '__main__':
    unittest.main()
