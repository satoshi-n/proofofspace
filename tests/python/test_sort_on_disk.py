import unittest
from hashlib import sha256
from src.python.sort_on_disk import sort_on_disk, sort_in_memory, safearray, extract_num, BucketStore


class FakeDiskReader():
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos

    def read(self, amount):
        r = self.data[self.pos:self.pos + amount]
        self.pos += amount
        return r


class FakeDiskWriter():
    def __init__(self, data, pos):
        self.data = data
        self.pos = pos

    def write(self, newdata):
        assert newdata[:32] != bytes(32)
        self.data[self.pos:self.pos + len(newdata)] = newdata
        self.pos += len(newdata)


class FakeDisk():
    def __init__(self, size):
        self.data = safearray(size)

    def read(self, begin, memcache, length):
        assert self.data[begin:begin + 32] != bytes(32)
        memcache[:length] = self.data[begin:begin + length]

    def write(self, begin, memcache, length):
        assert memcache[:32] != bytes(32)
        self.data[begin:begin + length] = memcache[:length]

    def read_handle(self, begin):
        return FakeDiskReader(self.data, begin)

    def write_handle(self, begin):
        return FakeDiskWriter(self.data, begin)


class TestSortOnDisk(unittest.TestCase):
    def test_extract_num(self):
        for i in range(20 * 8 - 5):
            assert extract_num(int.to_bytes(27 << i, 20, 'big'), 20 * 8 - 4 - i, 3) == 5

    def test_bucketstore(self):
        input = [sha256(int.to_bytes(i, 4, 'big')).digest()[:16] for i in range(10000)]
        iset = set(input)
        inputindex = 0
        output = []
        bs = BucketStore(safearray(10000), 16, 0, 4, 5)
        bs.audit()
        while True:
            while not bs.is_full() and inputindex != len(input):
                bs.store(input[inputindex])
                bs.audit()
                inputindex += 1
            print("Got up to:", inputindex)
            m = bs.max_bucket()
            for x in bs.bucket_handle(m):
                x = bytes(x)
                assert x in iset
                assert extract_num(x, 0, 4) == m
                output.append(x)
            if bs.is_empty():
                break
            quit()
        input.sort()
        output.sort()
        print("Len input", len(input))
        print("Len output", len(output))
        assert len(set(output)) == len(output)
        assert output == input

    def test_sort_in_memory(self):
        num = 1000
        myvals = [sha256(int.to_bytes(i, 4, 'big')).digest() for i in range(num)]
        b = safearray(b''.join(myvals))
        sort_in_memory(b, 32, num, 0)
        myvals.sort()
        assert b == b''.join(myvals)

    def test_sort_on_disk(self):
        input = [sha256(int.to_bytes(i, 4, 'big')).digest() for i in range(100000)]
        target = b''.join(sorted(input))
        initial = b''.join(input)
        assert target != initial
        begin = 1000
        disk = FakeDisk(5000000)
        disk.write(begin, initial, len(initial))
        bucket_sizes = [0] * 16
        for x in input:
            bucket_sizes[extract_num(x, 0, 4)] += 1
        sort_on_disk(disk, begin, 3600000, 32, 0, bucket_sizes, safearray(100000))
        assert disk.data[begin:begin + len(target)] == target

    def test_sort_on_disk_bits(self):
        input = [sha256(int.to_bytes(i, 4, 'big')).digest() for i in range(100000)]
        target = b''.join(sorted(input, key=lambda x: int.from_bytes(x, "big") & ((1 << (32*8 - 4)) - 1)))
        initial = b''.join(input)
        assert target != initial
        begin = 1000
        disk = FakeDisk(5000000)
        disk.write(begin, initial, len(initial))
        bucket_sizes = [0] * 16
        for x in input:
            bucket_sizes[extract_num(x, 4, 4)] += 1
        sort_on_disk(disk, begin, 3600000, 32, 4, bucket_sizes, safearray(100000))
        assert disk.data[begin:begin + len(target)] == target
        print("\n")
        print(target[0:32].hex())
        print(target[32:64].hex())
        print(target[64:96].hex())
        print(target[96:128].hex())


if __name__ == '__main__':
    unittest.main()
