from bitstring import BitArray
from math import sqrt


def square_to_line_point(x, y):
    if y > x:
        x, y = y, x
    return x * (x-1) // 2 + y


def line_point_to_square(index):
    x = int((sqrt(8 * index + 1) + 1) // 2)
    return int(x), int(index - int(x * (x-1) // 2))


def arithmetic_encode_deltas(deltas):
    s = BitArray()
    for d in deltas:
        if d != 0:
            s += BitArray(uint=0, length=d)
        s += BitArray(uint=1, length=1)
    return s


def arithmetic_decode_deltas(bits, max_deltas):
    deltas = []
    counter = 0
    for i in range(len(bits)):
        new_bit = bits[i]
        if new_bit == 1:
            deltas.append(counter)
            counter = 0
        else:
            counter += 1
    return deltas[:max_deltas]


def byte_align(num_bits):
    return (num_bits + (8 - ((num_bits) % 8)) % 8)


# Parses an entry in the following format:
# [k bits of y][pos_size bits of pos][offset_size bits of offset][metadata_size bits of metadata][padding]
def parse_entry(entry, k, pos_size, offset_size, metadata_size):
    entry_size = byte_align(k + pos_size + offset_size + metadata_size)
    padding_size = entry_size - (k + pos_size + offset_size + metadata_size)

    y_mask = (((1 << (entry_size)) - 1) -
              ((1 << (entry_size - k)) - 1))
    pos_mask = (((1 << (padding_size + metadata_size + offset_size + pos_size)) - 1) -
                ((1 << (padding_size + metadata_size + offset_size)) - 1))
    offset_mask = (((1 << (padding_size + metadata_size + offset_size)) - 1) -
                   ((1 << (padding_size + metadata_size)) - 1))
    metadata_mask = (((1 << (padding_size + metadata_size)) - 1) -
                     ((1 << (padding_size)) - 1))

    uint_y = (int.from_bytes(entry, "big") & y_mask) >> (entry_size - k)
    uint_pos = (int.from_bytes(entry, "big") & pos_mask) >> (offset_size + metadata_size + padding_size)
    uint_offset = (int.from_bytes(entry, "big") & offset_mask) >> (metadata_size + padding_size)
    uint_metadata = (int.from_bytes(entry, "big") & metadata_mask) >> (padding_size)

    if k != 0:
        y = BitArray(uint=uint_y, length=k)
    else:
        y = None
    if pos_size != 0:
        pos = BitArray(uint=uint_pos, length=pos_size)
    else:
        pos = None
    if offset_size != 0:
        offset = BitArray(uint=uint_offset, length=offset_size)
    else:
        offset = None
    if metadata_size != 0:
        metadata = BitArray(uint=uint_metadata, length=metadata_size)
    else:
        metadata = None
    return (y, pos, offset, metadata)
