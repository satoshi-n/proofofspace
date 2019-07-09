def sort_on_disk(disk, disk_begin, spare_begin, entry_len, bitsbegin,
                 bucket_sizes, mem):
    """
    Sort a disk using an external sort.

    disk: object that supports .read and .write
    disk_begin: starting position on disk to sort from
    spare_begin: starting position on disk for working memory
    entry_len: parameterized size of entries
    bitsbegin: how many bits before considered value starts in each entry
    bucket_sizes: initial (populated) list of how many entries on the disk
                  are in each bucket.  An entry belongs to bucket b, where
                  b is the first log_2(len(bucket_sizes)) bits of the entry.
                  len(bucket_sizes) must be a power of 2.
    mem: array-like for in memory storage
    """

    length = len(mem) // entry_len
    totalsize = sum(bucket_sizes)
    N_buckets = len(bucket_sizes)

    assert disk_begin + totalsize * entry_len <= spare_begin, \
        "Writing to this disk would overwrite the space allocated as spare"

    if bitsbegin >= entry_len * 8:
        return

    # If we can sort all entries in memory, do so.
    if totalsize <= length:
        disk.read(disk_begin, mem, totalsize * entry_len)
        sort_in_memory(mem, entry_len, totalsize, bitsbegin)
        disk.write(disk_begin, mem, totalsize * entry_len)
        return

    bucket_begins = [0]
    total = 0
    for i in range(N_buckets - 1):
        total += bucket_sizes[i]
        bucket_begins.append(total)

    # Assumed len(bucket_sizes) is a power of 2.
    bucketlog = N_buckets.bit_length() - 1

    # Move the beginning of each bucket into the spare.
    spare_written = 0
    consumed_per_bucket = [0] * N_buckets
    # The spare stores about 5 * N_buckets * len(mem) entries.
    unit = int(length / N_buckets * 5)
    for i, b_size in enumerate(bucket_sizes):
        to_consume = min(unit, b_size)
        while to_consume > 0:
            next_amount = min(length, to_consume)
            disk.read(disk_begin + (bucket_begins[i] + consumed_per_bucket[i])
                    * entry_len, mem, next_amount * entry_len)
            disk.write(spare_begin + spare_written * entry_len,
                    mem, next_amount * entry_len)
            to_consume -= next_amount
            spare_written += next_amount
            consumed_per_bucket[i] += next_amount

    # Populate BucketStore from spare
    spare_consumed = 0
    bstore = BucketStore(mem, entry_len, bitsbegin, bucketlog)
    handle = disk.read_handle(spare_begin)
    while not bstore.is_full() and spare_consumed < spare_written:
        bstore.store(handle.read(entry_len))
        spare_consumed += 1

    # Repeatedly write out largest bucket until nothing left
    written_per_bucket = [0] * N_buckets
    subbucket_sizes = [[0] * N_buckets for i in range(N_buckets)]
    counter = 0
    while not bstore.is_empty():
        # print(bstore.num_free())
        # Write out largest buckets first
        for b in bstore.buckets_by_size():  # [!] order is frozen here - not dynamic
            if written_per_bucket[b] >= consumed_per_bucket[b]:
                continue
            # Sequentially write everything to this handle
            handle = disk.write_handle(disk_begin +
                (bucket_begins[b] + written_per_bucket[b]) * entry_len)

            for val in bstore.bucket_handle(b):
                handle.write(val)
                written_per_bucket[b] += 1
                subbucket_sizes[b][extract_num(val, bitsbegin + bucketlog, bucketlog)] += 1

                if written_per_bucket[b] == consumed_per_bucket[b]:
                    break


        # Sweep new elements in order of smallest gap
        for i in sorted(range(N_buckets), key =
                lambda i: consumed_per_bucket[i] - written_per_bucket[i]):
            if consumed_per_bucket[i] == bucket_sizes[i]:
                continue
            handle2 = disk.read_handle(disk_begin +
                (bucket_begins[i] + consumed_per_bucket[i]) * entry_len)
            while not bstore.is_full() and consumed_per_bucket[i] < bucket_sizes[i]:
                bstore.store(handle2.read(entry_len))
                consumed_per_bucket[i] += 1
            if bstore.is_full():
                break

        else: # Refill from spare
            handle3 = disk.read_handle(spare_begin + spare_consumed * entry_len)
            while not bstore.is_full() and spare_consumed < spare_written:
                bstore.store(handle3.read(entry_len))
                spare_consumed += 1

    for i in range(len(bucket_sizes)):
        sort_on_disk(disk, disk_begin + bucket_begins[i] * entry_len, spare_begin,
                entry_len, bitsbegin + bucketlog, subbucket_sizes[i], mem)


class BucketStore:
    """
    Store values bucketed by their leading bits into an array-like memcache.

    The memcache stores stacks of values, one for each bucket.
    The stacks are broken into segments, where each segment has content
    all from the same bucket, and a 4 bit pointer to its previous segment.
    The most recent segment is the head segment of that bucket.

    Additionally, empty segments form a linked list: 4 bit pointers of
    empty segments point to the next empty segment in the memcache.

    Each segment has size entries_per_seg * entry_len + 4, and consists of:
    [4 bit pointer to segment id] + [entries of length entry_len]*
    """

    def __init__(self, mem, entry_len, bitsbegin, bucketlog, entries_per_seg=100):
        self.mem = mem
        self.entry_len = entry_len
        self.bitsbegin = bitsbegin
        self.bucketlog = bucketlog
        self.entries_per_seg = entries_per_seg
        self.bucket_sizes = [0] * 2 ** bucketlog

        self.segsize = 4 + self.entry_len * entries_per_seg
        self.length = len(mem) // self.segsize
        for i in range(self.length):
            self._set_seg_id(i, i+1)

        self.first_empty_seg_id = 0  # Head of a linked list of empty segments
        self.bucket_head_ids = [self.length] * len(self.bucket_sizes)
        self.bucket_head_counts = [0] * len(self.bucket_sizes)

    def _set_seg_id(self, i, v):
        self.mem[i * self.segsize: i * self.segsize + 4] = int.to_bytes(v, 4, 'big')

    def _get_seg_id(self, i):
        return int.from_bytes(
                self.mem[i * self.segsize : i * self.segsize + 4], 'big')

    def _get_entry_pos(self, b):
        # The right edge of entries in bucket b.
        return (self.bucket_head_ids[b] * self.segsize + 4 +
                self.bucket_head_counts[b] * self.entry_len)

    def audit(self):
        count = 0
        pos = self.first_empty_seg_id
        while pos != self.length:
            count += 1
            pos = self._get_seg_id(pos)
        for pos in self.bucket_head_ids:
            while pos != self.length:
                count += 1
                pos = self._get_seg_id(pos)
        assert count == self.length

    def num_free(self):
        used = self._get_seg_id(self.first_empty_seg_id)
        return (len(self.bucket_sizes) - used) * self.entries_per_seg

    def is_empty(self):
        return not any(self.bucket_sizes)

    def is_full(self):
        return self.first_empty_seg_id == self.length

    def store(self, newval):
        assert len(newval) == self.entry_len
        assert self.first_empty_seg_id != self.length
        b = extract_num(newval, self.bitsbegin, self.bucketlog)  # bucket id
        self.bucket_sizes[b] += 1

        # If this is the first element of this bucket, or the segment
        # for this bucket is full, assign a new segment.
        if (self.bucket_head_ids[b] == self.length or
                self.bucket_head_counts[b] == self.entries_per_seg):

            old_seg_id = self.bucket_head_ids[b]
            self.bucket_head_ids[b] = self.first_empty_seg_id
            self.first_empty_seg_id = self._get_seg_id(self.first_empty_seg_id)
            self._set_seg_id(self.bucket_head_ids[b], old_seg_id)
            self.bucket_head_counts[b] = 0

        # Write newval to the head segment of bucket id b.
        pos = self._get_entry_pos(b)
        self.mem[pos : pos + self.entry_len] = newval
        self.bucket_head_counts[b] += 1

    def max_bucket(self):
        return max(range(len(self.bucket_sizes)),
                   key = self.bucket_sizes.__getitem__)

    def buckets_by_size(self):
        return sorted(range(len(self.bucket_sizes)),
                      key = self.bucket_sizes.__getitem__, reverse = True)

    def bucket_handle(self, b):
        L = self.entry_len

        # For each segment of this bucket, exhaust the segment
        while self.bucket_head_ids[b] != self.length:
            start_pos = self._get_entry_pos(b) - L
            end_pos = start_pos - self.bucket_head_counts[b] * L

            # [!] It is not guaranteed we finish iterating, so we
            # are careful to handle exiting anywhere gracefully.

            # For most positions
            for pos in range(start_pos, end_pos + L, -L):
                self.bucket_sizes[b] -= 1
                self.bucket_head_counts[b] -= 1
                yield self.mem[pos: pos+L]

            # Move to next segment of this bucket...
            # Link this free segment to head of linked list of empty segments
            next_full_seg_id = self._get_seg_id(self.bucket_head_ids[b])
            self._set_seg_id(self.bucket_head_ids[b], self.first_empty_seg_id)
            self.first_empty_seg_id = self.bucket_head_ids[b]
            self.bucket_head_ids[b] = next_full_seg_id

            if next_full_seg_id == self.length:
                self.bucket_head_counts[b] = 0
            else:
                self.bucket_head_counts[b] = self.entries_per_seg

            # If one more position, do it
            if start_pos != end_pos:
                self.bucket_sizes[b] -= 1
                yield self.mem[end_pos + L: end_pos + 2*L]

        assert self.bucket_sizes[b] == 0


class safearray(bytearray):
    def __setitem__(self, index, thing):
        if type(index) is slice:
            start = index.start
            if start is None:
                start = 0
            stop = index.stop
            if stop is None:
                stop = len(self)
            assert index.step is None
            assert start >= 0
            assert stop >= 0
            assert start < len(self)
            assert stop <= len(self)
            assert stop - start == len(thing)
        else:
            assert index >= 0
            assert index < len(self)
        bytearray.__setitem__(self, index, thing)

    def __str__(self):
        return str([b for b in self])


def extract_num(bytes_, begin, take=None):
    """
    Return an int representing bits taken from bytes_.
    The bits taken are bytes_[begin : begin + take].
    If take is None, it takes all the bits.
    """
    L = 8 * len(bytes_) - begin
    ans = int.from_bytes(bytes_, 'big') & ((1 << L) - 1)
    if take is not None and take < L:
        ans >>= L - take
    return ans


def sort_in_memory(mem, entry_len, num_entries, bitsbegin):
    _sort_in_memory_inner(mem, entry_len, bitsbegin, 0, num_entries)


def _sort_in_memory_inner(mem, L, bitsbegin, begin, end):
    """
    mem: Array-like, with entries mem[i * L : (i+1) * L]
    L: entry length
    bitsbegin: parameter for casting entries to int
    begin, end: paramaters specifying to sort mem[begin * L : end * L]
    """

    if end - begin <= 1:
        return

    lo, hi = begin, end - 1
    pivotraw = mem[hi * L : end * L]
    pivot = extract_num(pivotraw, bitsbegin)
    leftside = True
    while lo < hi:
        if leftside:
            if extract_num(mem[lo*L : (lo+1)*L], bitsbegin) < pivot:
                lo += 1
            else:
                mem[hi*L : (hi+1)*L] = mem[lo*L : (lo+1)*L]
                hi -= 1
                leftside = False
        else:
            if extract_num(mem[hi*L : (hi+1)*L], bitsbegin) > pivot:
                hi -= 1
            else:
                mem[lo*L : (lo+1)*L] = mem[hi*L : (hi+1)*L]
                lo += 1
                leftside = True

    mem[lo*L : (lo+1)*L] = pivotraw
    if lo - begin <= end - lo:
        _sort_in_memory_inner(mem, L, bitsbegin, begin, lo)
        _sort_in_memory_inner(mem, L, bitsbegin, lo+1, end)
    else:
        _sort_in_memory_inner(mem, L, bitsbegin, lo+1, end)
        _sort_in_memory_inner(mem, L, bitsbegin, begin, lo)
