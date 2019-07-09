from bitstring import BitArray
from .plotter_disk import (id_len, checkpoint_1_interval, checkpoint_2_interval,
                           ProofOfSpaceDiskPlotter, EPP, format_desc)
from .utils import byte_align, parse_entry, arithmetic_decode_deltas, line_point_to_square
from .calculate_bucket import F1Calculator, FxCalculator


class ProofOfSpaceDiskProver():
    def __init__(self, filename):
        self.disk_file = open(filename, "rb")
        self.disk_file.seek(19)

        self.id = self.disk_file.read(id_len)
        self.k = int.from_bytes(self.disk_file.read(1), "big")

        format_desc_len = int.from_bytes(self.disk_file.read(2), "big")
        self.format_desc_read = self.disk_file.read(format_desc_len)
        if self.format_desc_read != format_desc:
            raise ("Incorrect format version: " + self.format_desc_read)
        memo_len = int.from_bytes(self.disk_file.read(2), "big")
        self.memo = self.disk_file.read(memo_len)

        self.table_begin_pointers = {}

        # Reads the table pointers
        for i in range(1, 11):
            self.table_begin_pointers[i] = int.from_bytes(self.disk_file.read(8), "big")

        # Reads the C2 table and loads it into memory
        self.C2 = []
        self.disk_file.seek(self.table_begin_pointers[9])
        for _ in range((self.table_begin_pointers[10] - self.table_begin_pointers[9]) // (byte_align(self.k)//8) - 1):
            self.C2.append(parse_entry(self.disk_file.read(byte_align(self.k)//8), self.k, 0, 0, 0)[0])

    def __del__(self):
        self.close()

    def close(self):
        self.disk_file.close()

    def get_memo(self):
        return self.memo

    def get_id(self):
        return self.id

    def get_size(self):
        return self.k

    def read_line_point(self, table_index, position):
        """
        Reads a line point from the plot file, at the specified position in the given table.
        For example, reads the 385th entry of table 4. It first reads the correct park into
        disk, calculates the deltas up the the right park, computes the line_point, and returns
        """
        park_index = position // EPP  # We can calculate which park, by dividing by entries per park
        park_size = ProofOfSpaceDiskPlotter.calculate_park_size(self.k)
        self.disk_file.seek(self.table_begin_pointers[table_index] + (park_size // 8) * park_index)

        # Parks start with a 2*k bit line point (checkpoint)
        line_point_size = byte_align(self.k * 2)
        line_point_bin = self.disk_file.read(line_point_size // 8)
        line_point = BitArray(uint=int.from_bytes(line_point_bin, "big"),
                              length=line_point_size)[:self.k*2].uint

        # Parks then contain EPP-1 k-bit stub entries (Except for final park)
        stubs_bin = BitArray(uint=int.from_bytes(self.disk_file.read(byte_align((EPP - 1) * self.k)
                                                                     // 8), "big"),
                             length=byte_align((EPP-1) * self.k))
        stubs = [0] + [stubs_bin[i*self.k:(i+1)*self.k].uint for i in range(EPP-1)]

        # The rest of park is composed of the deltas. The first stub and delta are not included,
        # since this information would be redundant with checkpoint.
        rest_of_park_size = park_size - byte_align(self.k * 2) - byte_align(EPP * (self.k-1))
        rest_of_park = self.disk_file.read(rest_of_park_size // 8)

        # The stubs are stored normally, but deltas are encoded due to their distribution
        deltas = arithmetic_decode_deltas(BitArray(uint=int.from_bytes(rest_of_park, "big"),
                                                   length=rest_of_park_size), EPP-1)
        deltas = [0] + deltas

        # Computes the value at position "position" using deltas, stubs, and checkpoint
        sum_deltas = sum([deltas[i] for i in range(min(len(deltas), (position % EPP) + 1))])
        sum_stubs = sum([stubs[i] for i in range(min(len(deltas), (position % EPP) + 1))])
        big_delta = (2**self.k) * sum_deltas + sum_stubs
        final_line_point = line_point + big_delta
        return final_line_point

    def get_qualities_for_challenge(self, challenge):
        """
        Returns quality BitArray.
        challenge is an arbitrary 256bit string generated from the blockchain. The purpose
        of the quality is to create a random metric by which proofs of space can be judged.
        Since the quality depends on the last bits of the challenge, the prover must actually
        recover 2 of the 64 total proof values (x values), based on these random bits, and
        it's not practical to store them all on disk. This requires the prover to have most
        (or the whole) plot to calculate their proof of space qualities.
        """

        # Gets all the p7 entries (each which corresponds to one solution, and one quality)
        # P7 is the last of the large tables. These are positions into table 6, which
        # contains a pair of entries, each positions to table 5, etc.
        p7_entries = self.get_p7_entries(challenge)

        if len(p7_entries) == 0:
            return []
        qualities = []
        last_5_bits = BitArray(uint=(challenge & 31), length=5)

        for i in range(len(p7_entries)):
            position = p7_entries[i].uint

            # Backpropages through the tables, going left or right based on
            # bits of the challenge. Only requires ~7 disk lookups, since we are
            # following one branch instead of all of them.
            for table_index in range(6, 1, -1):
                line_point = self.read_line_point(table_index, position)

                # Converts the line point (size ~2k) into two k bit integers, each
                # being a position in the previous table.
                x, y = line_point_to_square(line_point)
                assert(x >= y)

                # If the bit is 0, we go left (y), which is the smaller value. If it is 1, we
                # go right (x).
                if last_5_bits[(7 - table_index) - 1] == 0:
                    position = y
                else:
                    position = x

            # Finally, reads the final two values from table 1, which are two proof values stored together.
            line_point = self.read_line_point(1, position)
            x1, x2 = line_point_to_square(line_point)

            # The quality is 2 out of the total 64 leaves of the proof of space.
            qualities.append(BitArray(uint=x2, length=self.k) + BitArray(uint=x1, length=self.k))
        return qualities

    def get_p7_entries(self, challenge):
        """
        Retrieves the p7 entries corresponding to a solution to the challenge.
        This does not fetch the proofs of space, which would require backpropagation to to
        p1 entries, and thus more seeks. It uses the correct checkpoints to find where in p7
        to look for the entries.
        """

        if len(self.C2) == 0:
            return []

        f7 = BitArray(uint=challenge, length=256)[:self.k]

        # Looks through the checkpoints stored in memory (C2 table) to find the C1 index to
        # search at
        c1_index = 0
        broke = False
        for c2_entry in self.C2:
            if f7.uint < c2_entry.uint:
                # If we pass our target f7 value, we are done.
                c1_index -= checkpoint_2_interval
                broke = True
                break
            c1_index += checkpoint_2_interval

        # No entries found, first C2 checkpoint is after our target entry.
        if c1_index < 0:
            return []

        # We are at the last C2 checkpoint
        if not broke:
            c1_index -= checkpoint_2_interval

        self.disk_file.seek(self.table_begin_pointers[8] + c1_index * byte_align(self.k) // 8)
        c1_entries_bytes = self.disk_file.read(checkpoint_1_interval * byte_align(self.k) // 8)

        # Seaches through C1 to find the correct place within C3 to search for.
        curr_f7 = c2_entry.uint
        prev_f7 = c2_entry.uint
        broke = False

        for start in range(0, len(c1_entries_bytes), byte_align(self.k) // 8):
            c1_entry_bytes = c1_entries_bytes[start:start + byte_align(self.k) // 8]
            read_f7 = parse_entry(c1_entry_bytes, self.k, 0, 0, 0)[0].uint

            # TODO: check edge cases
            if start != 0 and read_f7 == 0:
                break
            curr_f7 = read_f7

            if f7.uint < curr_f7:
                # If we pass our target f7 value, we are done, so back up by one
                curr_f7 = prev_f7
                c1_index -= 1
                broke = True
                break

            c1_index += 1
            prev_f7 = curr_f7
        if not broke:
            # If we reached the end of the C1 block of entries without breaking, decrement
            c1_index -= 1

        # Reads a block of bits from C3, corresponding to all the p7 entries between the two
        # C1 entries c1_index and c1_index + 1. Bits determine a map from p7_position to f7.
        c3_entry_size = ProofOfSpaceDiskPlotter.calculate_c3_size(self.k)

        # Entry is in two bitfields, so we need to find out the previous thing
        double_entry = f7.uint == curr_f7 and c1_index > 0
        if double_entry:
            c1_index -= 1
            self.disk_file.seek(self.table_begin_pointers[8] + c1_index * byte_align(self.k + 1) // 8)
            c1_entry_bytes = self.disk_file.read(byte_align(self.k) // 8)
            next_f7 = curr_f7
            curr_f7 = parse_entry(c1_entry_bytes, self.k, 0, 0, 0)[0].uint

            self.disk_file.seek(self.table_begin_pointers[10] + c1_index * c3_entry_size)
            bit_mask = BitArray(self.disk_file.read(c3_entry_size))
            next_bit_mask = BitArray(self.disk_file.read(c3_entry_size))

        else:
            self.disk_file.seek(self.table_begin_pointers[10] + c1_index * c3_entry_size)
            bit_mask = BitArray(self.disk_file.read(c3_entry_size))

        curr_p7_pos = c1_index * checkpoint_1_interval - 1
        p7_positions = []
        p7_tmp_positions = []

        # Now iterate through the rest of the bits (the first is a 0, for the checkpoint)
        bit_index = 0
        while f7.uint >= curr_f7:
            bit = bit_mask[bit_index]
            if curr_p7_pos >= ((c1_index + 1) * checkpoint_1_interval) - 1:
                if double_entry:
                    c1_index += 1
                    bit_mask = next_bit_mask
                    curr_f7 = next_f7
                    bit_index = 0
                    double_entry = False
                    continue

                # Stop when we have read checkpoint_1_interval entries
                break
            if bit == 0:
                # A 0 means that there is an entry, with the current f7.
                curr_p7_pos += 1
                if curr_f7 == f7.uint:
                    p7_tmp_positions.append(curr_p7_pos)
            if bit == 1:
                # A 1 means increment the current f by one, because we are done seeing elements
                # with the previous f7.
                curr_f7 += 1
                p7_positions += p7_tmp_positions
                p7_tmp_positions = []
            bit_index += 1

        # There are no proofs of space for this challenge.
        if len(p7_positions) == 0:
            return []

        entry_size_bytes = byte_align(self.k + 1) // 8
        self.disk_file.seek(self.table_begin_pointers[7] + p7_positions[0] * entry_size_bytes)

        p7_entries = []
        # Calculate the last minus the first, +1, to see how many p7 entries to read
        for _ in range(p7_positions[-1] - p7_positions[0] + 1):
            p7_entries.append(BitArray(uint=int.from_bytes(self.disk_file.read(entry_size_bytes), "big"),
                                       length=entry_size_bytes*8)[:self.k+1])

        return p7_entries

    def get_full_proof(self, challenge, index):
        """
        Returns 'proof', an 8*k byte value, composed of 64 k bit x values.
        Takes in the challenge (from previous block), and the index of the proof,
        since the plot can have more than one valid proof of space.
        Index comes from previous results of a get_qualities_for_challenge() call
        """

        # Goes through the C tables and finds all P7 entries which match
        # with the challenge
        p7_entries = self.get_p7_entries(challenge)
        xs = self.get_inputs(p7_entries[index].uint, 6)

        # This reordering has to be done, because the proof fetches from disk is in
        # disk ordering. This means that it's stored to minimize space, but not stored
        # in proof order.
        xs_sorted = self.reorder_proof(xs)
        return sum(xs_sorted).tobytes()

    def reorder_proof(self, xs):
        """
        Takes in a 64 element array (xs). Returns with proof ordering, instead of disk
        ordering. Proof order can be recovered by trying the f functions on pairs of
        entries, and seeing which ones match.
        """

        f1 = F1Calculator(self.k, self.id)
        results = [f1.calculate_bucket(xs[i]) for i in range(64)]
        xs = [r[1] for r in results]

        # Disk ordering just changes the order of pairs of entries during plotting. For example,
        # and entry (e1, e2) might be stored as (e1, e2) or (e2, e1). Therefore, we must try
        # both options to see which one creates a match in the next table.
        for table_index in range(2, 8):
            new_xs = []
            new_results = []
            f = FxCalculator(self.k, table_index, self.id)
            for i in range(0, len(results), 2):
                if results[i][0].uint < results[i+1][0].uint:
                    new_y = f.f(results[i][1], results[i+1][1]) ^ results[i][0]
                    new_metadata = f.compose(results[i][1], results[i+1][1])
                    new_xs.append((xs[i], xs[i+1]))
                else:
                    new_y = f.f(results[i+1][1], results[i][1]) ^ results[i+1][0]
                    new_metadata = f.compose(results[i+1][1], results[i][1])
                    new_xs.append((xs[i+1], xs[i]))

                assert(new_y is not None)
                new_results.append((new_y, new_metadata))
            results = new_results
            xs = new_xs

        for i in range(6):
            xs = list(sum(xs, ()))
        return xs

    def get_inputs(self, position, depth):
        """
        Recursive function to go through tables P6 to P1, and get all proof values.
        """

        line_point = self.read_line_point(depth, position)
        x, y = line_point_to_square(line_point)

        if depth == 1:
            # The two leaf values are stored together. Therefore, there are 32 pairs
            # of two x values that are fetched for this proof.
            return [BitArray(uint=y, length=self.k), BitArray(uint=x, length=self.k)]
        else:
            return self.get_inputs(y, depth - 1) + self.get_inputs(x, depth - 1)
