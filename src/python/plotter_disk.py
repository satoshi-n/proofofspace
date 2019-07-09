import math
import os
from bitstring import BitArray

from .calculate_bucket import F1Calculator, FxCalculator, EXTRA_BITS, vector_lens, BC_param
from .sort_on_disk import extract_num, sort_on_disk, safearray
from .utils import byte_align, parse_entry, square_to_line_point, arithmetic_encode_deltas

id_len = 32
# min_plot_size = 35
min_plot_size = 9
max_plot_size = 60
memory_size = 1000000
num_sort_buckets = 16
offset_size = 8
checkpoint_1_interval = 10000   # Every _ B entries, there is a checkpoint in C1
checkpoint_2_interval = 10000   # Every _ C1 entries, there is a checkpoint in C2
format_desc = "alpha-v0.1".encode("utf-8")

# f-table batch sizes, f will be computed 2^batch_sizes evaluations at a time
batch_sizes = 8

# Entries per park
EPP = 256

# The average delta size for a park, will not exceed this number of bits. This can be lower,
# if EPP is increased
max_average_delta = 2


# Disk interface required for sort_on_disk
class Disk():
    def __init__(self, f):
        self.file = f

    def read(self, begin, memcache, length):
        self.file.seek(begin)
        memcache[:length] = self.file.read(length)

    def write(self, begin, memcache, length):
        self.file.seek(begin)
        self.file.write(memcache[:length])

    def read_handle(self, begin):
        self.file.seek(begin)
        return self.file

    def write_handle(self, begin):
        self.file.seek(begin)
        return self.file


class ProofOfSpaceDiskPlotter():
    @staticmethod
    def calculate_c3_size(k):
        # TODO: remove when bigger plots
        if k < 20:
            return byte_align(8 * checkpoint_1_interval) // 8
        else:
            # TODO(alex): tighten this bound, based on formula
            return byte_align(4 * checkpoint_1_interval) // 8

    @staticmethod
    def calculate_park_size(k):
        return byte_align(2*k) + byte_align((EPP - 1) * max_average_delta) + byte_align((EPP - 1) * k)

    def __init__(self):
        self.i = 5

    def write_header(self, plot_file, memo, id, k):
        # 19 bytes  - "Proof of Space Plot" (utf-8)
        # 32 bytes  - unique plot id
        # 1 byte    - k
        # 2 bytes   - format description length
        # x bytes   - format description
        # 2 bytes   - memo length
        # x bytes   - memo
        total_written = 0
        total_written += plot_file.write("Proof of Space Plot".encode("utf-8"))
        assert(total_written == 19)

        total_written += plot_file.write(id)
        # Write k into one byte
        total_written += plot_file.write((k).to_bytes(1, "big"))

        # Write format description and memo
        total_written += plot_file.write(len(format_desc).to_bytes(2, byteorder="big"))
        total_written += plot_file.write(format_desc)
        total_written += plot_file.write(len(memo).to_bytes(2, byteorder="big"))
        total_written += plot_file.write(memo)

        # 80 bytes of 0s, representing c1egin to p1begin
        total_written += plot_file.write(bytes((10 * 8) * [0]))

        assert(total_written == 19 + id_len + 1 + 2 + len(format_desc) + 2 + len(memo) + 10*8)
        return total_written

    def write_plot_file(self, plot_filename, k, id, memo):
        """
        The purpose of the forward propagation phase, is to generate all possible matches in the tables,
        and compute all proofs of space. This requires significantly more space than the final plot will
        use, and also is the slowest step, since matches must be computed.
        """

        # Open with read + write, in binary mode
        plot_file = open(plot_filename, "r+b")

        # Write the file header
        # memo id size c1begin c2begin c3begin p7begin p6begin p5begin p4begin p3begin p2begin p1begin
        header_size = self.write_header(plot_file, memo, id, k)

        f1 = F1Calculator(k, id)
        # Now we can start to write table 1
        x = BitArray(uint=0, length=k)

        bucket_sizes = [0] * num_sort_buckets
        right_bucket_sizes = [0] * num_sort_buckets

        num_buckets = (2**(k + EXTRA_BITS)) // BC_param + 1

        print("Computing table 1")
        for _ in range(2**(k-batch_sizes) + 1):
            for y, L in f1.calculate_buckets(x, 2**batch_sizes):
                # Writes y + L to the file, for every consecutive L
                (y + L).tofile(plot_file)

                bucket_sizes[extract_num(y.tobytes(), 0, int(math.log(num_sort_buckets, 2)))] += 1

                if x.uint + 1 > 2**k - 1:
                    # If we are overflowing the BitArray, we're done
                    break

                x = BitArray(uint=x.uint + 1, length=k)
            if x.uint + 1 > 2**k - 1:
                break

        # One blank entry to mark the end of the table
        BitArray(uint=0, length=byte_align(2*k)).tofile(plot_file)

        # This is the offset from the start, due to the header
        begin_byte = header_size

        total_table_entries = 2**k

        # Used later, for backpropagation
        plot_table_begin_pointers = {1: begin_byte}

        # Allow enough space in case we have more than 2**k entries
        pos_size = k + 1

        # For tables 1 through 6, sort the table, calculate matches, and write
        # the next table
        for table_index in range(1, 7):
            metadata_size = vector_lens[table_index + 1] * k
            right_metadata_size = vector_lens[table_index + 2] * k
            if table_index == 1:
                # y, L
                entry_size_bits = byte_align(k + EXTRA_BITS + metadata_size)
            else:
                # y, offset, pos, L, R
                entry_size_bits = byte_align(k + EXTRA_BITS + pos_size + offset_size + metadata_size)

            right_entry_size_bits = byte_align(k + EXTRA_BITS + pos_size + offset_size + right_metadata_size)
            if table_index + 1 == 7:
                # For the last table, we need some extra bits, for the compression phase. For other tables,
                # metadata is written so the size will be greater than k+3*pos_size.
                num_bits_required = byte_align((k + 3 * pos_size))
                right_entry_size_bits = max(byte_align(k + pos_size + offset_size + right_metadata_size),
                                            num_bits_required)

            # End of this table, beginning of the next. One 0 entry in between
            begin_byte_next = begin_byte + ((entry_size_bits // 8) * (total_table_entries + 1))

            # Now start counting from 0, the number of entries in the next table
            total_table_entries = 0

            print("Sorting table", table_index)
            # Sorts by y, pos
            sort_on_disk(Disk(plot_file), begin_byte, begin_byte_next, entry_size_bits // 8,
                         0, bucket_sizes, safearray(memory_size))

            # plot_file_2 will be used for writing the next table while we iterate through the current
            # plot_file reads and writes from the left table, and plot_file_2 reads and writes from the
            # right table.
            plot_file.seek(begin_byte)

            plot_file_2 = open(plot_filename, "r+b")
            plot_file_2.seek(begin_byte_next)
            print("Computing table", table_index + 1, "at position", hex(begin_byte_next))

            # Start calculating the next table
            f = FxCalculator(k, table_index + 1, id)

            # We will compare each thing in bucket_L with each thing in bucket_R
            # bucket_L = []
            # bucket_R = []
            bucket_L = []
            bucket_R = []

            # Start at left table pos = 0 and iterate through the whole table. Note that the left table
            # will already be sorted by y (curr_bucket).
            curr_bucket = 0
            pos = 0
            end_of_table = False
            num_matches = 0
            while not end_of_table:
                # Read each y and x
                left_entry = plot_file.read(entry_size_bits//8)
                if table_index == 1:
                    y, _, _, metadata = parse_entry(left_entry, k + EXTRA_BITS, 0, 0, metadata_size)
                else:
                    y, _, _, metadata = parse_entry(left_entry, k + EXTRA_BITS, pos_size, offset_size, metadata_size)

                end_of_table = (y.uint == 0 and metadata.uint == 0)  # Technically incorrect, but statistically safe

                if not end_of_table and table_index == 1:
                    # If we haven't hit the end of table marker
                    f1.reload_key()
                    assert(f1.calculate_bucket(metadata)[0] == y)
                    f.reload_key()

                read_bucket = y.uint // BC_param
                # Keep reading left table entries into bucket_L and R, until we run out of things
                if read_bucket == curr_bucket:
                    bucket_L.append((y, metadata, pos))
                elif read_bucket == curr_bucket + 1:
                    bucket_R.append((y, metadata, pos))
                else:
                    # This is reached when we have finished adding stuff to bucket_R and bucket_L,
                    # so now we can compare every pair and try to find matches. If therbucket match,
                    # it is written to the right table.
                    if len(bucket_L) > 0 and len(bucket_R) > 0:

                        matches = f.find_matches([x[0] for x in bucket_L], [x[0] for x in bucket_R], k)

                        for offset_L, offset_R in matches:
                            y1, L, pos1 = bucket_L[offset_L]
                            y2, R, pos2 = bucket_R[offset_R]
                            newY = f.f(L, R) ^ y1
                            newL = f.compose(L, R)

                            if table_index + 1 == 7:
                                newY = newY[:k]
                            new_pos = BitArray(uint=pos1, length=pos_size)
                            new_offset = BitArray(uint=(pos2-pos1), length=offset_size)
                            num_matches += 1
                            total_table_entries += 1
                            right_entry = newY + new_pos + new_offset + newL

                            # Makes sure to write the exact number of bits (right_entry_size_bits)
                            if byte_align(len(right_entry)) < right_entry_size_bits:
                                right_entry += BitArray(uint=0, length=right_entry_size_bits -
                                                        byte_align(len(right_entry)))
                            right_entry.tofile(plot_file_2),
                            right_bucket_sizes[extract_num(right_entry.tobytes(), 0,
                                                           int(math.log(num_sort_buckets, 2)))] += 1

                    # Now we have seen something which is further than L and R
                    if read_bucket == curr_bucket + 2:
                        bucket_L = bucket_R
                        bucket_R = [(y, metadata, pos)]
                        curr_bucket += 1
                    else:
                        bucket_L = [(y, metadata, pos)]
                        bucket_R = []
                        curr_bucket = read_bucket
                pos += 1

            print("Total matches:", str(num_matches) + ".", "Per bucket:", (num_matches / num_buckets))

            # 8 bytes each pointer. Points to the beginning of each table
            plot_file.seek(header_size - 8 * (12 - table_index))
            BitArray(uint=begin_byte_next, length=8*8).tofile(plot_file)
            plot_table_begin_pointers[table_index + 1] = begin_byte_next

            begin_byte = begin_byte_next
            bucket_sizes = right_bucket_sizes

            right_bucket_sizes = [0] * num_sort_buckets
            BitArray(uint=0, length=right_entry_size_bits).tofile(plot_file_2)

            plot_file_2.close()

        plot_file.close()

        for key, value in plot_table_begin_pointers.items():
            print("     Table", key, hex(value))

        return plot_table_begin_pointers

    def write_C_tables(self, k, pos_size, final_file_writer_1, final_file_writer_2, final_file_writer_3,
                       plot_file_reader, plot_table_begin_pointers, final_table_begin_pointers, total_f7_entries,
                       right_entry_size_bits):
        """
        Writes the checkpoint tables. The purpose of these tables, is to store a list of ~2^k values
        of size k (the proof of space outputs from table 7), in a way where they can be looked up for
        proofs, but also efficiently. To do this, we assume table 7 is sorted by f7, and we write the
        deltas between each f7 (which will be mostly 1s and 0s), with a variable encoding scheme (C3).
        Furthermore, we create C1 checkpoints along the way.  For example, every 10,000 f7 entries,
        we can have a C1 checkpoint, and a C3 delta encoded entry with 10,000 deltas.

        Since we can't store all the checkpoints in
        memory for large plots, we create checkpoints for the checkpoints (C2), that are meant to be
        stored in memory during proving. For example, every 10,000 C1 entries, we can have a C2 entry.

        The final table format for the checkpoints will be:
        C1 (checkpoint values)
        C2 (checkpoint values into)
        C3 (deltas of f7s between C1 checkpoints)
        """

        # Ceiling because we have 1 for every checkpoint_1_interval entries, and add one for the EOT
        # marker.
        begin_byte_C1 = final_table_begin_pointers[7] + (total_f7_entries * byte_align(pos_size)//8)
        total_C1_entries = math.ceil(total_f7_entries / checkpoint_1_interval)
        begin_byte_C2 = begin_byte_C1 + (total_C1_entries + 1) * (byte_align(k) // 8)
        total_C2_entries = math.ceil(total_C1_entries / checkpoint_2_interval)
        begin_byte_C3 = begin_byte_C2 + (total_C2_entries + 1) * (byte_align(k) // 8)

        size_C3 = self.calculate_c3_size(k)

        final_table_begin_pointers[8] = begin_byte_C1
        final_table_begin_pointers[9] = begin_byte_C2
        final_table_begin_pointers[10] = begin_byte_C3

        plot_file_reader.seek(plot_table_begin_pointers[7])
        final_file_writer_1.seek(begin_byte_C1)
        final_file_writer_2.seek(begin_byte_C3)
        final_file_writer_3.seek(final_table_begin_pointers[7])

        prev_y = 0
        C2 = []
        num_C1_entries = 0
        to_write = BitArray()
        print("Starting to write C1 and C3 tables")
        for f7_position in range(total_f7_entries):
            entry = plot_file_reader.read(right_entry_size_bits // 8)[:byte_align(k + 2*pos_size)//8]

            # If this is a checkpoint for C1, writes it
            y, _, new_pos, _ = parse_entry(entry, k, pos_size, pos_size, 0)

            # Writes the position to the P7 table
            # TODO(mariano): Don't byte align here, bit align instead (wasting some bits)
            new_pos.tofile(final_file_writer_3)

            if f7_position % checkpoint_1_interval == 0:
                y.tofile(final_file_writer_1)

                if num_C1_entries > 0:  # No writing for the first checkpoint, because we haven't read entries yet

                    # Writes the bit mask for entries between previous and current checkpoints
                    final_file_writer_2.seek(begin_byte_C3 + (num_C1_entries - 1) * size_C3)
                    print("Writing c3 enty...index:", num_C1_entries)
                    assert(len(to_write) <= size_C3 * 8)
                    to_write.tofile(final_file_writer_2)

                # If this is a checkpoint for C2, adds it to the list
                if f7_position % (checkpoint_1_interval * checkpoint_2_interval) == 0:
                    C2.append(y)
                to_write = BitArray()
                num_C1_entries += 1
            else:
                # Writes mask bits to C3
                if y.uint == prev_y:
                    to_write += BitArray(uint=0, length=1)
                else:
                    difference = y.uint - prev_y
                    assert difference > 0
                    to_write += BitArray(uint=2**(difference) - 1, length=difference + 1)
            prev_y = y.uint

        # Writes C3 entry (bit mask) after the final checkpoint
        if len(to_write) != 0:
            print("Writing final park at", begin_byte_C3 + (num_C1_entries - 1) * size_C3)
            print("Writing ", size_C3, "bytes")
            to_write += BitArray(uint=0, length=size_C3*8 - len(to_write))
            final_file_writer_2.seek(begin_byte_C3 + (num_C1_entries - 1) * size_C3)

            to_write.tofile(final_file_writer_2)

        # End of table C1
        BitArray(uint=0, length=k).tofile(final_file_writer_1)
        assert(total_C1_entries == num_C1_entries)
        print("Finished writing C1 and C3 tables")
        print("Writing C2 table")

        # Writes all the C2 entries to the file
        for C2_entry in C2:
            C2_entry.tofile(final_file_writer_1)

        # End of table C2
        BitArray(uint=0, length=k).tofile(final_file_writer_1)
        print("Finished writing C2 table")

    def backpropagate(self, k, plot_table_begin_pointers, plot_filename, id, memo):
        """
        The backpropagation step goes from table 7 to table 1, seeing which entries in each
        table did not lead to any matches in the next table. These entries are dropped,
        and all positions are updated. Furthermore, useless metadata is dropped.
        """

        print("First pass of plotting done")
        print("Starting backpropagation step....")

        # Allocate an extra bit for positions, because tables can exceed 2^k entries
        pos_size = k + 1

        # After pruning, we don't need the extra bit, so we store pointers in k bits
        final_pos_size = k

        bucket_sizes_pos = [0] * num_sort_buckets

        # This is the spare pointer, where temporary data will be written while sorting on disk.
        # This is set to a pointer after all the tables, so we don't override anything
        spare_pointer = 8 * byte_align(pos_size + offset_size) * (2**k)

        # This algorithm iterates through two tables at a time, a left table and a right table.
        # each table will have it's own read and write heads that point to entries in that table.
        for table_index in range(7, 1, -1):
            left_reader = open(plot_filename, "rb")
            left_writer = open(plot_filename, "r+b")
            right_reader = open(plot_filename, "rb")
            right_writer = open(plot_filename, "r+b")
            print("Backpropagating on table", table_index)

            # These will be used for sorting by pos
            new_bucket_sizes_pos = [0] * num_sort_buckets

            left_metadata_size = vector_lens[table_index] * k
            if table_index > 2:
                left_entry_size_bits = byte_align(k + EXTRA_BITS + pos_size + offset_size + left_metadata_size)
            else:
                # If this is the first table (table_index = 2), there are no pos and offsets
                left_entry_size_bits = byte_align(k + EXTRA_BITS + left_metadata_size)

            # Metadata will not be rewritten to the right table
            right_y_size = k

            new_left_entry_size_bits = max(byte_align(k + EXTRA_BITS + pos_size + offset_size),
                                           byte_align((k + EXTRA_BITS + 2 * final_pos_size + pos_size)))

            right_num_bits_required = byte_align((k + EXTRA_BITS + 2 * final_pos_size + pos_size))
            if table_index == 7:
                # For the last table, our ys are smaller (k instead of K+EXTRA_BITS), but our positions
                # are bigger. Since we can have over 2**k entries, we need an extra bit.
                right_num_bits_required = byte_align((k + 3 * pos_size))
            else:
                # For table 7, y is k bits.
                # For all other tables, y is larger
                right_y_size += EXTRA_BITS
            right_entry_size_bits = max(byte_align(right_y_size + pos_size + offset_size), right_num_bits_required)

            if table_index != 7:
                # Sorts right table by position, so we can iterate through it at the same time as
                # iterating through the left table. Table 7 is already sorted by pos.
                print("\tSorting table", table_index)

                sort_plot_file = open(plot_filename, "r+b")
                sort_on_disk(Disk(sort_plot_file), plot_table_begin_pointers[table_index], spare_pointer,
                             right_entry_size_bits // 8, right_y_size, bucket_sizes_pos, safearray(memory_size))
                sort_plot_file.close()

            left_reader.seek(plot_table_begin_pointers[table_index - 1])
            left_writer.seek(plot_table_begin_pointers[table_index - 1])
            right_reader.seek(plot_table_begin_pointers[table_index])
            right_writer.seek(plot_table_begin_pointers[table_index])

            print("\tReading R table:", table_index, "at address", hex(plot_table_begin_pointers[table_index]))

            cached_positions_size = 1024
            half_false = [False] * (cached_positions_size // 2)
            used_positions = half_false + half_false

            # When we iterate through the right table, we read a larger pos, and thus we cache it until
            # current_pos reaches that pos
            should_read_entry = True
            cached_entry = (None, None, None)

            left_entry_counter = 0

            # The left entries will be rewritten without the useless ones, so their positions
            # will change.
            new_positions = {}

            # The read pointer will be this much ahead of the write pointer, in number
            # of entries (for the right table).
            read_minus_write = 256

            old_ys = [[]] * read_minus_write
            old_offsets = [[]] * read_minus_write
            end_of_right_table = False

            current_pos = 0        # The current pos of the left tale
            end_of_table_pos = 0   # Position at which the left table ends
            greatest_pos = 0       # Greatest position we have seen
            while not end_of_right_table or (current_pos - end_of_table_pos <= read_minus_write):
                # Resets the cached data of (current_pos - 256) so we can use them for current_pos
                old_offsets[current_pos % read_minus_write] = []
                old_ys[current_pos % read_minus_write] = []

                # Resets the used_positions that we will not need anymore, so we can use them
                # for future entries
                if current_pos % (cached_positions_size // 2) == 0:
                    if current_pos % cached_positions_size == 0:
                        used_positions[(cached_positions_size // 2):] = half_false
                    else:
                        used_positions[:(cached_positions_size // 2)] = half_false

                # Continues until the end of the left table.
                if not end_of_right_table or current_pos <= greatest_pos:
                    # Reads all the entries that have position == current_pos
                    while not end_of_right_table:
                        if should_read_entry:
                            entry = right_reader.read(right_entry_size_bits // 8)[:byte_align(right_y_size + pos_size
                                                                                              + offset_size)//8]
                            y, pos, offset, _ = parse_entry(entry, right_y_size, pos_size, offset_size, 0)
                        elif cached_entry[1].uint == current_pos:
                            y, pos, offset = cached_entry
                        else:
                            break

                        should_read_entry = True
                        if pos.uint + offset.uint > greatest_pos:
                            greatest_pos = pos.uint + offset.uint
                        if y.uint == 0 and pos.uint == 0 and offset.uint == 0:
                            # We have read the end of table marker
                            end_of_right_table = True
                            end_of_table_pos = current_pos
                            break
                        elif pos.uint == current_pos:
                            # Have read an entry with the current position
                            used_positions[pos.uint % cached_positions_size] = True
                            used_positions[(pos.uint + offset.uint) % cached_positions_size] = True

                            # Actually stores positions instead of offsets
                            old_ys[pos.uint % read_minus_write].append(y)
                            old_offsets[pos.uint % read_minus_write].append(pos.uint + offset.uint)
                        else:
                            # We have read an entry with the next pos, so break
                            should_read_entry = False
                            cached_entry = (y, pos, offset)
                            break

                    # Reads an entry from the left table, and if it's used in the right table, it rewrites it
                    entry = left_reader.read(left_entry_size_bits // 8)
                    if table_index > 2:
                        y, pos, offset, metadata = parse_entry(entry, k + EXTRA_BITS, pos_size,
                                                               offset_size, left_metadata_size)
                    else:
                        y, pos, offset, metadata = parse_entry(entry, k + EXTRA_BITS, 0, 0, left_metadata_size)
                    if used_positions[current_pos % cached_positions_size]:
                        if table_index > 2:
                            new_left_entry = (y + pos + offset)
                            if byte_align(len(new_left_entry)) < new_left_entry_size_bits:
                                new_left_entry += BitArray(uint=0, length=new_left_entry_size_bits -
                                                           byte_align(len(new_left_entry)))
                        else:
                            new_left_entry = (y + metadata)

                        new_left_entry.tofile(left_writer)

                        new_bucket_sizes_pos[extract_num(new_left_entry.tobytes(), k + EXTRA_BITS,
                                                         int(math.log(num_sort_buckets, 2)))] += 1

                        # TODO optimize to a list instead of map
                        new_positions[current_pos] = left_entry_counter
                        left_entry_counter += 1

                # The old left position of where the write pointer currently is
                write_pointer_pos = current_pos - read_minus_write + 1

                # Rewrite the right table entries with updated positions
                if write_pointer_pos >= 0 and write_pointer_pos in new_positions:
                    # We will rewrite the entries that have pos == current_pos - 256. So when
                    # pos == 256, we rewrite entries with pos 0
                    new_pos = new_positions[write_pointer_pos]
                    new_pos_bin = BitArray(uint=new_pos, length=pos_size)
                    for counter in range(len(old_offsets[write_pointer_pos % read_minus_write])):
                        new_offset_pos = new_positions[old_offsets[write_pointer_pos % read_minus_write][counter]]
                        new_offset_bin = BitArray(uint=new_offset_pos-new_pos, length=offset_size)
                        new_y = old_ys[write_pointer_pos % read_minus_write][counter]
                        to_write = (new_y + new_pos_bin + new_offset_bin)
                        if byte_align(len(to_write)) < right_entry_size_bits:
                            to_write = to_write + BitArray(uint=0, length=(right_entry_size_bits -
                                                                           byte_align(len(to_write))))
                        to_write.tofile(right_writer)
                current_pos += 1

            # End of table markers
            BitArray(uint=0, length=right_entry_size_bits).tofile(right_writer)
            BitArray(uint=0, length=(new_left_entry_size_bits)).tofile(left_writer)

            left_reader.close()
            left_writer.close()
            right_reader.close()
            right_writer.close()

            bucket_sizes_pos = new_bucket_sizes_pos
        print("Finished backpropagation")

    def write_park_to_file(self, writer, table_start, park_index, park_size_bytes, first_line_point,
                           park_deltas, park_stubs, k):
        """
        This writes a number of entries into a file, in the final, optimized format. The park contains
        a checkpoint value (whicch is a 2k bits line point), as well as EPP (entries per park) entries.
        These entries are each divded into stub and delta section. The stub bits are encoded as is, but
        the delta bits are optimized into a variable encoding scheme. Since we have many entries in each
        park, we can approximate how much space each park with take.
        Format is: [2k bits of first_line_point]  [EPP-1 stubs] [EPP-1 deltas]....  [first_line_point] ....
        """
        writer.seek(table_start + park_index * park_size_bytes)
        BitArray(uint=first_line_point, length=2*k).tofile(writer)
        park_stubs_bin = sum([BitArray(uint=j, length=k) for j in park_stubs])
        if len(park_stubs_bin) < (EPP - 1) * k:
            park_stubs_bin += BitArray(uint=0, length=((EPP-1)*k - len(park_stubs_bin)))
        park_stubs_bin.tofile(writer)
        arithmetic_encode_deltas(park_deltas).tofile(writer)

    def compress_tables(self, k, plot_table_begin_pointers, filename, plot_filename, id, memo):
        """
        Compresses the plot file tables into the final file. In order to do this, entries must be
        reorganized from the (pos, offset) bucket sorting order, to a more free line_point sorting
        order. In (pos, offset ordering), we store two pointers two the previous table, (x, y) which
        are very close together, by storing  (x, y-x), or (pos, offset), which can be done in about k + 8 bits,
        since y is in the next bucket as x. In order to decrease this, We store the actual entries from the
        previous table (e1, e2), instead of pos, offset pointers, and sort the entire table by (e1,e2).
        Then, the deltas between each (e1, e2) can be stored, which require around k bits.

        Converting into this format requires a few passes and sorts on disk. It also assumes that the
        backpropagation step happened, so there will be no more dropped entries. See the design
        document for more details on the algorithm.
        """

        print("Starting compression step.")
        header_writer = open(filename, "wb")

        # Write the file header
        header_size = self.write_header(header_writer, memo, id, k)

        # Allocate an extra bit for positions, because tables can exceed 2^k entries
        pos_size = k + 1

        # After pruning, we don't need the extra bit
        final_pos_size = k

        # Length of our parks
        park_size_bits = self.calculate_park_size(k)
        park_size_bytes = park_size_bits // 8

        # Table pointers for tables in the final plot file
        final_table_begin_pointers = {
            1: header_size
        }

        # header_writer = open(filename, "wb")
        header_writer.seek((header_size - 10*8))
        BitArray(uint=final_table_begin_pointers[1], length=8*8).tofile(header_writer)
        header_writer.close()

        # TODO(mariano): adjust this
        spare_pointer = 8 * byte_align(pos_size + offset_size) * (2**k)
        assert(spare_pointer > plot_table_begin_pointers[7])

        for table_index in range(1, 7):
            print("Compressing tables", table_index, "and", table_index + 1)
            left_reader = open(plot_filename, "rb")
            right_reader = open(plot_filename, "rb")
            right_writer = open(plot_filename, "r+b")

            bucket_sizes = [0] * num_sort_buckets

            left_y_size = k + EXTRA_BITS
            if table_index == 6:
                right_y_size = k
                right_new_pos_size = pos_size
            else:
                right_y_size = k + EXTRA_BITS
                right_new_pos_size = final_pos_size

            if table_index == 1:
                left_entry_size_bits = byte_align(left_y_size + k)
                left_entry_disk_size_bits = left_entry_size_bits
            else:
                left_entry_size_bits = byte_align(left_y_size + pos_size + final_pos_size)
                left_entry_disk_size_bits = max(byte_align(left_y_size + 2 * final_pos_size + pos_size),
                                                left_entry_size_bits)

            right_entry_disk_size_bits = byte_align(right_y_size + 2*right_new_pos_size + pos_size)

            left_reader.seek(plot_table_begin_pointers[table_index])
            right_reader.seek(plot_table_begin_pointers[table_index + 1])
            right_writer.seek(plot_table_begin_pointers[table_index + 1])

            cached_positions_size = 1024

            # When we iterate through the right table, we read a larger pos, and thus we cache it until
            # current_pos reaches that pos
            should_read_entry = True
            cached_entry = (None, None, None)

            left_new_pos = [0] * cached_positions_size

            # The read pointer will be this much ahead of the write pointer, in number
            # of entries (for the right table).
            read_minus_write = 256

            old_ys = [[]] * read_minus_write
            old_offsets = [[]] * read_minus_write
            end_of_right_table = False

            current_pos = 0        # The current pos of the left tale
            end_of_table_pos = 0   # Position at which the left table ends
            greatest_pos = 0       # Greatest position we have seen

            while not end_of_right_table or (current_pos - end_of_table_pos <= read_minus_write):
                # Resets the cached data of (current_pos - 256) so we can use them for current_pos
                old_offsets[current_pos % read_minus_write] = []
                old_ys[current_pos % read_minus_write] = []

                # Continues until the end of the left table.
                if not end_of_right_table or current_pos <= greatest_pos:
                    # Reads all the entries that have position == current_pos
                    while not end_of_right_table:
                        if should_read_entry:
                            entry = (right_reader.read(right_entry_disk_size_bits // 8)
                                     [:byte_align(right_y_size + pos_size + offset_size)//8])
                            y, pos, offset, _ = parse_entry(entry, right_y_size, pos_size, offset_size, 0)

                        elif cached_entry[1].uint == current_pos:
                            y, pos, offset = cached_entry
                        else:
                            break

                        should_read_entry = True
                        if pos.uint + offset.uint > greatest_pos:
                            greatest_pos = pos.uint + offset.uint
                        if y.uint == 0 and pos.uint == 0 and offset.uint == 0:
                            # We have read the end of table marker
                            end_of_right_table = True
                            end_of_table_pos = current_pos
                            break
                        elif pos.uint == current_pos:
                            # Have read an entry with the current position
                            # Actually stores positions instead of offsets
                            old_ys[pos.uint % read_minus_write].append(y)
                            old_offsets[pos.uint % read_minus_write].append(pos.uint + offset.uint)
                        else:
                            # We have read an entry with the next pos, so break
                            should_read_entry = False
                            cached_entry = (y, pos, offset)
                            break

                    # Reads an entry from the left table, only use the fist left_entry_size_bits bits
                    entry = left_reader.read(left_entry_disk_size_bits // 8)[:left_entry_size_bits//8]
                    if table_index == 1:
                        _, _, _, new_pos = parse_entry(entry, left_y_size, 0, 0, k)
                    else:
                        _, _, new_pos, _ = parse_entry(entry, left_y_size, pos_size, pos_size, 0)
                    left_new_pos[current_pos % cached_positions_size] = new_pos.uint

                # The old left position of where the write pointer currently is
                write_pointer_pos = current_pos - read_minus_write + 1

                if write_pointer_pos >= 0:
                    # We will rewrite the entries that have pos == current_pos - 256. So when
                    # pos == 256, we rewrite entries with pos 0
                    left_new_pos_1 = left_new_pos[write_pointer_pos % cached_positions_size]
                    left_new_pos_1_bin = BitArray(uint=left_new_pos_1, length=final_pos_size)

                    # We take the two old table_i positions, and see the new positions that they now have,
                    # after the new sorting. We then compress these two positions (x, y) into a line point,
                    # which is written in table_i+1, and it's what will be written to the final plot file.
                    # However, table_i+1 must first be sorted by line_point before we can store it efficiently
                    # in the final plot file.
                    for counter in range(len(old_offsets[write_pointer_pos % read_minus_write])):
                        left_new_pos_2 = left_new_pos[old_offsets[write_pointer_pos % read_minus_write][counter]
                                                      % cached_positions_size]
                        left_new_pos_2_bin = BitArray(uint=left_new_pos_2, length=final_pos_size)
                        new_y = old_ys[write_pointer_pos % read_minus_write][counter]

                        old_pos = BitArray(uint=write_pointer_pos, length=pos_size)
                        line_point = square_to_line_point(left_new_pos_1, left_new_pos_2)
                        assert(line_point.bit_length() <= len(left_new_pos_1_bin) + len(left_new_pos_2_bin))
                        line_point_bin = BitArray(uint=line_point, length=2*k)
                        to_write = (line_point_bin + new_y + old_pos)
                        assert(byte_align(len(to_write)) == right_entry_disk_size_bits)
                        to_write.tofile(right_writer)
                        bucket_sizes[extract_num(to_write.tobytes(), 0,
                                                 int(math.log(num_sort_buckets, 2)))] += 1

                current_pos += 1

            BitArray(uint=0, length=right_entry_disk_size_bits).tofile(right_writer)
            right_reader.close()
            right_writer.close()

            # Sorts table_i+1 by line point. Recall that each line_point is two positions to the previous table.
            # The most efficient way to store this list of line_points is to sort, and then store the deltas
            # between each one, using arithmetic encoding.
            sort_plot_file = open(plot_filename, "r+b")
            sort_on_disk(Disk(sort_plot_file), plot_table_begin_pointers[table_index + 1], spare_pointer,
                         right_entry_disk_size_bits // 8, 0, bucket_sizes, safearray(memory_size))
            sort_plot_file.close()

            right_reader_2 = open(plot_filename, "rb")
            right_writer_2 = open(plot_filename, "r+b")
            right_reader_2.seek(plot_table_begin_pointers[table_index + 1])
            right_writer_2.seek(plot_table_begin_pointers[table_index + 1])

            final_table_writer = open(filename, "r+b")
            final_table_writer.seek(final_table_begin_pointers[table_index])
            final_entries_written = 0

            new_bucket_sizes = [0] * num_sort_buckets
            park_deltas = []
            park_stubs = []
            checkpoint_line_point = 0
            last_line_point = 0
            park_index = 0

            # Iterate through right table, rewriting each entry as (f, old_pos, index). Also, every EPP
            # entries, as park is written to disk, with the final, compressed entries.
            for index in range(sum(bucket_sizes)):
                entry = right_reader_2.read(right_entry_disk_size_bits // 8)[:byte_align(2*k + right_y_size +
                                                                                         pos_size)//8]

                # Reads the line point (which we used for sorting), the f and old pos which will be used for
                # sorting later.
                line_point_bin, f, old_pos, _ = parse_entry(entry, 2 * k, right_y_size,
                                                            pos_size, 0)
                line_point = line_point_bin.uint

                # One extra bit for table 7, since we can have more than 2^k entries
                to_write = f + old_pos + BitArray(uint=index, length=k+1)

                # Makes sure to write enough data, since we will be rewriting over it in the next iteration
                if len(to_write) < right_entry_disk_size_bits:
                    to_write += BitArray(uint=0, length=right_entry_disk_size_bits - len(to_write))
                to_write.tofile(right_writer_2)

                new_bucket_sizes[extract_num(to_write.tobytes(), 0,
                                             int(math.log(num_sort_buckets, 2)))] += 1
                # Every EPP entries (Except for the first), a park is written.
                if index % EPP == 0:
                    if index != 0:
                        self.write_park_to_file(final_table_writer, final_table_begin_pointers[table_index],
                                                park_index, park_size_bytes, checkpoint_line_point, park_deltas,
                                                park_stubs, k)
                        park_index += 1
                        final_entries_written += (len(park_stubs) + 1)
                    park_deltas = []
                    park_stubs = []

                    checkpoint_line_point = line_point

                # Deltas between each point are divided into the delta and stub, for encoding reasons
                big_delta = line_point - last_line_point

                stub = big_delta % (2**k)
                small_delta = (big_delta - stub) >> k

                # Don't write the first, as it's redundant because of the checkpoint
                if index % EPP != 0:
                    park_deltas.append(small_delta)
                    park_stubs.append(stub)

                last_line_point = line_point

            right_reader_2.close()
            right_writer_2.close()

            # The last park did not get written to disk, so write it if it exists
            if len(park_deltas) > 0:
                self.write_park_to_file(final_table_writer, final_table_begin_pointers[table_index],
                                        park_index, park_size_bytes, checkpoint_line_point, park_deltas,
                                        park_stubs, k)
                final_entries_written += (len(park_stubs) + 1)

            print("\tWrote", final_entries_written, "entries")

            final_table_begin_pointers[table_index + 1] = (final_table_begin_pointers[table_index] +
                                                           (park_index + 1) * park_size_bytes)

            # Table 1 position is the first 8 bytes, table 2 is the next 8 bytes, etc.
            final_table_writer.seek(header_size - 8 * (10 - table_index))
            BitArray(uint=final_table_begin_pointers[table_index + 1], length=8*8).tofile(final_table_writer)

            final_table_writer.close()
            # Table_i+1 must be sorted by (y,pos), because in the next iteration, we will be iterating through
            # table_i+1 and table_i+2 simultaneously, and the positions in table_i+2 refer to a table
            # that was sorted by (y,pos).
            sort_plot_file = open(plot_filename, "r+b")
            sort_on_disk(Disk(sort_plot_file), plot_table_begin_pointers[table_index + 1], spare_pointer,
                         right_entry_disk_size_bits // 8, 0, new_bucket_sizes, safearray(memory_size))
            sort_plot_file.close()

        final_file_writer_1 = open(filename, "r+b")
        final_file_writer_2 = open(filename, "r+b")
        final_file_writer_3 = open(filename, "r+b")
        plot_file_reader = open(plot_filename, "rb")
        self.write_C_tables(k, pos_size, final_file_writer_1, final_file_writer_2,
                            final_file_writer_3, plot_file_reader, plot_table_begin_pointers,
                            final_table_begin_pointers, final_entries_written, right_entry_disk_size_bits)

        for key, value in final_table_begin_pointers.items():
            print("     Table", key, hex(value))

        # Write table pointers to the beginning of file
        final_file_writer_1.seek(header_size - 8 * (3))
        BitArray(uint=final_table_begin_pointers[8], length=8*8).tofile(final_file_writer_1)
        BitArray(uint=final_table_begin_pointers[9], length=8*8).tofile(final_file_writer_1)
        BitArray(uint=final_table_begin_pointers[10], length=8*8).tofile(final_file_writer_1)

        plot_file_reader.close()
        final_file_writer_1.close()
        final_file_writer_2.close()
        final_file_writer_3.close()

    # This method creates a plot on disk with the filename. A temporary file, "plotting" + filename,
    # is created and will be larger than the final plot file. This file is deleted at the end of
    # the process.
    def create_plot_disk(self, filename, k, memo, id):
        assert(len(id) == id_len)
        assert(k >= min_plot_size)
        assert(k <= max_plot_size)

        # TODO: check if file exists, if so, resume plotting
        plot_filename = filename + ".tmp"
        f = open(filename, "wb")
        f.close()
        f = open(plot_filename, "wb")
        f.close()

        # Write the tables 1 through 7 in the plot file. This is the forward_computation phase.
        plot_table_begin_pointers = self.write_plot_file(plot_filename, k, id, memo)

        # Backpropagation phase.
        self.backpropagate(k, plot_table_begin_pointers, plot_filename, id, memo)

        # Compression and C table creation phase.
        self.compress_tables(k, plot_table_begin_pointers, filename, plot_filename, id, memo)

        os.remove(plot_filename)
