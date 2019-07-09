from bitstring import BitArray
from .calculate_bucket import BC_param, EXTRA_BITS


class ProofOfSpaceMemoryProver():
    def __init__(self, myfile):
        self.myfile = myfile
        self.k = self.myfile.k

    def close(self):
        pass

    def get_memo(self):
        return self.myfile.memo

    def get_id(self):
        return self.myfile.id

    def get_size(self):
        return self.k

    # returns ['quality']
    # challenge is an arbitrary byte string and gets hashed
    def get_qualities_for_challenge(self, challenge):
        f7 = BitArray(uint=(challenge >> (256-self.k)), length=(self.k))
        # These bits will determine which pair of x inputs we use
        last_5_bits = BitArray(uint=(challenge & 31), length=5)
        return self.iterate_qualities(f7, None, None, 7, last_5_bits)

    # returns 'proof'
    # index comes from previous results of a get_qualities_for_challenge() call
    def get_full_proof(self, challenge, index):
        f7 = BitArray(uint=(challenge >> (256-self.k)), length=(self.k))
        return self.iterate(f7, None, None, 7, index).tobytes()

    # Retrieves the next group and offsets
    def get_info_prev(self, depth, group, offset):
        info_prev = zip(self.myfile.f_ys[depth][group],
                        self.myfile.f_offset_L[depth][group],
                        self.myfile.f_offset_R[depth][group],
                        self.myfile.f_buckets_prev[depth][group])
        info_prev = sorted(info_prev, key=lambda x:x[0].uint)

        offset_prev1 = info_prev[offset][1]
        offset_prev2 = info_prev[offset][2]
        group_prev = info_prev[offset][3]

        return (offset_prev1, offset_prev2, group_prev)

    # Retrieves two leaves (x values), given group and two offsets
    def get_x_values(self, group_L, offset_L, offset_R):
        zipped_L = zip(self.myfile.f_ys[1][group_L], self.myfile.f_metadata[1][group_L])
        sorted_L = sorted(zipped_L, key=lambda x:x[0].uint)

        zipped_R = zip(self.myfile.f_ys[1][group_L + 1], self.myfile.f_metadata[1][group_L + 1])
        sorted_R = sorted(zipped_R, key=lambda x:x[0].uint)

        return (sorted_L[offset_L][1] + sorted_R[offset_R][1])

    def iterate_qualities(self, y_group, offset_L, offset_R, depth, last_5_bits):
        # Base case, return the two correct x values
        if depth == 1:
            return self.get_x_values(y_group, offset_L, offset_R)

        # Special case last table, because there is only one thing
        # (We don't combine it with something else to go to an 8th table)
        if depth == 7:
            qualities = []
            assert len(y_group) == self.k
            group_L_floor = (y_group.uint << EXTRA_BITS) // BC_param
            for group_L in range(group_L_floor, group_L_floor + 2):
                if len(self.myfile.f_ys[depth]) <= (group_L): continue
                for index, element in enumerate(sorted(self.myfile.f_ys[depth][group_L], key=lambda x:x.uint)):
                    if element[:self.k] == y_group:
                        offset_prev_1L, offset_prev_2L, group_prev_L = self.get_info_prev(depth, group_L, index)
                        quality = self.iterate_qualities(group_prev_L, offset_prev_1L, offset_prev_2L,
                                                        depth - 1, last_5_bits)
                        qualities.append(quality)
            return qualities

        if last_5_bits[6 - depth] == 0:
            # Left pointer
            offset_prev1, offset_prev2, group_prev = self.get_info_prev(depth, y_group, offset_L)
        else:
            # Right pointer
            offset_prev1, offset_prev2, group_prev = self.get_info_prev(depth, y_group + 1, offset_R)

        return self.iterate_qualities(group_prev, offset_prev1, offset_prev2, depth - 1,
                                      last_5_bits)

    def iterate(self, y_group, offset_L, offset_R, depth, index_input=0):
        # Base case, return the two correct x values
        if depth == 1:
            return self.get_x_values(y_group, offset_L, offset_R)

        # Special case last table, because there is only one thing
        # (We don't combine it with something else to go to an 8th table)
        if depth == 7:
            index = 0
            saved_group_offset = 0
            curr_index = 0
            assert len(y_group) == self.k
            group_L = (y_group.uint << EXTRA_BITS) // BC_param
            for group_offset in range(0, 2):
                if len(self.myfile.f_ys[depth]) <= (group_L + group_offset): continue
                for i, element in enumerate(sorted(self.myfile.f_ys[depth][group_L + group_offset], key=lambda x:x.uint)):
                    if element[:self.k] == y_group:
                        if index_input == curr_index:
                            index = i
                            saved_group_offset = group_offset
                            break
                        curr_index += 1
            group_L += saved_group_offset

            offset_prev_1L, offset_prev_2L, group_prev_L = self.get_info_prev(depth, group_L, index)
            return self.iterate(group_prev_L, offset_prev_1L, offset_prev_2L, depth - 1)

        # Left pointer
        offset_prev_1L, offset_prev_2L, group_prev_L = self.get_info_prev(depth, y_group, offset_L)
        # Right pointer
        offset_prev_1R, offset_prev_2R, group_prev_R = self.get_info_prev(depth, y_group + 1, offset_R)

        return (self.iterate(group_prev_L, offset_prev_1L, offset_prev_2L, depth - 1) +
                self.iterate(group_prev_R, offset_prev_1R, offset_prev_2R, depth - 1))
