from bitstring import BitArray
from .calculate_bucket import F1Calculator, FxCalculator, BC_param, EXTRA_BITS


class ProofOfSpaceMemoryPlotter():
    def create_plot_memory(self, filename, k, memo, id):
        self.num_buckets = (2**(k + EXTRA_BITS)) // BC_param + 1
        self.k = k
        self.memo = memo
        self.id = id
        self.f_metadata = [[[] for n in range(self.num_buckets)]
                           for d in range(0, 8)]
        self.f_ys = [[[] for n in range(self.num_buckets)]
                     for d in range(0, 8)]
        self.f_buckets_prev = [[[] for n in range(2**(k-1))]
                               for d in range(0, 8)]
        self.f_offset_L = [[[] for n in range(2**(k-1))]
                           for d in range(0, 8)]
        self.f_offset_R = [[[] for n in range(2**(k-1))]
                           for d in range(0, 8)]
        f1 = F1Calculator(k, id)

        x1 = BitArray(uint=0, length=k)
        while True:
            results = f1.calculate_buckets(x1, 128)
            for (y1, L) in results:
                key = y1.uint // BC_param
                self.f_metadata[1][key] += [L]
                self.f_ys[1][key] += [y1]
                if x1.uint == 2**k - 1:
                    break
                x1 = BitArray(uint=(x1.uint + 1), length=len(x1))

            if x1.uint == 2**k - 1:
                break

        # For each table
        for depth in range(2, 8):
            print("Table:", depth)
            f2 = FxCalculator(k, depth, id)
            num_matches = []

            # For every bucket in the table
            for bucket_index in range(self.num_buckets - 1):
                matches_i = 0

                # Gets two adjacent buckets
                bucket_L = zip(self.f_ys[depth - 1][bucket_index], self.f_metadata[depth - 1][bucket_index])
                bucket_R = zip(self.f_ys[depth - 1][bucket_index + 1], self.f_metadata[depth - 1][bucket_index + 1])

                # Sorts the table by y
                bucket_L = sorted(bucket_L, key=lambda x: x[0].uint)
                bucket_R = sorted(bucket_R, key=lambda x: x[0].uint)

                # Finds the matches between those two buckets
                matches = f2.find_matches([x[0] for x in bucket_L], [x[0] for x in bucket_R], k)

                # For every match, computes the f function and stores everything in memory
                for offset, offset2 in matches:
                    y1, L = bucket_L[offset]
                    y2, R = bucket_R[offset2]
                    yp1 = f2.f(L, R) ^ y1
                    Lp = f2.compose(L, R)

                    matches_i += 1
                    key = yp1.uint // BC_param  # This is the bucket of the new output
                    bucket_index_L = y1.uint // BC_param
                    assert len(y1) == self.k + EXTRA_BITS

                    self.f_metadata[depth][key] += [Lp]
                    self.f_ys[depth][key] += [yp1]
                    self.f_buckets_prev[depth][key] += [bucket_index_L]
                    self.f_offset_L[depth][key] += [offset]
                    self.f_offset_R[depth][key] += [offset2]
                num_matches += [matches_i]

            print(sum(num_matches) / self.num_buckets, "per bucket")
            print("Total:", sum(num_matches))
        print(self.f_ys[7])
