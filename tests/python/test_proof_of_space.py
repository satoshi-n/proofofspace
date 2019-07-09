import unittest
import random
import os
from hashlib import sha256

from src.python.plotter_disk import ProofOfSpaceDiskPlotter
from src.python.plotter_memory import ProofOfSpaceMemoryPlotter
from src.python.prover_disk import ProofOfSpaceDiskProver
from src.python.prover_memory import ProofOfSpaceMemoryProver
from src.python.verifier import validate_proof

plot_id_1 = bytes([35, 2, 52, 4, 51, 55, 23, 84, 91, 10, 111, 12, 13,
                   222, 151, 16, 228, 211, 254, 45, 92, 198, 204, 10, 9, 10,
                   11, 129, 139, 171, 15, 23])

plot_id_2 = bytes([31, 2, 52, 4, 51, 55, 23, 84, 91, 10, 111, 12, 13,
                   222, 151, 16, 228, 211, 254, 45, 92, 198, 204, 10, 9, 10,
                   11, 129, 139, 171, 15, 23])

plot_id_3 = bytes([5, 104, 52, 4, 51, 55, 23, 84, 91, 10, 111, 12, 13,
                   222, 151, 16, 228, 211, 254, 45, 92, 198, 204, 10, 9, 10,
                   11, 129, 139, 171, 15, 23])


class TestPlot(unittest.TestCase):
    def test_memory_plot_1(self):
        """
        Creates a plot on memory and creates proofs using the plot.
        """
        k = 12
        plotter = ProofOfSpaceMemoryPlotter()
        plotter.create_plot_memory("filename.dat", k, "memo", plot_id_1)

        prover = ProofOfSpaceMemoryProver(plotter)

        success = 0
        iterations = 100
        random.seed(1)
        for i in range(iterations):
            challenge_bytes = sha256(int.to_bytes(i, 4, 'big')).digest()
            challenge = int.from_bytes(challenge_bytes, "big")

            qualities = prover.get_qualities_for_challenge(challenge)
            if len(qualities) > 0:
                print("Challenge", challenge)
                print("qualities", i, qualities)

            for index in range(len(qualities)):
                proof = prover.get_full_proof(challenge, index)
                print("\tProof:", proof.hex())
                assert validate_proof(plot_id_1, k, challenge, proof) is not None
                success += 1

        print("Success:", str(success) + "/" + str(iterations),  str(100 * (success/iterations)) + "%")
        assert(success == 26)

    def test_disk_plot_1(self):
        """
        Creates a plot.
        """
        k = 12
        memo = bytes([1, 2, 3, 4, 5])
        plotter = ProofOfSpaceDiskPlotter()
        filename = "python-test-plot.dat"
        plotter.create_plot_disk(filename, k, memo, plot_id_1)
        prover = ProofOfSpaceDiskProver(filename)

        success = 0
        iterations = 100
        for i in range(iterations):
            challenge_bytes = sha256(int.to_bytes(i, 4, 'big')).digest()
            challenge = int.from_bytes(challenge_bytes, "big")

            qualities = prover.get_qualities_for_challenge(challenge)
            if len(qualities) > 0:
                print("Challenge", challenge)
                print("Qualities", i, qualities)
                success += len(qualities)

            for index in range(len(qualities)):
                proof = prover.get_full_proof(challenge, index)
                print("\tProof:", proof.hex())
                quality_string = validate_proof(plot_id_1, k, challenge, proof)
                assert quality_string
                assert quality_string == qualities[index]

        print("Success:", str(success) + "/" + str(iterations),  str(100 * (success/iterations)) + "%")
        os.remove(filename)
        assert(success == 26)

    def test_disk_plot_2(self):
        k = 13
        memo = bytes([1, 2, 3, 4, 5])
        plotter = ProofOfSpaceDiskPlotter()
        filename = "python-test-plot-small.dat"
        plotter.create_plot_disk(filename, k, memo, plot_id_3)
        prover = ProofOfSpaceDiskProver(filename)

        success = 0
        iterations = 500
        for i in range(iterations):
            challenge_bytes = sha256(int.to_bytes(i, 4, 'big')).digest()
            challenge = int.from_bytes(challenge_bytes, "big")

            qualities = prover.get_qualities_for_challenge(challenge)
            if len(qualities) > 0:
                print("Challenge", challenge)
                print("Qualities", i, qualities)
                success += len(qualities)

            for index in range(len(qualities)):
                proof = prover.get_full_proof(challenge, index)
                print("\tProof:", proof.hex())
                quality_string = validate_proof(plot_id_3, k, challenge, proof)
                assert quality_string
                assert quality_string == qualities[index]

        print("Success:", str(success) + "/" + str(iterations),  str(100 * (success/iterations)) + "%")
        os.remove(filename)
        assert(success == 339)

    def test_disk_plot_3(self):
        k = 10
        memo = bytes([1, 2, 3, 4, 5])
        plotter = ProofOfSpaceDiskPlotter()
        plotter.create_plot_disk("python-test-plot.dat", k, memo, plot_id_2)
        prover = ProofOfSpaceDiskProver("python-test-plot.dat")

        success = 0
        iterations = 100
        random.seed(1)
        for i in range(iterations):
            challenge = random.randrange(0, 2**256 - 1)

            qualities = prover.get_qualities_for_challenge(challenge)

            if len(qualities) > 0:
                print("Qualities", i, qualities)
                success += len(qualities)

            for index in range(len(qualities)):
                proof = prover.get_full_proof(challenge, index)
                quality_string = validate_proof(plot_id_2, k, challenge, proof)
                assert quality_string
                assert quality_string == qualities[index]

        print("Success:", str(success) + "/" + str(iterations),  str(100 * (success/iterations)) + "%")


if __name__ == '__main__':
    unittest.main()
