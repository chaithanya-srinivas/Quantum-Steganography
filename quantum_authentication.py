from qiskit import QuantumCircuit

def create_authenticated_circuit(message_bits):
    """
    Create a quantum circuit with quantum authentication added.
    """
    n = len(message_bits)
    qc = QuantumCircuit(2 * n, 2 * n)

    # Encode message bits into the first n qubits
    for i, bit in enumerate(message_bits):
        if bit == '1':
            qc.x(i)

    # Add authentication qubits (n to 2n-1)
    for i in range(n):
        qc.h(i)  # Hadamard on message qubit
        qc.cx(i, n + i)  # Entangle with auth qubit

    # Measurement of both message qubits and authentication qubits
    qc.barrier()
    qc.measure(range(2 * n), range(2 * n))

    return qc

if __name__ == "__main__":
    example_bits = "1010"
    qc = create_authenticated_circuit(example_bits)
    print(qc.draw("text"))