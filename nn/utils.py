from modules.autograd import Node

def matmul(A, B):

    rows_A = len(A)
    cols_A = len(A[0])

    rows_B = len(B)
    cols_B = len(B[0])

    assert cols_A == rows_B, "Incompatible matrix dimensions"

    return [
        [
            sum(
                (A[i][k] * B[k][j] for k in range(cols_A)),
                start=Node(0)
            )
            for j in range(cols_B)
        ]
        for i in range(rows_A)
    ]