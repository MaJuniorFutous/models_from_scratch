from typing import LiteralString, Any

#TODO: Add type checking for all inputs to all methods
class Node:

    def __init__(
            self,
            data,
            parents: tuple = (), 
            ops: str = '',
            label: str | None = None):
        self.data = data
        self._backward = lambda: None
        self.parents = parents
        self.ops = ops
        self.grad = 0
        self.label = label # debugging

    def __repr__(self) -> LiteralString:
        return f"Node({self.data}, ops={self.ops}, label={self.label})"

    def __add__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(data=other)
        out = Node(self.data + other.data, parents=(self, other), ops='+')

        def backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            
        out._backward = backward

        return out

    def __sub__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(data=other)
        out = Node(self.data - other.data, parents=(self, other), ops='-')

        def backward() -> None:
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = backward

        return out
    
    def __mul__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(data=other)
        out = Node(self.data * other.data, parents=(self, other), ops='*')

        def backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        
        return out

    def __truediv__(self, other) -> "Node":
        other = other if isinstance(other, Node) else Node(data=other)
        out = Node(self.data / other.data, parents=(self, other), ops='/')

        # Bottom * derivative of top − Top * derivative of bottom, all divided by bottom squared.
        def backward() -> None:
            self.grad += ((1*other.data - self.data*0) / other.data**2) * out.grad
            other.grad += self.data * out.grad

        out._backward = backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Node(self.data**other, (self,), f'**{other}')
        # dy/dx = x^2
        def backward() -> None:
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = backward

        return out

    # loss functions
    def mse(self, y) -> "Node":
        y = y if isinstance(y, Node) else Node(data=y)
        out = Node((y.data - self.data)**2, parents=(self, y), ops='mse')

        def backward() -> None:
            self.grad += (-2*(y.data-self.data)) * out.grad
            y.grad += (2*(y.data-self.data)) * out.grad

        out._backward = backward
        return out

    def backward(self) -> None:
        # meant to be called on the last node (ouptut of loss function)
        seen = set()
        topo = []

        def dfs(node) -> None:
            if node not in seen:
                seen.add(node)
                for parent in node.parents: dfs(parent)
                topo.append(node)
        dfs(self)
        self.grad = 1.0  # to kick off the backward propagation from last node
        for node in reversed(topo): node._backward()

    # activation functions
    def relu(self):
        out = Node(max(0, self.data), parents=(self,), ops='relu')

        def backward():
            self.grad += (1.0 if self.data > 0 else 0) * out.grad
            
        out._backward = backward
        return out

    #TODO: Add tanh, sigmoid

    #TODO: Continue adding rest of __r* dunder methods
    def __rmul__(self, other) -> "Node": return self * other

    def __radd__(self, other) -> "Node": return self + other

    def __rsub__(self, other) -> "Node": return self - other

    def __gt__(self, other) -> bool:
        return self.data > (other.data if isinstance(other, Node) else other)
    
    def __lt__(self, other) -> bool:
        return self.data < (other.data if isinstance(other, Node) else other)

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