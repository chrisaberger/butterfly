import torch
import math
import operator


class Block2x2Diag(torch.nn.Module):
    """
    Block matrix of size n x n of the form [[A, B], [C, D]] where each of A, B,
    C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, complex=False, ABCD=None):
        """
        Parameters:
            size:    size of butterfly matrix
            complex: real or complex matrix
            ABCD:    block of [[A, B], [C, D]], of shape (2, 2, size//2) if real 
                     or (2, 2, size//2, 2) if complex
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.complex = complex
        self.mul_op = complex_mul if complex else operator.mul
        ABCD_shape = (2, 2, size // 2) if not complex else (2, 2, size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = torch.nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD
        print(self.ABCD)

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        if not self.complex:
            return ((self.ABCD * input.view(input.shape[:-1] +
                                            (1, 2, self.size // 2))).sum(
                                                dim=-2)).view(input.shape)
        else:
            return (self.mul_op(
                self.ABCD,
                input.view(input.shape[:-2] + (1, 2, self.size // 2, 2))).sum(
                    dim=-3)).view(input.shape)


class Block2x2DiagProduct(torch.nn.Module):
    """
    Product of block 2x2 diagonal matrices.
    """

    def __init__(self, size, complex=False, decreasing_size=True):
        super().__init__()
        m = int(math.log2(size))
        assert size == 1 << m, "size must be a power of 2"
        self.size = size
        self.complex = complex
        sizes = [size >> i for i in range(m)
                ] if decreasing_size else [size >> i for i in range(m)[::-1]]

        print(sizes)
        self.factors = torch.nn.ModuleList(
            [Block2x2Diag(size_, complex=complex) for size_ in sizes])

    def forward(self, input):
        """
        Parameters:
            input: (..., size) if real or (..., size, 2) if complex
        Return:
            output: (..., size) if real or (..., size, 2) if complex
        """
        output = input.contiguous()
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(
                    output.view(output.shape[:-1] + (-1, factor.size))).view(
                        output.shape)
            else:
                output = factor(
                    output.view(output.shape[:-2] + (-1, factor.size, 2))).view(
                        output.shape)
        return output


block = Block2x2DiagProduct(8)

inp = torch.randn(8, 8)
out = block.forward(inp)
