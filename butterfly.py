import math
import operator
import torch
import torchvision
import torchvision.transforms as transforms
import argparse

########################  START Tri Code #######################################


class Block2x2DiagRectangular(torch.nn.Module):
    """Block matrix of size k n x k n of the form [[A, B], [C, D]] where each 
    of A, B, C, D are diagonal. This means that only the diagonal and the n//2-th
    subdiagonal and superdiagonal are nonzero.
    """

    def __init__(self, size, stack=1, complex=False, ABCD=None):
        """
        Parameters:
            size: input has shape (stack, ..., size)
            stack: number of stacked components, output has shape 
                   (stack, ..., size)
            complex: real or complex matrix
            ABCD: block of [[A, B], [C, D]], of shape (stack, 2, 2, size//2) if 
                  real or (stack, 2, 2, size//2, 2) if complex
        """
        super().__init__()
        assert size % 2 == 0, 'size must be even'
        self.size = size
        self.stack = stack
        self.complex = complex
        ABCD_shape = (stack, 2, 2, size // 2) if not complex else (stack, 2, 2,
                                                                   size // 2, 2)
        scaling = 1.0 / 2 if complex else 1.0 / math.sqrt(2)
        if ABCD is None:
            self.ABCD = torch.nn.Parameter(torch.randn(ABCD_shape) * scaling)
        else:
            assert ABCD.shape == ABCD_shape, f'ABCD must have shape {ABCD_shape}'
            self.ABCD = ABCD

    def forward(self, input):
        """
        Parameters:
            input: (stack, ..., size) if real or (stack, ..., size, 2) if complex
        Return:
            output: (stack, ..., size) if real or (stack, ..., size, 2) if complex
        """
        if not self.complex:
            return ((self.ABCD.unsqueeze(1) * input.view(
                self.stack, -1, 1, 2, self.size // 2)).sum(dim=-2)).view(
                    input.shape)
        else:
            return (complex_mul(
                self.ABCD.unsqueeze(1),
                input.view(self.stack, -1, 1, 2, self.size // 2,
                           2)).sum(dim=-3)).view(input.shape)


class Block2x2DiagProductRectangular(torch.nn.Module):
    """Product of block 2x2 diagonal matrices.
    """

    def __init__(self, in_size, out_size, complex=False, decreasing_size=True):
        super().__init__()
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        self.in_size_extended = 1 << m    # Will zero-pad input if in_size is not a power of 2
        self.out_size = out_size
        self.stack = int(math.ceil(out_size / self.in_size_extended))
        self.complex = complex
        in_sizes = [self.in_size_extended >> i for i in range(m)
                   ] if decreasing_size else [
                       self.in_size_extended >> i for i in range(m)[::-1]
                   ]
        self.factors = torch.nn.ModuleList([
            Block2x2DiagRectangular(
                in_size_, stack=self.stack, complex=complex)
            for in_size_ in in_sizes
        ])

    def forward(self, input):
        """
        Parameters:
            input: (..., in_size) if real or (..., in_size, 2) if complex
        Return:
            output: (..., out_size) if real or (..., out_size, 2) if complex
        """
        output = input.contiguous()
        if self.in_size != self.in_size_extended:    # Zero-pad
            if not self.complex:
                output = torch.cat((output,
                                    torch.zeros(
                                        output.shape[:-1] +
                                        (self.in_size_extended - self.in_size,),
                                        dtype=output.dtype,
                                        device=output.device)),
                                   dim=-1)
            else:
                output = torch.cat(
                    (output,
                     torch.zeros(
                         output.shape[:-2] +
                         (self.in_size_extended - self.in_size, 2),
                         dtype=output.dtype,
                         device=output.device)),
                    dim=-2)
        output = output.unsqueeze(0).expand((self.stack,) + output.shape)
        for factor in self.factors[::-1]:
            if not self.complex:
                output = factor(
                    output.view(output.shape[:-1] + (-1, factor.size))).view(
                        output.shape)
            else:
                output = factor(
                    output.view(output.shape[:-2] + (-1, factor.size, 2))).view(
                        output.shape)
        if not self.complex:
            output = output.permute(
                tuple(range(1,
                            output.dim() - 1)) +
                (0, -1)).reshape(input.shape[:-1] + (
                    self.stack * self.in_size_extended,))[..., :self.out_size]
        else:
            output = output.permute(
                tuple(range(1,
                            output.dim() - 2)) + (0, -2, -1)
            ).reshape(input.shape[:-2] + (self.stack * self.in_size_extended,
                                          2))[..., :self.out_size, :]
        return output


########################  END Tri Code #########################################

torch.manual_seed(7)

# Simple MNIST Logistic Regression Code.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    required=True,
    choices={"butterfly", "linear"},
    help="mode to run logistic regression in")
args = parser.parse_args()

# Hyper-parameters
input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(
    root='../../data',
    train=True,
    transform=transforms.ToTensor(),
    download=True)

test_dataset = torchvision.datasets.MNIST(
    root='../../data', train=False, transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Logistic regression model
if args.mode == "butterfly":
    model = Block2x2DiagProductRectangular(input_size, num_classes)
else:
    model = torch.nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Reshape images to (batch_size, input_size)
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                    float(correct * 100) / total))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the model on the 10000 test images: {} %'.format(
        100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
