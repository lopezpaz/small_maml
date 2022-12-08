# Section 5.1 from MAML paper: https://arxiv.org/abs/1703.03400
from matplotlib import pyplot as plt
import torch


class SinusoidTask:
    def __init__(self, n=10):
        # amplitude
        self.a = torch.ones(1).uniform_(0.1, 5.0).item()
        # phase
        self.b = torch.ones(1).uniform_(0, 3.14159)
        # data for inner update
        self.x_tr = torch.zeros(n, 1).uniform_(-5, 5)
        self.y_tr = self.a * self.x_tr.add(self.b).sin()
        # data for outer update
        self.x_va = torch.zeros(n, 1).uniform_(-5, 5)
        self.y_va = self.a * self.x_va.add(self.b).sin()
        # testing data
        self.x_te = torch.linspace(-5, 5, 100).view(-1, 1)
        self.y_te = self.a * self.x_te.add(self.b).sin()


def train(
    TaskGenerator=SinusoidTask,
    outer_updates=70000,
    inner_updates=1,
    test_updates=10,
    alpha=0.01,
    k=10):

    test_task = TaskGenerator(k)

    network = torch.nn.Sequential(
        torch.nn.Linear(1, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 40),
        torch.nn.ReLU(),
        torch.nn.Linear(40, 1))

    loss = torch.nn.MSELoss()

    def foo(p, x):
        return torch.nn.utils.stateless.functional_call(network, p, x)

    outer_optimizer = torch.optim.Adam(
        network.parameters(), lr=alpha, weight_decay=1e-4)

    for outer_update in range(outer_updates):
        train_task = TaskGenerator(k)
        outer_optimizer.zero_grad()

        # start of MAML
        parameters = dict(network.named_parameters())
        for _ in range(inner_updates):
            inner_loss = loss(foo(parameters, train_task.x_tr), train_task.y_tr)
            # we cannot use PyTorch's optimizers here, since their steps are
            # in-place, and therefore erase gradient's history
            grads = torch.autograd.grad(
                inner_loss, parameters.values(), create_graph=True)
            for name, grad in zip(parameters, grads):
                parameters[name] = parameters[name] - alpha * grad
        # end of MAML

        loss(foo(parameters, train_task.x_va), train_task.y_va).backward()
        outer_optimizer.step()

    for _ in range(test_updates):
        outer_optimizer.zero_grad()
        loss(network(test_task.x_va), test_task.y_va).backward()
        outer_optimizer.step()

    return network, test_task


if __name__ == "__main__":
    torch.manual_seed(0)
    network, test_task = train()
    plt.plot(test_task.x_te, test_task.y_te, lw=2, color="black")
    plt.plot(test_task.x_tr, test_task.y_tr, '.', ms=20, color="black")
    plt.plot(test_task.x_te, network(test_task.x_te).detach(), lw=2, color="red")
    plt.show()
