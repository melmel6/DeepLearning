import numpy as np

import torch
from torch import nn

def NIG_NLL(y, gamma, v, alpha, beta, reduce=False):

    twoBlambda = 2*beta*(1+v)

    nll = 0.5*torch.log(np.pi/v)  \
        - alpha*torch.log(twoBlambda)  \
        + (alpha+0.5) * torch.log(v*(y-gamma)**2 + twoBlambda)  \
        + torch.lgamma(alpha)  \
        - torch.lgamma(alpha+0.5)

    return torch.mean(nll) if reduce else nll

def KL_NIG(mu1, v1, a1, b1, mu2, v2, a2, b2):

    KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
        + 0.5*v2/v1  \
        - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
        - 0.5 + a2*torch.log(b1/b2)  \
        - (torch.lgamma(a1) - torch.lgamma(a2))  \
        + (a1 - a2)*torch.digamma(a1)  \
        - (b1 - b2)*a1/b1
    return KL

def NIG_Reg(y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
    # error = tf.stop_gradient(tf.abs(y-gamma))
    # print("hello 3")

    error = torch.abs(y-gamma)

    if kl:
        kl = KL_NIG(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
        reg = error*kl
    else:
        evi = 2*v+(alpha)
        reg = error*evi

    return torch.mean(reg) if reduce else reg


def EvidentialRegression(y_true, evidential_output, coeff=1.0):
    # print("hello 1")

    # print(evidential_output)
    # gamma, v, alpha, beta = tf.split(evidential_output, 4, axis=-1)
    gamma, v, alpha, beta  = torch.split(evidential_output, 1, dim=-1)

    # print(gamma, v, alpha, beta )

    loss_nll = NIG_NLL(y_true, gamma, v, alpha, beta)
    loss_reg = NIG_Reg(y_true, gamma, v, alpha, beta)

    loss = loss_nll + coeff * loss_reg

    # print("************** LOSS ****************")
    # print(loss)
    # print(loss.shape)
    # print("************** END LOSS ****************")

    return torch.mean(loss)


############################################################

# import torch
# from torch.distributions import Normal
# from torch import nn
# import numpy as np

# MSE = nn.MSELoss(reduction='mean')


# def reduce(val, reduction):
#     if reduction == 'mean':
#         val = val.mean()
#     elif reduction == 'sum':
#         val = val.sum()
#     elif reduction == 'none':
#         pass
#     else:
#         raise ValueError(f"Invalid reduction argument: {reduction}")
#     return val


# def RMSE(y, y_):
#     return MSE(y, y_).sqrt()


# def Gaussian_NLL(y, mu, sigma, reduction='mean'):
#     dist = Normal(loc=mu, scale=sigma)
#     # TODO: refactor to mirror TF implementation due to numerical instability
#     logprob = -1. * dist.log_prob(y)
#     return reduce(logprob, reduction=reduction)


# def NIG_NLL(y: torch.Tensor,
#             gamma: torch.Tensor,
#             nu: torch.Tensor,
#             alpha: torch.Tensor,
#             beta: torch.Tensor, reduction='mean'):
#     inter = 2 * beta * (1 + nu)

#     nll = 0.5 * (np.pi / nu).log() \
#           - alpha * inter.log() \
#           + (alpha + 0.5) * (nu * (y - gamma) ** 2 + inter).log() \
#           + torch.lgamma(alpha) \
#           - torch.lgamma(alpha + 0.5)
#     return reduce(nll, reduction=reduction)


# def NIG_Reg(y, gamma, nu, alpha, reduction='mean'):
#     error = (y - gamma).abs()
#     evidence = 2. * nu + alpha
#     return reduce(error * evidence, reduction=reduction)


# def EvidentialRegression(y: torch.Tensor, evidential_output: torch.Tensor, lmbda=1.):
#     print("*********** LOSS *************")
#     print(evidential_output)
#     gamma, nu, alpha, beta = evidential_output
#     loss_nll = NIG_NLL(y, gamma, nu, alpha, beta)
#     loss_reg = NIG_Reg(y, gamma, nu, alpha)
#     return loss_nll, lmbda * loss_reg

