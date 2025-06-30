from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.utils.cholesky import psd_safe_cholesky
from botorch.models.transforms import Normalize, Standardize
from botorch.acquisition.analytic import LogExpectedImprovement
from torch.distributions import Dirichlet
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement
#from botorch.acquisition.objective import LinearPosteriorTransform
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedPosteriorTransform

# --- assume EIUU is defined as discussed earlier ---


class EIUU(AcquisitionFunction):
    def __init__(self, model, M: int = 16):
        super().__init__(model=model)
        self.model = model
        self.M = M

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # 1) sample utilities
        d_out = self.model.num_outputs
        Y_train = torch.cat([m.train_targets for m in self.model.models], dim=-1)
        D = Y_train.shape[-1]
        Ys = [m.train_targets.unsqueeze(-1) if m.train_targets.ndim==1 else m.train_targets for m in self.model.models]

        # 2) stack into (n_samples, n_outputs)
        Y_train = torch.cat(Ys, dim=1)

        print("––> Y_train.shape", Y_train.shape, "=> D =", D)
        assert D == self.model.num_outputs, \
            f"num_outputs={self.model.num_outputs}, but Y_train has {D}"
        # now sample weight vectors of length D, in the same dtype/device as Y_train
        W = Dirichlet(torch.ones(D, device=Y_train.device,
                                 dtype=Y_train.dtype)).sample((self.M,))

        #W = Dirichlet(torch.ones(d_out, device=X.device, dtype=X.dtype)).sample((self.M,))
 
        eis = []
        # 2) loop over each sampled w
        #for w in W:
        #    # compute best scalarized f so far
        #    w = w.to(dtype=X.dtype, device=X.device)
        #    Y_train = torch.cat([m.train_targets for m in self.model.models], dim=-1)
        for w in W:
            # 2a) form w as a column and compute current best_f
            w_col = w.unsqueeze(-1)                 # [D,1]
            scalarized = Y_train.matmul(w_col).squeeze(-1)  # [n_samples]
            best_f = scalarized.max().item()

            # set up the “linear” posterior transform
            pt = ScalarizedPosteriorTransform(weights=w)


            # build analytic EI under that transform
            ei_acq = LogExpectedImprovement(
                model=self.model,
                best_f=best_f,
                posterior_transform=pt,
            )
            eis.append(ei_acq(X).view(-1))

        # 3) average over w’s
        return torch.stack(eis, dim=0).mean(dim=0)