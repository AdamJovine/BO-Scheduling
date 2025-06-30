from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import torch
from optimization.CArBO import CostAwareSchedulingOptimizer
class RandomBaselineOptimizer(CostAwareSchedulingOptimizer):
    """
    Runs the same parameter‐evaluation pipeline as CostAwareSchedulingOptimizer
    but purely with random existing‐block samples.
    """
    def __init__(self, *args, max_workers=None, **kwargs):
        super().__init__(*args, **kwargs)
        # how many threads to use for evaluation
        self.max_workers = max_workers or max(len(self.licenses), 1)

    def run_random_baseline(self, n_iterations: int = 50):
        """
        Draw n_iterations samples via _sample_existing, evaluate them all,
        and return a DataFrame with objective values + a Pareto flag.
        """
        # 1) Sample purely existing‐block points
        device = torch.device("cpu")
        X = self._sample_existing(n_iterations, device=device)  # [n_iter, dim]

        # 2) Evaluate each in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self.evaluate_parameters, x) for x in X]
            # evaluate_parameters returns (y, cost)
            results = [f.result()[0] for f in futures]

        # 3) Stack into a tensor
        Y = torch.stack([torch.tensor(y, dtype=torch.double) for y in results])  # [n_iter, n_obj]

        # 4) Pareto mask
        pareto_mask = self._pareto_mask(Y.cpu().numpy())

        # 5) Build DataFrame
        df = pd.DataFrame(X.cpu().numpy(), columns=self.param_names)
        for i in range(Y.size(1)):
            df[f"obj_{i}"] = Y[:, i].cpu().numpy()
        df['pareto'] = pareto_mask

        return df
