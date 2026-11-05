"""Strict max-row RMS verification; prefix commits stop at the first rejection."""
def acceptance_metrics(corrected, proposals, epsilon, mode='row_rms', coordinate_dim=-1):
    if mode != 'row_rms':
        raise ValueError('Only the frozen row_rms policy is supported')
    residual = corrected.float() - proposals.float()
    metric = residual.square().mean(coordinate_dim).sqrt().amax(dim=1)
    return metric, metric < epsilon


def prefix_length(passing):
    count = 0
    for value in passing.tolist():
        if not value:
            break
        count += 1
    return count
