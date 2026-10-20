"""Independent Monte Carlo check against deterministic area quadrature.

Run with Python + NumPy. This checks the estimator mathematics, not GPU output.
The reference integrates the geometry term directly; it does not use MIS PDFs.
"""
import numpy as np


def check(diffuse, height=1.49, survival=1.0, blocked=False):
    count = 400_000
    rng = np.random.default_rng(20261020)
    half_extent, area, albedo = 0.4, 0.64, 0.7

    # Independent midpoint quadrature of Le * f * cos(theta) * cos(light) / r^2.
    grid = (np.arange(768) + 0.5) / 768 * (2 * half_extent) - half_extent
    r2 = grid[:, None] ** 2 + grid[None, :] ** 2 + height ** 2
    diffuse_reference = area * np.mean(height ** 2 / (np.pi * r2 ** 2))
    reference = albedo * (diffuse * diffuse_reference + (1 - diffuse))
    if blocked:
        reference = 0.0

    # One uniform-area light sample at every surface with a diffuse lobe.
    xy = rng.uniform(-half_extent, half_extent, (count, 2))
    r2 = np.sum(xy ** 2, axis=1) + height ** 2
    cosine = abs(height) / np.sqrt(r2)
    light_pdf = r2 / (area * cosine)
    bsdf_pdf = diffuse * cosine / np.pi
    light_weight = light_pdf ** 2 / (light_pdf ** 2 + bsdf_pdf ** 2)
    direct = albedo * diffuse * cosine / np.pi / light_pdf * light_weight

    # Independent cosine-hemisphere sample, mixed with an ideal mirror aimed
    # straight at the lamp. Delta hits must receive weight one.
    is_diffuse = rng.random(count) < diffuse
    radial_squared = rng.random(count)
    phi = rng.random(count) * (2 * np.pi)
    cosine = np.sqrt(1 - radial_squared)
    radial = np.sqrt(radial_squared)
    x = abs(height) * radial * np.cos(phi) / cosine
    y = abs(height) * radial * np.sin(phi) / cosine
    hits_lamp = (np.abs(x) <= half_extent) & (np.abs(y) <= half_extent)
    light_pdf = (height ** 2 / cosine ** 2) / (area * cosine)
    bsdf_pdf = diffuse * cosine / np.pi
    bsdf_weight = bsdf_pdf ** 2 / (bsdf_pdf ** 2 + light_pdf ** 2)
    continuation = np.where(is_diffuse, hits_lamp * bsdf_weight, 1.0) * albedo
    # RR affects continuation only. The saved PDF remains the pre-RR BSDF PDF;
    # inverse-survival throughput compensation preserves its expectation.
    survives = rng.random(count) < survival
    continuation *= survives / survival
    estimates = direct + continuation
    naive = np.where(is_diffuse, hits_lamp, True) * albedo * survives / survival
    if blocked:
        estimates *= 0
        naive *= 0
    mean = estimates.mean()
    error = estimates.std(ddof=1) / np.sqrt(count)
    assert abs(mean - reference) < 6 * error + 2e-6, (mean, reference, error)
    if diffuse == 1 and not blocked and survival == 1:
        assert estimates.var() < naive.var() * 0.2
    print(f'diffuse={diffuse}, height={height}, RR={survival}, blocked={blocked}: '
          f'MIS={mean:.6f}, reference={reference:.6f}, SE={error:.6f}, '
          f'variance ratio={estimates.var() / max(naive.var(), 1e-30):.4f}')


if __name__ == '__main__':
    for mixture in (1.0, 0.5, 0.0):
        check(mixture)
    check(0.5, survival=0.6)
    check(1.0, height=-1.49)  # Existing emitter is two-sided.
    check(0.5, blocked=True)
    print('All estimator checks passed.')
