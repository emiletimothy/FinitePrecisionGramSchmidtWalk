import numpy as np
import _bootstrap  # noqa: F401  # repo root on sys.path
from core import gsw


def main() -> None:
    m, n = 3, 30
    rng = np.random.default_rng()
    B = rng.standard_normal((m, n))
    B /= np.linalg.norm(B, axis=0, keepdims=True)

    result = gsw.gram_schmidt_walk(
        B,
        # noise=lambda n: np.random.normal(0, np.sqrt(0.01), n),
        record_trajectory=True,
    )
    T = len(result.trajectory)

    np.set_printoptions(precision=4, suppress=True, linewidth=120)
    print(f"B:\n{B}")
    print(f"B shape: ({m}, {n})  |  iterations: {T}")
    print(f"Bz:              {result.Bz}")
    print(f"assignment:      {result.assignment}")
    print()
    print("Press Enter to step through iterations (q + Enter to quit).")
    print()

    for t, (z_t, u, delta_t) in enumerate(result.trajectory):
        raw = input(f"[{t + 1}/{T}] ")
        if raw.strip().lower() == "q":
            break
        z_after = z_t + delta_t * u
        decided = np.abs(z_after) >= 1 - 1e-9
        partial_Bz = B @ (z_after * decided)

        z_str = str(z_t)
        if "\n" not in z_str:
            print(f"  z_t:           {z_str}")
        u_str = str(u)
        if "\n" not in u_str:
            print(f"  u_t:           {u_str}")
        print(f"  Bu:            {B @ u}")
        print(f"  full Bz:       {B @ z_after}")
        print(f"  partial Bz:    {partial_Bz}")
        print()

    print(f"Final assignment: {result.assignment}")
    print(f"Bz:               {result.Bz}")


if __name__ == "__main__":
    main()
