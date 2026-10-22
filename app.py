import io, os, time, warnings, contextlib, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.special import jv
from scipy.optimize import least_squares
warnings.filterwarnings("ignore")

# Solver settings retained from the supplied implementation
RANDOM_SEED = 12345
IS_BONDED = True
NUMBER_OF_STARTS = 5
CACHE_DECIMAL_PLACES = 12
MAX_NFEV = 200
MAX_BOUND_EXPANSIONS = 100
BOUND_EXPANSION_MPA = 1000.0
USE_PROGRESSIVE_BOUND_EXPANSION = True
LOCAL_WINDOW_HALF_WIDTH_MPA = 10.0
LOCAL_WINDOW_GROWTH_FACTOR = 2.0
HISTORY_DEDUP_RTOL = 1e-9
HISTORY_DEDUP_ATOL = 1e-8
BOUND_TOLERANCE = 0.02
ABSOLUTE_MIN_MODULUS = 1.0
ABSOLUTE_MAX_MODULUS = 50000.0

# Optional Streamlit callback used to render live solver progress.
LIVE_PROGRESS_CALLBACK = None

def r_hat(

    n,

    thickness,

    E,

    v,

    isbonded,

    m,

    points

):


    # ============================================================
    # LAYER DEPTHS
    # ============================================================

    z = np.cumsum(
        thickness
    )


    H = z[-1]


    # ============================================================
    # LAMBDA VALUES
    # ============================================================

    lambda_values = np.zeros(
        n
    )


    for i in range(n - 1):

        lambda_values[i] = (

            z[i]
            /
            H

        )


    lambda_values[n - 1] = np.inf


    # ============================================================
    # F AND R
    # ============================================================

    F = np.zeros(
        n
    )


    R = np.zeros(
        n - 1
    )


    F[0] = np.exp(

        -m
        *
        lambda_values[0]

    )


    R[0] = (

        E[0]
        /
        E[1]

        *

        (1 + v[1])
        /
        (1 + v[0])

    )


    for i in range(

        1,
        n - 1

    ):


        F[i] = np.exp(

            -m

            *

            (

                lambda_values[i]

                -

                lambda_values[i - 1]

            )

        )


        R[i] = (

            E[i]
            /
            E[i + 1]

            *

            (1 + v[i + 1])
            /
            (1 + v[i])

        )


    F[n - 1] = 0.0


    # ============================================================
    # INTERFACE MATRICES
    # ============================================================

    M_list = []


    for i in range(n - 1):


        M1 = np.zeros(
            (4, 4)
        )


        M2 = np.zeros(
            (4, 4)
        )


        # ========================================================
        # BONDED
        # ========================================================

        if isbonded:


            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 1


            M1[0, 1] = F[i]
            M1[1, 1] = -F[i]
            M1[2, 1] = F[i]
            M1[3, 1] = -F[i]


            M1[0, 2] = (

                -(
                    1
                    -
                    2 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[1, 2] = (

                2 * v[i]
                +
                m * lambda_values[i]

            )


            M1[2, 2] = (

                1
                +
                m * lambda_values[i]

            )


            M1[3, 2] = (

                -(
                    2
                    -
                    4 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[0, 3] = (

                1
                -
                2 * v[i]
                +
                m * lambda_values[i]

            ) * F[i]


            M1[1, 3] = (

                2 * v[i]
                -
                m * lambda_values[i]

            ) * F[i]


            M1[2, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            M1[3, 3] = (

                -(
                    2
                    -
                    4 * v[i]
                    +
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            # ====================================================
            # M2
            # ====================================================

            M2[0, 0] = F[i + 1]

            M2[1, 0] = F[i + 1]


            M2[2, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[3, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[0, 1] = 1

            M2[1, 1] = -1


            M2[2, 1] = R[i]

            M2[3, 1] = -R[i]


            M2[0, 2] = (

                -(
                    1
                    -
                    2 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                F[i + 1]

            )


            M2[1, 2] = (

                2 * v[i + 1]
                +
                m * lambda_values[i]

            ) * F[i + 1]


            M2[2, 2] = (

                (
                    1
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[3, 2] = (

                -(
                    2
                    -
                    4 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[0, 3] = (

                1
                -
                2 * v[i + 1]
                +
                m * lambda_values[i]

            )


            M2[1, 3] = (

                2 * v[i + 1]
                -
                m * lambda_values[i]

            )


            M2[2, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

            )


            M2[3, 3] = (

                -(
                    2
                    -
                    4 * v[i + 1]
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

            )


        # ========================================================
        # UNBONDED
        # ========================================================

        else:


            M1[0, 0] = 1
            M1[1, 0] = 1
            M1[2, 0] = 1
            M1[3, 0] = 0


            M1[0, 1] = F[i]
            M1[1, 1] = F[i]
            M1[2, 1] = -F[i]
            M1[3, 1] = 0


            M1[0, 2] = (

                -(
                    1
                    -
                    2 * v[i]
                    -
                    m * lambda_values[i]
                )

            )


            M1[1, 2] = (

                1
                +
                m * lambda_values[i]

            )


            M1[2, 2] = (

                2 * v[i]
                +
                m * lambda_values[i]

            )


            M1[3, 2] = 0


            M1[0, 3] = (

                1
                -
                2 * v[i]
                +
                m * lambda_values[i]

            ) * F[i]


            M1[1, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                F[i]

            )


            M1[2, 3] = (

                2 * v[i]
                -
                m * lambda_values[i]

            ) * F[i]


            M1[3, 3] = 0


            M2[0, 0] = F[i + 1]


            M2[1, 0] = (

                R[i]
                *
                F[i + 1]

            )


            M2[2, 0] = 0


            M2[3, 0] = F[i + 1]


            M2[0, 1] = 1


            M2[1, 1] = R[i]


            M2[2, 1] = 0


            M2[3, 1] = -1


            M2[0, 2] = (

                -(
                    1
                    -
                    2 * v[i + 1]
                    -
                    m * lambda_values[i]
                )

                *

                F[i + 1]

            )


            M2[1, 2] = (

                (
                    1
                    +
                    m * lambda_values[i]
                )

                *

                R[i]

                *

                F[i + 1]

            )


            M2[2, 2] = 0


            M2[3, 2] = (

                2 * v[i + 1]
                +
                m * lambda_values[i]

            ) * F[i + 1]


            M2[0, 3] = (

                1
                -
                2 * v[i + 1]
                +
                m * lambda_values[i]

            )


            M2[1, 3] = (

                -(
                    1
                    -
                    m * lambda_values[i]
                )

                *

                R[i]

            )


            M2[2, 3] = 0


            M2[3, 3] = (

                2 * v[i + 1]
                -
                m * lambda_values[i]

            )


        # ========================================================
        # MATRIX SOLUTION
        # ========================================================

        try:


            Mi = np.linalg.solve(

                M1,

                M2

            )


        except np.linalg.LinAlgError:


            Mi = (

                np.linalg.pinv(M1)

                @

                M2

            )


        M_list.append(
            Mi
        )


    # ============================================================
    # MATRIX PRODUCT
    # ============================================================

    MM = np.eye(4)


    for Mi in M_list:

        MM = (

            MM
            @
            Mi

        )


    MM = MM[:, [1, 3]]


    # ============================================================
    # BOUNDARY CONDITIONS
    # ============================================================

    b11 = np.exp(

        -lambda_values[0]
        *
        m

    )


    b21 = np.exp(

        -lambda_values[0]
        *
        m

    )


    b12 = 1.0

    b22 = -1.0


    c11 = (

        -(1 - 2 * v[0])

        *

        np.exp(

            -m
            *
            lambda_values[0]

        )

    )


    c21 = (

        2 * v[0]

        *

        np.exp(

            -m
            *
            lambda_values[0]

        )

    )


    c12 = 1 - 2 * v[0]

    c22 = 2 * v[0]


    k11 = (

        b11 * MM[0, 0]
        +
        b12 * MM[1, 0]
        +
        c11 * MM[2, 0]
        +
        c12 * MM[3, 0]

    )


    k12 = (

        b11 * MM[0, 1]
        +
        b12 * MM[1, 1]
        +
        c11 * MM[2, 1]
        +
        c12 * MM[3, 1]

    )


    k21 = (

        b21 * MM[0, 0]
        +
        b22 * MM[1, 0]
        +
        c21 * MM[2, 0]
        +
        c22 * MM[3, 0]

    )


    k22 = (

        b21 * MM[0, 1]
        +
        b22 * MM[1, 1]
        +
        c21 * MM[2, 1]
        +
        c22 * MM[3, 1]

    )


    determinant = (

        k11 * k22
        -
        k12 * k21

    )


    if abs(determinant) < 1e-14:

        raise np.linalg.LinAlgError(

            "Singular boundary matrix."

        )


    # ============================================================
    # COEFFICIENTS
    # ============================================================

    A = np.zeros(n)

    B = np.zeros(n)

    C = np.zeros(n)

    D = np.zeros(n)


    A[n - 1] = 0.0


    B[n - 1] = (

        k22
        /
        determinant

    )


    C[n - 1] = 0.0


    if abs(k21) > 1e-14:


        D[n - 1] = (

            1.0

            /

            (

                k12

                -

                k22
                *
                k11
                /
                k21

            )

        )


    else:

        D[n - 1] = 0.0


    # ============================================================
    # BACK SUBSTITUTION
    # ============================================================

    for i in range(

        n - 2,
        -1,
        -1

    ):


        X = (

            M_list[i]

            @

            np.array(

                [

                    A[i + 1],

                    B[i + 1],

                    C[i + 1],

                    D[i + 1]

                ]

            )

        )


        A[i] = X[0]

        B[i] = X[1]

        C[i] = X[2]

        D[i] = X[3]


    # ============================================================
    # DISPLACEMENT
    # ============================================================

    results = []


    for pt in points:


        rho = (

            pt["r"]
            /
            H

        )


        lmm = (

            pt["z"]
            /
            H

        )


        ii = (

            pt["layer"]
            -
            1

        )


        if ii != 0:


            exp1 = np.exp(

                -m

                *

                (

                    lambda_values[ii]

                    -

                    lmm

                )

            )


            exp2 = np.exp(

                -m

                *

                (

                    lmm

                    -

                    lambda_values[ii - 1]

                )

            )


        else:


            exp1 = np.exp(

                -m

                *

                (

                    lambda_values[ii]

                    -

                    lmm

                )

            )


            exp2 = np.exp(

                -m
                *
                lmm

            )


        w = (

            -H

            *

            (1 + v[ii])

            /

            E[ii]

            *

            jv(

                0,

                m
                *
                rho

            )

            *

            (

                (

                    A[ii]

                    -

                    C[ii]

                    *

                    (

                        2

                        -

                        4 * v[ii]

                        -

                        m * lmm

                    )

                )

                *

                exp1

                -

                (

                    B[ii]

                    +

                    D[ii]

                    *

                    (

                        2

                        -

                        4 * v[ii]

                        +

                        m * lmm

                    )

                )

                *

                exp2

            )

        )


        results.append(

            {

                "w":
                    w

            }

        )


    return results


# ================================================================
# 6. FORWARD CALCULATION
# ================================================================

def calculate_deflections(

    thickness,

    E

):


    z = np.cumsum(
        thickness
    )


    H = z[-1]


    alpha = (

        LOAD_RADIUS_MM
        /
        H

    )


    dl = (

        0.1
        /
        alpha

    )


    deflections = np.zeros(

        len(POINTS)

    )


    # ============================================================
    # GAUSS INTEGRATION POINTS
    # ============================================================

    gauss_points = [

        (-0.86114, 0.34786),

        (-0.33998, 0.65215),

        (0.33998, 0.65215),

        (0.86114, 0.34786)

    ]


    l = 0.0


    while l <= 20.0 + 1e-12:


        for node, weight in gauss_points:


            m = (

                l

                +

                dl / 2.0

                +

                node
                *
                dl
                /
                2.0

            )


            fc = (

                weight

                *

                dl

                /

                2.0

            )


            if m <= 1e-14:

                continue


            try:


                rhat_results = r_hat(

                    N,

                    thickness,

                    E,

                    POISSON,

                    IS_BONDED,

                    m,

                    POINTS

                )


            except Exception:


                return np.full(

                    len(POINTS),

                    np.nan

                )


            scale = (

                fc

                *

                PRESSURE_MPA

                *

                alpha

                *

                jv(

                    1,

                    m
                    *
                    alpha

                )

                /

                m

            )


            for j in range(

                len(POINTS)

            ):


                deflections[j] += (

                    scale

                    *

                    rhat_results[j]["w"]

                )


        l += dl


    return deflections


# ================================================================
# 7. CALCULATE R-SQUARED
# ================================================================

def calculate_r2(

    target,

    calculated

):


    target = np.asarray(

        target,

        dtype=float

    )


    calculated = np.asarray(

        calculated,

        dtype=float

    )


    ss_res = np.sum(

        (

            target
            -
            calculated

        )

        ** 2

    )


    ss_tot = np.sum(

        (

            target
            -
            np.mean(target)

        )

        ** 2

    )


    if ss_tot <= 1e-30:

        return np.nan


    return (

        1.0

        -

        ss_res
        /
        ss_tot

    )


# ================================================================
# 8. CHECK BOUND HIT
# ================================================================

def check_bound_hit(

    E,

    lower_bounds,

    upper_bounds,

    tolerance

):


    E = np.asarray(

        E,

        dtype=float

    )


    lower_bounds = np.asarray(

        lower_bounds,

        dtype=float

    )


    upper_bounds = np.asarray(

        upper_bounds,

        dtype=float

    )


    current_range = (

        upper_bounds
        -
        lower_bounds

    )


    lower_threshold = (

        lower_bounds

        +

        tolerance
        *
        current_range

    )


    upper_threshold = (

        upper_bounds

        -

        tolerance
        *
        current_range

    )


    hit_lower = (

        E <= lower_threshold

    )


    hit_upper = (

        E >= upper_threshold

    )


    return (

        hit_lower,

        hit_upper

    )


# ================================================================
# 9. EXPAND BOUNDS
# ================================================================
#
# IMPORTANT:
# This function expands ONLY the modulus that actually hits a bound.
#
# Example:
#     Current E1 = 2000 MPa at upper bound
#     BOUND_EXPANSION_MPA = 1000
#     New E1 upper bound = 3000 MPa
#
# The other modulus bounds are handled separately by
# create_next_stage_bounds(), where their previous recovered
# values are protected inside a local window.
# ================================================================

def expand_bounds(
    lower_bounds,
    upper_bounds,
    hit_lower,
    hit_upper
):

    lower_bounds = np.asarray(
        lower_bounds,
        dtype=float
    )

    upper_bounds = np.asarray(
        upper_bounds,
        dtype=float
    )

    new_lower = lower_bounds.copy()
    new_upper = upper_bounds.copy()

    for i in range(len(lower_bounds)):

        # ----------------------------------------------------------
        # EXPAND LOWER BOUND
        # ----------------------------------------------------------

        if hit_lower[i]:

            interval = max(
                upper_bounds[i] - lower_bounds[i],
                BOUND_EXPANSION_MPA
            )
            expansion = (
                max(BOUND_EXPANSION_MPA, interval)
                if USE_PROGRESSIVE_BOUND_EXPANSION
                else BOUND_EXPANSION_MPA
            )
            new_lower[i] = max(
                ABSOLUTE_MIN_MODULUS,
                lower_bounds[i] - expansion
            )

        # ----------------------------------------------------------
        # EXPAND UPPER BOUND
        # ----------------------------------------------------------

        if hit_upper[i]:

            interval = max(
                upper_bounds[i] - lower_bounds[i],
                BOUND_EXPANSION_MPA
            )
            expansion = (
                max(BOUND_EXPANSION_MPA, interval)
                if USE_PROGRESSIVE_BOUND_EXPANSION
                else BOUND_EXPANSION_MPA
            )
            new_upper[i] = min(
                ABSOLUTE_MAX_MODULUS,
                upper_bounds[i] + expansion
            )

    return (
        new_lower,
        new_upper
    )


# ================================================================
# 9A. CREATE NEXT-STAGE BOUNDS
# ================================================================
#
# This is the key change for your idea.
#
# Suppose:
#
#     Stage result:
#         E1 = 2000 MPa  -> upper bound hit
#         E2 = 70 MPa    -> correct / not at bound
#         E3 = 20 MPa    -> correct / not at bound
#
# Then the next stage becomes approximately:
#
#     E1 =  ... existing lower bound to 3000 MPa
#     E2 = 60 to 80 MPa
#     E3 = 10 to 30 MPa
#
# This prevents E2 and E3 from freely jumping back to their
# original large search ranges when E1 is expanded.
# ================================================================

def create_next_stage_bounds(
    current_lower_bounds,
    current_upper_bounds,
    best_E,
    hit_lower,
    hit_upper
):
    """
    Create the next search region.

    Only bounds that were actually hit are expanded:
      * lower-bound hit: lower bound decreases by 1000 MPa
      * upper-bound hit: upper bound increases by 1000 MPa

    Every modulus that did not hit a bound retains its complete
    current range unchanged. No +/-10 MPa local window is applied.
    """
    current_lower_bounds = np.asarray(current_lower_bounds, dtype=float)
    current_upper_bounds = np.asarray(current_upper_bounds, dtype=float)
    hit_lower = np.asarray(hit_lower, dtype=bool)
    hit_upper = np.asarray(hit_upper, dtype=bool)

    next_lower = current_lower_bounds.copy()
    next_upper = current_upper_bounds.copy()

    for i in range(len(next_lower)):
        if hit_lower[i]:
            next_lower[i] = max(
                ABSOLUTE_MIN_MODULUS,
                current_lower_bounds[i] - BOUND_EXPANSION_MPA
            )

        if hit_upper[i]:
            next_upper[i] = min(
                ABSOLUTE_MAX_MODULUS,
                current_upper_bounds[i] + BOUND_EXPANSION_MPA
            )

    return next_lower, next_upper


# ================================================================
# 10. CREATE STARTING POINTS
# ================================================================

def get_start_points(

    lower_bounds,

    upper_bounds,

    previous_best,

    stage_number

):


    rng = np.random.default_rng(

        RANDOM_SEED

        +

        stage_number
        *
        7919

    )


    starts = []


    # ============================================================
    # CENTER POINT
    # ============================================================

    center = (

        lower_bounds
        +
        upper_bounds

    ) / 2.0


    starts.append(
        center
    )


    # ============================================================
    # PREVIOUS BEST POINT
    # ============================================================

    if previous_best is not None:


        previous_start = np.clip(

            previous_best,

            lower_bounds,

            upper_bounds

        )


        starts.append(
            previous_start
        )


    # ============================================================
    # RANDOM STARTING POINTS
    # ============================================================

    while len(starts) < NUMBER_OF_STARTS:


        random_start = rng.uniform(

            lower_bounds,

            upper_bounds

        )


        starts.append(
            random_start
        )


    return starts


# ================================================================
# 11. ADAPTIVE BACKWARD SOLVER
# ================================================================

def adaptive_backward_solver(


    thickness,

    target,

    initial_lower_bounds,

    initial_upper_bounds


):


    # ============================================================
    # INPUT CONVERSION
    # ============================================================

    thickness = np.asarray(

        thickness,

        dtype=float

    )


    target = np.asarray(

        target,

        dtype=float

    )


    lower_bounds = np.asarray(

        initial_lower_bounds,

        dtype=float

    ).copy()


    upper_bounds = np.asarray(

        initial_upper_bounds,

        dtype=float

    ).copy()


    # ============================================================
    # HISTORY
    # ============================================================

    history = []


    evaluation_number = 0


    # ============================================================
    # GLOBAL BEST
    # ============================================================

    global_best_E = None

    global_best_calculated = None

    global_best_rmse = np.inf

    global_best_optimizer = None

    global_best_lower_bounds = None

    global_best_upper_bounds = None


    # ============================================================
    # EXPANSION LOOP
    # ============================================================

    for stage in range(

        MAX_BOUND_EXPANSIONS + 1

    ):


        print()

        print("=" * 80)

        print(

            f"BOUND EXPANSION STAGE {stage}"

        )

        print("=" * 80)


        print()

        print(
            "Current Modulus Bounds:"
        )


        for i in range(N):


            print(

                f"E{i + 1}: "

                f"{lower_bounds[i]:.4f}"

                f" to "

                f"{upper_bounds[i]:.4f}"

                f" MPa"

            )


        # ========================================================
        # STAGE BEST
        # ========================================================

        stage_best_E = None

        stage_best_calculated = None

        stage_best_rmse = np.inf

        stage_best_optimizer = None

        # Last modulus vector written to the history for this stage.
        # This prevents repeated rows such as 2250, 2250, 2250, 2250.
        last_history_E = None

        # Per-stage cache: key = rounded modulus vector, value =
        # calculated deflections, error vector, and RMSE.
        evaluation_cache = {}

        def evaluate_cached(E):

            E = np.asarray(E, dtype=float)

            cache_key = tuple(
                np.round(E, CACHE_DECIMAL_PLACES)
            )

            if cache_key in evaluation_cache:
                cached = evaluation_cache[cache_key]
                return (
                    cached[0].copy(),
                    cached[1].copy(),
                    cached[2]
                )

            calculated = calculate_deflections(
                thickness,
                E
            )

            if not np.all(np.isfinite(calculated)):
                error = np.ones(len(target)) * 1e6
                rmse = float(1e6)
            else:
                error = calculated - target
                rmse = float(
                    np.sqrt(np.mean(error ** 2))
                )

            evaluation_cache[cache_key] = (
                calculated.copy(),
                error.copy(),
                rmse
            )

            return calculated, error, rmse


        # ========================================================
        # RESIDUAL FUNCTION
        # ========================================================

        def residual(E):


            nonlocal evaluation_number, last_history_E


            calculated, error, rmse = evaluate_cached(E)


            # ----------------------------------------------------
            # INVALID RESULT            # ----------------------------------------------------
            # STORE HISTORY WITHOUT DUPLICATE MODULUS VECTORS
            # ----------------------------------------------------
            # least_squares can evaluate the same E several times.
            # Do not create a new Excel row unless the modulus vector
            # has materially changed from the previous recorded row
            # in this optimization stage.
            if (
                last_history_E is None
                or not np.allclose(
                    E,
                    last_history_E,
                    rtol=HISTORY_DEDUP_RTOL,
                    atol=HISTORY_DEDUP_ATOL
                )
            ):

                evaluation_number += 1

                hit_lower_live, hit_upper_live = check_bound_hit(
                    E, lower_bounds, upper_bounds, BOUND_TOLERANCE
                )
                hit_layers_live = [
                    f"E{i + 1} {'lower' if hit_lower_live[i] else 'upper'}"
                    for i in range(N)
                    if hit_lower_live[i] or hit_upper_live[i]
                ]
                history_row = {
                    "Stage": stage,
                    "Evaluation": evaluation_number,
                    "RMSE": rmse,
                    "Boundary_Hit": bool(hit_layers_live),
                    "Boundary_Hit_Layers": ", ".join(hit_layers_live),
                }


                for i in range(N):


                    history_row[
                        f"E{i + 1}"
                    ] = float(
                        E[i]
                    )


                    history_row[
                        f"Lower_Bound_E{i + 1}"
                    ] = float(
                        lower_bounds[i]
                    )


                    history_row[
                        f"Upper_Bound_E{i + 1}"
                    ] = float(
                        upper_bounds[i]
                    )


                history.append(
                    history_row
                )

                last_history_E = np.asarray(
                    E,
                    dtype=float
                ).copy()

                # Update Streamlit's live plots during the actual optimizer run.
                # The callback is optional, so the numerical solver remains usable
                # outside the Streamlit interface as well.
                if LIVE_PROGRESS_CALLBACK is not None:
                    try:
                        LIVE_PROGRESS_CALLBACK(
                            history=list(history),
                            E=np.asarray(E, dtype=float).copy(),
                            calculated=np.asarray(calculated, dtype=float).copy(),
                            target=np.asarray(target, dtype=float).copy(),
                            offsets=np.asarray(OFFSETS, dtype=float).copy(),
                            lower_bounds=np.asarray(lower_bounds, dtype=float).copy(),
                            upper_bounds=np.asarray(upper_bounds, dtype=float).copy(),
                            stage=stage,
                        )
                    except Exception as live_exc:
                        # A display issue must not interrupt the back-calculation.
                        print(f"Live graph update warning: {live_exc}")

            return error


        # ========================================================
        # STARTING POINTS
        # ========================================================

        starts = get_start_points(

            lower_bounds,

            upper_bounds,

            global_best_E,

            stage

        )


        # ========================================================
        # MULTIPLE START OPTIMIZATION
        # ========================================================

        for start_number, x0 in enumerate(

            starts,
            start=1

        ):


            print()

            print(

                f"Running start "

                f"{start_number}"

                f"/"

                f"{len(starts)}"

            )


            try:


                optimizer_result = least_squares(


                    residual,


                    x0=x0,


                    bounds=(

                        lower_bounds,

                        upper_bounds

                    ),


                    method="trf",


                    jac="2-point",


                    x_scale="jac",


                    loss="linear",


                    ftol=1e-10,


                    xtol=1e-10,


                    gtol=1e-10,


                    max_nfev=MAX_NFEV,


                    verbose=0


                )


                # ------------------------------------------------
                # CANDIDATE MODULUS
                # ------------------------------------------------

                candidate_E = np.clip(

                    optimizer_result.x,

                    lower_bounds,

                    upper_bounds

                )


                # ------------------------------------------------
                # FORWARD CALCULATION
                # ------------------------------------------------

                candidate_calculated, candidate_error, candidate_rmse = (
                    evaluate_cached(candidate_E)
                )


                print(

                    f"Start {start_number} RMSE: "

                    f"{candidate_rmse:.10e}"

                )


                # ------------------------------------------------
                # STAGE BEST
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <
                    stage_best_rmse

                ):


                    stage_best_rmse = (

                        candidate_rmse

                    )


                    stage_best_E = (

                        candidate_E.copy()

                    )


                    stage_best_calculated = (

                        candidate_calculated.copy()

                    )


                    stage_best_optimizer = (

                        optimizer_result

                    )


                # ------------------------------------------------
                # GLOBAL BEST
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <
                    global_best_rmse

                ):


                    global_best_rmse = (

                        candidate_rmse

                    )


                    global_best_E = (

                        candidate_E.copy()

                    )


                    global_best_calculated = (

                        candidate_calculated.copy()

                    )


                    global_best_optimizer = (

                        optimizer_result

                    )


                    global_best_lower_bounds = (

                        lower_bounds.copy()

                    )


                    global_best_upper_bounds = (

                        upper_bounds.copy()

                    )


                # ------------------------------------------------
                # TARGET REACHED
                # ------------------------------------------------

                if (

                    candidate_rmse
                    <=
                    TARGET_RMSE

                ):

                    break


            except Exception as exc:


                print()

                print(

                    "Optimization error:",

                    exc

                )


                continue


        # ========================================================
        # CHECK STAGE RESULT
        # ========================================================

        if stage_best_E is None:


            print()

            print(

                "No valid solution found in this stage."

            )


            break


        print()

        print("-" * 80)

        print(

            f"BEST RESULT AT STAGE {stage}"

        )

        print("-" * 80)


        for i in range(N):


            print(

                f"E{i + 1} = "

                f"{stage_best_E[i]:.6f} MPa"

            )


        print()

        print(

            f"RMSE = "

            f"{stage_best_rmse:.10e} mm"

        )


        # ========================================================
        # TARGET ACHIEVED
        # ========================================================

        if (

            stage_best_rmse
            <=
            TARGET_RMSE

        ):


            print()

            print(

                "TARGET RMSE ACHIEVED."

            )


            break


        # ========================================================
        # CHECK BOUND HITS
        # ========================================================

        hit_lower, hit_upper = (

            check_bound_hit(

                stage_best_E,

                lower_bounds,

                upper_bounds,

                BOUND_TOLERANCE

            )

        )


        print()

        print(

            "BOUNDARY CHECK"

        )


        any_bound_hit = False


        for i in range(N):


            if hit_lower[i]:


                print(

                    f"E{i + 1} is near the LOWER bound."

                )


                any_bound_hit = True


            if hit_upper[i]:


                print(

                    f"E{i + 1} is near the UPPER bound."

                )


                any_bound_hit = True


        # ========================================================
        # NO BOUND HIT
        # ========================================================

        if not any_bound_hit:


            print()

            print(

                "RMSE target was not achieved, but the solution "
                "is not hitting any modulus boundary."

            )


            print(

                "Therefore, automatic expansion is stopped."

            )


            break


        # ========================================================
        # CREATE NEXT-STAGE BOUNDS
        # ========================================================
        #
        # KEY CHANGE:
        #
        # 1. The modulus that hit a bound is expanded.
        #
        # 2. Every modulus that did NOT hit a bound is kept near
        #    its previous recovered value.
        #
        # Example:
        #
        #     E1 hits upper bound -> E1 upper + 1000
        #     E2 = 70             -> E2 = 60 to 80
        #     E3 = 20             -> E3 = 10 to 30
        #
        # This allows the forward solver to re-adjust E2 and E3,
        # but prevents them from jumping over the entire original
        # search range.
        # ========================================================

        new_lower_bounds, new_upper_bounds = (

            create_next_stage_bounds(

                lower_bounds,

                upper_bounds,

                stage_best_E,

                hit_lower,

                hit_upper

            )

        )


        # ========================================================
        # CHECK WHETHER ANY BOUND CHANGED
        # ========================================================

        bounds_changed = (

            not np.allclose(

                new_lower_bounds,

                lower_bounds

            )

            or

            not np.allclose(

                new_upper_bounds,

                upper_bounds

            )

        )


        if not bounds_changed:

            print()

            print(

                "Bounds cannot be expanded further."

            )

            break


        # ========================================================
        # DISPLAY NEXT-STAGE BOUNDS
        # ========================================================

        print()

        print(

            "CREATING NEXT OPTIMIZATION SEARCH REGION..."

        )


        for i in range(N):

            if hit_lower[i]:

                print(

                    f"E{i + 1}: LOWER bound hit -> "

                    f"expanded to "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )

            elif hit_upper[i]:

                print(

                    f"E{i + 1}: UPPER bound hit -> "

                    f"expanded to "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )

            else:

                print(

                    f"E{i + 1}: preserved around "

                    f"{stage_best_E[i]:.4f} MPa -> "

                    f"{new_lower_bounds[i]:.4f} to "

                    f"{new_upper_bounds[i]:.4f} MPa"

                )


        # ========================================================
        # UPDATE BOUNDS
        # ========================================================

        lower_bounds = new_lower_bounds

        upper_bounds = new_upper_bounds


    # ============================================================
    # FINAL CHECK
    # ============================================================

    if global_best_E is None:


        return {

            "success":
                False,

            "history":
                history

        }


    # ============================================================
    # FINAL RESULTS
    # ============================================================

    final_E = (

        global_best_E.copy()

    )


    final_calculated = (

        calculate_deflections(

            thickness,

            final_E

        )

    )


    final_error = (

        final_calculated
        -
        target

    )


    absolute_error = np.abs(

        final_error

    )


    rmse = float(

        np.sqrt(

            np.mean(

                final_error ** 2

            )

        )

    )


    mae = float(

        np.mean(

            absolute_error

        )

    )


    maximum_error = float(

        np.max(

            absolute_error

        )

    )


    percentage_error = np.where(

        np.abs(target) > 1e-15,

        absolute_error
        /
        np.abs(target)
        *
        100.0,

        0.0

    )


    r2 = calculate_r2(

        target,

        final_calculated

    )


    return {


        "success":
            True,


        "Back_E":
            final_E,


        "Calculated":
            final_calculated,


        "Target":
            target,


        "Error":
            final_error,


        "Percentage_Error":
            percentage_error,


        "RMSE":
            rmse,


        "MAE":
            mae,


        "Maximum_Error":
            maximum_error,


        "R2":
            r2,


        "Target_Achieved":

            rmse <= TARGET_RMSE,


        "Best_Lower_Bounds":

            global_best_lower_bounds,


        "Best_Upper_Bounds":

            global_best_upper_bounds,


        "NFEV":

            (

                global_best_optimizer.nfev

                if global_best_optimizer is not None

                else np.nan

            ),


        "History":
            history

    }


# ================================================================
# 12. CREATE RESULTS DATAFRAME
# ================================================================

def create_results_dataframe(

    result

):


    row = {}


    # ============================================================
    # RECOVERED MODULI
    # ============================================================

    for i in range(N):


        row[

            f"Back E{i + 1} (MPa)"

        ] = result["Back_E"][i]


    # ============================================================
    # BOUNDS
    # ============================================================

    for i in range(N):


        row[

            f"Initial Lower E{i + 1}"

        ] = INITIAL_LOWER_BOUNDS[i]


        row[

            f"Initial Upper E{i + 1}"

        ] = INITIAL_UPPER_BOUNDS[i]


        row[

            f"Best Lower Bound E{i + 1}"

        ] = result["Best_Lower_Bounds"][i]


        row[

            f"Best Upper Bound E{i + 1}"

        ] = result["Best_Upper_Bounds"][i]


    # ============================================================
    # ERROR STATISTICS
    # ============================================================

    row["RMSE (mm)"] = result["RMSE"]

    row["MAE (mm)"] = result["MAE"]

    row["Maximum Error (mm)"] = (

        result["Maximum_Error"]

    )

    row["R2"] = result["R2"]

    row["Target Achieved"] = (

        result["Target_Achieved"]

    )

    row["NFEV"] = result["NFEV"]


    # ============================================================
    # DEFLECTION RESULTS
    # ============================================================

    for i in range(NUMBER_OF_POINTS):


        radius = OFFSETS[i]


        row[

            f"Target Point {i + 1}"

        ] = result["Target"][i]


        row[

            f"Calculated Point {i + 1}"

        ] = result["Calculated"][i]


        row[

            f"Error Point {i + 1}"

        ] = result["Error"][i]


        row[

            f"Percentage Error Point {i + 1}"

        ] = result["Percentage_Error"][i]


        row[

            f"Radius Point {i + 1} (mm)"

        ] = radius


    return pd.DataFrame(

        [row]

    )


# ================================================================
# 13. CREATE DEFLECTION GRAPH
# ================================================================

def create_deflection_graph(

    result

):


    plt.figure(

        figsize=(9, 6)

    )


    plt.plot(

        OFFSETS,

        result["Target"],

        marker="o",

        linewidth=2,

        label="Target Deflection"

    )


    plt.plot(

        OFFSETS,

        result["Calculated"],

        marker="s",

        linestyle="--",

        linewidth=2,

        label="Back-calculated Deflection"

    )


    plt.xlabel(

        "Radial Distance (mm)"

    )


    plt.ylabel(

        "Deflection (mm)"

    )


    plt.title(

        "Target vs Back-calculated Deflections"

    )


    plt.grid(

        True,

        alpha=0.3

    )


    plt.legend()


    plt.tight_layout()


    graph_file = os.path.join(

        OUTPUT_FOLDER,

        "Deflection_Comparison.png"

    )


    plt.savefig(

        graph_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()


# ================================================================
# 14. CREATE RMSE CONVERGENCE GRAPH
# ================================================================

def create_rmse_graph(

    history

):


    if len(history) == 0:

        return


    history_df = pd.DataFrame(

        history

    )


    if "RMSE" not in history_df.columns:

        return


    plt.figure(

        figsize=(9, 6)

    )


    plt.plot(

        history_df["Evaluation"],

        history_df["RMSE"],

        linewidth=1.5

    )


    plt.xlabel(

        "Function Evaluation"

    )


    plt.ylabel(

        "RMSE (mm)"

    )


    plt.title(

        "RMSE Convergence"

    )


    plt.yscale(
        "log"
    )


    plt.grid(

        True,

        alpha=0.3

    )


    plt.tight_layout()


    graph_file = os.path.join(

        OUTPUT_FOLDER,

        "RMSE_Convergence.png"

    )


    plt.savefig(

        graph_file,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()




def run_streamlit_app():
    global N, THICKNESS, INITIAL_LOWER_BOUNDS, INITIAL_UPPER_BOUNDS, POISSON
    global PRESSURE_MPA, LOAD_RADIUS_MM, NUMBER_OF_POINTS, OFFSETS, POINTS, TARGET
    global TARGET_RMSE, OUTPUT_FOLDER

    st.set_page_config(page_title="Pavement Back-Calculation", page_icon="🛣️", layout="wide")
    st.title("IITKGP-Style Multi-Layer Pavement Back-Calculation")
    st.caption("Adaptive modulus-bound expansion · numerical solver retained from your supplied Python code")
    st.info("Enter model parameters and target deflections, then run the back-calculation. The results are displayed here and can be downloaded as an Excel workbook.")

    top1, top2 = st.columns(2)
    with top1:
        N = st.number_input("Number of layers (minimum 2)", min_value=2, max_value=10, value=2, step=1)
    with top2:
        NUMBER_OF_POINTS = st.number_input("Number of evaluation points", min_value=1, max_value=100, value=1, step=1)

    with st.form("calculation_inputs"):
        st.subheader("Layer properties")
        thickness_values=[]
        lower_values=[]
        upper_values=[]
        poisson_values=[]
        for i in range(int(N)-1):
            c1,c2=st.columns(2)
            with c1:
                thickness_values.append(st.number_input(f"Layer {i+1} thickness (mm)", min_value=0.001, value=0.001, step=10.0, key=f"th_{i}"))
            with c2:
                st.write(f"Layer {i+1} is finite. Layer {int(N)} is semi-infinite.")
        st.markdown("**Initial modulus bounds (MPa) and Poisson ratios**")
        for i in range(int(N)):
            a,b,c=st.columns(3)
            with a:
                lo_default=1.0
                lower_values.append(st.number_input(f"E{i+1} minimum (MPa)", min_value=1.0, value=float(lo_default), step=10.0, key=f"lo_{i}"))
            with b:
                hi_default=2.0
                upper_values.append(st.number_input(f"E{i+1} maximum (MPa)", min_value=1.001, value=float(hi_default), step=10.0, key=f"hi_{i}"))
            with c:
                poisson_values.append(st.number_input(f"Layer {i+1} Poisson ratio", min_value=0.001, max_value=0.499, value=0.001, step=0.01, format="%.3f", key=f"nu_{i}"))

        st.subheader("Loading")
        c1,c2,c3=st.columns(3)
        with c1: pressure=st.number_input("Loading pressure (MPa)", min_value=0.000001, value=0.000001, format="%.6f")
        with c2: radius=st.number_input("Load radius (mm)", min_value=0.001, value=0.001, step=5.0)
        with c3: target_rmse=st.number_input("Target RMSE (mm)", min_value=1e-15, value=1e-15, format="%.12g")

        st.subheader("Deflection evaluation points")
        st.caption("Enter radius, depth, layer number, and target deflection for each point. Depth and radius are in mm; deflection is in mm.")
        default_r=[0.0]
        default_targets=[0.0]
        point_rows=[]
        for i in range(int(NUMBER_OF_POINTS)):
            st.markdown(f"**Point {i+1}**")
            a,b,c,d=st.columns(4)
            with a: r=st.number_input(f"Radius r (mm), point {i+1}", min_value=0.0, value=0.0, step=10.0, key=f"r_{i}")
            with b: z=st.number_input(f"Depth z (mm), point {i+1}", min_value=0.0, value=0.0, step=10.0, key=f"z_{i}")
            with c: layer=st.number_input(f"Layer number, point {i+1}", min_value=1, max_value=int(N), value=1, step=1, key=f"layer_{i}")
            with d: target=st.number_input(f"Target deflection (mm), point {i+1}", value=default_targets[i] if i<len(default_targets) else 0.0, format="%.12g", key=f"target_{i}")
            point_rows.append((r,z,int(layer),target))

        submitted=st.form_submit_button("Run back-calculation", type="primary", use_container_width=True)

    if not submitted:
        st.markdown("### What this app returns")
        st.markdown("- Recovered layer moduli and error statistics\n- Target vs calculated deflections\n- Optimization history and adaptive-bound stages\n- Downloadable Excel workbook and convergence plots")
        return

    # Validate all entered values before starting the numerical solver.
    if any(upper_values[i] <= lower_values[i] for i in range(int(N))):
        st.error("Each maximum modulus must be greater than its minimum modulus.")
        return
    if any(layer < 1 or layer > int(N) for _,_,layer,_ in point_rows):
        st.error("Each evaluation point's layer number must be between 1 and the number of layers.")
        return

    THICKNESS=np.asarray(thickness_values,dtype=float)
    INITIAL_LOWER_BOUNDS=np.asarray(lower_values,dtype=float)
    INITIAL_UPPER_BOUNDS=np.asarray(upper_values,dtype=float)
    POISSON=np.asarray(poisson_values,dtype=float)
    PRESSURE_MPA=float(pressure)
    LOAD_RADIUS_MM=float(radius)
    TARGET_RMSE=float(target_rmse)
    OFFSETS=np.asarray([p[0] for p in point_rows],dtype=float)
    POINTS=[{"r":float(r),"z":float(z),"layer":int(layer)} for r,z,layer,_ in point_rows]
    TARGET=np.asarray([p[3] for p in point_rows],dtype=float)

    # Check depth is consistent with the declared layer. Finite layer boundaries are cumulative thicknesses.
    boundaries=np.cumsum(THICKNESS)
    for idx,(r,z,layer,tgt) in enumerate(point_rows):
        if z < 0:
            st.error(f"Point {idx+1}: depth cannot be negative."); return
        if layer < int(N) and z > boundaries[layer-1] + 1e-9:
            st.warning(f"Point {idx+1}: depth {z:g} mm lies below the bottom of declared Layer {layer}; check your layer assignment.")

    global LIVE_PROGRESS_CALLBACK
    OUTPUT_FOLDER=tempfile.mkdtemp(prefix="pavement_backcalc_")
    start_time=time.time()
    progress=st.progress(0, text="Running adaptive multi-start least-squares optimization…")
    live_status=st.empty()
    live_moduli=st.empty()
    live_cols=st.columns(2)
    with live_cols[0]:
        st.markdown("#### Live deflection bowl")
        live_deflection_plot=st.empty()
    with live_cols[1]:
        st.markdown("#### Live RMSE convergence")
        live_rmse_plot=st.empty()
    log=io.StringIO()
    last_live_update = {"evaluation": 0}

    def update_live_plots(history, E, calculated, target, offsets, lower_bounds, upper_bounds, stage):
        if not history:
            return
        current_eval = len(history)
        # Avoid excessive redraws while still showing meaningful live movement.
        # Always redraw boundary hits and the final point; otherwise redraw every 2 rows.
        row = history[-1]
        boundary_hit = bool(row.get("Boundary_Hit", False))
        if current_eval != 1 and current_eval % 2 != 0 and not boundary_hit:
            return
        last_live_update["evaluation"] = current_eval
        live_status.info(f"Live optimization · stage {stage} · evaluation {row.get('Evaluation', current_eval)} · RMSE = {row.get('RMSE', np.nan):.6e} mm")
        live_moduli.markdown("**Current moduli (MPa):** " + "  |  ".join(f"E{i+1} = {value:.6f}" for i, value in enumerate(E)))

        fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
        ax1.plot(offsets, target, marker="o", linewidth=2, label="Target deflection")
        ax1.plot(offsets, calculated, marker="s", linestyle="--", linewidth=2, label="Current back-calculated deflection")
        ax1.set_xlabel("Radial Distance (mm)")
        ax1.set_ylabel("Deflection (mm)")
        ax1.set_title("Target vs Current Back-calculated Deflections")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        fig1.tight_layout()
        live_deflection_plot.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        hist_df = pd.DataFrame(history)
        valid = np.isfinite(pd.to_numeric(hist_df["RMSE"], errors="coerce")) & (pd.to_numeric(hist_df["RMSE"], errors="coerce") > 0)
        if valid.any():
            xs = pd.to_numeric(hist_df.loc[valid, "Evaluation"], errors="coerce")
            ys = pd.to_numeric(hist_df.loc[valid, "RMSE"], errors="coerce")
            ax2.plot(xs, ys, linewidth=1.8, label="RMSE")
            hit_mask = hist_df.loc[valid, "Boundary_Hit"].fillna(False).astype(bool) if "Boundary_Hit" in hist_df else pd.Series(False, index=xs.index)
            if hit_mask.any():
                ax2.scatter(xs[hit_mask], ys[hit_mask], color="crimson", marker="X", s=75, zorder=5, label="Modulus boundary hit")
            ax2.scatter([xs.iloc[-1]], [ys.iloc[-1]], color="darkorange", s=45, zorder=6, label="Current evaluation")
            ax2.set_yscale("log")
        else:
            ax2.text(0.5, 0.5, "Waiting for valid RMSE evaluations…", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_xlabel("Function Evaluation")
        ax2.set_ylabel("RMSE (mm)")
        ax2.set_title("Live RMSE Convergence")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best")
        fig2.tight_layout()
        live_rmse_plot.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    LIVE_PROGRESS_CALLBACK = update_live_plots
    try:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result=adaptive_backward_solver(THICKNESS,TARGET,INITIAL_LOWER_BOUNDS,INITIAL_UPPER_BOUNDS)
        elapsed=time.time()-start_time
        progress.progress(100, text="Calculation finished.")
        live_status.success("Optimization finished. The live graphs above show the final solver state.")
    except Exception as exc:
        progress.empty()
        st.error(f"Calculation failed: {exc}")
        with st.expander("Diagnostic log"): st.code(log.getvalue() or repr(exc))
        return
    finally:
        LIVE_PROGRESS_CALLBACK = None

    if not result.get("success"):
        st.error("Back-calculation failed: no valid solution was found.")
        with st.expander("Optimization log"): st.code(log.getvalue())
        return

    st.subheader("Final results")
    status="Target RMSE achieved" if result["Target_Achieved"] else "Best solution found; target RMSE not achieved"
    if result["Target_Achieved"]: st.success(status)
    else: st.warning(status)
    metrics=st.columns(4)
    metrics[0].metric("RMSE (mm)",f"{result['RMSE']:.6e}")
    metrics[1].metric("MAE (mm)",f"{result['MAE']:.6e}")
    metrics[2].metric("Maximum absolute error (mm)",f"{result['Maximum_Error']:.6e}")
    metrics[3].metric("R²",f"{result['R2']:.10f}" if np.isfinite(result['R2']) else "N/A")
    modulus_df=pd.DataFrame({"Layer":[f"E{i+1}" for i in range(int(N))],"Recovered modulus (MPa)":result["Back_E"],"Initial minimum (MPa)":INITIAL_LOWER_BOUNDS,"Initial maximum (MPa)":INITIAL_UPPER_BOUNDS,"Best-stage lower bound (MPa)":result["Best_Lower_Bounds"],"Best-stage upper bound (MPa)":result["Best_Upper_Bounds"]})
    st.markdown("**Recovered layer moduli**")
    st.dataframe(modulus_df,use_container_width=True,hide_index=True)

    comparison_df=pd.DataFrame({"Point":np.arange(1,len(TARGET)+1),"Radius (mm)":OFFSETS,"Depth (mm)":[p["z"] for p in POINTS],"Layer":[p["layer"] for p in POINTS],"Target deflection (mm)":result["Target"],"Calculated deflection (mm)":result["Calculated"],"Error (mm)":result["Error"],"Absolute error (mm)":np.abs(result["Error"]),"Percentage error (%)":result["Percentage_Error"]})
    st.markdown("**Deflection comparison**")
    st.dataframe(comparison_df,use_container_width=True,hide_index=True)
    st.line_chart(comparison_df.set_index("Radius (mm)")[["Target deflection (mm)","Calculated deflection (mm)"]])

    results_df=create_results_dataframe(result)
    history_df=pd.DataFrame(result["History"])
    # Keep the original workbook sheets and add readable detailed tables.
    excel_buffer=io.BytesIO()
    with pd.ExcelWriter(excel_buffer,engine="openpyxl") as writer:
        results_df.to_excel(writer,sheet_name="Final Results",index=False)
        history_df.to_excel(writer,sheet_name="Optimization History",index=False)
        modulus_df.to_excel(writer,sheet_name="Recovered Moduli",index=False)
        comparison_df.to_excel(writer,sheet_name="Deflection Comparison",index=False)
        inputs_df=pd.DataFrame({"Parameter":["Number of layers","Finite thicknesses (mm)","Poisson ratios","Pressure (MPa)","Load radius (mm)","Target RMSE (mm)","Bonded interfaces","Max bound expansions","Elapsed time (s)"],"Value":[int(N),", ".join(map(str,THICKNESS)),", ".join(map(str,POISSON)),PRESSURE_MPA,LOAD_RADIUS_MM,TARGET_RMSE,IS_BONDED,MAX_BOUND_EXPANSIONS,elapsed]})
        inputs_df.to_excel(writer,sheet_name="Input Summary",index=False)
    excel_bytes=excel_buffer.getvalue()
    st.download_button("Download Excel results",data=excel_bytes,file_name="Adaptive_Back_Calculation_Results.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",type="primary",use_container_width=True)

    create_deflection_graph(result)
    create_rmse_graph(result["History"])
    c1,c2=st.columns(2)
    deflection_png=Path(OUTPUT_FOLDER)/"Deflection_Comparison.png"
    rmse_png=Path(OUTPUT_FOLDER)/"RMSE_Convergence.png"
    with c1:
        if deflection_png.exists():
            st.markdown("**Deflection comparison plot**")
            st.image(str(deflection_png),use_container_width=True)
            st.download_button("Download deflection plot",deflection_png.read_bytes(),file_name="Deflection_Comparison.png",mime="image/png")
    with c2:
        if rmse_png.exists():
            st.markdown("**RMSE convergence plot**")
            st.image(str(rmse_png),use_container_width=True)
            st.download_button("Download RMSE plot",rmse_png.read_bytes(),file_name="RMSE_Convergence.png",mime="image/png")
    st.caption(f"Calculation time: {elapsed:.2f} seconds · Optimization evaluations recorded: {len(history_df)}")
    with st.expander("Optimization stages and terminal log"):
        st.code(log.getvalue() or "No solver log was emitted.")

if __name__ == "__main__":
    run_streamlit_app()
