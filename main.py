#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from price_coord_demo.experiment import ExperimentConfig, ExperimentRunner
from price_coord_demo.pybullet_exec import PyBulletExecutor
from price_coord_demo.streaming import OnlineTargetStreamDemo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price-guided coordination demo for multi-arm manipulation in PyBullet"
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--num-targets", type=int, default=6, help="Number of target points (4~8 recommended)")
    parser.add_argument("--grid-size", type=int, default=10, help="Cubic workspace discretization, e.g. 10 means 10x10x10")
    parser.add_argument("--robot-base-radius", type=float, default=0.66, help="Radius of the 4-arm ring layout around the workspace center")
    parser.add_argument("--robot-home-radius", type=float, default=0.20, help="Nominal end-effector home radius used to seed path generation")
    parser.add_argument("--max-iters", type=int, default=12, help="Max pricing iterations")
    parser.add_argument("--min-pricing-iters", type=int, default=6, help="Minimum pricing iterations before convergence")
    parser.add_argument("--alpha", type=float, default=1.0, help="Price update step size")
    parser.add_argument("--decay", type=float, default=0.12, help="Price decay rate")
    parser.add_argument("--price-weight", type=float, default=1.6, help="Price penalty weight in score")
    parser.add_argument("--max-price", type=float, default=25.0, help="Price clipping upper bound")
    parser.add_argument("--switch-threshold", type=float, default=1.5, help="Hysteresis margin to avoid path flip-flop")
    parser.add_argument("--time-horizon", type=int, default=24, help="Number of future time bins in the space-time pricing grid")
    parser.add_argument("--time-bin-size", type=float, default=0.8, help="Seconds represented by each time bin in the space-time pricing grid")
    parser.add_argument("--nominal-speed", type=float, default=0.32, help="Nominal end-effector speed used to stamp path occupancy into future time bins")
    parser.add_argument("--array-backend", choices=("numpy", "cupy"), default="numpy", help="Array backend for price/demand tensors")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots")
    parser.add_argument("--skip-pybullet", action="store_true", help="Skip PyBullet execution layer")
    parser.add_argument("--gui", action="store_true", help="Run PyBullet in GUI mode")
    parser.add_argument("--joint-max-velocity", type=float, default=0.9, help="Max joint speed used by the PyBullet execution layer")
    parser.add_argument("--real-time-factor", type=float, default=4.0, help="GUI simulation speedup relative to wall-clock time; 1.0 is near real-time")
    parser.add_argument("--stream-targets", action="store_true", help="Run an online random target stream in PyBullet")
    parser.add_argument("--batch-targets", action="store_true", help="Spawn a fixed batch of random targets at t=0 and measure how long it takes to clear them")
    parser.add_argument("--stream-points", type=int, default=100, help="How many random targets to generate in stream mode")
    parser.add_argument("--stream-duration", type=float, default=100.0, help="How many simulated seconds to run the stream mode")
    parser.add_argument("--batch-points", type=int, default=40, help="How many random targets to spawn at t=0 in batch mode")
    parser.add_argument("--batch-max-duration", type=float, default=120.0, help="Maximum simulated seconds to allow for clearing the initial batch")
    parser.add_argument("--stream-replan-interval", type=float, default=1.0, help="Seconds between online replanning updates in stream mode")
    parser.add_argument("--stream-touch-threshold", type=float, default=0.16, help="Distance threshold for considering a streaming target touched")
    parser.add_argument("--stream-min-deadline", type=float, default=8.0, help="Minimum countdown duration for a streaming target")
    parser.add_argument("--stream-max-deadline", type=float, default=12.0, help="Maximum countdown duration for a streaming target")
    parser.add_argument("--stream-urgency-bonus", type=float, default=12.0, help="Bonus added as a target approaches its deadline")
    parser.add_argument("--stream-overtime-rate", type=float, default=3.0, help="Per-second bonus added after a target misses its deadline")
    parser.add_argument("--stream-overtime-saturation", type=float, default=12.0, help="Saturation timescale for overdue utility growth in stream mode")
    parser.add_argument("--stream-max-ik-error", type=float, default=0.18, help="Reject streamed assignments whose coarse IK error exceeds this threshold")
    parser.add_argument("--stream-assignment-touch-slack", type=float, default=0.01, help="Extra slack beyond touch threshold allowed by the assignment-time IK gate")
    parser.add_argument("--stream-progress-epsilon", type=float, default=0.015, help="Minimum distance improvement that counts as progress in stream mode")
    parser.add_argument("--stream-stall-timeout", type=float, default=4.0, help="Release a streamed assignment if it makes no progress for this many seconds")
    parser.add_argument("--stream-reassign-cooldown", type=float, default=6.0, help="Seconds before a robot retries a streamed target it previously failed to reach")
    parser.add_argument("--stream-max-assignment-age", type=float, default=20.0, help="Hard cap on how long a robot will chase one streamed target")
    parser.add_argument("--stream-soft-touch-margin", type=float, default=0.02, help="If a stalled robot is this close beyond the touch threshold, count the target as softly completed")
    parser.add_argument("--strict-touch", action="store_true", help="Require true touch-threshold contact; disables soft completion and removes extra assignment slack")
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=5.0,
        help="When GUI is enabled, keep scene open for a short duration after execution",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        num_targets=args.num_targets,
        grid_size=args.grid_size,
        robot_base_radius=args.robot_base_radius,
        robot_home_radius=args.robot_home_radius,
        max_iters=args.max_iters,
        min_pricing_iters=args.min_pricing_iters,
        alpha=args.alpha,
        decay=args.decay,
        price_weight=args.price_weight,
        max_price=args.max_price,
        switch_threshold=args.switch_threshold,
        time_horizon=args.time_horizon,
        time_bin_size=args.time_bin_size,
        nominal_speed=args.nominal_speed,
        array_backend=args.array_backend,
        output_dir=args.output_dir,
    )

    runner = ExperimentRunner(config)
    online_mode = args.batch_targets or args.stream_targets
    outcome = None if online_mode else runner.run(verbose=True)

    if args.skip_pybullet:
        if online_mode:
            print("[PyBullet skipped] batch/stream modes require the PyBullet execution layer; remove --skip-pybullet.")
        return

    executor = PyBulletExecutor(
        gui=args.gui,
        joint_max_velocity=args.joint_max_velocity,
        real_time_factor=args.real_time_factor,
    )
    try:
        executor.connect()
        robot_bases, home_positions = runner._robot_layout()
        executor.setup_scene(robot_bases)
        if args.batch_targets:
            stream_workspace = runner._create_workspace()
            stream_workspace.reset_prices()
            stream_demo = OnlineTargetStreamDemo(
                runner=runner,
                executor=executor,
                workspace=stream_workspace,
                seed=args.seed + 1000,
                robot_bases=robot_bases,
                home_positions=home_positions,
                replan_interval=args.stream_replan_interval,
                touch_threshold=args.stream_touch_threshold,
                min_deadline=args.stream_min_deadline,
                max_deadline=args.stream_max_deadline,
                urgency_bonus=args.stream_urgency_bonus,
                overtime_bonus_rate=args.stream_overtime_rate,
                overtime_saturation=args.stream_overtime_saturation,
                max_ik_error=args.stream_max_ik_error,
                assignment_touch_slack=args.stream_assignment_touch_slack,
                progress_epsilon=args.stream_progress_epsilon,
                stall_timeout=args.stream_stall_timeout,
                reassign_cooldown=args.stream_reassign_cooldown,
                max_assignment_age=args.stream_max_assignment_age,
                soft_touch_margin=args.stream_soft_touch_margin,
                strict_touch=args.strict_touch,
            )
            batch_result = stream_demo.run_batch(
                total_points=args.batch_points,
                max_duration=args.batch_max_duration,
                verbose=True,
            )
            clear_text = (
                f"{batch_result.clear_time:.1f}s"
                if batch_result.clear_time is not None
                else "not cleared"
            )
            print(
                "\n=== Batch Summary ===\n"
                f"generated={batch_result.generated}, touched={batch_result.touched}, "
                f"soft_touched={batch_result.soft_touched}, "
                f"remaining={batch_result.remaining}, simulated_seconds={batch_result.simulated_seconds:.1f}, "
                f"cleared={batch_result.cleared}, clear_time={clear_text}"
            )
        elif args.stream_targets:
            stream_workspace = runner._create_workspace()
            stream_workspace.reset_prices()
            stream_demo = OnlineTargetStreamDemo(
                runner=runner,
                executor=executor,
                workspace=stream_workspace,
                seed=args.seed + 1000,
                robot_bases=robot_bases,
                home_positions=home_positions,
                replan_interval=args.stream_replan_interval,
                touch_threshold=args.stream_touch_threshold,
                min_deadline=args.stream_min_deadline,
                max_deadline=args.stream_max_deadline,
                urgency_bonus=args.stream_urgency_bonus,
                overtime_bonus_rate=args.stream_overtime_rate,
                overtime_saturation=args.stream_overtime_saturation,
                max_ik_error=args.stream_max_ik_error,
                assignment_touch_slack=args.stream_assignment_touch_slack,
                progress_epsilon=args.stream_progress_epsilon,
                stall_timeout=args.stream_stall_timeout,
                reassign_cooldown=args.stream_reassign_cooldown,
                max_assignment_age=args.stream_max_assignment_age,
                soft_touch_margin=args.stream_soft_touch_margin,
                strict_touch=args.strict_touch,
            )
            stream_result = stream_demo.run(
                total_points=args.stream_points,
                duration=args.stream_duration,
                verbose=True,
            )
            print(
                "\n=== Stream Summary ===\n"
                f"generated={stream_result.generated}, touched={stream_result.touched}, "
                f"soft_touched={stream_result.soft_touched}, "
                f"remaining={stream_result.remaining}, simulated_seconds={stream_result.simulated_seconds:.1f}"
            )
        else:
            assert outcome is not None
            executor.add_target_markers(outcome.targets)
            executor.draw_paths(outcome.agents_pricing, selected_paths=outcome.selected_paths_pricing)
            executor.draw_high_price_cells(outcome.workspace_pricing, threshold=2.0, max_cells=28)
            executor.execute_paths(outcome.selected_paths_pricing, waypoint_steps=90)
        if args.gui and args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
    except RuntimeError as exc:
        print(f"[PyBullet skipped] {exc}")
    finally:
        executor.disconnect()


if __name__ == "__main__":
    main()
