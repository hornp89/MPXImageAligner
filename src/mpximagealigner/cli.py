"""
CLI entry point for MPXImageAligner.

Usage:
    MPXImageAligner gui          — Open the graphical user interface
    MPXImageAligner align ...    — Run image alignment from the command line
"""

import argparse


def _build_align_parser(argparser: argparse.ArgumentParser) -> None:
    """Populate *parser* with all alignment arguments."""
    
    argparser.add_argument(
        "src_dir",
        help="Source directory containing images to align."
    )
    argparser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <src_dir>_aligned)."
    )
    argparser.add_argument(
         "--search_ref", action="store_true",
         help="Search for best reference image by incrementing ref_file_no until loss threshold is met "
              "(default: False)."
    )
    argparser.add_argument(
        "--mode", choices=["single", "batch"], default="single",
        help="Run mode: 'single' for processing one folder, 'batch' to process all folders (default: single)."
    )
    argparser.add_argument(
        "--ref_file_no", type=int, default=0,
        help="Index of the reference image file (default: 0)."
    )
    argparser.add_argument(
        "--method", choices=["rigid", "affine"], default="affine",
        help="Registration method (default: affine)."
    )
    argparser.add_argument(
        "--size_factor", type=int, default=4,
        help="Downsample factor for registration (default: 4). "
             "Good choices for WSI on a 4 GB GPU are 4 for affine and 2 for rigid."
    )
    argparser.add_argument(
        "--lr", type=float, default=1,
        help="Learning rate (default: 1)."
    )
    argparser.add_argument(
        "--num_epochs", type=int, default=5,
        help="Number of training epochs (default: 5)."
    )
    argparser.add_argument(
        "--device", default=None,
        help="Compute device: 'cpu' or 'cuda' (default: auto-detect)."
    )
    argparser.add_argument(
            "--tile_size", type=int, default=4096,
            help="Tile size in pixels for warping the full-resolution images (default: 4096). "
                "Smaller tiles reduce VRAM usage but may increase processing time."
        )
    argparser.add_argument(
        "--random_starts", type=int, default=24,
        help="Number of random initializations for registration (default: 24)."
    )
    argparser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility (default: 0)."
    )
    argparser.add_argument(
        "--no_plot_show", action="store_true",
        help="Do not display the training loss plot."
    )
    argparser.add_argument(
        "--no_plot_save", action="store_true",
        help="Do not save the training loss plot."
    )
    argparser.add_argument(
        "--no_save_loss", action="store_true",
        help="Do not save the training losses to a CSV file."
    )


def main() -> None:
    argparser = argparse.ArgumentParser(
        prog="MPXImageAligner",
        description="Multiplexed image alignment tool.",
    )
    subparsers = argparser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    subparsers.add_parser("gui", help="Open the graphical user interface.")

    align_parser = subparsers.add_parser(
        "align",
        help="Align multiplexed images using GPU-accelerated image registration.",
    )
    _build_align_parser(align_parser)

    args = argparser.parse_args()

    if args.command == "gui":
        from mpximagealigner.app.main import run_gui
        run_gui()
    else:
        from mpximagealigner.alignment import run_alignment
        run_alignment(
            src_dir=args.src_dir,
            out_dir=args.out_dir,
            ref_file_no=args.ref_file_no,
            mode=args.mode,
            method=args.method,
            search_ref=args.search_ref,
            size_factor=args.size_factor,
            lr=args.lr,
            num_epochs=args.num_epochs,
            device=args.device,
            tile_size=args.tile_size,
            plot_show=not args.no_plot_show,
            plot_save=not args.no_plot_save,
            save_loss=not args.no_save_loss,
            random_starts=args.random_starts,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
