import argparse


def add_dataset_base_config(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset_base_path", type=str, default="", help="Base path of the dataset (single-dataset mode).")
    parser.add_argument("--dataset_metadata_path", type=str, default=None, help="Path to the metadata file of the dataset.")
    parser.add_argument("--dataset_repeat", type=int, default=1, help="Number of times to repeat the dataset per epoch.")
    parser.add_argument("--dataset_num_workers", type=int, default=0, help="Number of workers for data loading.")
    return parser


def add_video_size_config(parser: argparse.ArgumentParser):
    parser.add_argument("--height", type=int, default=None, help="Height of images.")
    parser.add_argument("--width", type=int, default=None, help="Width of images.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames per video (M + N).")
    return parser


def add_model_config(parser: argparse.ArgumentParser):
    parser.add_argument("--model_paths", type=str, default=None, help="Paths to load models. In JSON format.")
    parser.add_argument("--model_id_with_origin_paths", type=str, default=None,
                        help="Model ID with origin paths, e.g. 'Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors'. Comma-separated for multiple.")
    parser.add_argument("--extra_inputs", default=None, help="Additional model inputs, comma-separated (e.g. 'input_image').")
    return parser


def add_training_config(parser: argparse.ArgumentParser):
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible training data sampling.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--patch_embedding_lr",
        type=float,
        default=None,
        help="Learning rate for DiT patch_embedding only. "
             "Default: same as --learning_rate. "
             "Set to 0 to freeze patch_embedding. "
             "Useful with --from_wan to keep Wan 36ch pe at a lower LR "
             "(e.g. 1e-5) while LoRA / raymap_pe / geom_pe stay at --learning_rate.",
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--steps_per_epoch", type=int, default=None,
                        help="Optimizer steps per epoch. If unset, derived from "
                        "--steps_per_epoch_base // world_size when that base is set.")
    parser.add_argument("--steps_per_epoch_base", type=int, default=None,
                        help="Global step budget per epoch before dividing by GPU count "
                        "(e.g. 10000 -> 1250 steps/epoch on 8 GPUs).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Stop after this many optimizer steps. Defaults to "
                        "num_epochs * steps_per_epoch when the latter is set.")
    parser.add_argument("--trainable_models", type=str, default=None, help="Models to fully train (no LoRA), comma-separated.")
    parser.add_argument("--find_unused_parameters", default=False, action="store_true", help="Whether to find unused parameters in DDP.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=["constant", "constant_with_warmup", "cosine"],
        help="Legacy LR schedule selector. 'constant' keeps the legacy "
             "ConstantLR behavior; "
             "'constant_with_warmup' linear-warms to --learning_rate then holds; "
             "'cosine' remains supported as an alias for --cos_lr "
             "(counted in real optimizer steps, robust to accelerate/DeepSpeed "
             "scheduler wrappers).",
    )
    parser.add_argument(
        "--cos_lr",
        default=False,
        action="store_true",
        help="Enable cosine decay after any independently configured warmup. "
             "Disabled by default; without it, the selected constant/warmup "
             "schedule is unchanged.",
    )
    parser.add_argument(
        "--cos_lr_total_steps",
        type=int,
        default=None,
        help="Horizon for --cos_lr (warmup + cosine span). Defaults to --max_steps. "
             "Use when stopping early via --max_steps but keeping the original "
             "full-schedule cosine curve (e.g. num_epochs * steps_per_epoch).",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=None,
        help="Final learning rate for --cos_lr (or legacy --lr_scheduler cosine). "
             "Defaults to learning_rate / 10.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Linear warmup steps for cosine / constant_with_warmup "
             "(0 = no warmup). Step k runs at learning_rate * k / warmup_steps.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Print training progress every N optimizer steps (0 to disable).",
    )
    return parser


def add_output_config(parser: argparse.ArgumentParser):
    parser.add_argument("--output_path", type=str, default="./models", help="Output save path.")
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.", help="Remove prefix in ckpt.")
    parser.add_argument("--save_steps", type=int, default=None, help="Step interval for intermediate checkpoints. If None, checkpoints are saved per-epoch.")
    parser.add_argument("--save_keep_first", type=int, default=5,
                        help="Always retain the first N step checkpoints; only the latest is kept afterwards.")
    return parser


def add_lora_config(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_base_model", type=str, default=None, help="Which submodule LoRA is added to (e.g. 'dit').")
    parser.add_argument("--lora_target_modules", type=str, default="q,k,v,o,ffn.0,ffn.2", help="Comma-separated LoRA target module names.")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA.")
    parser.add_argument("--lora_checkpoint", type=str, default=None, help="Path to a LoRA checkpoint to initialise from.")
    return parser


def add_gradient_config(parser: argparse.ArgumentParser):
    parser.add_argument("--use_gradient_checkpointing", default=False, action="store_true", help="Use gradient checkpointing.")
    parser.add_argument("--use_gradient_checkpointing_offload", default=False, action="store_true", help="Offload gradient checkpointing to CPU memory.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    return parser


def add_wandb_config(parser: argparse.ArgumentParser):
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name. If set, enables wandb logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name.")
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                        help="TensorBoard log directory. Defaults to <output_path>/tensorboard.")
    parser.add_argument("--no_tensorboard", default=False, action="store_true",
                        help="Disable TensorBoard logging.")
    parser.add_argument("--vis_steps", type=int, default=1000,
                        help="Log gt-vs-pred visualization every N optimizer steps (0 to disable).")
    parser.add_argument("--vis_inference_steps", type=int, default=20,
                        help="Diffusion steps used for gt-vs-pred visualization.")
    parser.add_argument("--vis_seed", type=int, default=42, help="Base seed for visualization inference.")
    return parser


def add_general_config(parser: argparse.ArgumentParser):
    parser = add_dataset_base_config(parser)
    parser = add_model_config(parser)
    parser = add_training_config(parser)
    parser = add_output_config(parser)
    parser = add_lora_config(parser)
    parser = add_gradient_config(parser)
    parser = add_wandb_config(parser)
    return parser
