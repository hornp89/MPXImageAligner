"""
Created on Jun 24 2025

@author: Paul Horn
"""
import gc
import os
import time

import tempfile

import numpy as np
import pandas as pd
# from scipy.signal import find_peaks

from scipy.ndimage import binary_dilation as dilation
from scipy.ndimage import generate_binary_structure

import torch

import matplotlib.pyplot as plt

from mpximagealigner.torchregister import torchregister as tr
from mpximagealigner.torchregister.utils import affine_warp_tiled
import torchvision.transforms as transforms
from torchvision.transforms import v2

import tifffile as tiff
from pylibCZIrw import czi as pyczi


def read_channel(file, channel_index):
    """
    Reads a single channel as a 2D array from a TIFF or CZI file.
    Args:
        file (str): Path to the image file.
        channel_index (int): Index of the channel to read.
    Returns:
        np.ndarray: The channel image as a 2D NumPy array.
    """
    ext = os.path.splitext(file)[1].lower()
    if ext == ".czi":
        with pyczi.open_czi(file) as czi:
            return czi.read(plane={"C": channel_index}).squeeze().copy()
    else:
        with tiff.TiffFile(file) as tif:
            return tif.series[0][channel_index].asarray().copy()


def get_channel_names(file):
    """
    Extracts channel names from a TIFF or CZI filenames.
    Args:
        file (str): Path to the image file.
    Returns:
        list: A list containing one list of channel name strings.
    """
    if file.lower().endswith(".ome.tif") or file.lower().endswith(".ome.tiff"):
        channels =  [os.path.splitext(os.path.basename(file))[0].split(".")[0].split("_")[-1].split(" ")]
    else:
        channels =  [os.path.splitext(os.path.basename(file))[0].split("_")[-1].split(" ")]

    return channels

def get_meta_data(file):
    """
    Extracts metadata 'date' and 'ID' from a TIFF or CZI file.
    Args:
        file (str): Path to the image file.
    Returns:
        dict: A dictionary containing metadata key-value pairs.
    """
    
    if file.lower().endswith("ome.tif") or file.lower().endswith("ome.tiff"):
        date = os.path.splitext(os.path.basename(file))[0].split(".")[0].split("_")[0]
        id = os.path.splitext(os.path.basename(file))[0].split(".")[0].split("_")[1]
        meta_data = {"date": date, "ID": id}
    
    else:  # file.lower().endswith(".czi"):
        date = os.path.splitext(os.path.basename(file))[0].split("_")[0]
        id = os.path.splitext(os.path.basename(file))[0].split("_")[1]
        meta_data = {"date": date, "ID": id}
    return meta_data
    

def read_dapi(file):
    """
    Reads the DAPI (first) channel from a TIFF or CZI file.
    Args:
        file (str): Path to the image file.
    Returns:
        np.ndarray: The DAPI image as a NumPy array.
    """
    return read_channel(file, 0)

def fill_background_with_noise(image):
    image = image.squeeze()  # Remove channel dimension if present
    
    black_background = image == 0
    
    if np.sum(black_background) <= 10:
        return image  # No background to fill
    
    else:
        structure = generate_binary_structure(2, 2)  # 2D connectivity
        
        dilated_background = dilation(black_background, structure=structure, iterations=20)
        background = dilated_background ^ black_background
        
        background_px = image[background]
        low_percentile = np.percentile(background_px, 1)
        high_percentile = np.percentile(background_px, 99)
        background_px = background_px[(background_px > low_percentile) & (background_px < high_percentile)]  # Remove outliers for better estimation

        mean_intensity = np.mean(background_px)
        std_intensity = np.std(background_px)

        image[black_background] = np.random.normal(loc=mean_intensity, scale=std_intensity, size=black_background.sum()).astype(image.dtype)
        
        return image

def preprocess_dapi(file, ref_shape, device, size_factor=4):
    """
    Preprocesses a DAPI image for registration by resizing and transformation to a float32 tensor.
    Args:
        file (str): Path to the DAPI image file.
        ref_shape (tuple): The shape of the reference image for resizing.
        device (torch.device): Device to place the tensor on.
        size_factor (numeric, optional): Factor by which to resize the image. Defaults to 4.
    Returns:
        torch.Tensor: Preprocessed DAPI image ready for registration.
    """
    image = read_channel(file, 0)

    size = int(np.min(ref_shape) / size_factor)
    
    # Compute the final downsampled shape directly so we never allocate a
    # full-resolution float32 tensor (which can exceed available RAM).
    min_ref = min(ref_shape)
    scale = size / min_ref
    target_h = max(1, round(ref_shape[0] * scale))
    target_w = max(1, round(ref_shape[1] * scale))

    # Pre-downscale the numpy array via strided slicing to ~2x the target
    # size.  This keeps the subsequent float32 conversion inside
    # transforms.Resize small enough to fit in memory.
    h, w = image.shape
    stride_h = max(1, h // (target_h * 2))
    stride_w = max(1, w // (target_w * 2))
    if stride_h > 1 or stride_w > 1:
        image = image[::stride_h, ::stride_w].copy()

    transform = transforms.Compose([
        transforms.ToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        transforms.Resize(size=(target_h, target_w)),
    ])
    
    # image = white_tophat_pytorch(image, kernel_size=31, device=device)
    
    image = fill_background_with_noise(image)
    
    image = transform(image).to(device)
      
    # high_percentile = np.quantile(image.cpu().numpy(), 0.99)
    # image = image / high_percentile if high_percentile > 0 else image
    # image = torch.clamp(image, 0.0, 1.0)
    
    image = image.view(1, 1, image.shape[1], image.shape[2]).to(torch.float32)
    
    return image

def build_dapi_cache(files, ref_shape, size_factor, device, cache_dir):
    """
    Pre-process all DAPI channels once and save to a temp directory.
    Returns a dict mapping file path → cached .pt path.
    """
    cache_map = {}
    for i, f in enumerate(files):
        tensor = preprocess_dapi(f, ref_shape, device, size_factor=size_factor)
        cache_path = os.path.join(cache_dir, f"{i}.pt")
        torch.save(tensor.cpu(), cache_path)
        cache_map[f] = cache_path
        del tensor
    gc.collect()
    torch.cuda.empty_cache()
    return cache_map


def load_cached_dapi(cache_path, device):
    """Load a preprocessed DAPI tensor from cache."""
    return torch.load(cache_path, map_location=device, weights_only=True)

def save_channel_names(channel_names, files_list, out_dir):
    """
    Saves a CSV file with channel names, date, ID, and channel index to the output directory.
    Args:
        channel_names (list): List of channel name strings.
        files_list (list): List of source file paths, parallel to channel_names.
        out_dir (str): Path to the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = [get_meta_data(f) for f in files_list]
    df = pd.DataFrame({
        "channel": channel_names,
        "date": [m["date"] for m in meta],
        "ID": [m["ID"] for m in meta],
    })
    df.index.name = "channel_index"
    df.to_csv(os.path.join(out_dir, "channel_names.csv"), index=True)

def get_affine_model(ref_file, target_file, ref_shape, size_factor, device, 
                     lr=1, max_epochs=5, dapi_cache=None):
    """
    Generates theta coefficients for affine registration of a target image to a reference image.

    Args:
        ref_file (str): Path to the reference image file.
        target_file (str): Path to the target image to be registered.
        ref_shape (array-like): The shape of the reference image, used for resizing and alignment.
        size_factor (int): Downsample factor for registration.
        device (torch.device): Device to run the model on.
        lr (float, optional): Learning rate for the optimization. Defaults to 1e-3.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 150.
        dapi_cache (dict, optional): Pre-built DAPI cache. Defaults to None.

    Returns:
        tuple: (theta, losses) optimized affine transformation parameters and training losses.
    """
    
    if dapi_cache is not None:
        ref_image = load_cached_dapi(dapi_cache[ref_file], device)
        target_image = load_cached_dapi(dapi_cache[target_file], device)
    else:
        ref_image = preprocess_dapi(ref_file, ref_shape, device, size_factor=size_factor)
        target_image = preprocess_dapi(target_file, ref_shape, device, size_factor=size_factor)

    ref_shape = ref_image.shape[2:]

    model = tr.Register(mode='affine', device=device, criterion=None)
    model.optim(target_image, ref_image, max_epochs=max_epochs, lr=lr)

    theta = model.theta
    losses = model.losses

    del model, ref_image, target_image
    gc.collect()
    torch.cuda.empty_cache()

    return theta, losses

def get_rigid_model(ref_file, target_file, ref_shape, size_factor, device, 
                    lr=1, max_epochs=5, dapi_cache=None):
    """
    Generates theta coefficients for rigid registration of a target image to a reference image.

    Args:
        ref_file (str): Path to the reference image file.
        target_file (str): Path to the target image to be registered.
        ref_shape (array-like): The shape of the reference image, used for resizing and alignment.
        size_factor (int): Downsample factor for registration.
        device (torch.device): Device to run the model on.
        lr (float, optional): Learning rate for the optimization. Defaults to 1.
        max_epochs (int, optional): Maximum number of epochs for training. Defaults to 5.
        dapi_cache (dict, optional): Pre-built DAPI cache. Defaults to None.

    Returns:
        tuple: (theta, losses) optimized rigid transformation parameters and training losses.
    """
    if dapi_cache is not None:
        ref_image = load_cached_dapi(dapi_cache[ref_file], device)
        target_image = load_cached_dapi(dapi_cache[target_file], device)
    else:
        ref_image = preprocess_dapi(ref_file, ref_shape, device, size_factor=size_factor)
        target_image = preprocess_dapi(target_file, ref_shape, device, size_factor=size_factor)

    ref_shape = ref_image.shape[2:]

    model = tr.Register(mode='rigid', device=device, criterion=None)
    model.optim(target_image, ref_image, max_epochs=max_epochs, lr=lr)

    theta = model.theta
    losses = model.losses

    del model, ref_image, target_image
    gc.collect()
    torch.cuda.empty_cache()

    return theta, losses

def pad_or_crop(image, target_shape):
    """
    Pads or crops a 2D image to exactly target_shape.
    Cropping is center-aligned; padding uses zeros on both sides.
    Args:
        image (np.ndarray): 2D input array.
        target_shape (tuple): Desired (height, width).
    Returns:
        np.ndarray: Array with shape == target_shape.
    """
    h, w = image.shape
    th, tw = target_shape

    # Crop height if too large
    if h > th:
        start = (h - th) // 2
        image = image[start:start + th, :]
    # Crop width if too large
    if w > tw:
        start = (w - tw) // 2
        image = image[:, start:start + tw]

    # Pad if too small in either dimension
    h, w = image.shape
    pad_h = th - h
    pad_w = tw - w
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2)),
            mode='reflect'
        )
    return image


def run_alignment(
    src_dir,
    out_dir=None,
    ref_file_no=0,
    mode="single",
    method="affine",
    search_ref=False,
    size_factor=4,
    lr=1,
    num_epochs=150,
    device=None,
    tile_size=4096,
    plot_show=True,
    plot_save=True,
    save_loss=True,
    cancelled=None,
):
    """
    Run the full multiplexed image alignment pipeline.

    Args:
        src_dir (str): Source directory containing images to align.
        out_dir (str, optional): Output directory. Defaults to <src_dir>_aligned.
        ref_file_no (int): Starting index of the reference image file.
        mode (str): Alignment mode: 'single' or 'batch'.
        method (str): Registration method: 'affine' or 'rigid'.
        search_ref (bool): Whether to search for the best reference image by iterating through ref_file_no until a loss threshold is met.
        size_factor (int): Downsample factor for registration.
        lr (float): Learning rate.
        num_epochs (int): Number of training epochs.
        per (float): Fraction of pixels used for training.
        device (str or torch.device, optional): Compute device. None/'auto' = auto-detect.
        plot_show (bool): Display the training loss plot interactively.
        plot_save (bool): Save the training loss plot to file.
        save_loss (bool): Save training losses to a CSV file.
        cancelled (callable, optional): Zero-argument callable; return True to abort between images.
    """
    
    if mode == "single":
        in_dirs = [src_dir]
        if out_dir is None:
            out_dir = src_dir + "_aligned"
        out_dirs = [out_dir]
    
    elif mode == "batch":
        in_dirs = [os.path.join(src_dir, d) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
        
        if out_dir is None:
            out_dir = src_dir + "_aligned"
        out_dirs = [os.path.join(out_dir, d) for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    for src_dir, out_dir in zip(in_dirs, out_dirs):
        if not os.path.isdir(src_dir):
            print(f"ERROR: Source directory '{src_dir}' does not exist.")
            return
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            
        if cancelled is None:
            cancelled = lambda: False

        if device is None or (isinstance(device, str) and device in ("auto", "")):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        print("Using device:", device)

        time_start = time.time()

        # Load file names
        files = [os.path.join(src_dir, file) for file in os.listdir(src_dir)]
        n_files = len(files)
        tried_refs = set()
        
        ref_file = files[ref_file_no % n_files]
        ref_shape = read_dapi(ref_file).shape

        # Tracks the best run: (max_final_loss, ref_file_no, ref_file, ref_shape)
        best_run = None

        with tempfile.TemporaryDirectory(prefix="mpx_dapi_cache_") as dapi_cache_dir:
        
            def _train_all(ref_file, ref_shape, size_factor=size_factor, 
                           lr=lr, num_epochs=num_epochs):
                """Train registration models for all non-reference files.
                Returns (thetas, losses), or (None, None) if cancelled."""
                thetas_out = {}
                losses_out = {}
                for f in files:
                    if cancelled():
                        return None, None
                    if f != ref_file:
                        if method == "rigid":
                            theta, loss = get_rigid_model(
                                ref_file, f, ref_shape=ref_shape, size_factor=size_factor,
                                device=device, lr=lr, max_epochs=num_epochs,
                                dapi_cache=dapi_cache,
                            )
                        else:
                            theta, loss = get_affine_model(
                                ref_file, f, ref_shape=ref_shape, size_factor=size_factor,
                                device=device, lr=lr, max_epochs=num_epochs,
                                dapi_cache=dapi_cache,
                            )
                        thetas_out[f] = theta
                        losses_out[f] = loss
                        
                    torch.cuda.empty_cache()
                        
                return thetas_out, losses_out        
            
            # Training loop: repeat with incremented ref_file_no if loss threshold is not met
            if search_ref:
                print("Starting training with ref_file_no =", ref_file_no)
                print("Writing cached DAPI tensors to:", dapi_cache_dir)
                dapi_cache = build_dapi_cache(files, ref_shape, size_factor=32, 
                                              device=device, cache_dir=dapi_cache_dir)
                while True:
                    if cancelled():
                        print("Alignment cancelled.")
                        return

                    ref_file_no_cur = ref_file_no % n_files
                    if ref_file_no_cur in tried_refs:
                        # All reference files exhausted — re-run training with the best reference
                        best_max_loss, best_ref_no, best_ref_file, best_ref_shape = best_run
                        print(
                            f"Re-running training with the best reference "
                            f"(ref_file_no={best_ref_no}, max final loss={best_max_loss:.6f})."
                        )
                        ref_file = best_ref_file
                        ref_shape = best_ref_shape
                        thetas, losses = _train_all(ref_file, ref_shape, size_factor=size_factor, lr=lr, num_epochs=num_epochs)
                        if thetas is None:
                            print("Alignment cancelled.")
                            return
                        break

                    tried_refs.add(ref_file_no_cur)

                    ref_file = files[ref_file_no_cur]
                    print(f"Using ref_file_no={ref_file_no_cur}: {os.path.basename(ref_file)}")

                    # Round up to even dimensions so all images are divisible by 2
                    ref_img = read_dapi(ref_file)
                    ref_shape = ((ref_img.shape[0] + 1) // 2 * 2, (ref_img.shape[1] + 1) // 2 * 2)
                    del ref_img
                    print("Reference image shape (even-padded):", ref_shape)

                    thetas, losses = _train_all(ref_file, ref_shape, size_factor=size_factor, 
                                                lr=lr, num_epochs=num_epochs)
                    if thetas is None:
                        print("Alignment cancelled.")
                        return

                    max_final_loss = max(loss1[-1] for loss1 in losses.values())

                    if best_run is None or max_final_loss < best_run[0]:
                        best_run = (max_final_loss, ref_file_no_cur, ref_file, ref_shape)

                    ref_file_no = ref_file_no_cur + 1
            else:
                print("Writing cached DAPI tensors to:", dapi_cache_dir)
                dapi_cache = build_dapi_cache(files, ref_shape, size_factor=size_factor, 
                                              device=device, cache_dir=dapi_cache_dir)
                thetas, losses = _train_all(ref_file, ref_shape, size_factor=size_factor, 
                                                lr=lr, num_epochs=num_epochs)
                if thetas is None:
                    print("Alignment cancelled.")
                    return

            # Plot training losses for each image for diagnostics
            for f, f_losses in losses.items():
                plt.plot(f_losses, label=os.path.basename(f))
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Losses")
            plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize="small")
            if plot_save:
                os.makedirs(out_dir, exist_ok=True)
                plt.savefig(os.path.join(out_dir, f'losses_{method}.png'), bbox_inches='tight')
            if plot_show:
                plt.show()
            plt.close()

            if save_loss:
                os.makedirs(out_dir, exist_ok=True)
                pd.DataFrame(losses).to_csv(os.path.join(out_dir, f'losses_{method}.csv'), index=False)

            # Build channel/file lists for output:
            # ref_file DAPI is always first; all other non-DAPI channels follow in file order.
            channel_list = ["DAPI"]
            files_list = [ref_file]

            for file in files:
                for ch in get_channel_names(file)[0]:
                    if ch != "DAPI":
                        channel_list.append(ch)
                        files_list.append(file)

            # Build channel name list upfront (needed for OME-TIFF metadata and CSV)
            num = 1
            channel_names_new = []
            for channel in channel_list:
                if channel == "Auto":
                    channel_names_new.append("Auto_" + str(num))
                    num += 1
                else:
                    channel_names_new.append(channel)

            print("Saving aligned images to:", out_dir)
            os.makedirs(out_dir, exist_ok=True)
            save_channel_names(channel_names_new, files_list, out_dir)

        # Stream each aligned channel directly to disk to avoid accumulating all channels in RAM.
        with tiff.TiffWriter(os.path.join(out_dir, os.path.basename(out_dir) + ".tiff"), bigtiff=True) as writer, \
            tiff.TiffWriter(os.path.join(out_dir, os.path.basename(out_dir) + "_DAPI.tiff"), bigtiff=True) as dapi_writer:

            # Pass 1: aligned non-DAPI channels (and the reference DAPI at index 0).
            for file, channel in zip(files_list, channel_list):
                print("Processing file:", file, "Channel:", channel)

                channels = get_channel_names(file)[0]
                index = channels.index(channel)
                image = read_channel(file, index)

                if file == ref_file:
                    image = pad_or_crop(image, ref_shape)
                    writer.write(image.astype(np.uint16), compression=True, tile=(512, 512), metadata=None)
                    del image
                    continue

                if file != ref_file:
                    img = pad_or_crop(image[:,:], ref_shape)
                    del image
                    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    output_np = affine_warp_tiled(thetas[file], img, tile_size=tile_size, train=False)
                    del img
                    torch.cuda.empty_cache()
                    
                    writer.write(output_np, compression=True, tile=(512, 512), metadata=None)
                    del output_np
                    gc.collect()

            # Pass 2: DAPI channels from all non-reference files.
            for file in list(np.unique(files_list)):
                print("Processing DAPI for file:", file)

                dapi_channels = get_channel_names(file)[0]
                index = dapi_channels.index("DAPI")
                image = read_channel(file, index)

                if file == ref_file:
                    image = pad_or_crop(image, ref_shape)
                    dapi_writer.write(image.astype(np.uint16), compression=True, tile=(512, 512),
                                    metadata=None)
                    # first_dapi_write = False
                    del image
                    continue

                if file != ref_file:
                    img = pad_or_crop(image[:,:], ref_shape)
                    del image
                    img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
                    output_np = affine_warp_tiled(thetas[file], img, tile_size=tile_size, train=False)
                    del img
                    torch.cuda.empty_cache()
                    
                    dapi_writer.write(output_np, compression=True, tile=(512, 512), metadata=None)
                    del output_np
                    gc.collect()

        time_end = time.time()
        
        if search_ref:
            best_max_loss, best_ref_no, _, _ = best_run
            print(f"(ref_file_no={best_ref_no}, max final loss={best_max_loss:.6f}).")
        
        del thetas, losses
        gc.collect()
        torch.cuda.empty_cache()

        print("Time taken:", round((time_end - time_start) / 60, 2), "minutes")