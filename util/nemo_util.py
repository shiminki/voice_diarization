import glob
import json
import math
import multiprocessing
import os
import shutil
from itertools import repeat
from math import ceil, floor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from pyannote.core import Annotation, Segment
from pyannote.metrics import detection
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from util.clusteringCuda import LongFormSpeakerClusteringCuda
from nemo.collections.asr.models import EncDecClassificationModel, EncDecFrameClassificationModel
from nemo.collections.common.parts.preprocessing.manifest import get_full_path
from nemo.utils import logging

import gc
import json
import math
import os
import shutil
from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import omegaconf
import soundfile as sf
import torch
from pyannote.core import Annotation, Segment
from tqdm import tqdm

from nemo.collections.asr.data.audio_to_label import repeat_signal
from nemo.collections.asr.parts.utils.offline_clustering import SpeakerClustering, get_argmin_mat, split_input_data
from nemo.utils import logging

from nemo.collections.asr.parts.utils.vad_utils import generate_overlap_vad_seq_per_file_star
from nemo.collections.asr.parts.utils.speaker_utils import (
    generate_cluster_labels,
    labels_to_rttmfile,
    labels_to_pyannote_object,
    rttm_to_labels,
    write_cluster_labels
)


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


"""
Majority of this code are from

url: https://nvidia.github.io/NeMo/
repository-code: https://github.com/NVIDIA/NeMo

We modified from the following components from the repository

Diarizer:
    * 

Utility:
    * nemo/collections/asr/parts/utils/vad_utils.py
    * nemo/collections/asr/parts/utils/speaker_utils.py
"""



def config_setup(paths2audio_files, manifest_filepath, durations, num_speakers=None):
    """
    Changes audio manifest file
    Arguments:
        * file_dir (str): directory to audio file
    """

    with open(manifest_filepath, 'w', encoding='utf-8') as fp:
        for i, audio_file in enumerate(paths2audio_files):
            audio_file = audio_file.strip()
            entry = {'audio_filepath': audio_file, 'offset': 0.0, 'text': '-', 'label': 'infer'}
            if num_speakers is not None:
                entry['num_speakers'] = num_speakers[i]
            if durations is not None:
                entry['duration'] = durations[i]
            fp.write(json.dumps(entry) + '\n')

def generate_overlap_vad_seq(
    frame_pred_dir: str,
    smoothing_method: str,
    overlap: float,
    window_length_in_sec: float,
    shift_length_in_sec: float,
    num_workers: int,
    out_dir: str = None,
) -> str:
    """
    Generate predictions with overlapping input windows/segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple windows. 
    Two common smoothing filters are supported: majority vote (median) and average (mean).
    This function uses multiprocessing to speed up. 
    Args:
        frame_pred_dir (str): Directory of frame prediction file to be processed.
        smoothing_method (str): median or mean smoothing filter.
        overlap (float): amounts of overlap of adjacent windows.
        window_length_in_sec (float): length of window for generating the frame.
        shift_length_in_sec (float): amount of shift of window for generating the frame.
        out_dir (str): directory of generated predictions.
        num_workers(float): number of process for multiprocessing
    Returns:
        overlap_out_dir(str): directory of the generated predictions.
    """

    frame_filepathlist = glob.glob(frame_pred_dir + "/*.frame")
    if out_dir:
        overlap_out_dir = out_dir
    else:
        overlap_out_dir = os.path.join(
            frame_pred_dir, "overlap_smoothing_output" + "_" + smoothing_method + "_" + str(overlap)
        )

    if not os.path.exists(overlap_out_dir):
        os.mkdir(overlap_out_dir)

    per_args = {
        "overlap": overlap,
        "window_length_in_sec": window_length_in_sec,
        "shift_length_in_sec": shift_length_in_sec,
        "out_dir": overlap_out_dir,
        "smoothing_method": smoothing_method,
    }
    if num_workers is not None and num_workers > 1:
        with multiprocessing.Pool(processes=num_workers) as p:
            inputs = zip(frame_filepathlist, repeat(per_args))
            results = list(
                tqdm(
                    p.imap(generate_overlap_vad_seq_per_file_star, inputs),
                    total=len(frame_filepathlist),
                    desc='generating preds',
                    leave=True,
                )
            )

    else:
        for frame_filepath in tqdm(frame_filepathlist, desc='generating preds', leave=False):
            generate_overlap_vad_seq_per_file(frame_filepath, per_args)

    return overlap_out_dir


def generate_overlap_vad_seq_per_file(frame_filepath: str, per_args: dict) -> str:
    """
    A wrapper for generate_overlap_vad_seq_per_tensor.
    """

    out_dir = per_args['out_dir']
    smoothing_method = per_args['smoothing_method']
    frame, name = load_tensor_from_file(frame_filepath)

    per_args_float: Dict[str, float] = {}
    for i in per_args:
        if type(per_args[i]) == float or type(per_args[i]) == int:
            per_args_float[i] = per_args[i]

    preds = generate_overlap_vad_seq_per_tensor(frame, per_args_float, smoothing_method)

    overlap_filepath = os.path.join(out_dir, name + "." + smoothing_method)
    with open(overlap_filepath, "w", encoding='utf-8') as f:
        for pred in preds:
            f.write(f"{pred:.4f}\n")

    return overlap_filepath

def load_tensor_from_file(filepath: str) -> Tuple[torch.Tensor, str]:
    """
    Load torch.Tensor and the name from file
    """
    frame = []
    with open(filepath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            frame.append(float(line))

    name = Path(filepath).stem
    return torch.tensor(frame), name

def generate_overlap_vad_seq_per_tensor(
    frame: torch.Tensor, per_args: Dict[str, float], smoothing_method: str
) -> torch.Tensor:
    """
    Use generated frame prediction (generated by shifting window of shift_length_in_sec (10ms)) to generate prediction with overlapping input window/segments
    See description in generate_overlap_vad_seq.
    Use this for single instance pipeline. 
    """
    # This function will be refactor for vectorization but this is okay for now

    overlap = per_args['overlap']
    window_length_in_sec = per_args['window_length_in_sec']
    shift_length_in_sec = per_args['shift_length_in_sec']
    frame_len = per_args.get('frame_len', 0.01)

    shift = int(shift_length_in_sec / frame_len)  # number of units of shift
    seg = int((window_length_in_sec / frame_len + 1))  # number of units of each window/segment

    jump_on_target = int(seg * (1 - overlap))  # jump on target generated sequence
    jump_on_frame = int(jump_on_target / shift)  # jump on input frame sequence

    if jump_on_frame < 1:
        raise ValueError(
            f"Note we jump over frame sequence to generate overlapping input segments. \n \
        Your input makes jump_on_frame={jump_on_frame} < 1 which is invalid because it cannot jump and will stuck.\n \
        Please try different window_length_in_sec, shift_length_in_sec and overlap choices. \n \
        jump_on_target = int(seg * (1 - overlap)) \n \
        jump_on_frame  = int(jump_on_frame/shift) "
        )

    target_len = int(len(frame) * shift)

    if smoothing_method == 'mean':
        preds = torch.zeros(target_len)
        pred_count = torch.zeros(target_len)

        for i, og_pred in enumerate(frame):
            if i % jump_on_frame != 0:
                continue
            start = i * shift
            end = start + seg
            preds[start:end] = preds[start:end] + og_pred
            pred_count[start:end] = pred_count[start:end] + 1

        preds = preds / pred_count
        last_non_zero_pred = preds[pred_count != 0][-1]
        preds[pred_count == 0] = last_non_zero_pred

    elif smoothing_method == 'median':
        preds = [torch.empty(0) for _ in range(target_len)]
        for i, og_pred in enumerate(frame):
            if i % jump_on_frame != 0:
                continue

            start = i * shift
            end = start + seg
            for j in range(start, end):
                if j <= target_len - 1:
                    preds[j] = torch.cat((preds[j], og_pred.unsqueeze(0)), 0)

        preds = torch.stack([torch.nanquantile(l, q=0.5) for l in preds])
        nan_idx = torch.isnan(preds)
        last_non_nan_pred = preds[~nan_idx][-1]
        preds[nan_idx] = last_non_nan_pred

    else:
        raise ValueError("smoothing_method should be either mean or median")

    return preds

def perform_clustering(
    embs_and_timestamps, AUDIO_RTTM_MAP, out_rttm_dir, clustering_params, device, verbose: bool = True
):
    """
    Performs spectral clustering on embeddings with time stamps generated from VAD output

    Args:
        embs_and_timestamps (dict): This dictionary contains the following items indexed by unique IDs.
            'embeddings' : Tensor containing embeddings. Dimensions:(# of embs) x (emb. dimension)
            'timestamps' : Tensor containing ime stamps list for each audio recording
            'multiscale_segment_counts' : Tensor containing the number of segments for each scale
        AUDIO_RTTM_MAP (dict): AUDIO_RTTM_MAP for mapping unique id with audio file path and rttm path
        out_rttm_dir (str): Path to write predicted rttms
        clustering_params (dict): clustering parameters provided through config that contains max_num_speakers (int),
        oracle_num_speakers (bool), max_rp_threshold(float), sparse_search_volume(int) and enhance_count_threshold (int)
        use_torch_script (bool): Boolean that determines whether to use torch.jit.script for speaker clustering
        device (torch.device): Device we are running on ('cpu', 'cuda').
        verbose (bool): Enable TQDM progress bar.

    Returns:
        all_reference (list[uniq_name,Annotation]): reference annotations for score calculation
        all_hypothesis (list[uniq_name,Annotation]): hypothesis annotations for score calculation

    """
    all_hypothesis = []
    all_reference = []
    no_references = False
    lines_cluster_labels = []

    cuda = True
    if device.type != 'cuda':
        logging.warning("cuda=False, using CPU for eigen decomposition. This might slow down the clustering process.")
        cuda = False

    speaker_clustering = LongFormSpeakerClusteringCuda(device=device)

    if clustering_params.get('export_script_module', False):
        speaker_clustering = torch.jit.script(speaker_clustering)
        torch.jit.save(speaker_clustering, 'speaker_clustering_script.pt')

    for uniq_id, audio_rttm_values in tqdm(AUDIO_RTTM_MAP.items(), desc='clustering', leave=True, disable=not verbose):
        uniq_embs_and_timestamps = embs_and_timestamps[uniq_id]

        if clustering_params.oracle_num_speakers:
            num_speakers = audio_rttm_values.get('num_speakers', None)
            if num_speakers is None:
                raise ValueError("Provided option as oracle num of speakers but num_speakers in manifest is null")
        else:
            num_speakers = -1

        base_scale_idx = uniq_embs_and_timestamps['multiscale_segment_counts'].shape[0] - 1

        cluster_labels = speaker_clustering._forward_infer(
            embeddings_in_scales=uniq_embs_and_timestamps['embeddings'],
            timestamps_in_scales=uniq_embs_and_timestamps['timestamps'],
            multiscale_segment_counts=uniq_embs_and_timestamps['multiscale_segment_counts'],
            multiscale_weights=uniq_embs_and_timestamps['multiscale_weights'],
            oracle_num_speakers=int(num_speakers),
            max_num_speakers=int(clustering_params.max_num_speakers),
            max_rp_threshold=float(clustering_params.max_rp_threshold),
            sparse_search_volume=int(clustering_params.sparse_search_volume),
            chunk_cluster_count=clustering_params.get('chunk_cluster_count', None),
            embeddings_per_chunk=clustering_params.get('embeddings_per_chunk', None),
        )

        del uniq_embs_and_timestamps
        if cuda:
            torch.cuda.empty_cache()
        else:
            gc.collect()
        timestamps = speaker_clustering.timestamps_in_scales[base_scale_idx]

        cluster_labels = cluster_labels.cpu().numpy()
        if len(cluster_labels) != timestamps.shape[0]:
            raise ValueError("Mismatch of length between cluster_labels and timestamps.")

        labels, lines = generate_cluster_labels(timestamps, cluster_labels)

        if out_rttm_dir:
            labels_to_rttmfile(labels, uniq_id, out_rttm_dir)
            lines_cluster_labels.extend([f'{uniq_id} {seg_line}\n' for seg_line in lines])
        hypothesis = labels_to_pyannote_object(labels, uniq_name=uniq_id)
        all_hypothesis.append([uniq_id, hypothesis])

        rttm_file = audio_rttm_values.get('rttm_filepath', None)
        if rttm_file is not None and os.path.exists(rttm_file) and not no_references:
            ref_labels = rttm_to_labels(rttm_file)
            reference = labels_to_pyannote_object(ref_labels, uniq_name=uniq_id)
            all_reference.append([uniq_id, reference])
        else:
            no_references = True
            all_reference = []

    if out_rttm_dir:
        write_cluster_labels(base_scale_idx, lines_cluster_labels, out_rttm_dir)

    return all_reference, all_hypothesis

