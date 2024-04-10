import os
import json
from typing import List
import pickle as pkl
from pathlib import Path
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from omegaconf import DictConfig
import torch
import shutil
from copy import deepcopy
from nemo.collections.asr.models import ClusteringDiarizer

from tqdm import tqdm

from nemo.collections.asr.parts.utils.speaker_utils import (
    audio_rttm_map,
    get_embs_and_timestamps,
    get_uniqname_from_filepath,
    write_rttm2manifest,
)
from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_segment_table,
    get_vad_stream_status,
)
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield

from util.nemo_util import *


class NeMoDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig, speaker_model=None):
        super().__init__(cfg, speaker_model)
        self.output_dir = cfg.diarizer.out_dir
        self.rttm_dir = os.path.join(self.output_dir, 'pred_rttms')
        self.json_dir = os.path.join(self.output_dir, 'pred_json')

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.json_dir):
            os.mkdir(self.json_dir)

        # Oracle speaker num
        self._cluster_params.oracle_num_speakers = True

        torch.set_default_device(self._cfg.device)

    def diarize(self, paths2audio_files: List[str], durations: List, num_speakers: List[int] = None, output_label: bool = True, batch_size: int = 0):
        """
        Diarize list of audio files in paths2audio_files
        Arguments:
            * paths2audio_files: list of audio files to be diarized
            * duration: duration of each audio file
            * num_speakers: if not None, then it corresponds to number of speakers in each of the audio input files
            * output_label: if True, saves text label file that can be used in Audacity application for visualization
            * batch_size: batch_sizeconsidered for extraction of speaker embedding and VAD computation
        """
        # setup manifest file
        config_setup(paths2audio_files, self._diarizer_params.manifest_filepath, durations, num_speakers=num_speakers)

        self._cluster_params.oracle_num_speakers = num_speakers is not None

        self._out_dir = self._diarizer_params.out_dir

        self._speaker_dir = os.path.join(self._diarizer_params.out_dir, 'speaker_outputs')

        if os.path.exists(self._speaker_dir):
            logging.warning("Deleting previous clustering diarizer outputs.")
            shutil.rmtree(self._speaker_dir, ignore_errors=True)
        os.makedirs(self._speaker_dir)

        if not os.path.exists(self._out_dir):
            os.mkdir(self._out_dir)

        self._vad_dir = os.path.join(self._out_dir, 'vad_outputs')
        self._vad_out_file = os.path.join(self._vad_dir, "vad_out.json")

        if batch_size:
            self._cfg.batch_size = batch_size

        if paths2audio_files:
            if type(paths2audio_files) is list:
                self._diarizer_params.manifest_filepath = os.path.join(self._out_dir, 'paths2audio_filepath.json')
                config_setup(paths2audio_files, self._diarizer_params.manifest_filepath, durations, num_speakers=num_speakers)
                # self.path2audio_files_to_manifest(paths2audio_files, self._diarizer_params.manifest_filepath)
            else:
                raise ValueError("paths2audio_files must be of type list of paths to file containing audio file")

        self.AUDIO_RTTM_MAP = audio_rttm_map(self._diarizer_params.manifest_filepath)

        out_rttm_dir = os.path.join(self._out_dir, 'pred_rttms')
        os.makedirs(out_rttm_dir, exist_ok=True)

        # Speech Activity Detection
        self._perform_speech_activity_detection()

        # Segmentation
        scales = self.multiscale_args_dict['scale_dict'].items()
        for scale_idx, (window, shift) in scales:

            # Segmentation for the current scale (scale_idx)
            self._run_segmentation(window, shift, scale_tag=f'_scale{scale_idx}')

            # Embedding Extraction for the current scale (scale_idx)
            self._extract_embeddings(self.subsegments_manifest_path, scale_idx, len(scales))

            self.multiscale_embeddings_and_timestamps[scale_idx] = [self.embeddings, self.time_stamps]

        embs_and_timestamps = get_embs_and_timestamps(
            self.multiscale_embeddings_and_timestamps, self.multiscale_args_dict
        )

        # Clustering
        all_reference, all_hypothesis = perform_clustering(
            embs_and_timestamps=embs_and_timestamps,
            AUDIO_RTTM_MAP=self.AUDIO_RTTM_MAP,
            out_rttm_dir=out_rttm_dir,
            clustering_params=self._cluster_params,
            device=self._speaker_model.device,
            verbose=self.verbose,
        )
        logging.info("Outputs are saved in {} directory".format(os.path.abspath(self._diarizer_params.out_dir)))


        # generate label file
        if output_label:
            for uniq_id, audio_rttm_values in self.AUDIO_RTTM_MAP.items():
            
                filename = audio_rttm_values.get('audio_filepath').split('/')[-1][:-4]
                labels = rttm_to_labels(os.path.join(self.rttm_dir, f'{filename}.rttm'))
                hypothesis = labels_to_pyannote_object(labels)

                last_label = {
                    'start' : None, 'end' : None, 'label' : None
                }

                with open(os.path.join(self.json_dir, f'{filename}_labels.txt'), 'w') as f:
                    for segment, track, label in hypothesis.itertracks(yield_label=True):
                        start, end = segment.start, segment.end
                        if label == last_label['label']:
                            last_label['end'] = end
                            continue
                        # write previous label
                        if last_label['label'] is not None:
                            f.write(f"{last_label['start']}\t{last_label['end']}\t{last_label['label']}\n")
                        last_label = {
                            'start' : start, 'end' : end, 'label' : label
                        }
                    f.write(f"{last_label['start']}\t{last_label['end']}\t{last_label['label']}\n")

    def _run_vad(self, manifest_file):
        """
        Run voice activity detection. 
        Get log probability of voice activity detection and smoothes using the post processing parameters. 
        Using generated frame level predictions generated manifest file for later speaker embedding extraction.
        input:
        manifest_file (str) : Manifest file containing path to audio file and label as infer

        """

        shutil.rmtree(self._vad_dir, ignore_errors=True)
        os.makedirs(self._vad_dir)

        self._vad_model.eval()

        time_unit = int(self._vad_window_length_in_sec / self._vad_shift_length_in_sec)
        trunc = int(time_unit / 2)
        trunc_l = time_unit - trunc
        all_len = 0
        data = []
        for line in open(manifest_file, 'r', encoding='utf-8'):
            file = json.loads(line)['audio_filepath']
            data.append(get_uniqname_from_filepath(file))

        status = get_vad_stream_status(data)
        for i, test_batch in enumerate(
            tqdm(self._vad_model.test_dataloader(), desc='vad', leave=True, disable=not self.verbose)
        ):
            test_batch = [x.to(self._vad_model.device) for x in test_batch]
            with autocast():
                log_probs = self._vad_model(input_signal=test_batch[0], input_signal_length=test_batch[1])
                probs = torch.softmax(log_probs, dim=-1)
                pred = probs[:, 1]
                if status[i] == 'start':
                    to_save = pred[:-trunc]
                elif status[i] == 'next':
                    to_save = pred[trunc:-trunc_l]
                elif status[i] == 'end':
                    to_save = pred[trunc_l:]
                else:
                    to_save = pred
                all_len += len(to_save)
                outpath = os.path.join(self._vad_dir, data[i] + ".frame")
                with open(outpath, "a", encoding='utf-8') as fout:
                    for f in range(len(to_save)):
                        fout.write('{0:0.4f}\n'.format(to_save[f]))
            del test_batch
            if status[i] == 'end' or status[i] == 'single':
                all_len = 0

        if not self._vad_params.smoothing:
            # Shift the window by 10ms to generate the frame and use the prediction of the window to represent the label for the frame;
            self.vad_pred_dir = self._vad_dir
            frame_length_in_sec = self._vad_shift_length_in_sec
        else:
            # Generate predictions with overlapping input segments. Then a smoothing filter is applied to decide the label for a frame spanned by multiple segments.
            # smoothing_method would be either in majority vote (median) or average (mean)
            logging.info("Generating predictions with overlapping input segments")
            smoothing_pred_dir = generate_overlap_vad_seq(
                frame_pred_dir=self._vad_dir,
                smoothing_method=self._vad_params.smoothing,
                overlap=self._vad_params.overlap,
                window_length_in_sec=self._vad_window_length_in_sec,
                shift_length_in_sec=self._vad_shift_length_in_sec,
                num_workers=self._cfg.num_workers,
            )
            self.vad_pred_dir = smoothing_pred_dir
            frame_length_in_sec = 0.01

        logging.info("Converting frame level prediction to speech/no-speech segment in start and end times format.")

        vad_params = self._vad_params if isinstance(self._vad_params, (DictConfig, dict)) else self._vad_params.dict()
        table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.vad_pred_dir,
            postprocessing_params=vad_params,
            frame_length_in_sec=frame_length_in_sec,
            num_workers=self._cfg.num_workers,
            out_dir=self._vad_dir,
        )

        AUDIO_VAD_RTTM_MAP = {}
        for key in self.AUDIO_RTTM_MAP:
            if os.path.exists(os.path.join(table_out_dir, key + ".txt")):
                AUDIO_VAD_RTTM_MAP[key] = deepcopy(self.AUDIO_RTTM_MAP[key])
                AUDIO_VAD_RTTM_MAP[key]['rttm_filepath'] = os.path.join(table_out_dir, key + ".txt")
            else:
                logging.warning(f"no vad file found for {key} due to zero or negative duration")

        write_rttm2manifest(AUDIO_VAD_RTTM_MAP, self._vad_out_file)
        self._speaker_manifest_path = self._vad_out_file

    def _extract_embeddings(self, manifest_file: str, scale_idx: int, num_scales: int):
        """
        This method extracts speaker embeddings from segments passed through manifest_file
        Optionally you may save the intermediate speaker embeddings for debugging or any use. 
        """
        logging.info("Extracting embeddings for Diarization")
        self._setup_spkr_test_data(manifest_file)
        self.embeddings = {}
        self._speaker_model.eval()
        self.time_stamps = {}

        all_embs = torch.empty([0]).cpu()
        for test_batch in tqdm(
            self._speaker_model.test_dataloader(),
            desc=f'[{scale_idx+1}/{num_scales}] extract embeddings',
            leave=True,
            disable=not self.verbose,
        ):
            test_batch = [x.to(self._speaker_model.device) for x in test_batch]
            audio_signal, audio_signal_len, labels, slices = test_batch
            with autocast():
                _, embs = self._speaker_model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
                emb_shape = embs.shape[-1]
                embs = embs.view(-1, emb_shape)
                all_embs = torch.cat((all_embs, embs.cpu().detach()), dim=0)
            del test_batch

        with open(manifest_file, 'r', encoding='utf-8') as manifest:
            for i, line in enumerate(manifest.readlines()):
                line = line.strip()
                dic = json.loads(line)
                uniq_name = get_uniqname_from_filepath(dic['audio_filepath'])
                if uniq_name in self.embeddings:
                    self.embeddings[uniq_name] = torch.cat((self.embeddings[uniq_name], all_embs[i].view(1, -1)))
                else:
                    self.embeddings[uniq_name] = all_embs[i].view(1, -1)
                if uniq_name not in self.time_stamps:
                    self.time_stamps[uniq_name] = []
                start = dic['offset']
                end = start + dic['duration']
                self.time_stamps[uniq_name].append([start, end])

        if self._speaker_params.save_embeddings:
            embedding_dir = os.path.join(self._speaker_dir, 'embeddings')
            if not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir, exist_ok=True)

            prefix = get_uniqname_from_filepath(manifest_file)
            name = os.path.join(embedding_dir, prefix)
            self._embeddings_file = name + f'_embeddings.pkl'
            pkl.dump(self.embeddings, open(self._embeddings_file, 'wb'))
            logging.info("Saved embedding files to {}".format(embedding_dir))
