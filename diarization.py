import os
import json
import pandas as pd
import logging
logging.disable(logging.CRITICAL)
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from omegaconf import OmegaConf
import torch

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from tqdm import tqdm

from nemo.utils import logging

from util.nemo_util import *

from model.NeMo_diarizer import *


if __name__ == '__main__':
    # Initalize and setup model
    device_id = int(input('GPU Device?:'))
    MODEL_CONFIG = os.path.join('config','model_config.yaml')
    config = OmegaConf.load(MODEL_CONFIG)
    config.device = f'cuda:{device_id}'
    config.verbose = False
    model = NeMoDiarizer(cfg=config)
    torch.set_default_device(config.device)

    audio_dir = 'conversations'
    filenames = []
    for file in os.listdir(audio_dir):
        if file[-4:] == '.wav':
            filenames.append(os.path.join(audio_dir, file))

    # get duration
    durations = []

    # get number of speakers
    num_speakers = []

    for filename in filenames:
        json_path = f'{filename[:-4]}.json'
        raw_json = json.loads(open(json_path, 'r').read())
        num_speaker = len(raw_json['participants'])
        for p in raw_json['participants']:
            if p['name'] in ('Hearth', 'participant', 'Participant'):
                num_speaker -= 1
        num_speakers.append(num_speaker)
        durations.append(raw_json['duration'])

    # Diarization
    print("Begin diarization")
    model.diarize(filenames, durations, num_speakers=num_speakers, batch_size=5)

    # Evaluate model performance
    print("Evaluating diariation performance")
    diarize_performance = {}
    for file in tqdm(filenames):
        # Merge discontinued labels
        merged_label = Annotation()

        filename = file.split('/')[-1]

        with open(os.path.join(config.diarizer.out_dir, 'pred_json', f'{filename[:-4]}_labels.txt')) as f:
            for line in f.readlines():
                start, end, speaker = line.split()
                start, end = float(start), float(end)
                merged_label[Segment(start, end)] = speaker

        # Evaluate metrics using merged label
        true_labels = rttm_to_labels(os.path.join(audio_dir, f'{filename[:-4]}.rttm'))
        reference = labels_to_pyannote_object(true_labels)

        performance = DiarizationErrorRate().compute_components(reference, merged_label)
        metrics = ['confusion', 'missed detection', 'false alarm']
        for metric in metrics:
            performance[metric] /= performance['total']
        performance['DER'] = sum(performance[metric] for metric in metrics)
        diarize_performance[filename] = performance

    df = {x : [] for x in performance}
    index = []

    for filename in diarize_performance:
        for x in diarize_performance[filename]:
            df[x].append(diarize_performance[filename][x])
        index.append(filename)

    df = pd.DataFrame(df, index=index)

    df.to_csv('diarize_performance.csv')
