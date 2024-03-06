import os
import json
import sys
from typing import List
sys.path.insert(1, "NeMo")
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from omegaconf import DictConfig, OmegaConf
import wget
import torch
from nemo.collections.asr.models import ClusteringDiarizer


def config_setup(audio_dir, filename) -> None:
    """
    Changes audio manifest file
    Arguments:
        * audio_dir (str): directory to audio files
        * filename (str): audio_dir must contain filename.wav and filename.json
            downloaded from cortico
    """
    json_path = os.path.join(audio_dir, f'{filename}.json')
    raw_json = json.loads(open(json_path, 'r').read())
    num_speakers = len(raw_json['participants'])
    for p in raw_json['participants']:
        if p['name'] in ('Hearth', 'participant', 'Participant'):
            num_speakers -= 1
    duration = raw_json['duration']
    rttm_dir = os.path.join(audio_dir, f'{filename}.rttm')
    create_rttm(json_path, audio_dir, filename)

    manifest = {
        'audio_filepath': os.path.join(audio_dir, f'{filename}.wav'),
        'offset': 0,
        'duration': duration,
        'label': 'infer',
        'text': '-',
        'num_speakers': num_speakers,
        'rttm_filepath': rttm_dir,
        'uem_filepath': None
    }
    config_path = 'config'
    manifest_path = os.path.join(config_path, 'manifest.json')

    json.dump(manifest, open(manifest_path, 'w'))


def create_rttm(json_path, rttm_dir, filename):
    raw_json = json.loads(open(json_path, "r").read())

    with open(os.path.join(rttm_dir, f"{filename}.rttm"), "wb") as f:
        for snippet in raw_json["snippets"]:
            speaker_id = snippet["speaker_id"]
            start = snippet["audio_start_offset"]
            duration = snippet["duration"]

            if duration == 0:
                continue
            
            fields = [
                "SPEAKER", filename, 1, start, duration, "<NA>", "<NA>", speaker_id, "<NA>"
            ]
            for i, field in enumerate(fields):
                fields[i] = str(field)

            line = " ".join(fields)
            f.write(line.encode('utf-8'))
            f.write(b'\n')
        f.close()

def create_reference_label(json_path, filename):
    raw_json = json.loads(open(json_path, "r").read())

    label_dir = os.path.join('outputs', 'pred_json', f'{filename}_reference_labels.txt')
    with open(label_dir, 'w') as f:
        for snippet in raw_json["snippets"]:
            speaker_id = snippet["speaker_id"]
            start = snippet["audio_start_offset"]
            end = snippet["audio_end_offset"]

            if start == end:
                continue

            f.write(f"{start}\t{end}\t{speaker_id}\n")


class NeMoDiarizer(ClusteringDiarizer):
    def __init__(self, cfg: DictConfig, speaker_model=None, device_id=0):
        super().__init__(cfg, speaker_model)
        self.output_dir = 'outputs'
        self.rttm_dir = os.path.join(self.output_dir, 'pred_rttms')
        self.json_dir = os.path.join(self.output_dir, 'pred_json')

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if not os.path.exists(self.json_dir):
            os.mkdir(self.json_dir)

        self._cfg.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'

        # Oracle speaker num
        self._cluster_params.oracle_num_speakers = True

    def diarize(self, paths2audio_files: List[str] = None, batch_size: int = 0):
        """
        generate json file in Cortico format
        """
        super().diarize(paths2audio_files, batch_size)

        # generate label file

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
                        f.write(f'{last_label['start']}\t{last_label['end']}\t{last_label['label']}\n')
                    last_label = {
                        'start' : start, 'end' : end, 'label' : label
                    }
                f.write(f'{last_label['start']}\t{last_label['end']}\t{last_label['label']}\n')
                    
        

if __name__ == '__main__':
    filename = 'conversation-67'
    json_dir = os.path.join('conversations', f'{filename}.json')
    create_reference_label(json_dir, filename)