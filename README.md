# Voice Diarization

## Data

The audio data and RTTM files are retrieved from the Cortico API. The audio files typically span about 1 hour and feature 4-6 speakers.

## Running the Application

This model facilitates diarization based on audio input and the specified number of speakers. To execute the model, place the desired audio files into a folder named "conversations". Additionally, include the participants' names following the JSON format outlined in the [Cortico API documentation](https://api.fora.io/docs#get-/v1/conversations/-conversation_id-). Then, execute the following command:

```
python diarization.py
```

## Model Performance

The table below summarizes the diarization performance of the model:

| Metric         | Missed Detection | False Alarm | Confusion Rate | DER     |
|----------------|------------------|-------------|----------------|---------|
| Mean           | 0.047562         | 0.024156    | 0.079058       | 0.150776|
| Standard Dev.  | 0.028238         | 0.011271    | 0.052352       | 0.059052|

We plan to further enhance the model performance by [utilizing post-processing techniques](https://arxiv.org/abs/2309.05248) for diarization results.
