# Voice Diarization

## Data

The audio data and RTTM files are retrieved from the Cortico API. The audio files typically span about 1 hour and feature 4-6 speakers.

## Running the Application

This model facilitates diarization based on audio input and the specified number of speakers. To execute the model, place the desired audio files into a folder named "conversations". Additionally, include the participants' names following the JSON format outlined in the [Cortico API documentation](https://api.fora.io/docs#get-/v1/conversations/-conversation_id-). Then, execute the following command:

```
python diarization.py
```
