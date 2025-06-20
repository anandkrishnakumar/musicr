# MUSICR

MUSICR is a simple note detection project written in Python. It contains a Streamlit
application that loads a small neural network model to predict which musical note
is played in an audio clip. You can record audio through the browser or upload an
MP3 file.

## Requirements

- Python 3.8+
- The packages listed in `requirements.txt`

## Installation

Install the dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Running the App

After installing the requirements, start the Streamlit app:

```bash
streamlit run app/app.py
```

You will be able to record audio directly in the browser or upload an MP3 file
and see the predicted note.

## Repository Layout

- `app/` – the Streamlit front‑end.
- `models/` – pre‑trained model files (`note_detect.h5`).
- `notebooks/` – experiments and helper modules.
- `requirements.txt` – minimal list of Python dependencies.

## License

This project is licensed under the Apache 2.0 License.
