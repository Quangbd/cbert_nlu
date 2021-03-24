# Cpp BERT nlu

## Requirements

- Step 1: Build [tflite](https://www.tensorflow.org/lite/guide/build_cmake#build_tensorflow_lite_c_library)
- Step 2: Build [utf8proc](https://github.com/JuliaStrings/utf8proc)

## Example

- Input: `add sabrina salerno to the grime instrumentals playlist`
- Output:
```json
{
  "intent": {
    "intent_name": "AddToPlaylist",
    "probability": 1.0
  },
  "slot_labels": [
    "O",
    "B-artist",
    "I-artist",
    "O",
    "O",
    "B-playlist",
    "I-playlist",
    "O"
  ],
  "tokens": [
    "add",
    "sabrina",
    "salerno",
    "to",
    "the",
    "grime",
    "instrumentals",
    "playlist"
  ]
}
```
