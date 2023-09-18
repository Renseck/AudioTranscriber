## Python command line audio transciber
This script transcribes audio from recorded files (no live transcription as yet) to .txt files. It adds punctuation and convenient time markings in the file as well.

## Motivation
Taking notes in long interviews is a painstaking process - why not let the computer do it for you? There are online services to help you with this, but in the interest of protecting IP and identities, I wanted to create a fully offline program to do it.

## Installation:

Open an Anaconda prompt, cd to the AudioTranscriber folder and run `pip install -r requirements.txt` to make sure
all important packages are installed. Optionally, you may go to the Vosk's [official website](https://alphacephei.com/vosk/models) and download a specific model to improve your results. 

E.g. for Dutch, the model named "vosk-model-nl-spraakherkenning-0.6" is used to improve results. Simply place it in the local folder where the .py file is located.

## Usage:

The program currently accepts most common audio files (such as the ones generated by Windows' Recorder program, .m4a). 
Simply place these directly in the AudioTranscriber folder, where the .py file is also located.
Again in an Anaconda prompt, `cd` to the AudioTransciber folder. There, the command to run the script is:

```python
python transcriber.py name_of_audio_file.extension name_of_transcipt_file.txt
```

The argument regarding the name of the transcript file is optional; the program generates a name for the transcript file automatically if this argument is left empty (recommended). It is important you include the ".extension", .e.g ".m4a", or the program will not find the file you are referring to.

### Changing the language

The default language of this program is Dutch, but you can change it to work for English audio as well. This is done simply by running the script as
```python
python transcriber.py --lan en name_of_audio_file.extension name_of_transcipt_file.txt
```

Supported languages:
1. Dutch (default)
2. English

## Credits and References
The pre-trained model to restore punctuation to the transcript was found at [here](https://github.com/oliverguhr/deepmultilingualpunctuation). This repo contains excellent instructions on how to use it further.

Vosk was used for the audio-transcription, more info can be found on the [github](https://github.com/alphacep/vosk-api) or the [official website](https://alphacephei.com/vosk/)

## License
MIT License 

Copyright (c) 2023 Rens van Eck

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
