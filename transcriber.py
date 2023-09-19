# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 09:55:02 2023

@author: rens_
"""
from deepmultilingualpunctuation import PunctuationModel
from vosk import Model, KaldiRecognizer, SetLogLevel
from tqdm import tqdm
from pydub import AudioSegment
import re
import os
import audioop
import json
import wave
import argparse
import warnings
warnings.simplefilter("ignore", UserWarning)

verbose = True
SetLogLevel(-1)

monoPath = "MonoAudio/"


def parse_arguments():
    """
    Parses the command line arguments.

    Returns
    -------
    LIST
        .parse_args() output.

    """
    parser = argparse.ArgumentParser(description='Your script description here')
    parser.add_argument('--lan', '--language', type=str, default='nl', choices=['nl', 'en'],
                        help='Specify the input language (nl or en)')
    parser.add_argument('input1', type=str, help='Input audio files')
    parser.add_argument('input2', type=str, nargs="?", default=None, help='Output text file')
    return parser.parse_args()


def hr_min_sec(secs):
    """
    Converts number of seconds to hours:minutes:seconds format

    Parameters
    ----------
    secs : INT/FLOAT
        Number of seconds.

    Returns
    -------
    STR
        {hours:02d}:{minutes:02d}:{secs:02d}

    """
    minutes, secs = divmod(secs, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def convert_to_wav(input_file):
    """
    Converts whatever (?) filetype to .wav for further use.

    Parameters
    ----------
    input_file : STR
        Name of the file, with extension.

    Returns
    -------
    STR
        New name of the file, i.e. with the .wav extension. 

    """
    input_split = input_file.split(".")
    input_ext = input_split[-1]
    wav_filename = input_split[0] + ".wav"

    # I've yet to run into a filetype which cannot be converted this way
    sound = AudioSegment.from_file(input_file, format=input_ext)
    file_handle = sound.export(wav_filename, format="wav")

    return file_handle.name


def wav_to_mono(wav_file, mono_wav_file):
    """
    Converts .wav files to mono channel format.

    Parameters
    ----------
    wav_file : STR
        File name of a .wav file, including extension.
    mono_wav_file : STR
        File name of the mono .wav file, including extension.

    Returns
    -------
    INT
        Number of sound bytes in the file (used for loading bars)

    """
    try:
        inFile = wave.open(wav_file, "rb")
        outFile = wave.open("MonoAudio/"+mono_wav_file, "wb")
        # force mono
        outFile.setnchannels(1)
        # set output file like the input file
        outFile.setsampwidth(inFile.getsampwidth())
        outFile.setframerate(inFile.getframerate())
        # read
        soundBytes = inFile.readframes(inFile.getnframes())
        #print("[INFO] frames read: {} length: {}".format(inFile.getnframes(),len(soundBytes)))
        # convert to mono and write file
        monoSoundBytes = audioop.tomono(soundBytes, inFile.getsampwidth(), 1, 1)
        outFile.writeframes(monoSoundBytes)

        inFile.close()
        outFile.close()

    except Exception as e:
        print(e)

    return len(soundBytes)


def transcribe(inputFile, language='nl'):
    """
    The actual transcription function

    Parameters
    ----------
    inputFile : STR
        Name of the input file, with extension.
    language : STR
        Language of the audio, either "en" or "nl"

    Returns
    -------
    textResults : LIST
        List of all words transcribed.

    """
    results = ""
    textResults = []
    converted = False

    inFileSplit = inputFile.split(".")
    monoFileName = inFileSplit[0] + "_mono.wav"

    if inFileSplit[-1] != "wav":
        inputFile = convert_to_wav(inputFile)
        converted = True

    wfLength = wav_to_mono(inputFile, monoFileName)
    wf = wave.open(os.path.join(monoPath, monoFileName))

    ##### Run model to recognise language #####
    if language == "nl":
        if "vosk-model-nl-spraakherkenning-0.6" in os.listdir(os.getcwd()):
            model = Model(model_path="vosk-model-nl-spraakherkenning-0.6")
        else:
            print("Large Dutch model not found - using the default Dutch model. See README for more info")
            model = Model(lang=language)
    elif language == "en":
        print("Language set to English.")
        model = Model(lang=language)

    recognizer = KaldiRecognizer(model, wf.getframerate())
    recognizer.SetWords(True)

    sample_rate = 4*48000

    pbar_format = "{desc}: {percentage:3.0f}% [{bar:40}] {n:f} s /{total:f} s | [{rate_fmt}, {elapsed}<{remaining}]"
    pbar = tqdm(total=wfLength/sample_rate, desc="Parsing audio...", bar_format=pbar_format)
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            recognizerResult = recognizer.Result()
            results = results + recognizerResult
            # convert the recognizerResult string into a dictionary
            # print(recognizerResult)
            resultDict = json.loads(recognizerResult)
            # save the 'text' value from the dictionary into a list
            textResults.append(resultDict.get("text", ""))

        if 0 <= pbar.n % 60 <= 0.03:
            textResults.append("[" + hr_min_sec(int(60 * round(pbar.n/60))) + "]")

        remainder = wfLength/sample_rate - pbar.n
        if remainder >= 16000/sample_rate:
            pbar.update(16000/sample_rate)
        else:
            pbar.update(remainder)

    pbar.close()

    # =============================================================================
    #     else:
    #         print(recognizer.PartialResult())
    # =============================================================================

    # process "final" result
    recognizerResult = recognizer.FinalResult()
    resultDict = json.loads(recognizerResult)
    # save the 'text' value from the dictionary into a list
    textResults.append(resultDict.get("text", ""))
    wf.close()
    print("[INFO] Results extracted.")

    if converted:
        os.remove(inputFile)

    return textResults


def capitalize_sentences(text):
    """
    Capitalises the first letter of every sentence in a string.

    Parameters
    ----------
    text : STR
        Not nicely capitalised string.

    Returns
    -------
    result_text : STR
        Nicely capitalised string.

    """
    # Split the text into sentences using a period (.) as the delimiter
    sentences = re.split(r'\.', text)

    # Define the regex pattern
    pattern = r'\b(?![^\[]*\])\w+\b'

    # Iterate through each sentence and find the first word not surrounded by square brackets
    for i, sentence in enumerate(sentences):
        match = re.search(pattern, sentence)
        if match:
            first_word = match.group()
            # Capitalize the first word
            capitalized_word = first_word.capitalize()
            # Replace the first word in the sentence with the capitalized version
            sentences[i] = sentence.replace(first_word, capitalized_word, 1)

    # Join the modified sentences back into a single string
    result_text = '.'.join(sentences)
    return result_text


def insert_newlines(text, max_line_length=80):
    """
    Clips lines to a maximum character length and inserts a new line \n

    Parameters
    ----------
    text : STR
        Any run-on string.
    max_line_length : INT
        Maximum length of text line, default is 80

    Returns
    -------
    STR
        String with enters added in.

    """
    lines = []
    words = text.split()
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "

    if current_line:
        lines.append(current_line.strip())

    return "\n".join(lines)


def make_folders():
    """
    Generates the subfolder in which the mono audio is temporarily kept.

    Returns
    -------
    None.

    """
    flag = False
    monoPathExist = os.path.exists(monoPath)

    if not monoPathExist:
        os.makedirs(monoPath)
    else:
        flag = len(os.listdir(monoPath)) > 0

    if flag:
        delete_stragglers()
        print("Cleared errant files from temp directories.")


def delete_stragglers():
    """
    Gets rid of temp files to save space.

    Returns
    -------
    None.

    """
    for monoFile in os.listdir(monoPath):
        os.remove(os.path.join(monoPath, monoFile))


def main():
    """
    Put it  all together.

    Returns
    -------
    None.

    """
    args = parse_arguments()
    language = args.lan

    make_folders()

    ##### Handle inputs #####
    inFileName = args.input1
    inFileSplit = inFileName.split(".")

    num_inputs = len([arg for arg in [args.input1, args.input2] if arg is not None])

    if num_inputs == 1:
        outfileText = inFileSplit[0] + "_transcript.txt"
    elif num_inputs == 2:
        outfileText = args.input2

    textResults = transcribe(inFileName, language)
    textOutput = " ".join(textResults)

    print("[INFO] Loading punctuation model...")
    # Credits to https://github.com/oliverguhr/deepmultilingualpunctuation

    if language == "nl":
        punctModel = PunctuationModel(model="oliverguhr/fullstop-dutch-punctuation-prediction")
    elif language == "en":
        punctModel = PunctuationModel(model="oliverguhr/fullstop-punctuation-multilang-large")

    print("[INFO] Adding punctuation...")
    resultsPunctuated = capitalize_sentences(punctModel.restore_punctuation(textOutput))
    print("[INFO] Punctuation added.")
    resultsFormatted = insert_newlines(resultsPunctuated, 80)

    if verbose:
        print_output = "[OUTPUT SAMPLE] " + resultsFormatted[:1000]
        if len(resultsFormatted) > 1000:
            print(print_output + "...")
        else:
            print(print_output)

    with open(outfileText, 'w') as output:
        output.write(resultsFormatted)

    print(f"\n[INFO] Text written to {outfileText}!")

    delete_stragglers()


if __name__ == "__main__":
    main()
