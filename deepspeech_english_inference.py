# De-biasing Transcribed Text from Automatic Speech Recognition Systems
# Copyright (C) 2021  Rigved Rakshit
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import csv
import wave
import deepspeech
import numpy as np

if len(sys.argv) < 7:
    print('Usage: python ' + sys.argv[0] + ' <clips_directory> <modified_clips_directory> <output_directory containing DeepSpeech models> <validated_file.csv> <output_hypothesis_file.txt> <output_gold_standard_file.txt>')
    sys.exit(1)

# Directory containing all the .wav voice recording clips
clips_directory = sys.argv[1]
# Directory containing all the modified .wav voice recording clips
modified_clips_directory = sys.argv[2]
# Directory to store the hypothesis outputs (for later use to calculate metrics like WER, WRR, and SER)
# This is also the directory where the DeepSpeech memory-mapped models are expected to be stored
output_directory = sys.argv[3]

# CSV file containing list of voice recordings, in the same format as Mozilla's validated.tsv file
validated_input_filename = os.path.join(output_directory, sys.argv[4])
# Files to store DeepSpeech model output for all the voice recordings in the above mentioned CSV file
hypothesis_output_filename = os.path.join(output_directory, sys.argv[5])
hypothesis_time_stretched_output_filename = os.path.join(output_directory, sys.argv[5].replace('.txt', '-time_stretched.txt'))
hypothesis_pitch_shifted_output_filename = os.path.join(output_directory, sys.argv[5].replace('.txt', '-pitch_shifted.txt'))
hypothesis_louder_output_filename = os.path.join(output_directory, sys.argv[5].replace('.txt', '-augmented.txt'))
# File to store gold standard for all the voice recordings in the above mentioned CSV file
gold_standard_output_filename = os.path.join(output_directory, sys.argv[6])

# Load the DeepSpeech memory-mapped model files
ds = deepspeech.Model(os.path.join(output_directory, 'deepspeech-0.9.3-models.pbmm'))
ds.enableExternalScorer(os.path.join(output_directory, 'deepspeech-0.9.3-models.scorer'))

# Output progress because performing ASR on many voice recordings can take a long time
progress_counter = 0
print('Progress: 0 instances completed')

with open(hypothesis_output_filename, 'w') as en_hypothesis_file:
    with open(hypothesis_time_stretched_output_filename, 'w') as en_hypothesis_time_stretched_file:
        with open(hypothesis_pitch_shifted_output_filename, 'w') as en_hypothesis_pitch_shifted_file:
            with open(hypothesis_louder_output_filename, 'w') as en_hypothesis_louder_file:
                with open(gold_standard_output_filename, 'w') as en_gold_standard_file:
                    with open(validated_input_filename, 'r') as validated_input_csv_file:
                        reader = csv.DictReader(validated_input_csv_file)

                        for line in reader:
                            with wave.open(os.path.join(clips_directory, line['path']), 'rb') as audio_input_file:
                                audio_input = np.frombuffer(
                                    audio_input_file.readframes(
                                        audio_input_file.getnframes()
                                    ), np.int16
                                )

                            audio_time_stretched_filename = os.path.join(modified_clips_directory, line['path'].replace('.wav', '-time_stretched.wav'))

                            with wave.open(audio_time_stretched_filename, 'rb') as audio_time_stretched_file:
                                audio_time_stretched = np.frombuffer(
                                    audio_time_stretched_file.readframes(
                                        audio_time_stretched_file.getnframes()
                                    ), np.int16
                                )

                            audio_pitch_shifted_filename = os.path.join(modified_clips_directory, line['path'].replace('.wav', '-pitch_shifted.wav'))

                            with wave.open(audio_pitch_shifted_filename, 'rb') as audio_pitch_shifted_file:
                                audio_pitch_shifted = np.frombuffer(
                                    audio_pitch_shifted_file.readframes(
                                        audio_pitch_shifted_file.getnframes()
                                    ), np.int16
                                )

                            audio_louder_filename = os.path.join(modified_clips_directory, line['path'].replace('.wav', '-augmented.wav'))

                            with wave.open(audio_louder_filename, 'rb') as audio_louder_file:
                                audio_louder = np.frombuffer(
                                    audio_louder_file.readframes(
                                        audio_louder_file.getnframes()
                                    ), np.int16
                                )

                            audio_transcription_hypothesis = ds.stt(audio_input)

                            en_hypothesis_file.write(audio_transcription_hypothesis + '\n')
                            en_gold_standard_file.write(line['sentence'] + '\n')

                            audio_transcription_hypothesis = ds.stt(audio_time_stretched)

                            en_hypothesis_time_stretched_file.write(audio_transcription_hypothesis + '\n')

                            audio_transcription_hypothesis = ds.stt(audio_pitch_shifted)

                            en_hypothesis_pitch_shifted_file.write(audio_transcription_hypothesis + '\n')

                            audio_transcription_hypothesis = ds.stt(audio_louder)

                            en_hypothesis_louder_file.write(audio_transcription_hypothesis + '\n')

                            progress_counter += 1

                            if (progress_counter % 10) == 0:
                                print('Progress: ' + str(progress_counter) + ' instances completed')

print('')
