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

if len(sys.argv) < 6:
    print('Usage: python ' + sys.argv[0] + ' <clips_directory> <output_directory containing DeepSpeech models> <validated_file.csv> <output_hypothesis_file.txt> <output_gold_standard_file.txt>')
    sys.exit(1)

# Directory containing all the .wav voice recording clips
clips_directory = sys.argv[1]
# Directory to store the hypothesis outputs (for later use to calculate metrics like WER, WRR, and SER)
# This is also the directory where the DeepSpeech memory-mapped models are expected to be stored
output_directory = sys.argv[2]

# CSV file containing list of voice recordings, in the same format as Mozilla's validated.tsv file
validated_input_filename = os.path.join(output_directory, sys.argv[3])
# File to store DeepSpeech model output for all the voice recordings in the above mentioned CSV file
hypothesis_output_filename = os.path.join(output_directory, sys.argv[4])
# File to store gold standard for all the voice recordings in the above mentioned CSV file
gold_standard_output_filename = os.path.join(output_directory, sys.argv[5])

# Load the DeepSpeech memory-mapped model files
ds = deepspeech.Model(os.path.join(output_directory, 'deepspeech-0.9.3-models.pbmm'))
ds.enableExternalScorer(os.path.join(output_directory, 'deepspeech-0.9.3-models.scorer'))

# Output progress because performing ASR on many voice recordings can take a long time
progress_counter = 0
print('Progress: 0 instances completed')

with open(hypothesis_output_filename, 'w') as en_hypothesis_file:
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

                audio_transcription_hypothesis = ds.stt(audio_input)

                en_hypothesis_file.write(audio_transcription_hypothesis + '\n')
                en_gold_standard_file.write(line['sentence'] + '\n')

                progress_counter += 1

                if (progress_counter % 100) == 0:
                    print('Progress: ' + str(progress_counter) + ' instances completed')

print('')
