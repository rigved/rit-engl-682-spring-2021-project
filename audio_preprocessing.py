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
import librosa.display
import librosa
import soundfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


if len(sys.argv) < 4:
    print('Usage: python ' + sys.argv[0] + ' <clips_directory> <modified_clips_directory> </path/to/validated.csv>')
    sys.exit(1)

# Directory containing all the original .wav voice recording clips
clips_directory = sys.argv[1]
# Directory where all the modified .wav voice recording clips will be stored
modified_clips_directory = sys.argv[2]
# Full path to CSV file containing list of voice recordings, in the same format as Mozilla's validated.tsv file
validated_input_filename = sys.argv[3]

# Output progress because modifying audio on many voice recordings can take a long time
progress_counter = 0
print('Progress: 0 instances completed')

with open(validated_input_filename, 'r') as validated_input_csv_file:
    reader = csv.DictReader(validated_input_csv_file)

    for line in reader:
        y, sr = librosa.load(os.path.join(clips_directory, line['path']), sr=16000)

        # Time stretch the audio spectrogram
        y_time_stretched = y.copy()
        tmp_time_stretched = librosa.effects.time_stretch(y, np.random.uniform(0.9, 1.1))
        min_len = min(y.shape[0], tmp_time_stretched.shape[0])
        y_time_stretched *= 0
        y_time_stretched[0:min_len] = tmp_time_stretched[0:min_len]

        # Pitch shift the audio spectrogram
        y_pitch_shifted = librosa.effects.pitch_shift(y, sr, n_steps=(4 * np.random.uniform()))

        # Increase the volume of the audio spectrogram
        y_louder = y * np.random.uniform(low=1.5, high=3.0)

        # Uncomment the following lines to view the audio wave-plots for the original and modified audio clips
        # plt.figure(figsize=(12, 4))
        # librosa.display.waveplot(y, sr=sr)
        # plt.show()
        #
        # plt.figure(figsize=(12, 4))
        # librosa.display.waveplot(y_time_stretched, sr=sr)
        # plt.show()
        #
        # plt.figure(figsize=(12, 4))
        # librosa.display.waveplot(y_pitch_shifted, sr=sr)
        # plt.show()
        #
        # plt.figure(figsize=(12, 4))
        # librosa.display.waveplot(y_louder, sr=sr)
        # plt.show()

        # Store the modified spectrograms in the modified clips directory
        soundfile.write(os.path.join(modified_clips_directory, line['path'].replace('.wav', '-time_stretched.wav')), y_time_stretched, sr, subtype='PCM_16')
        soundfile.write(os.path.join(modified_clips_directory, line['path'].replace('.wav', '-pitch_shifted.wav')), y_pitch_shifted, sr, subtype='PCM_16')
        soundfile.write(os.path.join(modified_clips_directory, line['path'].replace('.wav', '-augmented.wav')), y_louder, sr, subtype='PCM_16')

        progress_counter += 1

        if (progress_counter % 100) == 0:
            print('Progress: ' + str(progress_counter) + ' instances completed')

print('')
