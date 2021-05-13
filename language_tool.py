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
import language_tool_python


if len(sys.argv) < 4:
    print('Usage: python ' + sys.argv[0] + ' <output_directory> <deepspeech_output_hypothesis_file.txt> <language_tool_output_hypothesis_file.txt>')
    sys.exit(1)

# Directory to store the hypothesis output (for later use to calculate metrics like WER, WRR, and SER)
output_directory = sys.argv[1]
# File containing DeepSpeech model outputs on which Language Tool post-processing is required
hypothesis_filename = os.path.join(output_directory, sys.argv[2])
# File to store output from Language Tool
language_tool_filename = os.path.join(output_directory, sys.argv[3])

# Initialize Language Tool
language_tool = language_tool_python.LanguageTool('en-us')

# Output progress because performing Language Tool post-processing on many voice recordings can take time
progress_counter = 0
print('Progress: 0 instances completed')

with open(hypothesis_filename, 'r') as hypothesis_file:
    with open(language_tool_filename, 'w') as language_tool_file:
        for deepspeech_hypothesis in hypothesis_file:
            deepspeech_hypothesis = deepspeech_hypothesis.strip()

            # Use Language Tool to correct the given sentence. Ignore capitalization.
            language_tool_hypothesis = language_tool.correct(deepspeech_hypothesis).lower()

            # Store the output
            language_tool_file.write(language_tool_hypothesis + '\n')

            progress_counter += 1

            if (progress_counter % 100) == 0:
                print('Progress: ' + str(progress_counter) + ' instances completed')

print('')
