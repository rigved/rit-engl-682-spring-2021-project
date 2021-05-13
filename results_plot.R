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

library(ggplot2)
library(reshape2)

set.seed(42)

# Store the results
results <- data.frame(Metric = c('WER', 'WRR', 'SER'), B = c(51.5, 50.1, 85.5), LT = c(51.6, 49.9, 85.5), G = c(51.6, 49.9, 85.2), TS = c(83.6, 16.8, 95.9), PS = c(87.9, 12.4, 97.1), VA = c(53.8, 47.7, 86.4))
mresults <- melt(results, id.vars = 'Metric', value.name = 'Percentage', variable.name = 'Tool')

# Display the results data frame
mresults

# Plot the results to show similarities and differences in the results
ggplot(mresults, aes(x = Metric, y = Percentage, color = Tool, group = Tool)) + geom_line() + scale_color_manual(values = c('B'='black', 'LT'='red', 'G'='orange', 'TS'='blue', 'PS'='yellow', 'VA'='green')) + scale_linetype_manual(values = c('B'='solid', 'LT'='dashed', 'G'='dashed', 'TS'='dotted', 'PS'='dotted', 'VA'='dotted'))
