# nanogpt-analysis
Measure the amount of algorithmic improvement in the nanogpt speedrun (https://github.com/KellerJordan/modded-nanogpt) versus the amount of effort applied.

analysis by tkwa

- Speed over time is super jagged
- Time vs LoC is 




prompt for claude

The goal of this project is to measure the amount of algorithmic improvement in the nanogpt speedrun (https://github.com/KellerJordan/modded-nanogpt) versus the amount of effort applied. We want to measure the training speed record vs calendar time, lines of code, and repository stars.

Info that may be helpful:
https://x.com/tamaybes/status/1890263324899848412
https://www.lesswrong.com/posts/j3gp8tebQiFJqzBgg/how-the-nanogpt-speedrun-wr-dropped-by-20-in-3-months

Use python and uv. Put all code in src/ and all data in data/
- Extract the table of timing data from the readme
- Create a graph of log(speed) over time, where each of the ~55 improvements is one data point, excluding the ones that are just retiming
- Star history
- Find the star history of the repo, probably using https://www.star-history.com/. If you cannot download the csv from here, tell me but continue with the rest
- Linearly interpolate to estimate stars at the exact dates of a run
- Lines of code
- Should be its own script
- We want two things, cumulative lines of code and lines of code for individual changes, but just implement the first one for now
- Cumulative lines of code are lines of TRAINING code in the PR that introduces the changes. I suspect you should ONLY count train_gpt.py. If there are other files that appear to be training code rather than data, document this.
- A file analysis.py should compile all this data into improvements.json, where each row is one improvement and the columns are date, days since start, record time, cumulative LoC, stars.
- In a python file plots.py which saves figures to plots/, fit a power law and an exponential. If appropriate, do transforms. Log the regression results to metrics.yaml. - Also visually inspect the graphs and note any anomalies.
- Also analyze the codebase to see how the listed date relates to the date when the changes were made, e.g. when the PR was opened
- Document the implementation details and observations in ai_notes.md
- Especially list each of the items you couldn’t finish or which have significant limitations

Ideas for future changes (do not implement yet)
- Normalize times
- Figure out how to track lines of code for the changes without PRs
- Lines of code for individual changes
- Figure out how to regress on individual changes data, similarly to the Tamay tweet
- Incorporate gpt-2 medium track (train_gpt_medium.py)
- More statistical modeling
- Tests
