# Goal #1

Stream the dataset bigcode/the-stack-dedup
Scan each row of data by the 'lang' column, make sure it's either Python, C, C++ or JS code
The actual code is in the 'content' folder.

Create 3 json files (as 3 datasets) with these columns:

- 'lang'
- 'content'
- 'stars' (copy from the 'max_stars_count' column of the original dataset)

This computer has 2 vCPU, 8GB & 1TB storage, so do parallel as you need

# Goal 2

Stream the dataset HuggingFaceFW/fineweb-edu

Scan each row of data on the 'text' and 'url' columns for the following strings when made all lower-case
"geometry", "3d printing", "object", "shape", "dimension", "angle", "volume", "openscad", "solid", "model", "vector", "mesh", "cad"

if the match is found, create a new database in json format with the following columns only

- 'text' from the original 'text' column
- 'url' from the original 'url' column
- 'key_words' are made of words that found matching

Make sure so that when it runs, it first checks to see if any existing fineweb.json file is existing and skip the rows already processed, that way it continues where left off.

Save the dataset to fineweb.json as the code runs. Add in ETA.

When reaching 5B words total in the dataset, stop running.

# Goal 3

Similar to goal 2, except with HuggingFaceTB/dclm-edu dataset.

File name is dclm.json, limit is 3B words.
