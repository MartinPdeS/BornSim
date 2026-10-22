# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/BornSim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|--------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| bornsim/angular\_data.py               |      143 |       17 |       80 |       16 |     85.20% |63, 77, 115, 118, 121, 124, 167, 173, 189, 202, 205, 226, 234, 257, 260, 272, 324 |
| bornsim/directions.py                  |       34 |        1 |       10 |        0 |     97.73% |        99 |
| bornsim/ensemble.py                    |       53 |        2 |       14 |        3 |     92.54% |37, 68, 134-\>137 |
| bornsim/ensemble\_sampling.py          |       32 |        1 |       12 |        1 |     95.45% |        69 |
| bornsim/geometry.py                    |      168 |       10 |       44 |       10 |     90.57% |74, 105, 108, 138, 141, 233, 243, 327, 371, 418 |
| bornsim/green.py                       |       38 |        1 |        4 |        1 |     95.24% |        82 |
| bornsim/grid.py                        |       45 |        3 |       14 |        2 |     91.53% |43, 62, 95 |
| bornsim/material.py                    |       24 |        3 |       10 |        2 |     85.29% |31, 37, 45 |
| bornsim/medium/\_\_init\_\_.py         |       13 |        7 |        4 |        0 |     35.29% |     14-24 |
| bornsim/medium/base.py                 |       17 |        4 |        0 |        0 |     76.47% |27, 47, 56, 62 |
| bornsim/medium/particles/spheres.py    |       58 |        1 |       20 |        1 |     97.44% |        55 |
| bornsim/medium/random/fields.py        |       74 |        2 |       14 |        0 |     97.73% |   33, 103 |
| bornsim/medium/structured/composite.py |      137 |       16 |       70 |       10 |     86.47% |66, 69, 112, 115, 131, 211, 235, 293, 298-304, 308-313 |
| bornsim/medium/structured/periodic.py  |       37 |        2 |        8 |        2 |     91.11% |    47, 50 |
| bornsim/results.py                     |      389 |       36 |      164 |       26 |     88.07% |132, 141, 144, 227, 230, 251, 254, 257, 266, 284, 294, 397-400, 407, 412-418, 526, 529, 532, 546, 791, 895, 898, 927, 933, 936, 939, 970, 998, 1088-1093, 1146 |
| bornsim/rotation.py                    |       34 |        2 |        6 |        2 |     90.00% |    45, 48 |
| bornsim/sampling.py                    |       73 |        4 |       16 |        1 |     94.38% |162-166, 332 |
| bornsim/series.py                      |       84 |        2 |       22 |        2 |     96.23% |   92, 271 |
| bornsim/solver.py                      |       74 |        6 |       22 |        4 |     89.58% |50, 60-62, 109, 134, 272 |
| bornsim/units.py                       |       34 |        1 |       20 |        1 |     96.30% |        66 |
| bornsim/volume.py                      |      190 |        4 |       64 |        4 |     96.85% |57, 79, 89, 514 |
| **TOTAL**                              | **1874** |  **125** |  **622** |   **88** | **91.07%** |           |

11 files skipped due to complete coverage.


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/MartinPdeS/BornSim/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/BornSim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/MartinPdeS/BornSim/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/BornSim/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FMartinPdeS%2FBornSim%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/MartinPdeS/BornSim/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.