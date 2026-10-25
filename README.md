# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/MartinPdeS/BornSim/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                   |    Stmts |     Miss |   Branch |   BrPart |      Cover |   Missing |
|--------------------------------------- | -------: | -------: | -------: | -------: | ---------: | --------: |
| bornsim/angular\_data.py               |      144 |       17 |       80 |       16 |     85.27% |63, 77, 119, 122, 125, 128, 171, 177, 193, 206, 209, 230, 238, 261, 264, 276, 328 |
| bornsim/convergence.py                 |       71 |       11 |       34 |        7 |     79.05% |40, 44, 46, 75-\>79, 77-\>79, 92, 95, 108-116 |
| bornsim/directions.py                  |       34 |        1 |       10 |        0 |     97.73% |        99 |
| bornsim/ensemble.py                    |       63 |        2 |       14 |        3 |     93.51% |41, 72, 152-\>155 |
| bornsim/ensemble\_sampling.py          |       32 |        1 |       12 |        1 |     95.45% |        69 |
| bornsim/geometry.py                    |      168 |       10 |       44 |       10 |     90.57% |74, 105, 108, 138, 141, 233, 243, 327, 371, 418 |
| bornsim/green.py                       |       38 |        1 |        4 |        1 |     95.24% |        82 |
| bornsim/grid.py                        |       45 |        3 |       14 |        2 |     91.53% |43, 62, 95 |
| bornsim/material.py                    |       29 |        4 |       12 |        3 |     82.93% |34, 41, 50, 58 |
| bornsim/medium/\_\_init\_\_.py         |       13 |        7 |        4 |        0 |     35.29% |     14-24 |
| bornsim/medium/base.py                 |       17 |        4 |        0 |        0 |     76.47% |27, 47, 56, 62 |
| bornsim/medium/particles/spheres.py    |       58 |        1 |       20 |        1 |     97.44% |        55 |
| bornsim/medium/random/fields.py        |       75 |        2 |       14 |        0 |     97.75% |   34, 105 |
| bornsim/medium/structured/composite.py |      149 |       16 |       74 |       10 |     87.44% |67, 70, 115, 118, 135, 219, 243, 306, 311-317, 321-326 |
| bornsim/medium/structured/periodic.py  |       37 |        2 |        8 |        2 |     91.11% |    47, 50 |
| bornsim/results.py                     |      411 |       40 |      178 |       33 |     87.27% |151, 156, 159, 211, 245, 248, 262, 264, 268, 282, 285, 288, 297, 309, 318, 328, 444, 454, 463, 468-474, 583, 586, 589, 607, 852, 956, 959, 988, 994, 997, 1000, 1031, 1059, 1149-1154, 1207 |
| bornsim/rotation.py                    |       34 |        2 |        6 |        2 |     90.00% |    45, 48 |
| bornsim/sampling.py                    |       78 |        4 |       16 |        1 |     94.68% |163-167, 362 |
| bornsim/series.py                      |       92 |        2 |       22 |        2 |     96.49% |  104, 300 |
| bornsim/solver.py                      |       74 |        6 |       22 |        4 |     89.58% |50, 60-62, 109, 134, 283 |
| bornsim/source.py                      |       61 |        1 |       22 |        1 |     97.59% |       143 |
| bornsim/units.py                       |       34 |        1 |       20 |        1 |     96.30% |        66 |
| bornsim/volume.py                      |      190 |        4 |       64 |        4 |     96.85% |57, 79, 89, 514 |
| **TOTAL**                              | **2098** |  **142** |  **708** |  **104** | **90.81%** |           |

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