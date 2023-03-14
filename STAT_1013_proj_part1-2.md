---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="9xZnRXM7x0Cv">

# CUHK-STAT1013: Practical Assignment Part 1: Sharing Your Idea and Data

</div>

<div class="cell markdown" id="9Fy05KAkyJI0">

## NBA Champion dataset background

**Description**:

Dataset describing the number of home wins of the champion on the
NBA-Finals.

**Github**:
<https://github.com/dashamet/NBA-Finals/blob/master/championsdata.csv>

**Sample size**: 227

**Feature documentation**:

| Feature | Class | Shape | Dtype   |
|:--------|:------|:------|:--------|
| Year    |       |       | int64   |
| Team    |       |       | object  |
| Game    |       |       | int64   |
| Win     |       |       | int64   |
| Home    |       |       | int64   |
| MP      |       |       | int64   |
| FG.     |       |       | int64   |
| FGA     |       |       | int64   |
| FGP     |       |       | float64 |
| TP      |       |       | int64   |
| TPA     |       |       | int64   |
| TPP     |       |       | float64 |
| FT      |       |       | int64   |
| FTA     |       |       | int64   |
| FTP     |       |       | float64 |
| ORB     |       |       | int64   |
| DRB     |       |       | int64   |
| TRB     |       |       | int64   |
| AST     |       |       | int64   |
| STL     |       |       | int64   |
| BLK     |       |       | int64   |
| TOV     |       |       | int64   |
| PF      |       |       | int64   |
| PTS     |       |       | int64   |

</div>

<div class="cell markdown" id="k85zO7zxys4H">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.
    -   We are interested in "*Did teams with home-field have a better
        chance of winning in NBA?*"
-   What two groups you are comparing:
    -   **G1**: Winning percentage of home-games; **G2**: Winning
        percentage of away-games
-   What you will be measuring (i.e., what your response variable will
    be)
    -   `The number of wins`
-   Is your response variable quantitative rather than categorical?
    -   `Number of wins` is binary data, with the order `1 > 0`, which
        can be regarded as a quantitative variable.
-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.
    -   We'd expect that **G1** \> **G2** since [Referee bias and the
        psychological impact of playing at home are two of the biggest
        factors.](https://www.madduxsports.com/library/nba/properly-understanding-nba-home-court-advantage.html).
-   Talk about how you will gather your data
    -   From Github link:
        <https://github.com/dashamet/NBA-Finals/blob/master/championsdata.csv>
-   If you had unlimited resources (time, money, staff, etc.) how would
    you collect your data?
    -   \(i\) Attempt to collect more data on NBA; (ii) investigate if
        the provided dataset is a good random sampling subset of the
        official statistics of nba.

</div>

<div class="cell markdown" id="3GOdPWT03PQB">

## Prepare your dataset

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:233}"
id="mUxJb4hxvpHQ" outputId="119c9141-6fb6-4ecf-e8fe-d8b28368de50">

``` python
## load dataset from github

import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/dashamet/NBA-Finals/master/championsdata.csv')
df.head(5)
```

<div class="output execute_result" execution_count="4">

       Year    Team  Game  Win  Home   MP  FG  FGA    FGP  TP  ...    FTP  ORB  \
    0  1980  Lakers     1    1     1  240  48   89  0.539   0  ...  0.867   12   
    1  1980  Lakers     2    0     1  240  48   95  0.505   0  ...  0.667   15   
    2  1980  Lakers     3    1     0  240  44   92  0.478   0  ...  0.767   22   
    3  1980  Lakers     4    0     0  240  44   93  0.473   0  ...  0.737   18   
    4  1980  Lakers     5    1     1  240  41   91  0.451   0  ...  0.788   19   

       DRB  TRB  AST  STL  BLK  TOV  PF  PTS  
    0   31   43   30    5    9   17  24  109  
    1   37   52   32   12    7   26  27  104  
    2   34   56   20    5    5   20  25  111  
    3   31   49   23   12    6   19  22  102  
    4   37   56   28    7    6   21  27  108  

    [5 rows x 24 columns]

</div>

</div>

<div class="cell markdown" id="55xAIxVa3hpQ">

-   Tell us what groups you want to compare in the dataset
    -   **G1** (Win \| Home = 1) vs. **G2** (Win \| Home = 0)

</div>

<div class="cell markdown" id="13PdL3ht3902">

-   Print first 5 records of each group, respectively.

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="UNL0WXav3hLj" outputId="3f3bebd8-b238-4a7b-9f55-a15c273a0339">

``` python
## First 5 records of G1 (Home = 1)
(df[df['Home'] == '1']['Win']).head(5)
```

<div class="output execute_result" execution_count="5">

    Series([], Name: Win, dtype: int64)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="dhe52HVB4T1O" outputId="d3ba2657-948a-4fd3-c8ba-f90fc83c3755">

``` python
## First 5 records of G2 (Home = 0)
(df[df['Home'] == '0']['Win']).head(5)
```

<div class="output execute_result" execution_count="6">

    Series([], Name: Win, dtype: int64)

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="zEgfWXaKGvNC" outputId="66d0ca98-294a-4c9a-b261-ac0a19b4a7f8">

``` python
## Any other data description and visualization you want to add.

## Open question, be flexible and no example can be provided.
##Number of games win in home-field in nba champion history

len(df[(df['Home'] == 1)&(df['Win'] == 1)])
```

<div class="output execute_result" execution_count="25">

    91

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Rf5vi7HMIgls" outputId="20e6ee30-ed01-46cc-cd9e-c229461b0dec">

``` python
##Number of games win in away-field in nba champion history

len(df[(df['Home'] == 0)&(df['Win'] == 1)])
```

<div class="output execute_result" execution_count="3">

    69

</div>

</div>
