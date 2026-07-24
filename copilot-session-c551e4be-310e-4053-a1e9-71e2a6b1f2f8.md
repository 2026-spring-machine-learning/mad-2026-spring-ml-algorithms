# Copilot CLI Session

> [!NOTE]
> - **Session ID:** `c551e4be-310e-4053-a1e9-71e2a6b1f2f8`  
> - **Started:** 7/24/2026, 4:24:18 PM  
> - **Duration:** 14m 56s  
> - **Exported:** 7/24/2026, 4:39:15 PM  

---

<sub>0s</sub>

### Info

No copilot-instructions.md found. Run /init to generate.

---

<sub>0s</sub>

### Info

Tip: /init

---

<sub>1m 36s</sub>

### User

init a uv project here with a venv and all the expected things

---

<sub>1m 38s</sub>

### Copilot

I’ll scaffold a standard `uv` Python project in this repo, create a local virtual environment, and wire the usual starter files.



---

<sub>2m 17s</sub>

### `bash`

**List repository root**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && ls -la

<details>
<summary>8 lines</summary>

```
total 16
drwxr-xr-x   6 tsteele  staff   192 Jul 24 14:42 .
drwxr-xr-x   6 tsteele  staff   192 Jul 24 14:42 ..
drwxr-xr-x  14 tsteele  staff   448 Jul 24 14:45 .git
-rw-r--r--   1 tsteele  staff  4688 Jul 24 14:42 .gitignore
drwxr-xr-x   4 tsteele  staff   128 Jul 24 14:42 classification
drwxr-xr-x   5 tsteele  staff   160 Jul 24 14:42 linear_regression
<shellId: 0 completed with exit code 0>
```

</details>


---

<sub>2m 17s</sub>

### `bash`

**Check uv availability**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv --version

```
uv 0.10.11 (006b56b12 2026-03-16)
<shellId: 1 completed with exit code 0>
```


---

<sub>2m 25s</sub>

### `bash`

**Inspect uv init options**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv init --help | sed -n '1,220p'

<details>
<summary>61 lines</summary>

```
Create a new project

Usage: uv init [OPTIONS] [PATH]

Arguments:
  [PATH]  The path to use for the project/script

Options:
      --name <NAME>                    The name of the project
      --bare                           Only create a `pyproject.toml`
      --package                        Set up the project to be built as a Python package
      --no-package                     Do not set up the project to be built as a Python package
      --app                            Create a project for an application
      --lib                            Create a project for a library
      --script                         Create a script
      --description <DESCRIPTION>      Set the project description
      --no-description                 Disable the description for the project
      --vcs <VCS>                      Initialize a version control system for the project [possible
                                       values: git, none]
      --build-backend <BUILD_BACKEND>  Initialize a build-backend of choice for the project [env:
                                       UV_INIT_BUILD_BACKEND=] [possible values: uv, hatch, flit,
                                       pdm, poetry, setuptools, maturin, scikit]
      --no-readme                      Do not create a `README.md` file
      --author-from <AUTHOR_FROM>      Fill in the `authors` field in the `pyproject.toml` [possible
                                       values: auto, git, none]
      --no-pin-python                  Do not create a `.python-version` file for the project
      --no-workspace                   Avoid discovering a workspace and create a standalone project

Python options:
  -p, --python <PYTHON>      The Python interpreter to use to determine the minimum supported Python
                             version. [env: UV_PYTHON=]
      --managed-python       Require use of uv-managed Python versions [env: UV_MANAGED_PYTHON=]
      --no-managed-python    Disable use of uv-managed Python versions [env: UV_NO_MANAGED_PYTHON=]
      --no-python-downloads  Disable automatic downloads of Python. [env:
                             "UV_PYTHON_DOWNLOADS=never"]

Cache options:
  -n, --no-cache               Avoid reading from or writing to the cache, instead using a temporary
                               directory for the duration of the operation [env: UV_NO_CACHE=]
      --cache-dir <CACHE_DIR>  Path to the cache directory [env: UV_CACHE_DIR=]

Global options:
  -q, --quiet...
          Use quiet output
  -v, --verbose...
          Use verbose output
      --color <COLOR_CHOICE>
          Control the use of color in output [possible values: auto, always, never]
      --native-tls
          Whether to load TLS certificates from the platform's native store [env: UV_NATIVE_TLS=]
      --offline
          Disable network access [env: UV_OFFLINE=]
      --allow-insecure-host <ALLOW_INSECURE_HOST>
          Allow insecure connections to a host [env: UV_INSECURE_HOST=]
      --no-progress
          Hide all progress outputs [env: UV_NO_PROGRESS=]
      --directory <DIRECTORY>
          Change to the given directory prior to running the command [env: UV_WORKING_DIR=]
      --project <PROJECT>
          Discover a project in the given directory [env: UV_PROJECT=]
      --config-file <CONFIG_FILE>
          The path to a `uv.toml` file to use for configuration [env: UV_CONFIG_FILE=]
      --no-config
          Avoid discovering configuration files (`pyproject.toml`, `uv.toml`) [env: UV_NO_CONFIG=]
  -h, --help
          Display the concise help for this command
<shellId: 2 completed with exit code 0>
```

</details>


---

<sub>2m 29s</sub>

### `bash`

**Initialize uv app project at repo root**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv init --app --vcs none --no-workspace .

```
Initialized project `mad-2026-spring-ml-algorithms` at `/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms`
<shellId: 3 completed with exit code 0>
```


---

<sub>2m 31s</sub>

### `bash`

**Create virtual environment and sync project deps**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv sync

```
Using CPython 3.14.3
Creating virtual environment at: .venv
Resolved 1 package in 19ms
Checked in 0.46ms
<shellId: 4 completed with exit code 0>
```


---

<sub>2m 33s</sub>

### `bash`

**List root after uv init**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && ls -la

<details>
<summary>14 lines</summary>

```
total 48
drwxr-xr-x  12 tsteele  staff   384 Jul 24 16:26 .
drwxr-xr-x   6 tsteele  staff   192 Jul 24 14:42 ..
drwxr-xr-x  14 tsteele  staff   448 Jul 24 14:45 .git
-rw-r--r--   1 tsteele  staff  4688 Jul 24 14:42 .gitignore
-rw-r--r--   1 tsteele  staff     5 Jul 24 16:26 .python-version
drwxr-xr-x   8 tsteele  staff   256 Jul 24 16:26 .venv
-rw-r--r--   1 tsteele  staff     0 Jul 24 16:26 README.md
drwxr-xr-x   4 tsteele  staff   128 Jul 24 14:42 classification
drwxr-xr-x   5 tsteele  staff   160 Jul 24 14:42 linear_regression
-rw-r--r--   1 tsteele  staff   107 Jul 24 16:26 main.py
-rw-r--r--   1 tsteele  staff   175 Jul 24 16:26 pyproject.toml
-rw-r--r--   1 tsteele  staff   149 Jul 24 16:26 uv.lock
<shellId: 5 completed with exit code 0>
```

</details>


---

<sub>2m 33s</sub>

### `bash`

**Show created files**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && git --no-pager status --short

<details>
<summary>6 lines</summary>

```
?? .python-version
?? README.md
?? main.py
?? pyproject.toml
?? uv.lock
<shellId: 6 completed with exit code 0>
```

</details>


---

<sub>2m 35s</sub>

### `rg`

**\.venv**

"\.venv" (.gitignore)

```
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.gitignore:124:#   in the .venv directory. It is recommended not to include this directory in version control.
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.gitignore:140:.venv
```


---

<sub>2m 36s</sub>

### Copilot

Done — this repo is now initialized as a `uv` project with a local virtual environment.

Created at the repo root: `pyproject.toml`, `uv.lock`, `.python-version`, `main.py`, `README.md`, and `.venv/` (via `uv sync`).  
To use the environment now: `source .venv/bin/activate`

---

<sub>5m 4s</sub>

### User

not sure toml includes dependencies from py files already in project

---

<sub>5m 8s</sub>

### Copilot

I’ll scan the existing Python files for third-party imports and compare them to `pyproject.toml`, then I’ll add any missing dependencies.



---

<sub>5m 11s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/pyproject.toml.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/pyproject.toml

<details>
<summary>8 lines</summary>

```
1. [project]
2. name = "mad-2026-spring-ml-algorithms"
3. version = "0.1.0"
4. description = "Add your description here"
5. readme = "README.md"
6. requires-python = ">=3.14"
7. dependencies = []
8.
```

</details>


---

<sub>5m 11s</sub>

### `rg`

**^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)**

"^(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)" (classification, linear_regression, main.py)

<details>
<summary>17 matches</summary>

```
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:1:import numpy as np
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:2:import pandas as pd
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:3:import sklearn.linear_model as lm
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:4:import sklearn.model_selection as ms
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:5:import sklearn.metrics as metrics
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:8:import matplotlib.pyplot as plt
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:9:from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:10:import seaborn as sns
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:1:import numpy as np
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:2:import pandas as pd
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:3:import sklearn.model_selection as ms
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:4:import sklearn.linear_model as lm
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:5:import sklearn.ensemble as es
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:6:import sklearn.metrics as metrics
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:7:import imblearn.over_sampling as ios
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:8:import matplotlib.pyplot as plt
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py:9:import seaborn as sns
```

</details>


---

<sub>5m 17s</sub>

### `bash`

**Add third-party dependencies used by existing scripts**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv add numpy pandas scikit-learn imbalanced-learn matplotlib seaborn

<details>
<summary>38 lines</summary>

```
Resolved 22 packages in 628ms
Downloading scikit-learn (7.9MiB)
Downloading numpy (5.1MiB)
Downloading pandas (9.5MiB)
Downloading fonttools (2.7MiB)
Downloading scipy (19.5MiB)
Downloading matplotlib (8.9MiB)
Downloading pillow (4.6MiB)
 Downloaded fonttools
 Downloaded pillow
 Downloaded numpy
 Downloaded scikit-learn
 Downloaded matplotlib
 Downloaded pandas
 Downloaded scipy
Prepared 11 packages in 3.17s
Installed 20 packages in 205ms
 + contourpy==1.3.3
 + cycler==0.12.1
 + fonttools==4.63.0
 + imbalanced-learn==0.14.2
 + joblib==1.5.3
 + kiwisolver==1.5.0
 + matplotlib==3.11.1
 + narwhals==2.24.0
 + numpy==2.5.1
 + packaging==26.2
 + pandas==3.0.5
 + pillow==12.3.0
 + pyparsing==3.3.2
 + python-dateutil==2.9.0.post0
 + scikit-learn==1.9.0
 + scipy==1.18.0
 + seaborn==0.13.2
 + six==1.17.0
 + sklearn-compat==0.1.6
 + threadpoolctl==3.6.0
<shellId: 7 completed with exit code 0>
```

</details>


---

<sub>5m 50s</sub>

### `bash`

**Smoke test installed dependencies**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv run python -c "import numpy,pandas,sklearn,imblearn,matplotlib,seaborn; print('ok')"

```
<command with shellId: 8 is still running after 30 seconds. The command is still running but hasn't produced output yet. You will be automatically notified when it completes; if you need the command to complete end your response with no tool calls to wait for the notification, or use stop_bash to stop it.>
```


---

<sub>6m 1s</sub>

### `read_bash`

**Waiting up to 10 seconds for command output**

<details>
<summary>Arguments</summary>

```json
{
  "shellId": "8",
  "delay": 10
}
```

</details>

```
<command with shellId: 8 is still running after 10 seconds. The command is still running but hasn't produced output yet. You will be automatically notified when it completes; if you need the command to complete end your response with no tool calls to wait for the notification, or use stop_bash to stop it.>
```


---

<sub>6m 3s</sub>

### `stop_bash`

<details>
<summary>Arguments</summary>

```json
{
  "shellId": "8"
}
```

</details>

```
<command with id: 8 stopped>
```


---

<sub>6m 31s</sub>

### `bash`

**Retry dependency import smoke test**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && MPLBACKEND=Agg uv run python -c "import numpy,pandas,sklearn,imblearn,matplotlib,seaborn; print('ok')"

```
ok
<shellId: 9 completed with exit code 0>
```


---

<sub>6m 32s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/pyproject.toml.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/pyproject.toml

<details>
<summary>15 lines</summary>

```
1. [project]
2. name = "mad-2026-spring-ml-algorithms"
3. version = "0.1.0"
4. description = "Add your description here"
5. readme = "README.md"
6. requires-python = ">=3.14"
7. dependencies = [
8.     "imbalanced-learn>=0.14.2",
9.     "matplotlib>=3.11.1",
10.     "numpy>=2.5.1",
11.     "pandas>=3.0.5",
12.     "scikit-learn>=1.9.0",
13.     "seaborn>=0.13.2",
14. ]
15.
```

</details>


---

<sub>6m 33s</sub>

### `bash`

**Show changed files**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && git --no-pager status --short

<details>
<summary>6 lines</summary>

```
?? .python-version
?? README.md
?? main.py
?? pyproject.toml
?? uv.lock
<shellId: 10 completed with exit code 0>
```

</details>


---

<sub>6m 34s</sub>

### Copilot

You were right — it didn’t initially include them, but it does now.

I added the third‑party imports used by your existing scripts to `pyproject.toml`: `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, and `seaborn`, and `uv.lock` was updated accordingly.

---

<sub>7m 26s</sub>

### User

what about main.py being a mini cli-ish to run other py files, either as a true, simple, cli, or just some sort of comment in/out to run specific py, what do you think

---

<sub>7m 29s</sub>

### Copilot

I’d do a **true tiny CLI** in `main.py`, not comment-in/out blocks.

Comment toggles are quick once, but they get brittle and easy to forget. A minimal argparse-based CLI stays simple and makes runs explicit, e.g. `uv run python main.py diabetes` vs `... cherry-tree`.

If you want, I can wire `main.py` to support commands for your current scripts (`classification/analyze_diabetes.py` and `linear_regression/analyze_cherry_tree.py`) with ~20–30 lines.

---

<sub>8m 28s</sub>

### User

go for it

---

<sub>8m 31s</sub>

### Copilot

I’ll implement a small argparse CLI in `main.py` that dispatches to your two analysis scripts by name.



---

<sub>8m 32s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/main.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/main.py

<details>
<summary>7 lines</summary>

```
1. def main():
2.     print("Hello from mad-2026-spring-ml-algorithms!")
3. 
4. 
5. if __name__ == "__main__":
6.     main()
7.
```

</details>


---

<sub>8m 32s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py (lines 1-220)

<details>
<summary>108 lines</summary>

```
1. import numpy as np
2. import pandas as pd
3. import sklearn.model_selection as ms
4. import sklearn.linear_model as lm
5. import sklearn.ensemble as es
6. import sklearn.metrics as metrics
7. import imblearn.over_sampling as ios
8. import matplotlib.pyplot as plt
9. import seaborn as sns
10. 
11. 
12. def compute_confusion_matrix_numbers(actual_data_df, prediction):
13.     confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
14.     command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
15.     command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
16.     command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
17.     true_negs = 0
18.     false_poss = 0
19.     false_negs = 0
20.     true_poss = 0
21.     sensitivity = 0
22.     specificity = 0
23.     if len(confusion_tuple.ravel()) == 4:
24.         (true_negs, false_poss, false_negs, true_poss) = confusion_tuple.ravel()
25.         if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
26.             sensitivity = true_poss / (true_poss + false_negs)
27.             specificity = true_negs / (true_negs + false_poss)
28.     return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)
29. 
30. 
31. def create_confusion_matrix(actual_data_df, prediction):
32.     (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
33.         = compute_confusion_matrix_numbers(actual_data_df, prediction)
34.     # if (sensitivity > 0) or (specificity > 0):
35.     #     print(f'tp: {true_poss}, fn: {false_negs}, tn: {true_negs}, fp: {false_poss}, sensitivity: {sensitivity}, specificity: {specificity}.')
36.     # print(command_line_display_as_accuracy_top_confusion_matrix)
37.     sns.heatmap(confusion_tuple, annot=True)
38.     plt.show()
39. 
40. 
41. def read_diabetes():
42.      """Read the pa_diabetes.csv file into a pandas DataFrame."""
43.      df = pd.read_csv("pa_diabetes.csv")
44.      return df
45. 
46. 
47. def show_accuracy(diabetes_response_testing, predictions):
48.     print(f"Accuracy: {(predictions == diabetes_response_testing.values).mean()}")
49.     # print((predictions == diabetes_response_testing.values).mean())
50. 
51. 
52. def perform_logistic_regression(predictors, response, balance_counter):
53.     # for random_state in range(0, 3):
54.     #     if balance_counter == 1:
55.     #         random_over_sampler = ios.RandomOverSampler(random_state=random_state)
56.     #         predictors, response \
57.     #             = random_over_sampler.fit_resample(predictors, response)
58. 
59.     (diabetes_predictors_training, diabetes_predictors_testing,
60.     diabetes_response_training, diabetes_response_testing) = \
61.         ms.train_test_split(predictors, response, test_size=0.2)
62.         # random_state=random_state) #, random_state=0)
63. 
64.     algorithm = lm.LogisticRegression(max_iter=100000)
65.     # algorithm = es.RandomForestClassifier()
66.     model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
67.     predictions = model.predict(diabetes_predictors_testing)
68.     show_accuracy(diabetes_response_testing, predictions)
69.     # show_accuracy(diabetes_response_testing, all_zeros)
70. 
71.     # create_confusion_matrix(diabetes_response_testing, predictions)
72. 
73. 
74. # Cross-validation. Sklearn makes it almost too simple. It's hard to remember
75. # that it's performing many steps that we often perform manually.
76. def perform_logistic_regression_with_cross_validation(predictors, response):    
77.     algorithm = lm.LogisticRegression(max_iter=100000)
78.     scores = \
79.         ms.cross_val_score(algorithm, predictors, response, scoring='accuracy', cv=10)
80.     print(f"Scores: {scores}, average: {np.mean(scores)}")
81. 
82. 
83. def analyze(diabetes_df):
84.     # diabetes_df = ['3']
85.     diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
86.     predictors = diabetes_df.drop('Outcome', axis='columns')
87.     response = diabetes_df['Outcome']
88.     # print(predictors.columns)
89. 
90.     # for balance_counter in range(2):
91.     #     if balance_counter == 0:
92.     #         print("Unbalanced:")
93.     #     else:
94.     #         print("Balanced:")
95.     #     perform_logistic_regression(predictors, response, balance_counter)
96. 
97.     perform_logistic_regression_with_cross_validation(predictors, response)
98. 
99. 
100. def main():
101.     df = read_diabetes()
102.     # print(df.head())
103.     analyze(df)
104. 
105. 
106. if __name__ == "__main__":
107.     main()
108.
```

</details>


---

<sub>8m 32s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py (lines 1-240)

<details>
<summary>96 lines</summary>

```
1. import numpy as np
2. import pandas as pd
3. import sklearn.linear_model as lm
4. import sklearn.model_selection as ms
5. import sklearn.metrics as metrics
6. 
7. # Add plotting imports
8. import matplotlib.pyplot as plt
9. from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
10. import seaborn as sns
11. 
12. 
13. def feature_select(tree_df):
14.     one_hot_encoded_columns = pd.get_dummies(tree_df[['Season']])
15.     # Won't work with Season as strings!
16.     # predictors_df = tree_df[['Height', 'Diam', 'Season']]
17.     predictors_df = tree_df[['Height', 'Diam']]
18.     # predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis='columns')
19. 
20.     # Most of this block is for linearity.
21.     predictors_and_response_df = pd.concat([predictors_df, tree_df['Volume']], axis='columns')
22.     print(predictors_and_response_df)
23.     correlation_matrix = predictors_and_response_df.corr()
24.     volume_correlation_matrix = correlation_matrix[['Volume']].sort_values(by='Volume', ascending=False)
25.     print(volume_correlation_matrix)
26.     # print(correlation_matrix)
27.     # sns.heatmap(volume_correlation_matrix, annot=True)
28.     # plt.show()
29. 
30.     # For independence.
31.     print(correlation_matrix)
32.     mask = abs(correlation_matrix < 0.3)
33.     # sns.heatmap(correlation_matrix, annot=True, mask=mask)
34.     # plt.show()
35. 
36.     response_series = tree_df['Volume'].to_numpy()
37. 
38.     return (predictors_df, response_series)
39. 
40. 
41. def predict(cherry_tree_df):
42.     print(cherry_tree_df)
43.     (predictors_df, response_series) = feature_select(cherry_tree_df)
44. 
45.     # print("Before print")
46.     # print(predictors_df)
47.     # print(response_series)
48.     # print("After print")
49. 
50.     (predictors_training_df, predictors_testing_df, \
51.         response_training_df, response_testing_df) \
52.             = ms.train_test_split(predictors_df, response_series, test_size=0.2)
53. 
54.     # Train and predict.
55.     algorithm = lm.LinearRegression()
56.     model = algorithm.fit(predictors_training_df, response_training_df)
57.     prediction = model.predict(predictors_testing_df)
58.     print("Prediction vs Actual:")
59.     print(prediction)
60.     print(response_testing_df)
61. 
62.     mse = metrics.mean_squared_error(response_testing_df, prediction)
63.     rmse = np.sqrt(mse)
64.     print(f"rmse: {rmse}.")
65. 
66.     # Calculate r-squared.
67.     r_squared = model.score(predictors_df, response_series)
68.     print(f"r^2: {r_squared}")
69. 
70.     # Plot: Height (x) vs Volume (y)
71.     height = cherry_tree_df["Height"].to_numpy()
72.     volume = cherry_tree_df["Volume"].to_numpy()
73. 
74.     # Best-fit line (simple linear regression: Volume ~ Height)
75.     slope, intercept = np.polyfit(height, volume, 1)
76.     height_line = np.linspace(height.min(), height.max(), 200)
77.     volume_line = slope * height_line + intercept
78. 
79.     plt.figure(figsize=(9, 7))
80.     plt.scatter(height, volume, s=40)
81.     plt.plot(height_line, volume_line)
82.     plt.xlabel("Height")
83.     plt.ylabel("Volume")
84.     plt.title("Cherry Tree Volume vs Height")
85.     plt.tight_layout()
86.     plt.show()
87. 
88. 
89. def main():
90.     cherry_tree_df = pd.read_csv("CherryTree.csv")
91.     predict(cherry_tree_df)
92. 
93. 
94. if __name__ == "__main__":
95.     main()
96.
```

</details>


---

<sub>8m 35s</sub>

### `glob`

****/*.csv**

"**/*.csv"

```
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/CherryTree.csv
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/pa_diabetes.csv
```


---

<sub>8m 35s</sub>

### `glob`

**classification/***

"classification/*"

```
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/pa_diabetes.csv
```


---

<sub>8m 35s</sub>

### `glob`

**linear_regression/***

"linear_regression/*"

```
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/CherryTree.csv
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/pay_bonuses_non_modular.py
```


---

<sub>9m 2s</sub>

### `apply_patch`

<details>
<summary>Arguments</summary>

```json
"*** Begin Patch\n*** Update File: /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/main.py\n@@\n-def main():\n-    print(\"Hello from mad-2026-spring-ml-algorithms!\")\n-\n-\n-if __name__ == \"__main__\":\n-    main()\n+import argparse\n+import subprocess\n+import sys\n+from pathlib import Path\n+\n+\n+PROJECT_ROOT = Path(__file__).resolve().parent\n+SCRIPTS = {\n+    \"diabetes\": PROJECT_ROOT / \"classification\" / \"analyze_diabetes.py\",\n+    \"cherry-tree\": PROJECT_ROOT / \"linear_regression\" / \"analyze_cherry_tree.py\",\n+    \"bonuses\": PROJECT_ROOT / \"linear_regression\" / \"pay_bonuses_non_modular.py\",\n+}\n+\n+\n+def run_script(script_key: str) -> int:\n+    script_path = SCRIPTS[script_key]\n+    result = subprocess.run(\n+        [sys.executable, script_path.name],\n+        cwd=script_path.parent,\n+        check=False,\n+    )\n+    return result.returncode\n+\n+\n+def build_parser() -> argparse.ArgumentParser:\n+    parser = argparse.ArgumentParser(\n+        description=\"Run project analysis scripts.\",\n+    )\n+    parser.add_argument(\n+        \"target\",\n+        nargs=\"?\",\n+        default=\"list\",\n+        choices=[\"list\", *SCRIPTS.keys()],\n+        help=\"Script to run, or 'list' to show available options.\",\n+    )\n+    return parser\n+\n+\n+def main() -> int:\n+    parser = build_parser()\n+    args = parser.parse_args()\n+\n+    if args.target == \"list\":\n+        print(\"Available scripts:\")\n+        for key, script_path in SCRIPTS.items():\n+            print(f\"- {key}: {script_path.relative_to(PROJECT_ROOT)}\")\n+        return 0\n+\n+    return run_script(args.target)\n+\n+\n+if __name__ == \"__main__\":\n+    raise SystemExit(main())\n*** End Patch\n"
```

</details>

```
Modified 1 file(s): /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/main.py
```


---

<sub>9m 10s</sub>

### `bash`

**Verify CLI lists and runs diabetes script**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && uv run python main.py list && MPLBACKEND=Agg uv run python main.py diabetes

<details>
<summary>7 lines</summary>

```
Available scripts:
- diabetes: classification/analyze_diabetes.py
- cherry-tree: linear_regression/analyze_cherry_tree.py
- bonuses: linear_regression/pay_bonuses_non_modular.py
Scores: [0.75342466 0.76712329 0.78082192 0.73972603 0.75       0.77777778
 0.76388889 0.79166667 0.76388889 0.83333333], average: 0.7721651445966514
<shellId: 11 completed with exit code 0>
```

</details>


---

<sub>9m 11s</sub>

### Copilot

Implemented: `main.py` is now a mini CLI that runs your project scripts by name.

You can now use:

- `uv run python main.py list`
- `uv run python main.py diabetes`
- `uv run python main.py cherry-tree`
- `uv run python main.py bonuses`

It dispatches each script from its own folder, so existing relative CSV paths still work.

---

<sub>9m 36s</sub>

### User

should we consider a cli library for simplicity, or is that the oposite of simplicity

---

<sub>9m 38s</sub>

### Copilot

For this project size, a CLI library is usually the **opposite of simplicity**.

`argparse` (what you have now) is stdlib, zero dependency, and totally fine for a few commands. I’d only switch to something like Typer/Click if you expect growth like nested subcommands, richer help UX, typed options, prompts, or shell completion.

---

<sub>10m 48s</sub>

### User

can we update readme to include how to get started with uv after cloning, and also how to run existing py files, brief, prefer clear commands with brief descriptions

---

<sub>10m 55s</sub>

### Copilot

I’ll add a concise README with copy/paste `uv` setup and run commands for both the new CLI and direct script execution.



---

<sub>10m 56s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md

<details>
<summary>5 lines</summary>

```diff

diff --git a/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md b/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md
index 0000000..0000000 100644
--- a/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md
+++ b/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md
@@ -1,0 +1,0 @@
```

</details>


---

<sub>11m 1s</sub>

### `apply_patch`

<details>
<summary>Arguments</summary>

````json
"*** Begin Patch\n*** Update File: /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md\n@@\n+# mad-2026-spring-ml-algorithms\n+\n+## Getting started (uv)\n+\n+```bash\n+# 1) Clone and enter the repo\n+git clone <repo-url>\n+cd mad-2026-spring-ml-algorithms\n+\n+# 2) Create/update .venv and install dependencies from pyproject.toml/uv.lock\n+uv sync\n+\n+# 3) (Optional) activate the virtual environment for manual python usage\n+source .venv/bin/activate\n+```\n+\n+## Running the project scripts\n+\n+```bash\n+# Show available script targets\n+uv run python main.py list\n+\n+# Run diabetes classification analysis\n+uv run python main.py diabetes\n+\n+# Run cherry tree linear regression analysis\n+uv run python main.py cherry-tree\n+\n+# Run bonus calculation script\n+uv run python main.py bonuses\n+```\n+\n+## Running scripts directly (without main.py dispatcher)\n+\n+```bash\n+# Diabetes analysis script\n+uv run python classification/analyze_diabetes.py\n+\n+# Cherry tree analysis script\n+uv run python linear_regression/analyze_cherry_tree.py\n+\n+# Bonus script\n+uv run python linear_regression/pay_bonuses_non_modular.py\n+```\n*** End Patch\n"
````

</details>

```
Modified 1 file(s): /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md
```


---

<sub>11m 2s</sub>

### Copilot

Updated `README.md` with brief, command-first setup and run instructions.

It now includes:

1. `uv` onboarding after clone (`uv sync`, optional `source .venv/bin/activate`)
2. Running scripts via the new CLI (`main.py list|diabetes|cherry-tree|bonuses`)
3. Running each existing Python file directly with `uv run`

---

<sub>12m 49s</sub>

### User

$ uv run python classification/analyze_diabetes.py
Traceback (most recent call last):
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py", line 107, in \<module>
    main()
    ~~~~^^
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py", line 101, in main
    df = read_diabetes()
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py", line 43, in read_diabetes
    df = pd.read_csv("pa_diabetes.csv")
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 873, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 300, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 1645, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.venv/lib/python3.14/site-packages/pandas/io/parsers/readers.py", line 1904, in _make_engine
    self.handles = get_handle(
                   ~~~~~~~~~~^
        f,
        ^^
    ...\<6 lines>...
        storage_options=self.options.get("storage_options", None),
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/.venv/lib/python3.14/site-packages/pandas/io/common.py", line 930, in get_handle
    handle = open(
        handle,
    ...\<3 lines>...
        newline="",
    )
FileNotFoundError: [Errno 2] No such file or directory: 'pa_diabetes.csv'

---

<sub>12m 52s</sub>

### Copilot

You’re right — that command currently runs from repo root, while the script expects its CSV in the current directory. I’ll make the scripts resolve data files relative to their own file paths and update README commands accordingly.



---

<sub>12m 54s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py (lines 1-160)

<details>
<summary>108 lines</summary>

```
1. import numpy as np
2. import pandas as pd
3. import sklearn.model_selection as ms
4. import sklearn.linear_model as lm
5. import sklearn.ensemble as es
6. import sklearn.metrics as metrics
7. import imblearn.over_sampling as ios
8. import matplotlib.pyplot as plt
9. import seaborn as sns
10. 
11. 
12. def compute_confusion_matrix_numbers(actual_data_df, prediction):
13.     confusion_tuple = metrics.confusion_matrix(actual_data_df, prediction)
14.     command_line_display_as_accuracy_top_confusion_matrix = confusion_tuple.T
15.     command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=0)
16.     command_line_display_as_accuracy_top_confusion_matrix = np.flip(command_line_display_as_accuracy_top_confusion_matrix, axis=1)
17.     true_negs = 0
18.     false_poss = 0
19.     false_negs = 0
20.     true_poss = 0
21.     sensitivity = 0
22.     specificity = 0
23.     if len(confusion_tuple.ravel()) == 4:
24.         (true_negs, false_poss, false_negs, true_poss) = confusion_tuple.ravel()
25.         if ((true_poss + false_negs) > 0) and ((true_negs + false_poss) > 0):
26.             sensitivity = true_poss / (true_poss + false_negs)
27.             specificity = true_negs / (true_negs + false_poss)
28.     return (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity)
29. 
30. 
31. def create_confusion_matrix(actual_data_df, prediction):
32.     (confusion_tuple, command_line_display_as_accuracy_top_confusion_matrix, true_negs, false_poss, false_negs, true_poss, sensitivity, specificity) \
33.         = compute_confusion_matrix_numbers(actual_data_df, prediction)
34.     # if (sensitivity > 0) or (specificity > 0):
35.     #     print(f'tp: {true_poss}, fn: {false_negs}, tn: {true_negs}, fp: {false_poss}, sensitivity: {sensitivity}, specificity: {specificity}.')
36.     # print(command_line_display_as_accuracy_top_confusion_matrix)
37.     sns.heatmap(confusion_tuple, annot=True)
38.     plt.show()
39. 
40. 
41. def read_diabetes():
42.      """Read the pa_diabetes.csv file into a pandas DataFrame."""
43.      df = pd.read_csv("pa_diabetes.csv")
44.      return df
45. 
46. 
47. def show_accuracy(diabetes_response_testing, predictions):
48.     print(f"Accuracy: {(predictions == diabetes_response_testing.values).mean()}")
49.     # print((predictions == diabetes_response_testing.values).mean())
50. 
51. 
52. def perform_logistic_regression(predictors, response, balance_counter):
53.     # for random_state in range(0, 3):
54.     #     if balance_counter == 1:
55.     #         random_over_sampler = ios.RandomOverSampler(random_state=random_state)
56.     #         predictors, response \
57.     #             = random_over_sampler.fit_resample(predictors, response)
58. 
59.     (diabetes_predictors_training, diabetes_predictors_testing,
60.     diabetes_response_training, diabetes_response_testing) = \
61.         ms.train_test_split(predictors, response, test_size=0.2)
62.         # random_state=random_state) #, random_state=0)
63. 
64.     algorithm = lm.LogisticRegression(max_iter=100000)
65.     # algorithm = es.RandomForestClassifier()
66.     model = algorithm.fit(diabetes_predictors_training, diabetes_response_training)
67.     predictions = model.predict(diabetes_predictors_testing)
68.     show_accuracy(diabetes_response_testing, predictions)
69.     # show_accuracy(diabetes_response_testing, all_zeros)
70. 
71.     # create_confusion_matrix(diabetes_response_testing, predictions)
72. 
73. 
74. # Cross-validation. Sklearn makes it almost too simple. It's hard to remember
75. # that it's performing many steps that we often perform manually.
76. def perform_logistic_regression_with_cross_validation(predictors, response):    
77.     algorithm = lm.LogisticRegression(max_iter=100000)
78.     scores = \
79.         ms.cross_val_score(algorithm, predictors, response, scoring='accuracy', cv=10)
80.     print(f"Scores: {scores}, average: {np.mean(scores)}")
81. 
82. 
83. def analyze(diabetes_df):
84.     # diabetes_df = ['3']
85.     diabetes_df = diabetes_df.drop(['DNR Order', 'Med Tech'], axis='columns')
86.     predictors = diabetes_df.drop('Outcome', axis='columns')
87.     response = diabetes_df['Outcome']
88.     # print(predictors.columns)
89. 
90.     # for balance_counter in range(2):
91.     #     if balance_counter == 0:
92.     #         print("Unbalanced:")
93.     #     else:
94.     #         print("Balanced:")
95.     #     perform_logistic_regression(predictors, response, balance_counter)
96. 
97.     perform_logistic_regression_with_cross_validation(predictors, response)
98. 
99. 
100. def main():
101.     df = read_diabetes()
102.     # print(df.head())
103.     analyze(df)
104. 
105. 
106. if __name__ == "__main__":
107.     main()
108.
```

</details>


---

<sub>12m 54s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py (lines 1-160)

<details>
<summary>96 lines</summary>

```
1. import numpy as np
2. import pandas as pd
3. import sklearn.linear_model as lm
4. import sklearn.model_selection as ms
5. import sklearn.metrics as metrics
6. 
7. # Add plotting imports
8. import matplotlib.pyplot as plt
9. from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
10. import seaborn as sns
11. 
12. 
13. def feature_select(tree_df):
14.     one_hot_encoded_columns = pd.get_dummies(tree_df[['Season']])
15.     # Won't work with Season as strings!
16.     # predictors_df = tree_df[['Height', 'Diam', 'Season']]
17.     predictors_df = tree_df[['Height', 'Diam']]
18.     # predictors_df = pd.concat([predictors_df, one_hot_encoded_columns], axis='columns')
19. 
20.     # Most of this block is for linearity.
21.     predictors_and_response_df = pd.concat([predictors_df, tree_df['Volume']], axis='columns')
22.     print(predictors_and_response_df)
23.     correlation_matrix = predictors_and_response_df.corr()
24.     volume_correlation_matrix = correlation_matrix[['Volume']].sort_values(by='Volume', ascending=False)
25.     print(volume_correlation_matrix)
26.     # print(correlation_matrix)
27.     # sns.heatmap(volume_correlation_matrix, annot=True)
28.     # plt.show()
29. 
30.     # For independence.
31.     print(correlation_matrix)
32.     mask = abs(correlation_matrix < 0.3)
33.     # sns.heatmap(correlation_matrix, annot=True, mask=mask)
34.     # plt.show()
35. 
36.     response_series = tree_df['Volume'].to_numpy()
37. 
38.     return (predictors_df, response_series)
39. 
40. 
41. def predict(cherry_tree_df):
42.     print(cherry_tree_df)
43.     (predictors_df, response_series) = feature_select(cherry_tree_df)
44. 
45.     # print("Before print")
46.     # print(predictors_df)
47.     # print(response_series)
48.     # print("After print")
49. 
50.     (predictors_training_df, predictors_testing_df, \
51.         response_training_df, response_testing_df) \
52.             = ms.train_test_split(predictors_df, response_series, test_size=0.2)
53. 
54.     # Train and predict.
55.     algorithm = lm.LinearRegression()
56.     model = algorithm.fit(predictors_training_df, response_training_df)
57.     prediction = model.predict(predictors_testing_df)
58.     print("Prediction vs Actual:")
59.     print(prediction)
60.     print(response_testing_df)
61. 
62.     mse = metrics.mean_squared_error(response_testing_df, prediction)
63.     rmse = np.sqrt(mse)
64.     print(f"rmse: {rmse}.")
65. 
66.     # Calculate r-squared.
67.     r_squared = model.score(predictors_df, response_series)
68.     print(f"r^2: {r_squared}")
69. 
70.     # Plot: Height (x) vs Volume (y)
71.     height = cherry_tree_df["Height"].to_numpy()
72.     volume = cherry_tree_df["Volume"].to_numpy()
73. 
74.     # Best-fit line (simple linear regression: Volume ~ Height)
75.     slope, intercept = np.polyfit(height, volume, 1)
76.     height_line = np.linspace(height.min(), height.max(), 200)
77.     volume_line = slope * height_line + intercept
78. 
79.     plt.figure(figsize=(9, 7))
80.     plt.scatter(height, volume, s=40)
81.     plt.plot(height_line, volume_line)
82.     plt.xlabel("Height")
83.     plt.ylabel("Volume")
84.     plt.title("Cherry Tree Volume vs Height")
85.     plt.tight_layout()
86.     plt.show()
87. 
88. 
89. def main():
90.     cherry_tree_df = pd.read_csv("CherryTree.csv")
91.     predict(cherry_tree_df)
92. 
93. 
94. if __name__ == "__main__":
95.     main()
96.
```

</details>


---

<sub>12m 54s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/pay_bonuses_non_modular.py.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/pay_bonuses_non_modular.py (lines 1-220)

<details>
<summary>44 lines</summary>

```
1. def calculate_bonus(sales):
2.     # sales = 20000
3.     bonus = sales * .1
4.     # Sales greater than 10000:
5.     if sales > 10000:
6.         bonus += sales * .05
7.     if sales > 30000:
8.         bonus += sales * .1
9. 
10.     return bonus
11. 
12. 
13. def main():
14.     name = 'Pablo'
15.     sales = 20000
16.     pablo_bonus = calculate_bonus(sales)
17.     # pablo_bonus = sales * .1
18.     # # Sales greater than 10000:
19.     # pablo_bonus += sales * .05
20. 
21.     name = 'Veronica'
22.     sales = 9500
23.     # veronica_bonus = sales * .1
24.     # Sales not greater than 10000:
25.     veronica_bonus = calculate_bonus(sales)
26. 
27.     name = 'Cindy'
28.     sales = 45000
29.     # cindy_bonus = sales * .1
30.     # # Sales greater than 10000:
31.     # cindy_bonus += sales * .05
32.     # # Sales greater than 30000:
33.     # cindy_bonus += sales * .1
34.     cindy_bonus = calculate_bonus(sales)
35. 
36.     # print(f'{name}\'s bonus is ${cindy_bonus:.2f}')
37.     print(f'Pablo\'s bonus is ${pablo_bonus:.2f}')
38.     print(f'Veronica\'s bonus is ${veronica_bonus:.2f}')
39.     print(f'{name}\'s bonus is ${cindy_bonus:.2f}')
40. 
41. 
42. if __name__ == "__main__":
43.     main()
44.
```

</details>


---

<sub>12m 54s</sub>

### `view`

**view the file at /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md.**

/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/README.md

<details>
<summary>44 lines</summary>

````
1. # mad-2026-spring-ml-algorithms
2. 
3. ## Getting started (uv)
4. 
5. ```bash
6. # 1) Clone and enter the repo
7. git clone <repo-url>
8. cd mad-2026-spring-ml-algorithms
9. 
10. # 2) Create/update .venv and install dependencies from pyproject.toml/uv.lock
11. uv sync
12. 
13. # 3) (Optional) activate the virtual environment for manual python usage
14. source .venv/bin/activate
15. ```
16. 
17. ## Running the project scripts
18. 
19. ```bash
20. # Show available script targets
21. uv run python main.py list
22. 
23. # Run diabetes classification analysis
24. uv run python main.py diabetes
25. 
26. # Run cherry tree linear regression analysis
27. uv run python main.py cherry-tree
28. 
29. # Run bonus calculation script
30. uv run python main.py bonuses
31. ```
32. 
33. ## Running scripts directly (without main.py dispatcher)
34. 
35. ```bash
36. # Diabetes analysis script
37. uv run python classification/analyze_diabetes.py
38. 
39. # Cherry tree analysis script
40. uv run python linear_regression/analyze_cherry_tree.py
41. 
42. # Bonus script
43. uv run python linear_regression/pay_bonuses_non_modular.py
44. ```
````

</details>


---

<sub>12m 58s</sub>

### `apply_patch`

<details>
<summary>Arguments</summary>

```json
"*** Begin Patch\n*** Update File: /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py\n@@\n import numpy as np\n import pandas as pd\n import sklearn.model_selection as ms\n import sklearn.linear_model as lm\n import sklearn.ensemble as es\n import sklearn.metrics as metrics\n import imblearn.over_sampling as ios\n import matplotlib.pyplot as plt\n import seaborn as sns\n+from pathlib import Path\n@@\n def read_diabetes():\n      \"\"\"Read the pa_diabetes.csv file into a pandas DataFrame.\"\"\"\n-     df = pd.read_csv(\"pa_diabetes.csv\")\n+     csv_path = Path(__file__).resolve().parent / \"pa_diabetes.csv\"\n+     df = pd.read_csv(csv_path)\n      return df\n*** End Patch\n"
```

</details>

```
Modified 1 file(s): /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/classification/analyze_diabetes.py
```


---

<sub>13m 1s</sub>

### `apply_patch`

<details>
<summary>Arguments</summary>

```json
"*** Begin Patch\n*** Update File: /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py\n@@\n import numpy as np\n import pandas as pd\n import sklearn.linear_model as lm\n import sklearn.model_selection as ms\n import sklearn.metrics as metrics\n+from pathlib import Path\n@@\n def main():\n-    cherry_tree_df = pd.read_csv(\"CherryTree.csv\")\n+    csv_path = Path(__file__).resolve().parent / \"CherryTree.csv\"\n+    cherry_tree_df = pd.read_csv(csv_path)\n     predict(cherry_tree_df)\n*** End Patch\n"
```

</details>

```
Modified 1 file(s): /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py
```


---

<sub>13m 16s</sub>

### `bash`

**Verify direct script runs from repo root**

$ cd /Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms && MPLBACKEND=Agg uv run python classification/analyze_diabetes.py && MPLBACKEND=Agg uv run python linear_regression/analyze_cherry_tree.py

<details>
<summary>83 lines</summary>

```
Scores: [0.75342466 0.76712329 0.78082192 0.73972603 0.75       0.77777778
 0.76388889 0.79166667 0.76388889 0.83333333], average: 0.7721651445966514
/Users/tsteele/Documents/GitHub/2026-fall/mad-2026-spring-ml-algorithms/linear_regression/analyze_cherry_tree.py:87: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
    Diam  Height  Volume  Season
0    8.3      70    10.3  Spring
1    8.6      65    10.3  Spring
2    8.8      63    10.2    Fall
3   10.5      72    16.4  Summer
4   10.7      81    18.8    Fall
5   10.8      83    19.7  Winter
6   11.0      66    15.6  Summer
7   11.0      75    18.2  Summer
8   11.1      80    22.6  Summer
9   11.2      75    19.9  Spring
10  11.3      79    24.2  Winter
11  11.4      76    21.0    Fall
12  11.4      76    21.4    Fall
13  11.7      69    21.3    Fall
14  12.0      75    19.1    Fall
15  12.9      74    22.2    Fall
16  12.9      85    33.8  Spring
17  13.3      86    27.4  Winter
18  13.7      71    25.7  Summer
19  13.8      64    24.9  Spring
20  14.0      78    34.5  Winter
21  14.2      80    31.7  Winter
22  14.5      74    36.3    Fall
23  16.0      72    38.3    Fall
24  16.3      77    42.6  Summer
25  17.3      81    55.4  Spring
26  17.5      82    55.7  Summer
27  17.9      80    58.3  Spring
28  18.0      80    51.5    Fall
29  18.0      80    51.0  Winter
30  20.6      87    77.0  Summer
    Height  Diam  Volume
0       70   8.3    10.3
1       65   8.6    10.3
2       63   8.8    10.2
3       72  10.5    16.4
4       81  10.7    18.8
5       83  10.8    19.7
6       66  11.0    15.6
7       75  11.0    18.2
8       80  11.1    22.6
9       75  11.2    19.9
10      79  11.3    24.2
11      76  11.4    21.0
12      76  11.4    21.4
13      69  11.7    21.3
14      75  12.0    19.1
15      74  12.9    22.2
16      85  12.9    33.8
17      86  13.3    27.4
18      71  13.7    25.7
19      64  13.8    24.9
20      78  14.0    34.5
21      80  14.2    31.7
22      74  14.5    36.3
23      72  16.0    38.3
24      77  16.3    42.6
25      81  17.3    55.4
26      82  17.5    55.7
27      80  17.9    58.3
28      80  18.0    51.5
29      80  18.0    51.0
30      87  20.6    77.0
          Volume
Volume  1.000000
Diam    0.967119
Height  0.598250
         Height      Diam    Volume
Height  1.00000  0.519280  0.598250
Diam    0.51928  1.000000  0.967119
Volume  0.59825  0.967119  1.000000
Prediction vs Actual:
[21.05170621 22.16273989 42.48698524 19.42960387 20.85589379 45.54301436
 36.42074817]
[19.7 24.2 38.3 18.2 21.3 42.6 31.7]
rmse: 2.8325742105266967.
r^2: 0.9471202099330456
<shellId: 12 completed with exit code 0>
```

</details>


---

<sub>13m 18s</sub>

### Copilot

Fixed — both data-loading scripts now resolve CSVs relative to their own file location, so running from repo root works.

`uv run python classification/analyze_diabetes.py` now works as written in the README, and I also applied the same fix to `linear_regression/analyze_cherry_tree.py` for consistency.

---

<sub>Generated by [GitHub Copilot CLI](https://github.com/features/copilot/cli)</sub>
