# Forecasting consultancy demo

Synthetic business cases for bakery production and beverage replenishment. The code compares forecasting methods, then simulates inventory, waste and costs. A separate bakery experiment handles censored demand.

Use Python 3.12 and install `requirements.txt`, or `requirements.lock.txt` for the recorded environment.

```sh
python run_demo.py --case all
```

Full runs fit several models and write local results. To rebuild the executed notebooks, run `python build_notebooks.py --case all`, followed by `python validate_project.py`.

All businesses, observations and financial effects are simulated. The notebooks in Git have their outputs cleared. MIT licensed.
