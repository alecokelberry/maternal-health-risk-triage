# Maternal health risk

1,014 rows from clinics in rural Bangladesh. Six vitals in. The label comes back low, mid, or high.

I trained a plain random forest, then tried 24 other setups. Cross-validation liked the plain one, 0.817 against 0.808, so `models/` is the plain one. Holdout weighted F1 is 0.8679. That number was scored after the pick.

| | precision | recall | f1 | rows |
| --- | ---: | ---: | ---: | ---: |
| low | 0.89 | 0.82 | 0.85 | 81 |
| mid | 0.77 | 0.87 | 0.82 | 67 |
| high | 0.96 | 0.95 | 0.95 | 55 |

Mid is the label it mixes up. Most splits are blood sugar, then systolic pressure, then age.

This ranks who should get a follow-up call first. It is not a diagnosis.

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m triage predict --age 25 --systolic 130 --diastolic 80 --bs 15 --temp 98 --heart-rate 86
```

```text
high risk
  high risk 0.84  mid risk 0.15  low risk 0.01
```

That's the first row of the CSV.

`python3 -m triage train` refits both and overwrites `models/` and `results/`. Tests are `python3 -m unittest discover -s tests -v`.

The rows were collected with an IoT monitor and published by Ahmed, Kashem, Rahman, and Khatun. UCI Maternal Health Risk, [10.24432/C5DP5D](https://doi.org/10.24432/C5DP5D), CC BY 4.0. Age in years, blood pressure in mmHg, blood sugar in mmol/L, temperature in Fahrenheit, heart rate in bpm.

Code is [MIT](LICENSE). The CSV stays CC BY 4.0.
