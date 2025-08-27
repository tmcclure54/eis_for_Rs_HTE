# eis\_for\_Rs\_HTE

Nyquist EIS plotting \& robust Rs extraction (CNLS).

\# Nyquist EIS (Robust Rs Extraction)



This repo provides `nyquist\_rs\_pro.py`:



\- GUI file picker

\- Nyquist plot (Z' vs -Z'')

\- Robust Rs via CNLS to Randles-like circuit (CPE, Warburg, optional L)

\- Guardrails + bootstrap 95% CI for Rs



\## Install

```bash

python -m venv .venv

.venv\\Scripts\\activate  # Windows

pip install -r requirements.txt



