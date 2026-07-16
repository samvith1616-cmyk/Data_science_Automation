import re

fake_llm_response = """
MODEL: RandomForestClassifier
REASONING: Handles mixed feature types well and is robust to outliers.

Here is the training script:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data.csv")
print(df.head())
```

Let me know if you need anything else.
"""

match = re.search(r"```python\s*\n(.*?)```", fake_llm_response, re.DOTALL)

if match:
    code = match.group(1).strip()
    print("Extracted code:")
    print(code)
else:
    print("No code block found!")