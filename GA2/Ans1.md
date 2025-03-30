# Step Analysis: Weekly Walking Data

This document analyzes the number of steps walked each day over a week, comparing trends over time and with friends.

## Methodology

Data was collected using a *fitness tracker* and exported as a CSV file. The analysis was performed using Python with `pandas` for data manipulation. **Important**: Only days with complete data were included.

### Data Collection
- Steps recorded from **Monday** to **Sunday**
- Synced via Bluetooth to the companion app
- Exported using the `export_steps()` function

## Weekly Step Count

Here’s the raw data for the week:

| Day      | Steps  |
|----------|--------|
| Monday   | 8,542  |
| Tuesday  | 7,231  |
| Wednesday| 10,876 |
| Thursday | 6,543  |
| Friday   | 9,210  |
| Saturday | 12,345 |
| Sunday   | 5,678  |

> **Note**: *Sunday’s low count was due to a rest day.*

## Comparison with Friends

Compared my average steps with two friends:

1. **Friend A**: 9,500 steps/day
2. **Friend B**: 7,800 steps/day

My average: `(8542 + 7231 + 10876 + 6543 + 9210 + 12345 + 5678) / 7 ≈ 8,632 steps/day`

## Visualization

![Step Trends](https://example.com/step_chart.png)  
*Chart generated using [Matplotlib](https://matplotlib.org/).*

## Code Example

```python
import pandas as pd

data = {
    "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "Steps": [8542, 7231, 10876, 6543, 9210, 12345, 5678]
}

df = pd.DataFrame(data)
print(f"Average steps: {df['Steps'].mean():.0f}")
