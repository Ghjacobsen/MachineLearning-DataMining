import pandas as pd

calories = pd.read_csv('../Dataset/calories.csv')
exercise = pd.read_csv('../Dataset/exercise.csv')
df = pd.merge(exercise, calories, on = 'User_ID')
df = df.reset_index()
df['Intercept'] = 1
df.head()

print(df.columns)
print(df.describe())