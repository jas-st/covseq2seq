import pandas as pd

df_path = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned.xlsx"
df1_path = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned_1.xlsx"
df2_path = "C:\\Users\\zhasmi00\\Downloads\\test\\august\\evols2022lineages_cleaned_2.xlsx"
df = pd.read_excel(df_path)


index = int(len(df)/2)

while "input" not in df.iloc[index]["seqName"]:
    index += 1


df.iloc[:index].to_excel(df1_path)
df.iloc[index:].to_excel(df2_path)