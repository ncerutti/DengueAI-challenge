import pandas as pd

def table_md(csv_path, out_file):
    data=pd.read_csv(csv_path)

    rows=[]
    header_row='''
| col name | type | value count | missing values |
| -------- | ---- | ----------- | -------------- | 
'''
    rows.append(header_row)

    for i in data.columns:

        row=f'| {i} | {data[i].dtype} | {data[i].nunique()} | {data[i].isnull().sum()} |\n'
        rows.append(row)

    with open(out_file, 'w+') as file:
        file.writelines(rows)


