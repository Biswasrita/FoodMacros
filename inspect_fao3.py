import pandas as pd
xls = pd.ExcelFile('Densitydatabasev2__1_.xls')
for s in xls.sheet_names:
    print('===', s, '===')
    df = pd.read_excel(xls, s)
    print(df.shape)
    print(list(df.columns))
    print(df.head(15))
    print()
