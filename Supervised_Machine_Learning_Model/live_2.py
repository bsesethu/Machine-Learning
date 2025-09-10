import pandas as pd

ss = pd.DataFrame({'Thing':['Sese. is A helpfull Guy!', 'So too @mthUNzi?']})

ss['Thing'] = ss['Thing'].str.replace('.','')
print(ss)