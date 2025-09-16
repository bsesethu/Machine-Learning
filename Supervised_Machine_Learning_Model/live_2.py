import pandas as pd

ss = pd.DataFrame({'Thing':['Sese. is A helpfull Guy!', 'So too @mthUNzi?']})

ss['Thing'] = ss['Thing'].str.replace('.','')
print(ss)

print("%s: Sese = %.2f%%; Bang = %.2f%%" % ('My Name IS', 100/3.3, 6.666666)) #NOTE New way to format a print statement