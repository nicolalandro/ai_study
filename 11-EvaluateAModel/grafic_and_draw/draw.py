import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

pd.set_option('max.columns', None)

df = pd.read_csv('Pokemon.csv', low_memory=False)
# print(df.info())
# print(df.head())
print('Legendary:', str(len(df[df['Legendary'] == True]) / len(df) * 100) + '%')

plt.title('Count Plot')
plt.xticks(rotation=45)
sns_plot_type1 = sns.countplot(df['Type 1'])
# sns_plot.figure.savefig("figure.png")
plt.close()

plt.title('Count Plot')
plt.xticks(rotation=45)
sns_plot_type2 = sns.countplot(df['Type 2'])
plt.close()

sns_plot_tot = sns.distplot(df['Total'])
plt.close()

sns_plot_x = sns.pairplot(df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
# sns_plot.savefig("figure.png")
plt.close()

corr = df.corr()
sns_plot = sns.heatmap(corr, cmap='coolwarm', annot=True)
sns_plot.figure.savefig("figure.png")
plt.close()

# sns.heatmap(corr, annot=True)
# plt.savefig("heatmap.png")
