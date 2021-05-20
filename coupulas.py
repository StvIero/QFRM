####Copulas Code####

#Import packages.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from pycopula.copula import ArchimedeanCopula
from pycopula.copula import StudentCopula
from pycopula.copula import GaussianCopula
from mpl_toolkits.mplot3d import Axes3D
from pycopula.visualization import pdf_2d, cdf_2d
from matplotlib import cm

#Import data, set date column as index.
df = pd.read_csv('/Users/connorstevens/Documents/GitHub/qfrm_code/Data3/data_main.csv', index_col= 1)

#Extract list of return names to plot.
assets = df.columns[df.columns.get_loc('ASML.AS_ret'): -1]
print(assets)

#Correlation matrix.
df_corr = df[assets].dropna().corr()

credit_bayer = np.array(df[['CSGN.SW_ret', 'BAYN.DE_ret']].dropna())
bayer_tyson = np.array(df[['BAYN.DE_ret', 'TSN_ret']].dropna())
amd_asml = np.array(df[['AMD_ret', 'ASML.AS_ret']].dropna())
sony_square = np.array(df[['SONY_ret', 'SQNXF_ret']].dropna())
sony_nintendo = np.array(df[['SONY_ret', 'NTDOY_ret']].dropna())

X = sony_nintendo * -1

#Gumbel Copula.
Gumbel = ArchimedeanCopula(family="gumbel", dim=2)
Gumbel.fit(X, method="cmle")

Clayton = ArchimedeanCopula(family="clayton", dim=2)
Clayton.fit(X, method="cmle")

Frank = ArchimedeanCopula(family="frank", dim=2)
Frank.fit(X, method="cmle")

Gaussian = GaussianCopula(dim=2)
Gaussian.fit(X, method="cmle")

# Visualization of CDF and PDF
u, v, c = cdf_2d(Gumbel)
u, v, c = pdf_2d(Gumbel)

# Plotting
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(221, projection='3d', title="Gumbel copula")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
ax.set_xlabel('Sony')
ax.set_ylabel('Nintendo')
#plt.show()

u, v, c = cdf_2d(Clayton)
u, v, c = pdf_2d(Clayton)

# Plotting
ax = fig.add_subplot(222, projection='3d', title="Clayton copula")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
ax.set_xlabel('Sony')
ax.set_ylabel('Nintendo')

u, v, c = cdf_2d(Frank)
u, v, c = pdf_2d(Frank)

# Plotting
ax = fig.add_subplot(223, projection='3d', title="Frank copula")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
ax.set_xlabel('Sony')
ax.set_ylabel('Nintendo')

u, v, c = cdf_2d(Gaussian)
u, v, c = pdf_2d(Gaussian)

# Plotting
ax = fig.add_subplot(224, projection='3d', title="Gaussian copula")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
ax.set_xlabel('Sony')
ax.set_ylabel('Nintendo')
plt.show()


############

#Clayton Copula.
Clayton = ArchimedeanCopula(family="clayton", dim=2)
Clayton.fit(test, method="cmle")

# Visualization of CDF and PDF
u, v, C = cdf_2d(Clayton)
# u, v, c = pdf_2d(Clayton)

# Plotting
ax = fig.add_subplot(121, projection='3d', title="Clayton copula CDF")
X, Y = np.meshgrid(u, v)

ax.set_zlim(0, 5)
ax.plot_surface(X, Y, c, cmap=cm.Blues)
ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)

plt.tight_layout()
plt.show()

#Define stocks to be on the x- and y-axis.
x_stocks = ['CSGN.SW_ret', 'TSN_ret', 'AMD_ret', 'SONY_ret']
y_stocks = ['BAYN.DE_ret', 'BAYN.DE_ret', 'ASML.AS_ret', 'SQNXF_ret']

#Define vector of degrees of freedom for each copula.
DoF = [5.467, 8.812, 8.763, 15.024]

##Loop through lists of stocks and plot copulas.
for count, (stock_x, stock_y) in enumerate(zip(x_stocks, y_stocks)):

    # Clayton Copula.
    clayton = ArchimedeanCopula(family="clayton", dim=2)
    clayton.fit(np.array(df[[stock_x, stock_y]].dropna()) * -1, method="cmle")

    #Gumbel Copula
    gumbel = ArchimedeanCopula(family="gumbel", dim=2)
    gumbel.fit(np.array(df[[stock_x, stock_y]].dropna()) * -1, method="cmle")

    #Frank copula
    frank = ArchimedeanCopula(family="frank", dim=2)
    frank.fit(np.array(df[[stock_x, stock_y]].dropna()) * -1, method="cmle")

    #Calculate covariance matrix for two assets for use in Gaussian and Student-t copulas.
    cov_df = np.array((df[[stock_x, stock_y]].dropna() *-1).cov())
    # #Student-t copula
    # student = StudentCopula(dim=2, df=DoF[count])
    # student.fit(np.array(df[[stock_x, stock_y]].dropna())*-1, method="mle")

    #Gaussian copula
    gaussian = GaussianCopula(dim=2)
    gaussian.fit(np.array(df[[stock_x, stock_y]].dropna()) * -1, method="cmle")

    #Clayton
    u1, v1, c1 = cdf_2d(clayton)

    #Gumbel
    u2, v2, c2 = cdf_2d(gumbel)

    #Frank
    u3, v3, c3 = cdf_2d(frank)

    #Gaussian
    u4, v4, c4 = cdf_2d(gaussian)

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    X1, Y1 = np.meshgrid(u1, v1)

    ax.set_zlim(0, 5)
    ax.plot_surface(X1, Y1, c1, cmap=cm.Blues)
    ax.plot_wireframe(X1, Y1, c1, color='black', alpha=0.3)

    ax = fig.add_subplot(222, projection='3d')
    X2, Y2 = np.meshgrid(u2, v2)

    ax.set_zlim(0, 5)
    ax.plot_surface(X2, Y2, c2, cmap=cm.Blues)
    ax.plot_wireframe(X2, Y2, c2, color='black', alpha=0.3)

    ax = fig.add_subplot(223, projection='3d')
    X, Y = np.meshgrid(u, v)

    ax.set_zlim(0, 5)
    ax.plot_surface(X, Y, c, cmap=cm.Blues)
    ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
    plt.show()

    ax = fig.add_subplot(224, projection='3d')
    X, Y = np.meshgrid(u, v)

    ax.set_zlim(0, 5)
    ax.plot_surface(X, Y, c, cmap=cm.Blues)
    ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
    plt.show()
    plt.show()

##Loop through lists of stocks and plot copulas.
for count, (stock_x, stock_y) in enumerate(zip(x_stocks, y_stocks)):
    def CopulaPlot(X, x1_name, x2_name, copula):

        # Clayton Copula.
        if copula == 'clayton':
            clayton = ArchimedeanCopula(family="clayton", dim=2)
            clayton.fit(X, method="cmle")
            cop =1

        # Gumbel Copula
        if copula == 'gumbel':
            gumbel = ArchimedeanCopula(family="gumbel", dim=2)
            gumbel.fit(X, method="cmle")
            cop =2

        # Frank copula
        if copula == 'frank':
            frank = ArchimedeanCopula(family="frank", dim=2)
            frank.fit(X, method="cmle")
            cop=3

        # Calculate covariance matrix for two assets for use in Gaussian and Student-t copulas.
        # cov_df = np.array((df[[stock_x, stock_y]].dropna() *-1).cov())
        # #Student-t copula
        # student = StudentCopula(dim=2, df=DoF[count])
        # student.fit(np.array(df[[stock_x, stock_y]].dropna())*-1, method="mle")

        # Gaussian copula
        if copula == 'gaussian':
            gaussian = GaussianCopula(dim=2)
            gaussian.fit(X, method="cmle")

        # Visualization of CDF and PDF
        u, v, c = cdf_2d()

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d', title=str(copula) + " copula")
        X, Y = np.meshgrid(u, v)

        ax.set_zlim(0, 5)
        ax.plot_surface(X, Y, c, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.plot_wireframe(X, Y, c, color='black', alpha=0.3)
        ax.set_xlabel('x1_name')
        ax.set_ylabel('x2_name')
        plt.show()