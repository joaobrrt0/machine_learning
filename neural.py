#%%
import pandas as pd
data = pd.read_csv("../data/clean_weather.csv", index_col =0)
data = data.ffill()

data.plot.scatter("tmax", "tmax_tomorrow")
# %%
data.corr()
# %%
import matplotlib.pyplot as plt

data.plot.scatter("tmax", "tmax_tomorrow")

prediction = lambda x, w1 =.82, b=11.99: x * w1 + b

plt.plot([30, 120], [prediction(30), prediction(120)], 'green')

# %%
import numpy as np

def mse(actual, predicted):
    return np.mean((actual - predicted)** 2)

print(mse(data["tmax_tomorrow"], prediction(data["tmax"])))
print(mse(data["tmax_tomorrow"], prediction(data["tmax"], .82, 13)))

# %%
tmax_bins = pd.cut(data["tmax"], 25)
ratios = (data["tmax_tomorrow"] - 11.99) / data["tmax"] 
binned_ratio = ratios.groupby(tmax_bins).mean()

binned_tmax = data["tmax"].groupby(tmax_bins).mean()

plt.scatter(binned_tmax, binned_ratio)
# %%
binned_ratio

# %%
#A nonlinear regression on top of the linear transformation
#Multiple layers, which can capture interactions between features
#Multiple hidden units, which each have slightly different linear and nonlinear transformations 
# %%
temps = np.arange(-50, 50)

plt.plot(temps, np.maximum(0, prediction(temps)))

# %%
#$y = w_(2) w_relu(w_(1)x + b_(1)) + b_(2)$
# %%
temps = np.arange(-50, 50)

layer1 = np.maximum(0, prediction(temps))
layer2 = prediction(layer1, .5, 10)

plt.plot(temps, layer2)

plt.ylim((0,40))
# %%
layer1 = np.maximum(0, prediction(temps))

layer1_2 = np.maximum(0, prediction(temps, .1, 10))

layer1_3 = np.maximum(0, prediction(temps, 2, -50))

layer2 = layer1 * .1 + layer1_2 * .3 + layer1_3 * .4 + 20

plt.plot(temps, layer2)
# %%
plt.plot(temps, layer1 +layer1_2 + layer1_3)

# %%
import tsensor
input = np.array([[80], [90], [100], [-20], [-10]])

l1_weights = np.array([[.81, .1]])

l1_bias = np.array([[11.99, 10]])

with tsensor.explain():
    l1_output = input @ l1_weights + l1_bias

# %%
#ORDEM CORRETA E LINHA COLUNA, 5 LINHAS E 2 COLUNAS
l1_output
# %%
l1_activated = np.maximum(l1_output, 0)
l1_activated
# %%
l2_weights = np.array([
    [.5],
    [.2]
])
l2_bias = np.array([[5]])

with tsensor.explain():
    output = l1_activated @l2_weights + l2_bias
output
# %%
t_max = np.array([[80], [90], [100], [-20], [-10]])
tmax_tomorrow = np.array([[83],[89],[95],[22],[-9]])

# %%
t_max
tmax_tomorrow
# %%
def mse(actual, predicted):
    return (actual - predicted) ** 2
# %%
mse(tmax_tomorrow, output)
# %%
def mse_grad(actual, predicted):
    return predicted - actual
# %%
mse_grad(tmax_tomorrow, output)
# %%
output_gradient = mse_grad(tmax_tomorrow, output)
# %%
from tsensor import explain as exp
with exp():
    l2_w_gradient = l1_activated.T @ output_gradient

l2_w_gradient
# %%
from sympy import diff, symbols

x, w = symbols('X, W')
sympy_output = x * w
diff(sympy_output, w)
# %%
with exp():
    l2_b_gradient = np.mean(output_gradient, axis=0)

l2_b_gradient
# %%
l2_weights
# %%
lr = 1e-5

with exp():
    l2_bias = l2_bias - l2_b_gradient * lr
    l2_weights = l2_weights - l2_w_gradient * lr

l2_weights
# %%
with exp():
    l1_activated_gradient = output_gradient @ l2_weights.T

l1_activated_gradient
# %%
temps = np.arange(-50, 50)
plt.plot(temps, np.maximum(0, temps))
# %%
activation = np.maximum(0, temps)

plt.plot(temps[1:], activation[1:] - np.roll(activation, 1)[1:])
# %%
with exp():
    l1_output_gradient = l1_activated_gradient * np.heaviside(l1_output, 0)

l1_output_gradient
# %%
l1_w_gradient = input.T @ l1_output_gradient
l1_b_gradient = np.mean(l1_output_gradient, axis= 0)

l1_weights -= l1_w_gradient * lr
l1_bias -= l1_b_gradient * lr
# %%
l1_weights
# %%
l1_bias
# %%
import numpy as np
from sklearn.preprocessing import StandardScaler
PREDICTORS = ['tmax', 'tmin', 'rain']
TARGET = 'tmax_tomorrow'

scaler = StandardScaler()
data[PREDICTORS] = scaler.fit_transform(data[PREDICTORS])       

split_data = np.split(data,[int(.7 * len(data)), int(.85 * len(data))]) 
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) =  [[d[PREDICTORS].to_numpy(), d[TARGET].to_numpy()]for d in split_data]  
# %%
def init_layers(inputs):
    layers = []
    for i in range(1, len(inputs)):
        layers.append([
            np.random.rand(inputs[i-1], inputs[i]) / 5 - .1,
            np.ones((1, inputs[i]))
        ])
    return layers

layer_conf = [3, 10 , 10, 1]

layers = init_layers(layer_conf)
# %%
layers
# %%
def forward(batch , layers):
    hiddens = [batch.copy()]
    for i in range (len(layers)):
        batch = np.matmul(batch, layers[i],[0]) + layers[i][1]
        if i < len(layers) - 1:
            batch = np.maximum(batch, 0)
        hiddens.append(batch.copy())
    return batch, hiddens

# %%
def mse (atcual, predicted):
    return (actual- predicted) **2

def mse_grad(actual, predicted):
    return predicted -actual

# %%
def backward(layers, hidden, grad, lr):
    for i in range(len(layers)- 1, -1, -1):
        if i  != len(layers) - 1:
            grad = np.multiply(grad, np.heaviside(hidden[i + 1], 0))
        
        w_grad = hidden[i].T @ grad
        b_grad = np.mean(grad, axis=0)
        layers[i][0] = w_grad * lr
        layers[i][1] = b_grad * lr

        grad = grad @ layers[i].T
    return layers
# %%
lr = 1e-6
epochs = 10
batch_size = 8  

layers = init_layers(layer_conf)

for epoch in range(epochs):
    epoch_loss = []

    for i in range(0, train_x.shape[0], batch_size):
        x_batch = train_x[i:(i+batch_size)]
        y_batch = train_y[i:(i+batch_size)]

        pred, hidden = forward(x_batch, layers)
        
        loss = mse_grad(y_batch, pred)
        epoch_loss += np.mean(loss ** 2)

        layers = backward(layers, hidden, loss, lr)
    
    valid_preds, _ = forward(valid_x, layers)

    print(f"Epoch {epoch} Train MSE: {epoch_loss / (train_x.shape[0]/batch_size)} Valid MSE: {np.mean(mse(valid_preds, valid_y))}")

# %%
