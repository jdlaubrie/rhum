import tensorflow as tf
import keras
from tensorflow.keras import backend
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import copy
import os
import json
import pandas as pd
#from sys import exit

#=======================================================================================#
epochs = 3000
batch_size = 8

### Choose regularization type & penalty amount
# Option: 'L1', 'L2'
reg = 'L1'
pen = 0.001  # Use 0 for no regularization

#=======================================================================================#
path2data = "./" + 'data/'
# setting working path
# Make path to save results to
def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)

filename = 'ferruzzi13_cca' # Change to keep track of different data e.g. Brain, Skin, Muscle, etc.
path2scratch = "../" + 'scratch/' + filename
makeDIR(path2scratch)

model_summary = path2scratch + '/cann_summary.txt'

#=======================================================================================#
# L1 and L2 regularization with penalty weight
def regularize(reg, pen):
    if reg == 'L2':
        return keras.regularizers.L2(pen)
    elif reg == 'L1':
        return keras.regularizers.L1(pen)
    else:
        raise ValueError("Regularization type must be 'L1' or 'L2'")

#=======================================================================================#
# Step 3: Prepare Data
mmhg2kpa = 0.133322
# Generate synthetic data
# read data from file. lagrangian stress
exp_data = pd.read_csv(path2data + '/' + filename + '.csv', dtype=float, decimal=',')

# stock experimental data into numpy arrays
outer_diameter = 1.0e-3*exp_data['diameter'].values  #input in um
pressure_d = mmhg2kpa*exp_data['press_d'].values     #input in mmHg
pressure_f = mmhg2kpa*exp_data['press_f'].values     #input in mmHg
force = 1.0e-3*exp_data['force'].values              #input in mN
pressure = 0.5*(pressure_f + pressure_d)

# according to Ferruzzi et al 2013, dividing pressure and force by the experimental
# average gives better results
pressure_avg = pressure.mean()
force_avg = force.mean()

# arrange experimental data to feed the fitting algorithm
input_train = outer_diameter
output_train = [pressure/pressure_avg, force/force_avg]
sample_weights = np.ones(outer_diameter.shape[0], dtype=np.float32)

#=======================================================================================#
# Processing data to then compute pressure and force from stress functions
# unloaded dimensions
outer_diameter_0 = 0.394     #unloaded outer diameter
thickness_0 = 0.094          #unloaded thickness
length_0 = 4.66              #in-vitro axial length
vol_0 = np.pi * (0.25 * outer_diameter_0 ** 2 - (0.5 * outer_diameter_0 - thickness_0) ** 2) * length_0
inner_diameter_0 = outer_diameter_0 - 2.0*thickness_0

# in-vivo stretch, the test is performed at constant axial stretch
stretch_long = 1.72

#=======================================================================================#
# Self defined activation functions for exp term
def activation_exp(x):
    return 1.0*(tf.math.exp(x) - 1.0)

# Step 4: Build the Model
initializer_def = 'glorot_normal'
initializer_exp = keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=np.random.randint(0,10000)) # use random integer as seed

# Define input layer. layer 0
input1_layer = keras.Input(shape=(1,), name='I1')
input2_layer = keras.Input(shape=(1,), name='I2')
input4_layer = keras.Input(shape=(1,), name='I4')
input5_layer = keras.Input(shape=(1,), name='I5')
input_layer = [input1_layer, input2_layer, input4_layer, input5_layer]

idi = 0
all_invariants = []
for id1, invariant in enumerate(input_layer):
    # this would be layer 1
    if (invariant is input4_layer) or (invariant is input5_layer):
        layer1_lin = keras.layers.Lambda(lambda x: (x - 1.0), name='diff1'+str(id1+1)+'_lin')(invariant)
        layer1_sqr = keras.layers.Lambda(lambda x: tf.math.square(x - 1.0), name='diff1'+str(id1+1)+'_sqr')(invariant)
    else:
        layer1_lin = keras.layers.Lambda(lambda x: (x - 3.0), name='diff1'+str(id1+1)+'_lin')(invariant)
        layer1_sqr = keras.layers.Lambda(lambda x: tf.math.square(x - 3.0), name='diff1'+str(id1+1)+'_sqr')(invariant)

    # Define multiple dense layers. this would be layer 2
    collect = []
    for id2, layer1 in enumerate([layer1_lin,layer1_sqr]):
        dense_1 = keras.layers.Dense(1, kernel_initializer=initializer_def,
                                        kernel_constraint=keras.constraints.NonNeg(),
                                        kernel_regularizer=regularize(reg, pen),
                                        use_bias=False, activation=None,
                                        name='w1_'+str(idi+id2*2+1))(layer1)
        dense_2 = keras.layers.Dense(1, kernel_initializer=initializer_exp,
                                        kernel_constraint=keras.constraints.NonNeg(),
                                        kernel_regularizer=regularize(reg, pen),
                                        use_bias=False, activation=activation_exp,
                                        name='w1_'+str(idi+id2*2+2))(layer1)
        collect.append(dense_1)
        collect.append(dense_2)

    collect = keras.layers.concatenate(collect, axis=1)
    idi += collect.shape[1]
    all_invariants.append(collect)

# Concatenate the outputs of the dense layers
concatenated = keras.layers.concatenate(all_invariants, axis=1)
terms = concatenated.shape[1]

# Add an output layer
output_layer = keras.layers.Dense(1, kernel_initializer=initializer_def,
                                     kernel_constraint=keras.constraints.NonNeg(),
                                     kernel_regularizer=regularize(reg, pen),
                                     use_bias=False, activation=None, name='w2_x')(concatenated)

# Create the model
psi_model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='psi')

# Print the model summary
psi_model.summary()

with open(model_summary, 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))  # summarize layers in architecture

#=======================================================================================#
phi_0 = 0.2*np.pi
phi_f = tf.constant(phi_0, dtype='float32')
two = tf.constant(2.0, dtype='float32')
four = tf.constant(2.0, dtype='float32')
stretch2 = tf.constant(stretch_long, dtype='float32')

# Tension stress
def stress_circ_th(inputs):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, stretch1, I1) = inputs
    stretch3 = stretch1*stretch2
    minus  = two*(dPsidI1 + dPsidI2 * (I1 - backend.pow(stretch3, 2))) * backend.pow(stretch3, 2)
    stress = two*(dPsidI1 + dPsidI2*(I1 - backend.pow(stretch1, 2)) + two*dPsidI4*backend.cos(phi_f)*backend.cos(phi_f) + four*dPsidI5*backend.pow(stretch1,2)*backend.cos(phi_f)*backend.cos(phi_f))*backend.pow(stretch1,2) - minus
    return stress

def stress_long_th(inputs):
    (dPsidI1, dPsidI2, dPsidI4, dPsidI5, stretch1, I1) = inputs
    stretch3 = stretch1*stretch2
    minus  = two*(dPsidI1 + dPsidI2*(I1 - backend.pow(stretch3, 2))) * backend.pow(stretch3, 2)
    stress = two*(dPsidI1 + dPsidI2*(I1 - backend.pow(stretch2, 2)) + two*dPsidI4*backend.sin(phi_f)*backend.sin(phi_f) + four*dPsidI5*backend.pow(stretch2,2)*backend.sin(phi_f)*backend.sin(phi_f))*backend.pow(stretch2,2) - minus
    return stress

# Gradient function
def my_gradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

# outer diameter as an input
outer_diameter_tf = keras.layers.Input(shape=(1,), name='d_o')

# computed-measured quantities. we use length invivo because the test is at invivo stretch level
inner_diameter_tf = keras.layers.Lambda(lambda x: 2.0*tf.math.sqrt(0.25*x**2-vol_0/(np.pi*length_0*stretch_long)),
                                           name='d_i')(outer_diameter_tf)
thickness_tf = keras.layers.Lambda(lambda x: 0.5*(x[0]-x[1]),
                                      name='h')([outer_diameter_tf, inner_diameter_tf])

# stretch as input
stretch1 = keras.layers.Lambda(lambda x: (x[0]-x[1])/(outer_diameter_0-thickness_0),
                                  name='stretch1')([outer_diameter_tf, thickness_tf])

# specific Invariants in uni-axial tension
I1_biax = keras.layers.Lambda(lambda x: x[0]**2 + x[1]**2 + 1.0/(x[0]*x[1])**2,
                                 name='I1_biax')([stretch1, stretch2])
I2_biax = keras.layers.Lambda(lambda x: 1.0/x[0]**2 + 1.0/x[1]**2 + (x[0]*x[1])**2,
                                 name='I2_biax')([stretch1, stretch2])
I4_biax = keras.layers.Lambda(lambda x: (x[0]*tf.math.cos(phi_0))**2 + (x[1]*tf.math.sin(phi_0))**2,
                                 name='I4_biax')([stretch1, stretch2])
I5_biax = keras.layers.Lambda(lambda x: (tf.math.cos(phi_0)*x[0]**2)**2 + (tf.math.sin(phi_0)*x[1]**2)**2,
                                 name='I5_biax')([stretch1, stretch2])

# load specific models
psi_biax = psi_model([I1_biax, I2_biax, I4_biax, I5_biax])

# derivative uniaxial tension
dWdI1_biax = keras.layers.Lambda(lambda x: my_gradient(x[0], x[1]))([psi_biax, I1_biax])
dWdI2_biax = keras.layers.Lambda(lambda x: my_gradient(x[0], x[1]))([psi_biax, I2_biax])
dWdI4_biax = keras.layers.Lambda(lambda x: my_gradient(x[0], x[1]))([psi_biax, I4_biax])
dWdI5_biax = keras.layers.Lambda(lambda x: my_gradient(x[0], x[1]))([psi_biax, I5_biax])

# Stress circumferential tension
stress_circ_tf = keras.layers.Lambda(function=stress_circ_th,
                                        name='stress_circ')([dWdI1_biax, dWdI2_biax, dWdI4_biax, dWdI5_biax, stretch1, I1_biax])
# Stress longitudinal tension
stress_long_tf = keras.layers.Lambda(function=stress_long_th,
                                        name='stress_long')([dWdI1_biax, dWdI2_biax, dWdI4_biax, dWdI5_biax, stretch1, I1_biax])

# theoretical values of ordinates in P-d and f-P graphs, pressure and transducer force
pressure_tf = keras.layers.Lambda(lambda x: (2.0*x[0]*x[2]/x[1])/pressure_avg,
                                     name='pressure_th')([stress_circ_tf, inner_diameter_tf, thickness_tf])
force_tf = keras.layers.Lambda(lambda x: 1.0e-3*np.pi*x[3]*(x[1]*(x[2]+x[3])-0.5*x[0]*x[2])/force_avg,
                                  name='force_th')([stress_circ_tf, stress_long_tf, inner_diameter_tf, thickness_tf])

# Define model training for different load case
model_press = keras.models.Model(inputs=outer_diameter_tf, outputs=pressure_tf)
model_force = keras.models.Model(inputs=outer_diameter_tf, outputs=force_tf)

# Combined model
#model_inflation = tf.keras.models.Model(inputs=[model_circ.inputs, model_long.inputs], outputs=[model_circ.outputs, model_long.outputs])
model_inflation = keras.models.Model(inputs=outer_diameter_tf, outputs=[pressure_tf, force_tf])

model_inflation.summary()

with open(model_summary, 'a') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model_inflation.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))  # summarize layers in architecture

#=======================================================================================#
# Step 5: Compile the Model
# Compile the model
opti1 = tf.optimizers.Adam(learning_rate=0.001)
mse_loss = keras.losses.MeanSquaredError()
metrics = [keras.metrics.MeanSquaredError(), keras.metrics.MeanSquaredError()]
model_inflation.compile(optimizer=opti1, loss=mse_loss, metrics=metrics)

# Step 6: Train the Model
history = model_inflation.fit(input_train, output_train, batch_size=batch_size, epochs=epochs, validation_split=0.0, shuffle=True, verbose=0, sample_weight=sample_weights)

loss_history = history.history['loss']
if np.any(np.isnan(loss_history)):
    raise ValueError("History loss resulted in NaN, something may be wrong here.")

# Plot loss function
fig, axe = plt.subplots(figsize=[6, 5])  # inches
axe.plot(loss_history)
axe.set_yscale('log')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(path2scratch + '/plot_loss' + '.pdf')
#plt.show()
plt.close()

# Step 7: Evaluate the Model
# Make predictions
press_predict = model_press.predict(input_train)
force_predict = model_force.predict(input_train)

# Show weights (remember: weights are output in the order they are built)
weight_matrix = np.empty((terms, 2))
for i in range(terms):
    value = psi_model.get_weights()[i][0][0]
    weight_matrix[i, 0] = value                                               # inner layer is first column
weight_matrix[:, 1] = psi_model.get_layer('w2_x').get_weights()[0].flatten()  # outer layer is second column
print("weight_matrix")
print("| w1_x | w2_x |")
print(weight_matrix)

# Get the trained weights
model_weights = psi_model.get_weights()

# Plot the results
plt.scatter(input_train, output_train[0], label='circ data')
plt.plot(input_train, press_predict, color='red', label='fitted circ')
plt.legend()
plt.savefig(path2scratch + '/plot_fitting' + '.pdf')
#plt.show()
plt.close()

#=======================================================================================#
plt.rcParams['xtick.major.pad'] = 14 # set plotting parameters
plt.rcParams['ytick.major.pad'] = 14
# Plot the contributions of each term to the output of the model
fig, axt = plt.subplots(figsize=(12.5, 8.33))
num_terms = terms
cmap = plt.get_cmap('jet_r', num_terms)  # define the colormap with the number of terms from the full network
# this way, we can use 1 or 2 term models and have the colors be the same for those terms
cmaplist = [cmap(i) for i in range(cmap.N)]
# ax.set_xticks([1, 1.02, 1.04, 1.06, 1.08, 1.1])
#axt.set_xlim(1, 2.0)
# ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
#axt.set_ylim(0, 20.0)
# colormap
predictions = np.zeros([input_train.shape[0], terms])
model_plot = copy.deepcopy(model_weights)  # deep copy model weights
for i in range(terms):
    model_plot[-1] = np.zeros_like(model_weights[-1])  # w2_x all set to zero
    model_plot[-1][i] = model_weights[-1][i]  # w2_x[i] set to trained value

    psi_model.set_weights(model_plot)
    lower = np.sum(predictions, axis=1)
    upper = lower + model_press.predict(input_train, verbose=0)[:].flatten()
    predictions[:, i] = model_press.predict(input_train, verbose=0)[:].flatten()
    axt.fill_between(input_train, lower.flatten(), upper.flatten(), lw=0, zorder=i+1, color=cmaplist[i], label=str(i+1))
    axt.plot(input_train, upper, lw=0.4, zorder=34, color='k')

axt.scatter(input_train, output_train[0], s=200, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
plt.title('contributions w2_x')
plt.tight_layout(pad=2)
plt.savefig(path2scratch + '/plot_contributions' + '.pdf')
#plt.show()
plt.legend()
plt.close()

R2_circ = r2_score(output_train[0], press_predict)
R2_long = r2_score(output_train[1], force_predict)
print('R2 circumferential = ', R2_circ)
print('R2 longitudinal = ', R2_long)

# Save trained weights and R2 values to txt file
config = {'Penalty': pen, "R2_circ": R2_circ, "R2_long": R2_long, "weights": weight_matrix.tolist()}
json.dump(config, open(path2scratch + "/config_file.txt", 'w') )
