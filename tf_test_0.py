import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import copy
import os
import json
#from sys import exit

#=======================================================================================#
epochs = 1000
batch_size = 8

### Choose regularization type & penalty amount
# Option: 'L1', 'L2'
reg = 'L1'
pen = 0.001  # Use 0 for no regularization

#=======================================================================================#
path = "./" # change to where you download this
# setting working path
# Make path to save results to
def makeDIR(path):
    if not os.path.exists(path):
        os.makedirs(path)

filename = 'tf_test' # Change to keep track of different data e.g. Brain, Skin, Muscle, etc.
path2scratch = path + 'scratch/' + filename
makeDIR(path2scratch)

model_summary = path2scratch + '/model_summary.txt'

#=======================================================================================#
# L1 and L2 regularization with penalty weight
def regularize(reg, pen):
    if reg == 'L2':
        return tf.keras.regularizers.L2(pen)
    elif reg == 'L1':
        return tf.keras.regularizers.L1(pen)
    else:
        raise ValueError("Regularization type must be 'L1' or 'L2'")

# Self defined activation functions for exp term
def activation_exp(x):
    return 1.0*(tf.math.exp(x) - 1.0)
# Self defined activation functions for ln term
def activation_log(x):
    return -1.0*tf.math.log(1.0 - x)

#=======================================================================================#
# Step 3: Prepare Data
# Generate synthetic data
np.random.seed(0)
input_train = np.linspace(1.0, 2.0, 25).astype(np.float32)
output_train = (2.5 * (input_train ** 2 - 1.0 / input_train) + 0.1 * np.random.randn(*input_train.shape)).astype(np.float32)

sample_weights = np.array([1.0] * input_train.shape[0], dtype=np.float32)

# Demo: Plot tension and compression
plt.figure(figsize=(12.5,8.33))
plt.plot(input_train, output_train, color='blue')
plt.xlabel('stretch')
plt.ylabel('stress')
plt.tight_layout(pad=2)
#plt.show()
plt.savefig(path2scratch + '/raw_data' + '.pdf')
plt.close()

#=======================================================================================#
# Step 4: Build the Model
initializer_def = 'glorot_normal'
initializer_exp = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.1, seed=np.random.randint(0,10000)) # use random integer as seed

# Define input layer. layer 0
input_layer = tf.keras.Input(shape=(1,), name='I1')
# this would be layer 1
layer_1_1 = tf.keras.layers.Lambda(lambda x: (x - 3.0), name='diff1')(input_layer)
layer_1_2 = tf.keras.layers.Lambda(lambda x: tf.math.square(x - 3.0), name='diff2')(input_layer)

# Define multiple dense layers. this would be layer 2
idi = 0
dense_1 = tf.keras.layers.Dense(1, kernel_initializer=initializer_def,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=None, name='w1_'+str(1+idi))(layer_1_1)
dense_2 = tf.keras.layers.Dense(1, kernel_initializer=initializer_exp,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=activation_exp, name='w1_'+str(2+idi))(layer_1_1)
dense_3 = tf.keras.layers.Dense(1, kernel_initializer=initializer_def,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=activation_log, name='w1_'+str(3+idi))(layer_1_1)
dense_4 = tf.keras.layers.Dense(1, kernel_initializer=initializer_def,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=None, name='w1_'+str(4+idi))(layer_1_2)
dense_5 = tf.keras.layers.Dense(1, kernel_initializer=initializer_exp,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=activation_exp, name='w1_'+str(5+idi))(layer_1_2)
dense_6 = tf.keras.layers.Dense(1, kernel_initializer=initializer_def,
                                kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_regularizer=regularize(reg, pen),
                                use_bias=False, activation=activation_log, name='w1_'+str(6+idi))(layer_1_2)

# Concatenate the outputs of the dense layers
concatenated = tf.keras.layers.concatenate([dense_1, dense_2, dense_3, dense_4, dense_5, dense_6], axis=1)
terms = concatenated.shape[1]

# Add an output layer
output_layer = tf.keras.layers.Dense(1, kernel_initializer=initializer_def,
                                     kernel_constraint=tf.keras.constraints.NonNeg(),
                                     kernel_regularizer=regularize(reg, pen),
                                     use_bias=False, activation=None, name='w2_x')(concatenated)

# Create the model
psi_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer, name='psi')

# Print the model summary
psi_model.summary()

with open(model_summary, 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    psi_model.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))  # summarize layers in architecture

#=======================================================================================#
# Tension stress
def stress_calc_tension(inputs):
    (dPsidI1, stretch) = inputs
    one = tf.constant(1.0,dtype='float32')
    two = tf.constant(2.0,dtype='float32')
    minus  = two * (dPsidI1 * one/tf.keras.backend.square(stretch))
    stress = two * (dPsidI1 * stretch) - minus
    return stress

# Gradient function
def myGradient(a, b):
    der = tf.gradients(a, b, unconnected_gradients='zero')
    return der[0]

# stretch as input
stretch = tf.keras.layers.Input(shape=(1,), name='stretch')

# specific Invariants in uni-axial tension
I1_tension = tf.keras.layers.Lambda(lambda x: x**2 + 2.0/x)(stretch)

# load specific models
psi_tension = psi_model(I1_tension)

# derivative tension
dWdI1_tension = tf.keras.layers.Lambda(lambda x: myGradient(x[0], x[1]))([psi_tension, I1_tension])

# Stress tension
stress_tension = tf.keras.layers.Lambda(function=stress_calc_tension, name='stress_tension')([dWdI1_tension, stretch])

# Define model training for different load case
model_tension = tf.keras.models.Model(inputs=stretch, outputs=stress_tension)

#=======================================================================================#
# Step 5: Compile the Model
# Compile the model
opti1 = tf.optimizers.Adam(learning_rate=0.001)
mse_loss = tf.keras.losses.MeanSquaredError()
metrics = [tf.keras.metrics.MeanSquaredError()]
model_tension.compile(optimizer=opti1, loss=mse_loss, metrics=metrics)

# Step 6: Train the Model
history = model_tension.fit(input_train, output_train, batch_size=batch_size, epochs=epochs, validation_split=0.0, shuffle=True, verbose=0, sample_weight=sample_weights)

# Plot loss function
loss_history = history.history['loss']
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
output_predict = model_tension.predict(input_train)

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
plt.scatter(input_train, output_train, label='Data')
plt.plot(input_train, output_predict, color='red', label='Fitted line')
plt.legend()
plt.savefig(path2scratch + '/plot_fitting' + '.pdf')
#plt.show()
plt.close()

#=======================================================================================#
plt.rcParams['xtick.major.pad'] = 14 # set plotting parameters
plt.rcParams['ytick.major.pad'] = 14
# Plot the contributions of each term to the output of the model
fig, axt = plt.subplots(figsize=(12.5, 8.33))
num_terms = 6
cmap = plt.get_cmap('jet_r', num_terms)  # define the colormap with the number of terms from the full network
# this way, we can use 1 or 2 term models and have the colors be the same for those terms
cmaplist = [cmap(i) for i in range(cmap.N)]
# ax.set_xticks([1, 1.02, 1.04, 1.06, 1.08, 1.1])
axt.set_xlim(1, 2.0)
# ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
axt.set_ylim(0, 20.0)
# colormap
predictions = np.zeros([input_train.shape[0], terms])
model_plot = copy.deepcopy(model_weights)  # deep copy model weights
for i in range(terms):
    model_plot[-1] = np.zeros_like(model_weights[-1])  # w2_x all set to zero
    model_plot[-1][i] = model_weights[-1][i]  # w2_x[i] set to trained value

    psi_model.set_weights(model_plot)
    lower = np.sum(predictions, axis=1)
    upper = lower + model_tension.predict(input_train, verbose=0)[:].flatten()
    predictions[:, i] = model_tension.predict(input_train, verbose=0)[:].flatten()
    axt.fill_between(input_train, lower.flatten(), upper.flatten(), lw=0, zorder=i+1, color=cmaplist[i], label=str(i+1))
    axt.plot(input_train, upper, lw=0.4, zorder=34, color='k')

axt.scatter(input_train, output_train, s=200, zorder=103, lw=3, facecolors='w', edgecolors='k', clip_on=False)
plt.title('contributions w2_x')
plt.tight_layout(pad=2)
plt.savefig(path2scratch + '/plot_contributions' + '.pdf')
#plt.show()
plt.legend()
plt.close()

R2_t = r2_score(output_train, output_predict)
print('R2 tension = ', R2_t)

# Save trained weights and R2 values to txt file
config = {'Penalty': pen, "R2_t": R2_t, "weights": weight_matrix.tolist()}
json.dump(config, open(path2scratch + "/config_file.txt", 'w'))
